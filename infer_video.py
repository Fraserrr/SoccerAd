import argparse
import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from model import DINOv3Segmentation, hidden_size
from config import ALL_CLASSES
from utils import (
    image_overlay,
    get_segment_labels,
    safe_torch_load,
    calculate_dinov3_dimensions
)


def process_mask_to_fill_holes(mask, width, height):
    """
    针对广告牌边缘检测的后处理填充算法。

    Args:
        mask: 原始预测的 mask (0/1矩阵), numpy array
        width: 视频宽
        height: 视频高
    Returns:
        filled_mask: 填充后的 mask
    """
    # 1. 调整到原始尺寸进行处理 (保证形态学操作的尺度对应实际像素)
    # 使用 INTER_NEAREST 保持二值特性
    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # 2. 定义形态学操作核
    # 策略：广告牌通常是水平长条状。
    # kernel_h: 横向膨胀核。(宽, 高)。用于连接水平方向断裂的像素。
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))

    # kernel_v: 纵向闭运算核。(宽, 高)。
    # 用于连接广告牌的上下边缘。如果上下边缘距离超过40像素，可能需要调大这个参数。
    # 这里的 (5, 40) 是一个经验值，根据视频分辨率可能需要调整（比如1080p下可能需要60-80）
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 35))

    # 3. 形态学操作
    # 步骤A: 横向膨胀，把碎点连成横线
    mask_processed = cv2.dilate(mask_resized, kernel_h, iterations=1)

    # 步骤B: 纵向闭运算，尝试“桥接”上下边缘
    mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel_v)

    # 4. 轮廓查找与凸包填充
    # 寻找外轮廓 (RETR_EXTERNAL 忽略内部的小洞，只看最外层)
    cnts, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_mask = np.zeros_like(mask_resized)

    for c in cnts:
        # 过滤噪声：面积太小的忽略 (例如误检的球员球鞋)
        if cv2.contourArea(c) < 3000:
            continue

        # 核心逻辑：计算凸包
        # 凸包就像用橡皮筋包住这些点，能完美填充上下边缘之间的空隙
        hull = cv2.convexHull(c)

        # 绘制填充的凸包 (颜色为1)
        cv2.drawContours(filled_mask, [hull], -1, 1, -1)

    return filled_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input/videos/short_cut.mp4', help='path to input video file')
    parser.add_argument('--output', default='outputs/short_result_filled_3.mp4', help='path to save output video')
    parser.add_argument('--model', default='outputs/kaggle_model_896/best_model_iou.pth', help='path to trained model')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[896, 896], help='inference size (width height)')
    parser.add_argument('--device', default='cuda:0', help='cuda or cpu')
    args = parser.parse_args()

    # 2. 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"🚀 Using device: {device}")

    # 3. 加载模型
    model = DINOv3Segmentation()
    # 重新初始化分割头
    model.decode_head.conv_seg = nn.Conv2d(hidden_size, len(ALL_CLASSES), kernel_size=(1, 1))

    print(f"📥 Loading model from {args.model}...")
    ckpt = safe_torch_load(args.model, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    # 4. 视频设置
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"❌ Error opening video file {args.input}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算推理尺寸
    req_w, req_h = args.imgsz[0], args.imgsz[1]
    infer_w, infer_h = calculate_dinov3_dimensions(req_w, req_h)
    print(f"📏 Inference size: {infer_w}x{infer_h} (Original: {width}x{height})")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("🎬 Starting inference with Region Filling...")
    pbar = tqdm(total=total_frames)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理
        frame_resized = cv2.resize(frame, (infer_w, infer_h))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # 推理
        current_size = [infer_w, infer_h]
        labels = get_segment_labels(frame_rgb, model, device, current_size)

        # --- 维度处理 ---
        labels = labels.squeeze()
        mask = labels.cpu().numpy().astype(np.uint8)
        while mask.ndim > 2:
            mask = mask[0]

        # ---------------------------------------------------------
        # 🛠️ 智能填充算法
        # 直接在 mask 层级进行填充，而不是 resize 之后再画框
        # 注意：这里传入 0/1 的 mask，索引 1 是广告牌类别
        # ---------------------------------------------------------

        # 提取广告牌 mask (类别索引 1 是广告牌，如果是其他类别请修改此处)
        binary_mask = (mask == 1).astype(np.uint8)

        # 如果画面中有检测到内容才处理
        if np.any(binary_mask):
            # 调用填充函数，得到填充后的完整 Mask (尺寸为 width x height)
            filled_mask = process_mask_to_fill_holes(binary_mask, width, height)

            # 生成绿色遮罩
            color_mask = np.zeros_like(frame)
            mask_bool = (filled_mask == 1)

            # 应用遮罩
            if np.any(mask_bool):
                color_mask[mask_bool] = [0, 255, 0]  # 绿色
                alpha = 0.5
                frame[mask_bool] = cv2.addWeighted(frame[mask_bool], 1 - alpha, color_mask[mask_bool], alpha, 0)

        # ---------------------------------------------------------

        out.write(frame)
        pbar.update(1)
        frame_count += 1

    cap.release()
    out.release()
    print(f"\n✅ Video saved to {args.output}")


if __name__ == '__main__':
    main()
