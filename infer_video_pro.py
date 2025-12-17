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
    改进版填充算法：移除凸包，使用形态学闭运算和多边形拟合。
    保留广告牌的透视形状和凹陷区域，避免包裹背景。
    """
    # 1. 调整到原始尺寸
    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # 2. 定义形态学操作核
    # 策略调整：不再依赖凸包来填充，而是通过“闭运算”让空心的框变成实心的条

    # 纵向核 (Kernel Vertical): 关键参数
    # 高度设得较大 (例如 50-80)，用于填满广告牌上下边缘之间的空隙
    # 宽度设得较小 (例如 3-5)，防止左右方向误连
    kv_height = int(height * 0.08)  # 动态计算，约为屏幕高度的 8%
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, kv_height))

    # 横向核 (Kernel Horizontal):
    # 用于连接横向断裂的文字或纹理
    kh_width = int(width * 0.05)  # 动态计算，约为屏幕宽度的 5%
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kh_width, 3))

    # 3. 形态学操作流程
    # 步骤A: 纵向闭运算 (Closing)
    # 这一步是核心：它会将上下两根线“吸”在一起，变成实心区域，但不会改变左右轮廓
    mask_processed = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel_v)

    # 步骤B: 横向闭运算
    # 连接断开的段落
    mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel_h)

    # 步骤C: 稍微膨胀一点点，弥补边缘的锯齿
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_processed = cv2.dilate(mask_processed, kernel_smooth, iterations=1)

    # 4. 轮廓查找与多边形拟合 (替代凸包)
    cnts, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask_resized)

    for c in cnts:
        # 过滤噪声
        if cv2.contourArea(c) < 3000:
            continue

        # --- 核心修改：使用 approxPolyDP 替代 convexHull ---
        # epsilon 是拟合精度，值越小越贴合原轮廓，值越大越平滑
        # 0.005 * 周长 是一个经验值，既能保持直线特征，又能保留弯曲/透视变化
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        # 绘制填充的多边形
        # 注意：这里直接画 approx，它允许凹多边形 (Concave)，
        # 所以草坪如果本来就没被形态学卷进去，这里也不会被画进去
        cv2.drawContours(filled_mask, [approx], -1, 1, -1)

    return filled_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input/videos/cut_video.mp4', help='path to input video file')
    parser.add_argument('--output', default='outputs/cut_result_filled_4.mp4', help='path to save output video')
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
