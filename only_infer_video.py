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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input/videos/short_cut.mp4', help='path to input video file')
    parser.add_argument('--output', default='outputs/short_result.mp4', help='path to save output video')
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

    print("🎬 Starting inference...")
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

        # --- 🛠️ 关键修复：强力降维 ---
        # 1. 移除所有维度为 1 的轴 (Batch, Channel)
        # 例如: (1, 1, 512, 512) -> (512, 512)
        labels = labels.squeeze()

        # 2. 转为 Numpy
        mask = labels.cpu().numpy().astype(np.uint8)

        # 3. 防御性编程：确保 Mask 绝对是 2D 的 (H, W)
        # 如果因为某些奇怪的原因它还是 3D (例如 C > 1)，我们强行取第一个通道
        while mask.ndim > 2:
            mask = mask[0]

        # 4. Debug 打印 (只在第一帧显示，确认形状)
        if frame_count == 0:
            tqdm.write(f"🔍 Debug - Mask Shape: {mask.shape}, Dtype: {mask.dtype}")
        # ---------------------------

        # 后处理：还原尺寸
        # 现在 mask 必然是 (512, 512) 这种 2D 矩阵，cv2.resize 不会再报错了
        mask_original = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # 生成绿色遮罩
        color_mask = np.zeros_like(frame)
        # 假设类别 1 是广告牌
        mask_bool = (mask_original == 1)

        if np.any(mask_bool):
            color_mask[mask_bool] = [0, 255, 0]  # 绿色
            alpha = 0.5
            frame[mask_bool] = cv2.addWeighted(frame[mask_bool], 1 - alpha, color_mask[mask_bool], alpha, 0)

        out.write(frame)
        pbar.update(1)
        frame_count += 1

    cap.release()
    out.release()
    print(f"\n✅ Video saved to {args.output}")


if __name__ == '__main__':
    main()