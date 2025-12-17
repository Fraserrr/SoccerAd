import argparse
import cv2
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

# 你的项目依赖
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
    智能填充算法
    """
    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    kv_height = int(height * 0.08)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, kv_height))
    kh_width = int(width * 0.05)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kh_width, 3))

    mask_processed = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel_v)
    mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel_h)

    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_processed = cv2.dilate(mask_processed, kernel_smooth, iterations=1)

    cnts, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask_resized)

    for c in cnts:
        if cv2.contourArea(c) < 3000: continue
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(filled_mask, [approx], -1, 1, -1)

    return filled_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input/videos/more_video.mp4', help='Input video path')
    parser.add_argument('--output_video', default='outputs/viz_videos/more_result.mp4', help='Output visualization video')
    parser.add_argument('--crops_dir', default='outputs/crops_more', help='Directory to save crop images')
    parser.add_argument('--model', default='outputs/kaggle_model_896/best_model_iou.pth', help='Path to trained model')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[896, 896], help='Inference size')
    parser.add_argument('--device', default='cuda:0', help='Device')
    # 注意：这里每一帧都推理并保存裁剪，OCR 阶段再决定要不要采样，这样灵活性最大
    # 后续可以修改实现隔帧采样
    args = parser.parse_args()

    # 0. 准备目录
    os.makedirs(args.output_video, exist_ok=True)
    os.makedirs(args.crops_dir, exist_ok=True)

    # 1. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"🚀 [Step 1] DINOv3 running on {device}")

    model = DINOv3Segmentation()
    model.decode_head.conv_seg = nn.Conv2d(hidden_size, len(ALL_CLASSES), kernel_size=(1, 1))
    ckpt = safe_torch_load(args.model, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    # 2. 视频设置
    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    req_w, req_h = args.imgsz[0], args.imgsz[1]
    infer_w, infer_h = calculate_dinov3_dimensions(req_w, req_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    metadata = []  # 记录裁剪图的信息

    print("🎬 Starting detection & extraction...")
    pbar = tqdm(total=total_frames)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- DINOv3 推理 ---
        frame_resized = cv2.resize(frame, (infer_w, infer_h))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        labels = get_segment_labels(frame_rgb, model, device, [infer_w, infer_h])

        labels = labels.squeeze()
        mask = labels.cpu().numpy().astype(np.uint8)
        while mask.ndim > 2: mask = mask[0]
        binary_mask = (mask == 1).astype(np.uint8)

        # 默认写入原始帧，如果有 mask 再覆盖绿色
        frame_viz = frame.copy()

        if np.any(binary_mask):
            filled_mask = process_mask_to_fill_holes(binary_mask, width, height)

            # 1. 绘制可视化 (绿色遮罩)
            mask_bool = (filled_mask == 1)
            if np.any(mask_bool):
                color_mask = np.zeros_like(frame_viz)
                color_mask[mask_bool] = [0, 255, 0]
                frame_viz[mask_bool] = cv2.addWeighted(frame_viz[mask_bool], 0.5, color_mask[mask_bool], 0.5, 0)

            # 2. 裁剪并保存 (Masked Crop)
            cnts, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            crop_count = 0
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w < 30 or h < 15: continue  # 忽略过小的

                pad = 5
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(width, x + w + pad), min(height, y + h + pad)

                roi_img = frame[y1:y2, x1:x2]
                roi_mask = filled_mask[y1:y2, x1:x2]

                # 关键：背景涂黑
                masked_roi = cv2.bitwise_and(roi_img, roi_img, mask=roi_mask)

                # 保存图片
                filename = f"frame_{frame_idx:06d}_crop_{crop_count}.jpg"
                save_path = os.path.join(args.crops_dir, filename)
                cv2.imwrite(save_path, masked_roi)

                metadata.append({
                    'frame_index': frame_idx,
                    'second': int(frame_idx / fps),
                    'crop_path': save_path
                })
                crop_count += 1

        out.write(frame_viz)
        pbar.update(1)
        frame_idx += 1

    cap.release()
    out.release()

    # 保存元数据
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(args.crops_dir, 'metadata.csv'), index=False)
    print(f"\n✅ Detection done! Video saved to {args.output_video}")
    print(f"✅ Crops saved to {args.crops_dir} (Total crops: {len(df)})")


if __name__ == '__main__':
    main()
