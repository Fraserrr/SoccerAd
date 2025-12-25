import cv2
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# ================= 引入 SAM 3 模块 =================
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ================= 配置区域 =================
VIDEO_PATH = 'input_video/cut_video.mp4'  # 视频路径
CSV_PATH = 'ad_timeline_validate.csv'  # CSV 文件路径
OUTPUT_DIR = 'sam3_results'  # 结果保存路径
CROPS_DIR = 'crops'  # Crop保存路径
TEXT_PROMPT = "Ad banner"  # 文本提示词
SAMPLE_FPS = 1.0  # 采样率
SCORE_THRESHOLD = 0.25  # 结果置信度阈值
# 自动检测设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================

def show_mask(mask, ax, random_color=False):
    """
    绘制 Mask：半透明填充 + 不透明边缘描边
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    # 确保 mask 维度正确 (H, W)
    if mask.ndim == 3:
        mask = mask.squeeze()

    h, w = mask.shape[-2:]

    # --- 1. 半透明填充 ---
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

    # --- 2. 边缘描边 (不透明) ---
    # 需要将 mask 转为 uint8 才能用于 findContours
    # 假设 mask 是 bool 或 0/1 float，先转换
    mask_uint8 = mask.astype(np.uint8)

    # 获取轮廓
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 提取不透明颜色 (去掉 Alpha 通道)
    stroke_color = color[:3]

    for contour in contours:
        # contour shape: (N, 1, 2) -> 压缩为 (N, 2) 方便绘图
        # 如果轮廓点太少可能无法绘制，加个简单判断
        if len(contour) > 2:
            path = contour.squeeze()
            # 注意：OpenCV 轮廓坐标是 (x, y)，直接 plot 即可
            # 如果 path 是一维的 (比如只有3个点)，可能需要 reshape，但 squeeze 通常够用
            if path.ndim == 1:
                path = path.reshape(-1, 2)

            # 闭合轮廓：将最后一个点和第一个点连起来
            path = np.vstack([path, path[0]])

            ax.plot(path[:, 0], path[:, 1], color=stroke_color, linewidth=2)


def show_box(box, ax):
    """
    在 Matplotlib axis 上绘制边界框。
    Box 格式: [x1, y1, x2, y2]
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor='none', lw=2))


def process_video_with_sam3():
    """
    主处理流程：读取 CSV -> 定位视频帧 -> SAM 3 推理 -> 保存可视化结果
    """
    # 1. 准备输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[Info] 创建输出目录: {OUTPUT_DIR}")

    # 2. 读取并筛选 CSV
    print(f"[Info] 正在读取 CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 筛选无效帧
    invalid_df = df[df['status'] != 'VALID'].copy()
    if invalid_df.empty:
        print("[Info] 未发现需要处理的帧。")
        return

    # 按时间排序优化 IO
    invalid_df = invalid_df.sort_values(by='second')
    print(f"[Info] 待处理总秒数: {len(invalid_df)}")

    # 3. 初始化 SAM 3 模型
    print(f"[Info] 正在加载 SAM 3 模型 (Device: {DEVICE})...")
    try:
        model = build_sam3_image_model()
        model.to(DEVICE)
        processor = Sam3Processor(model)
        print("[Info] 模型加载成功。")
    except Exception as e:
        print(f"[Error] 模型加载失败: {e}")
        return

    # 4. 打开视频资源
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Error] 无法打开视频: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0

    # 5. 开始推理循环
    print(f"[Info] 开始处理，提示词: '{TEXT_PROMPT}'")

    for _, row in tqdm(invalid_df.iterrows(), total=len(invalid_df), desc="Inference"):
        target_second = row['second']
        timestamp_str = row['timestamp']

        # 定位帧
        frame_index = int(target_second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            continue

        # --- 核心图像预处理 (Strict Compliance with Official Demo) ---
        # 1. BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 2. Numpy -> PIL Image
        # 3. .convert("RGB"): 显式强制转换为3通道，防止任何潜在的格式不匹配报错
        image_pil = Image.fromarray(frame_rgb).convert("RGB")

        # --- SAM 3 推理 ---
        try:
            # Step 1: 设置图像上下文
            inference_state = processor.set_image(image_pil)

            # Step 2: 文本提示分割
            output = processor.set_text_prompt(state=inference_state, prompt=TEXT_PROMPT)

            # Step 3: 提取结果
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]

        except Exception as e:
            print(f"[Error] 推理出错 ({timestamp_str}): {e}")
            continue

        # --- 数据转换 (Tensor -> Numpy) ---
        if isinstance(masks, torch.Tensor):
            masks_np = masks.cpu().detach().numpy()
            boxes_np = boxes.cpu().detach().numpy()
            scores_np = scores.cpu().detach().numpy()
        else:
            masks_np, boxes_np, scores_np = masks, boxes, scores

        # --- 可视化与保存 ---
        if len(masks_np) > 0:
            found_valid_obj = False

            # 1. 准备 Crop 保存目录
            if not os.path.exists(CROPS_DIR):
                os.makedirs(CROPS_DIR)

            # 创建画布 (用于整帧可视化)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(image_pil)
            # 去除坐标轴和边距
            ax.set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

            # 将 PIL 转为 Numpy 方便进行 Crop 的遮罩处理
            full_img_np = np.array(image_pil)

            for i, score in enumerate(scores_np):
                if score > SCORE_THRESHOLD:
                    # --- A. 可视化绘制 ---
                    show_mask(masks_np[i], ax, random_color=True)
                    found_valid_obj = True

                    # --- B. Crop 裁剪与处理 ---
                    # 获取 Box 坐标 (x1, y1, x2, y2) 并取整
                    x1, y1, x2, y2 = boxes_np[i].astype(int)

                    # 边界安全检查 (防止越界)
                    h_img, w_img = full_img_np.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)

                    if x2 > x1 and y2 > y1:
                        # 1. 裁剪原图
                        crop_img = full_img_np[y1:y2, x1:x2]

                        # 2. 裁剪 Mask (注意 Mask 维度可能需要处理)
                        # masks_np[i] 可能是 (H, W) 或 (1, H, W)
                        mask_full = masks_np[i]
                        if mask_full.ndim == 3: mask_full = mask_full.squeeze()
                        crop_mask = mask_full[y1:y2, x1:x2]

                        # 3. 应用 Mask (背景填充纯黑)
                        # 创建纯黑背景
                        masked_crop = np.zeros_like(crop_img)
                        # 仅在 Mask 为 True 的地方复制原图像素
                        # expand_dims 是为了匹配 RGB 通道
                        masked_crop[crop_mask > 0] = crop_img[crop_mask > 0]

                        # 4. 保存 Crop
                        # 命名格式: 秒数_时间戳_索引.jpg
                        crop_filename = f"{target_second:05d}_{timestamp_str.replace(':', '-')}_idx{i}.jpg"
                        crop_save_path = os.path.join(CROPS_DIR, crop_filename)

                        # 使用 PIL 保存 (OpenCV 需要转 BGR，PIL 直接存 RGB)
                        Image.fromarray(masked_crop).save(crop_save_path)

            # 仅保存包含有效目标的结果
            if found_valid_obj:
                save_name = f"{target_second:05d}_{timestamp_str.replace(':', '-')}_result.jpg"
                save_path = os.path.join(OUTPUT_DIR, save_name)
                # 修改关键点: pad_inches=0 去除白边
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

            # --- 清理资源 ---
            plt.close(fig)

            # 清理推理状态
            del inference_state, output, masks, boxes, scores

    cap.release()
    print("[Info] 全部处理完成。")


if __name__ == "__main__":
    # Windows 下多进程保护
    torch.multiprocessing.set_start_method('spawn', force=True)
    process_video_with_sam3()