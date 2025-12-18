import cv2
import numpy as np
import torch
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import LineString, box
import time
import os

# === 调试选项 ===
DEBUG_VISUALIZE = True
DEBUG_OUTPUT_DIR = 'debug_output'

# === 导入工程模块 ===
try:
    from tvcalib.infer.module import TvCalibInferModule
    from main import preprocess_image_tvcalib, IMAGE_SHAPE, SEGMENTATION_MODEL_PATH
    from visualizer import create_minimap_view
except ImportError as e:
    print("请确保在工程根目录下运行，并能访问 tvcalib, visualizer 等模块")
    raise e

# === 常量 (必须与 ad_config_builder.py 和 visualizer.py 一致) ===
FIELD_LENGTH_YARDS = 114.83
FIELD_WIDTH_YARDS = 74.37
EXPECTED_H, EXPECTED_W = 720, 1280

# === 配置 ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG_FILE = 'ad_map_config.json'
OUTPUT_CSV = 'ad_timeline.csv'
SAMPLE_FPS = 1.0
OPTIM_STEPS = 100


def load_ad_config():
    if not Path(CONFIG_FILE).exists():
        raise FileNotFoundError(f"找不到 {CONFIG_FILE}，请先运行 ad_config_builder.py")
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_minimap_metrics(minimap_size=(1280, 720)):
    """获取 World(Yards) -> Minimap(Pixels) 的变换矩阵 S"""
    minimap_h, minimap_w = minimap_size[1], minimap_size[0]

    scale_x = minimap_w / FIELD_LENGTH_YARDS
    scale_y = minimap_h / FIELD_WIDTH_YARDS
    scale = min(scale_x, scale_y) * 0.9

    field_width_px = int(FIELD_WIDTH_YARDS * scale)
    field_length_px = int(FIELD_LENGTH_YARDS * scale)
    offset_x = (minimap_w - field_length_px) // 2
    offset_y = (minimap_h - field_width_px) // 2

    S = np.array([
        [scale, 0, offset_x],
        [0, scale, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)
    return S


def check_ads_in_image_space(homography, ad_configs, img_w, img_h):
    """
    将广告牌从世界坐标投影回图像坐标进行判定。
    homography: Image -> World (Yards)
    """
    visible_sponsors = set()  # set 自动去重
    debug_draw_data = []

    # 1. 计算逆矩阵 H_inv: World (Yards) -> Image (Pixels)
    try:
        H_inv = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return [], []

    # 2. 定义图像的矩形区域 (Shapely Box)
    image_rect = box(0, 0, img_w, img_h)

    for ad in ad_configs:
        # ad['coords'] 是 World Coordinates (Yards) [(x1,y1), (x2,y2)]
        world_pts = np.array(ad['coords'], dtype=np.float32).reshape(-1, 1, 2)

        # 3. 投影到图像空间
        img_pts_trans = cv2.perspectiveTransform(world_pts, H_inv)
        p1_img = img_pts_trans[0][0]
        p2_img = img_pts_trans[1][0]

        # 4. 构建图像空间的线段
        line_img = LineString([p1_img, p2_img])

        # 5. 判定：投影后的线段是否与图像矩形相交
        is_visible = False
        if image_rect.intersects(line_img):
            is_visible = True
            visible_sponsors.add(ad['name'])

        # 存储原始的世界坐标，用于后续在 Minimap 上绘制 Debug 结果
        debug_draw_data.append({
            "world_pts": ad['coords'],
            "name": ad['name'],
            "visible": is_visible
        })

    return list(visible_sponsors), debug_draw_data


def transform_points(points, Matrix):
    pts_arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts_arr, Matrix)
    return transformed.reshape(-1, 2)


def analyze_video(video_path):
    print(f"开始分析视频: {video_path}")
    if DEBUG_VISUALIZE:
        os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

    ad_configs = load_ad_config()
    print(f"已加载 {len(ad_configs)} 个广告牌配置")

    # 准备 Minimap 变换矩阵 S
    S_matrix = get_minimap_metrics((EXPECTED_W, EXPECTED_H))

    print("初始化定标模型...")
    model = TvCalibInferModule(
        segmentation_checkpoint=SEGMENTATION_MODEL_PATH,
        image_shape=IMAGE_SHAPE,
        optim_steps=OPTIM_STEPS,
        lens_dist=False
    )
    model_device = next(model.model_calib.parameters()).device

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps / SAMPLE_FPS)
    if frame_interval < 1: frame_interval = 1

    results = []
    pbar = tqdm(total=total_frames)

    current_frame_idx = 0
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if current_frame_idx % frame_interval == 0:
            second_int = int(current_frame_idx / fps)

            try:
                # === 1. 推理 ===
                image_tensor, img_bgr_resized, img_rgb_resized = preprocess_image_tvcalib(frame_bgr)
                image_tensor = image_tensor.to(model_device)

                with torch.no_grad():
                    keypoints = model._segment(image_tensor)

                # Homography: Image(Pixel) -> World(Yards)
                homography = model._calibrate(keypoints)

                if isinstance(homography, torch.Tensor):
                    homography_np = homography.detach().cpu().numpy()
                else:
                    homography_np = np.array(homography)

                # === 2. 核心逻辑：Image 空间求交 ===
                h_img, w_img = img_bgr_resized.shape[:2]
                current_sponsors, debug_data = check_ads_in_image_space(
                    homography_np, ad_configs, w_img, h_img
                )

                # === 3. 调试可视化 ===
                if DEBUG_VISUALIZE:
                    # 使用 visualizer 生成带有白色视锥的底图
                    minimap_viz = create_minimap_view(img_rgb_resized, homography_np)

                    if minimap_viz is not None:
                        minimap_bgr = cv2.cvtColor(minimap_viz, cv2.COLOR_RGB2BGR)

                        # 在 Minimap 上绘制结果 (利用 debug_data 中的世界坐标 + S 矩阵)
                        for item in debug_data:
                            world_pts = item["world_pts"]
                            # 将 World 坐标转为 Minimap 坐标进行绘制
                            map_pts = transform_points(world_pts, S_matrix)

                            pt1 = tuple(map_pts[0].astype(int))
                            pt2 = tuple(map_pts[1].astype(int))

                            # 绿色=可见，红色=不可见
                            color = (0, 255, 0) if item["visible"] else (0, 0, 255)
                            thickness = 3 if item["visible"] else 1

                            # 只有当线段在 Minimap 范围内才绘制，避免绘制溢出
                            cv2.line(minimap_bgr, pt1, pt2, color, thickness)

                            # if item["visible"]:
                            #     mid_pt = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                            #     # 稍微偏移文字以免重叠
                            #     text_pos = (mid_pt[0] + 5, mid_pt[1] + 5)
                            #     cv2.putText(minimap_bgr, item["name"], text_pos,
                            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                        # 保存
                        h_map = minimap_bgr.shape[0]
                        scale_img = h_map / img_bgr_resized.shape[0]
                        img_resized_disp = cv2.resize(img_bgr_resized, (0, 0), fx=scale_img, fy=scale_img)
                        combined = np.hstack([img_resized_disp, minimap_bgr])
                        cv2.imwrite(f"{DEBUG_OUTPUT_DIR}/sec_{second_int:04d}_check.jpg", combined)

                # === 4. 记录结果 ===
                existing_entry = next((item for item in results if item['second'] == second_int), None)
                if existing_entry:
                    existing_set = set(existing_entry['sponsors_list'])
                    existing_set.update(current_sponsors)
                    existing_entry['sponsors_list'] = list(existing_set)
                else:
                    results.append({
                        'second': second_int,
                        'timestamp': time.strftime('%H:%M:%S', time.gmtime(second_int)),
                        'sponsors_list': current_sponsors
                    })

            except Exception as e:
                print(f"Frame {second_int} processing error: {e}")
                pass

        current_frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # === 保存 CSV ===
    csv_rows = []
    for row in results:
        sponsors_str = ", ".join(sorted(row['sponsors_list']))
        csv_rows.append({
            'second': row['second'],
            'timestamp': row['timestamp'],
            'sponsors': sponsors_str
        })

    df = pd.DataFrame(csv_rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"完成！结果已保存至 {OUTPUT_CSV}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to MP4 video')
    args = parser.parse_args()

    analyze_video(args.video)