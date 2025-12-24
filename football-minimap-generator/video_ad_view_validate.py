import cv2
import numpy as np
import torch
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import LineString, box, Polygon
import time
import os
import math

# === 调试选项 ===
DEBUG_VISUALIZE = True
DEBUG_OUTPUT_DIR = 'debug_output_v2'

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
OUTPUT_CSV = 'ad_timeline_validate.csv'
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


def get_line_intersection(p1, p2, p3, p4):
    """计算直线(p1,p2)与直线(p3,p4)的交点"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None  # 平行线

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return (x, y)


def validate_view_geometry(homography, img_w, img_h):
    """
    校验视角有效性
    """
    # === 1. 采样关键点 ===
    # y_100: 底部 (1.0h)
    # y_75:  下 3/4 处 (0.75h)
    # y_50:  中部 (0.5h)
    y_100 = img_h
    y_75 = img_h * 0.75
    y_50 = img_h * 0.5

    uv_points = np.array([
        # Near Plane (Bottom)
        [0, y_100], [img_w, y_100],
        # Mid-Near Plane (3/4)
        [img_w / 2, y_75],  # 只取中点计算纵深
        # Mid Plane (1/2)
        [0, y_50], [img_w, y_50],
        [img_w / 2, y_50]  # 中点
    ], dtype=np.float32).reshape(-1, 1, 2)

    try:
        world_pts = cv2.perspectiveTransform(uv_points, homography).reshape(-1, 2)
    except Exception:
        return False, "PROJECTION_ERROR", None

    p_bl, p_br = world_pts[0], world_pts[1]  # Bottom Left/Right
    p_center_75 = world_pts[2]  # Center at 0.75h
    p_ml, p_mr = world_pts[3], world_pts[4]  # Mid Left/Right
    p_center_50 = world_pts[5]  # Center at 0.50h

    # 辅助: 计算两点距离
    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    # === 规则 1: 坐标数值稳定域 ===
    # 允许稍大的缓冲区，防止边缘误杀，但要拦截飞出天际的点
    valid_min_x, valid_max_x = -200, FIELD_LENGTH_YARDS + 200
    valid_min_y, valid_max_y = -200, FIELD_WIDTH_YARDS + 200
    for pt in world_pts:
        if not (valid_min_x < pt[0] < valid_max_x and valid_min_y < pt[1] < valid_max_y):
            return False, "REJECT_COORDS_OUT_OF_BOUNDS", {"corners": world_pts}

    # === 规则 2: 底边水平约束 ===
    # 主视角摄像机是横着拍球场的，所以图像底边(BL->BR)在Minimap上应该是横向的。
    # 也就是 X 轴跨度 应该远大于 Y 轴跨度。
    # 如果 dy > dx，说明变成了竖向切片（如 sec_0021, sec_0017）。
    dx_near = abs(p_bl[0] - p_br[0])
    dy_near = abs(p_bl[1] - p_br[1])

    # 阈值：dy 不能超过 dx 的 0.6 倍 (允许约30度倾斜，适应某些斜侧机位)
    # 很多错误特写 dy 甚至大于 dx (比例 > 1.0)
    if dx_near == 0 or (dy_near / dx_near > 0.6):
        return False, f"REJECT_BAD_ALIGNMENT (dy/dx={dy_near / max(1e-3, dx_near):.2f})", {"corners": world_pts}

    # === 规则 3: 物理尺度校验 ===
    # 图像底边覆盖的球场实际宽度不能太宽
    w_near = dist(p_bl, p_br)
    if w_near > 100.0:
        return False, f"REJECT_TOO_WIDE ({w_near:.1f}y)", {"corners": world_pts}

    # === 规则 4: 梯形扩张性校验 ===
    # 在世界坐标系下，图像中间（Mid）的视野宽度必须显著大于图像底部（Near）的视野宽度
    w_mid = dist(p_ml, p_mr)
    expansion_ratio = w_mid / w_near
    if expansion_ratio < 1.1:
        return False, f"REJECT_NO_EXPANSION (Ratio:{expansion_ratio:.2f})", {"corners": world_pts}

    # === 规则 5: 透视纵深梯度 (Perspective Depth Gradient) ===
    # 真正的摄像机看地面，越往远处看，同样的像素高度代表的物理距离越长。
    # Segment 1: 图像 100% -> 75% 的物理纵深 (近处)
    # Segment 2: 图像 75% -> 50% 的物理纵深 (远处)

    # 使用底边中点 (近似) 到 p_center_75 的距离
    p_center_100 = (p_bl + p_br) / 2
    depth_near_segment = dist(p_center_100, p_center_75)
    depth_far_segment = dist(p_center_75, p_center_50)

    # 正常透视：depth_far 应该明显大于 depth_near
    # 错误平面投影 (sec_0009)：两者几乎相等

    if depth_near_segment == 0: return False, "REJECT_ZERO_DEPTH", None

    depth_ratio = depth_far_segment / depth_near_segment

    # 阈值：1.2 (表示远处那 1/4 图像覆盖的距离至少比近处那 1/4 多 20%)
    # 正常视角通常 > 1.5，特写/错误平面投影通常 ~ 1.0
    if depth_ratio < 1.2:
        return False, f"REJECT_FLAT_PROJECTION (DepthRatio:{depth_ratio:.2f})", {"corners": world_pts}

    # === 规则 6: 摄像机位置校验 ===
    cam_pos = get_line_intersection(p_bl, p_ml, p_br, p_mr)

    if cam_pos is not None:
        cx, cy = cam_pos

        # A. 摄像机 X轴 归位约束
        # 主摄像机必须在两条底线之间。如果 cx 超出球场长度范围 (比如在角旗外侧)，
        # 说明是球门视角或错误的斜视 (如 sec_0013)。
        # 给予 -10 ~ +10 的容错
        if not (-10 < cx < FIELD_LENGTH_YARDS + 10):
            return False, f"REJECT_CAM_X_RANGE (x={cx:.1f})", {"corners": world_pts, "cam": cam_pos}

        # B. 摄像机 Y轴 场内检测
        if (-5 < cx < FIELD_LENGTH_YARDS + 5) and (-5 < cy < FIELD_WIDTH_YARDS + 5):
            return False, "REJECT_CAM_INSIDE_FIELD", {"corners": world_pts, "cam": cam_pos}

        # C. 摄像机距离限制
        center_x, center_y = FIELD_LENGTH_YARDS / 2, FIELD_WIDTH_YARDS / 2
        dist_to_center = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        if dist_to_center > 150:
            return False, "REJECT_CAM_TOO_FAR", {"corners": world_pts, "cam": cam_pos}

    else:
        # 平行线拒绝
        return False, "REJECT_PARALLEL_LINES", {"corners": world_pts}

    return True, "VALID", {"corners": [p_ml, p_mr, p_br, p_bl], "cam": cam_pos}


def check_ads_in_image_space(homography, ad_configs, img_w, img_h):
    """
    将广告牌从世界坐标投影回图像坐标进行判定。
    homography: Image -> World (Yards)
    """
    visible_sponsors = set()  # set 自动去重
    debug_draw_data = []

    try:
        H_inv = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return [], []

    image_rect = box(0, 0, img_w, img_h)

    for ad in ad_configs:
        world_pts = np.array(ad['coords'], dtype=np.float32).reshape(-1, 1, 2)

        # 投影到图像空间
        try:
            img_pts_trans = cv2.perspectiveTransform(world_pts, H_inv)
            p1_img = img_pts_trans[0][0]
            p2_img = img_pts_trans[1][0]

            line_img = LineString([p1_img, p2_img])

            is_visible = False
            if image_rect.intersects(line_img):
                is_visible = True
                visible_sponsors.add(ad['name'])

            debug_draw_data.append({
                "world_pts": ad['coords'],
                "name": ad['name'],
                "visible": is_visible
            })
        except Exception:
            continue

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

            # 默认状态
            status = "UNKNOWN"
            current_sponsors = []
            debug_data = []

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

                # === 2. 几何有效性校验 (新增步骤) ===
                h_img, w_img = img_bgr_resized.shape[:2]
                is_valid, reason, geom_info = validate_view_geometry(homography_np, w_img, h_img)
                status = reason

                if is_valid:
                    # === 3. 只有合法的帧才进行广告判定 ===
                    current_sponsors, debug_data = check_ads_in_image_space(
                        homography_np, ad_configs, w_img, h_img
                    )
                else:
                    # 无效帧，跳过广告检测，保留 debug_data 为空
                    current_sponsors = []

                # === 4. 调试可视化 ===
                if DEBUG_VISUALIZE:
                    # 使用 visualizer 生成带有白色视锥的底图
                    minimap_viz = create_minimap_view(img_rgb_resized, homography_np)

                    if minimap_viz is not None:
                        minimap_bgr = cv2.cvtColor(minimap_viz, cv2.COLOR_RGB2BGR)

                        # --- 绘制广告检测结果 (仅有效帧) ---
                        for item in debug_data:
                            world_pts = item["world_pts"]
                            map_pts = transform_points(world_pts, S_matrix)
                            pt1 = tuple(map_pts[0].astype(int))
                            pt2 = tuple(map_pts[1].astype(int))
                            color = (0, 255, 0) if item["visible"] else (0, 0, 255)
                            thickness = 3 if item["visible"] else 1
                            cv2.line(minimap_bgr, pt1, pt2, color, thickness)

                        # --- 绘制几何校验辅助信息 ---
                        # 无论有效无效，都尝试绘制推断出的摄像机位置，便于Debug
                        if geom_info and "cam" in geom_info and geom_info["cam"] is not None:
                            cam_world = np.array([geom_info["cam"]], dtype=np.float32)
                            cam_map = transform_points(cam_world, S_matrix)[0].astype(int)

                            # 绘制摄像机点
                            cam_color = (255, 0, 0) if is_valid else (0, 0, 255)  # 蓝=有效, 红=无效
                            cv2.circle(minimap_bgr, tuple(cam_map), 8, cam_color, -1)
                            cv2.putText(minimap_bgr, "CAM", tuple(cam_map), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 1)

                            # 绘制视锥连线 (Near corners -> Camera)
                            if "corners" in geom_info:
                                p_bl = geom_info["corners"][3]
                                p_br = geom_info["corners"][2]
                                p_bl_map = transform_points([p_bl], S_matrix)[0].astype(int)
                                p_br_map = transform_points([p_br], S_matrix)[0].astype(int)
                                cv2.line(minimap_bgr, tuple(cam_map), tuple(p_bl_map), cam_color, 1)
                                cv2.line(minimap_bgr, tuple(cam_map), tuple(p_br_map), cam_color, 1)

                        # 如果无效，在图上打大叉或写原因
                        if not is_valid:
                            h_map, w_map = minimap_bgr.shape[:2]
                            cv2.putText(minimap_bgr, f"INVALID: {reason}", (20, h_map - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            # 画个大红框
                            cv2.rectangle(minimap_bgr, (0, 0), (w_map - 1, h_map - 1), (0, 0, 255), 5)

                        # 保存
                        h_map = minimap_bgr.shape[0]
                        scale_img = h_map / img_bgr_resized.shape[0]
                        img_resized_disp = cv2.resize(img_bgr_resized, (0, 0), fx=scale_img, fy=scale_img)
                        combined = np.hstack([img_resized_disp, minimap_bgr])
                        cv2.imwrite(f"{DEBUG_OUTPUT_DIR}/sec_{second_int:04d}_check.jpg", combined)

                # === 5. 记录结果 ===
                existing_entry = next((item for item in results if item['second'] == second_int), None)
                if existing_entry:
                    # 如果有重复帧，合并逻辑 (仅在状态都为VALID时合并广告，否则保留错误状态)
                    if status == "VALID" and existing_entry['status'] == "VALID":
                        existing_set = set(existing_entry['sponsors_list'])
                        existing_set.update(current_sponsors)
                        existing_entry['sponsors_list'] = list(existing_set)
                    elif status != "VALID":
                        # 如果新的一帧无效，覆盖为无效？或者保留之前的有效？
                        # 策略：优先保留有效帧。如果都是无效，保留最后的。
                        if existing_entry['status'] == "VALID":
                            pass  # 已经有一帧有效了，忽略当前的无效帧
                        else:
                            existing_entry['status'] = status
                else:
                    results.append({
                        'second': second_int,
                        'timestamp': time.strftime('%H:%M:%S', time.gmtime(second_int)),
                        'status': status,
                        'sponsors_list': current_sponsors
                    })

            except Exception as e:
                print(f"Frame {second_int} processing error: {e}")
                pass

        current_frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # === 保存 CSV (包含 Status 列) ===
    csv_rows = []
    for row in results:
        sponsors_str = ", ".join(sorted(row['sponsors_list'])) if row['status'] == 'VALID' else ""
        csv_rows.append({
            'second': row['second'],
            'timestamp': row['timestamp'],
            'status': row['status'],
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
