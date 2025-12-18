import cv2
import numpy as np
import torch
from pathlib import Path
import time
import traceback
from ultralytics import YOLO

# 导入其他模块
try:
    from tvcalib.infer.module import TvCalibInferModule
    from main import preprocess_image_tvcalib, IMAGE_SHAPE, SEGMENTATION_MODEL_PATH, YOLO_MODEL_PATH, BALL_CLASS_INDEX
    from visualizer import (
        create_minimap_view,
        create_minimap_with_offset_skeletons,
        DYNAMIC_SCALE_MIN_MODULATION,
        DYNAMIC_SCALE_MAX_MODULATION
    )
    from pose_estimator import get_player_data
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保tvcalib、main、visualizer、pose_estimator模块可访问")
    raise e

# 配置设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {DEVICE}")

# 检查模型文件是否存在
if not SEGMENTATION_MODEL_PATH.exists():
    print(f"警告: 未找到分割模型: {SEGMENTATION_MODEL_PATH}")
    print("应用可能无法正常工作")

if not YOLO_MODEL_PATH.exists():
    print(f"警告: 未找到YOLO模型: {YOLO_MODEL_PATH}")
    print("应用可能无法正常工作")


def process_single_image(image_path, optim_steps=500, target_avg_scale=1.0, output_prefix="output"):
    """
    处理单个图像并生成小地图
    
    参数:
        image_path: 输入图像路径
        optim_steps: 优化步数
        target_avg_scale: 目标平均骨骼比例
        output_prefix: 输出文件前缀
    """
    print(f"\n=== 开始处理: {image_path} ===")
    print(f"参数: optim_steps={optim_steps}, target_avg_scale={target_avg_scale}")

    # 1. 读取图像
    if not Path(image_path).exists():
        print(f"错误: 图像文件不存在: {image_path}")
        return

    print("读取图像...")
    input_image_bgr = cv2.imread(image_path)
    if input_image_bgr is None:
        print(f"错误: 无法读取图像: {image_path}")
        return

    print(f"图像尺寸: {input_image_bgr.shape}")

    # 检查模型文件
    if not SEGMENTATION_MODEL_PATH.exists():
        print(f"错误: 分割模型不存在: {SEGMENTATION_MODEL_PATH}")
        return

    if not YOLO_MODEL_PATH.exists():
        print(f"错误: YOLO模型不存在: {YOLO_MODEL_PATH}")
        return

    try:
        # 2. 初始化TvCalib模型
        print("初始化TvCalibInferModule...")
        start_init = time.time()
        model = TvCalibInferModule(
            segmentation_checkpoint=SEGMENTATION_MODEL_PATH,
            image_shape=IMAGE_SHAPE,
            optim_steps=int(optim_steps),
            lens_dist=False
        )
        model_device = next(model.model_calib.parameters()).device
        print(f"✓ 模型加载完成 (设备: {model_device}) 耗时: {time.time() - start_init:.3f}s")

        # 3. 图像预处理
        print("图像预处理...")
        start_preprocess = time.time()
        image_tensor, image_bgr_resized, image_rgb_resized = preprocess_image_tvcalib(input_image_bgr)
        image_tensor = image_tensor.to(model_device)
        print(f"预处理耗时: {time.time() - start_preprocess:.3f}s")

        # 4. YOLO足球检测
        print("YOLO足球检测...")
        start_yolo = time.time()
        ball_ref_point_img = None

        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            results = yolo_model.predict(image_bgr_resized, classes=[BALL_CLASS_INDEX], verbose=False)

            if results and len(results[0].boxes) > 0:
                best_ball_box = results[0].boxes[results[0].boxes.conf.argmax()]
                x1, y1, x2, y2 = map(int, best_ball_box.xyxy[0].tolist())
                conf = best_ball_box.conf[0].item()

                # 计算参考点（边界框底部中心）
                ball_ref_point_img = np.array([(x1 + x2) / 2, y2], dtype=np.float32)
                print(f"✓ 检测到足球 (置信度: {conf:.2f})")
                print(f"  边界框: [{x1},{y1},{x2},{y2}]")
                print(f"  参考点: {ball_ref_point_img}")
            else:
                print("未检测到足球")

        except Exception as e_yolo:
            print(f"YOLO检测错误: {e_yolo}")

        print(f"YOLO检测耗时: {time.time() - start_yolo:.3f}s")

        # 5. 执行分割
        print("执行分割...")
        start_segment = time.time()
        with torch.no_grad():
            keypoints = model._segment(image_tensor)
        print(f"分割耗时: {time.time() - start_segment:.3f}s")

        # 6. 执行校准（优化）
        print("执行校准（优化）...")
        start_calibrate = time.time()
        homography = model._calibrate(keypoints)
        print(f"校准耗时: {time.time() - start_calibrate:.3f}s")

        if homography is None:
            print("错误: 无法计算单应性矩阵")
            return

        if isinstance(homography, torch.Tensor):
            homography_np = homography.detach().cpu().numpy()
        else:
            homography_np = np.array(homography)

        print("✓ 单应性矩阵计算完成")
        print(f"单应性矩阵:\n{homography_np}")

        # 7. 提取球员数据
        print("提取球员数据（姿态+颜色）...")
        start_pose = time.time()
        player_list = get_player_data(image_bgr_resized)
        print(f"球员数据提取耗时: {time.time() - start_pose:.3f}s")
        print(f"检测到 {len(player_list)} 名球员")

        # 8. 计算基础比例
        print("计算基础比例...")
        avg_modulation_expected = DYNAMIC_SCALE_MIN_MODULATION + \
                                  (DYNAMIC_SCALE_MAX_MODULATION - DYNAMIC_SCALE_MIN_MODULATION) * (1.0 - 0.5)
        estimated_base_scale = target_avg_scale
        if avg_modulation_expected != 0:
            estimated_base_scale = target_avg_scale / avg_modulation_expected
        print(f"目标平均比例: {target_avg_scale:.3f}")
        print(f"估算的基础比例: {estimated_base_scale:.3f}")

        # 9. 生成小地图
        print("生成小地图...")
        start_viz = time.time()

        # 原始投影小地图
        minimap_original = create_minimap_view(image_rgb_resized, homography_np)

        # 带骨骼偏移的小地图（包含足球）
        minimap_offset_skeletons, actual_avg_scale = create_minimap_with_offset_skeletons(
            player_list,
            homography_np,
            base_skeleton_scale=estimated_base_scale,
            ball_ref_point_img=ball_ref_point_img
        )

        print(f"小地图生成耗时: {time.time() - start_viz:.3f}s")

        if actual_avg_scale is not None:
            print(f"最终应用的实际平均比例: {actual_avg_scale:.3f}")

        # 10. 保存结果
        print("保存结果...")

        # 保存原始小地图
        orig_output_path = f"{output_prefix}_original_minimap.png"
        if minimap_original is not None:
            cv2.imwrite(orig_output_path, cv2.cvtColor(minimap_original, cv2.COLOR_RGB2BGR))
            print(f"✓ 保存原始小地图: {orig_output_path}")
        else:
            print("错误: 原始小地图生成失败")

        # 保存骨骼小地图
        skel_output_path = f"{output_prefix}_skeleton_minimap.png"
        if minimap_offset_skeletons is not None:
            cv2.imwrite(skel_output_path, cv2.cvtColor(minimap_offset_skeletons, cv2.COLOR_RGB2BGR))
            print(f"✓ 保存骨骼小地图: {skel_output_path}")
        else:
            print("错误: 骨骼小地图生成失败")

        # 可选：保存中间结果
        # 保存预处理后的图像
        preprocess_output_path = f"{output_prefix}_preprocessed.png"
        cv2.imwrite(preprocess_output_path, image_bgr_resized)
        print(f"✓ 保存预处理图像: {preprocess_output_path}")

        # 在图像上绘制足球检测结果（如果检测到）
        if ball_ref_point_img is not None:
            annotated_image = image_bgr_resized.copy()
            x1, y1, x2, y2 = map(int, [ball_ref_point_img[0] - 10, ball_ref_point_img[1] - 20,
                                       ball_ref_point_img[0] + 10, ball_ref_point_img[1]])
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated_image, (int(ball_ref_point_img[0]), int(ball_ref_point_img[1])),
                       5, (0, 0, 255), -1)
            ball_output_path = f"{output_prefix}_ball_detection.png"
            cv2.imwrite(ball_output_path, annotated_image)
            print(f"✓ 保存足球检测可视化: {ball_output_path}")

        print(f"\n=== 处理完成 ===")
        print(f"输出文件:")
        print(f"  - 原始小地图: {orig_output_path}")
        print(f"  - 骨骼小地图: {skel_output_path}")
        print(f"  - 预处理图像: {preprocess_output_path}")
        if ball_ref_point_img is not None:
            print(f"  - 足球检测: {ball_output_path}")

        return minimap_original, minimap_offset_skeletons, homography_np

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        traceback.print_exc()
        return None, None, None


def process_batch_images(image_dir, optim_steps=500, target_avg_scale=1.0):
    """
    批量处理目录中的所有图像
    
    参数:
        image_dir: 图像目录路径
        optim_steps: 优化步数
        target_avg_scale: 目标平均骨骼比例
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"错误: 目录不存在: {image_dir}")
        return

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # 查找所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"未在目录中找到图像文件: {image_dir}")
        return

    print(f"找到 {len(image_files)} 个图像文件")

    # 创建输出目录
    output_dir = image_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # 处理每个图像
    for i, image_path in enumerate(image_files):
        print(f"\n{'=' * 50}")
        print(f"处理图像 {i + 1}/{len(image_files)}: {image_path.name}")
        print('=' * 50)

        output_prefix = output_dir / image_path.stem
        process_single_image(
            str(image_path),
            optim_steps=optim_steps,
            target_avg_scale=target_avg_scale,
            output_prefix=str(output_prefix)
        )


def main():
    """
    主函数 - 处理单个图像或批量图像
    """
    import argparse

    parser = argparse.ArgumentParser(description='足球场图像处理与小地图生成')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像路径或目录路径')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理目录中的所有图像')
    parser.add_argument('--optim_steps', type=int, default=500,
                        help='TvCalib优化步数 (默认: 500)')
    parser.add_argument('--target_scale', type=float, default=1.0,
                        help='目标平均骨骼比例 (默认: 1.0)')
    parser.add_argument('--output_prefix', type=str, default='output',
                        help='输出文件前缀 (默认: output)')

    args = parser.parse_args()

    if args.batch:
        # 批量处理模式
        process_batch_images(
            image_dir=args.input,
            optim_steps=args.optim_steps,
            target_avg_scale=args.target_scale
        )
    else:
        # 单个图像处理模式
        process_single_image(
            image_path=args.input,
            optim_steps=args.optim_steps,
            target_avg_scale=args.target_scale,
            output_prefix=args.output_prefix
        )


if __name__ == "__main__":
    main()
