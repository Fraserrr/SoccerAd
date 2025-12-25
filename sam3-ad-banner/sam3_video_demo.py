import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

# ================= 配置区域 =================
# 1. 输入视频路径
VIDEO_PATH = "input_video/cut_video.mp4"

# 2. 输出视频保存路径
OUTPUT_VIDEO_PATH = "tracking_result.mp4"

# 3. 提示词 (想要分割对象的英文描述)
PROMPT_TEXT = "Ad banner"

# 4. 可视化时的帧率 (建议与原视频保持一致)
FPS = 30
# =======================================================

# 设置 Matplotlib 后端为 Agg，用于非交互式环境（直接保存图片/视频）
plt.switch_backend('Agg')


def propagate_in_video(predictor, session_id):
    """从 notebook 中提取的传播函数"""
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def load_video_frames(video_path):
    """加载视频帧用于可视化"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件未找到: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    print(f"正在加载视频帧: {video_path} ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV 默认是 BGR，SAM3 可视化需要 RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"视频加载完成，共 {len(frames)} 帧。")
    return frames


def save_result_video(frames, outputs_per_frame, output_path, fps):
    """将可视化结果渲染并保存为 MP4"""
    print("正在渲染并保存结果视频...")

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 准备掩膜数据
    formatted_outputs = prepare_masks_for_visualization(outputs_per_frame)

    # 遍历每一帧进行渲染
    for frame_idx in range(len(frames)):
        if frame_idx not in formatted_outputs:
            # 如果某帧没有输出，直接写入原图（转回 BGR）
            out.write(cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2BGR))
            continue

        # 创建 Matplotlib 图形
        # figsize 根据图像比例大致设置，dpi 设置稍高以保证清晰度
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)

        # 调用 SAM3 官方可视化函数
        visualize_formatted_frame_output(
            frame_idx,
            frames,
            outputs_list=[formatted_outputs],
            titles=None,  # 不显示标题以保持视频干净
            figsize=(width / 100, height / 100),
        )

        # 去除坐标轴
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        # 将 Matplotlib 图形转换为 OpenCV 图像
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        X = np.asarray(buf)
        img_vis = cv2.cvtColor(X, cv2.COLOR_RGBA2BGR)

        # Resize 到原视频尺寸 (matplotlib 渲染可能会有轻微尺寸偏差)
        img_vis = cv2.resize(img_vis, (width, height))

        out.write(img_vis)
        plt.close(fig)  # 关闭图形释放内存

        if frame_idx % 10 == 0:
            print(f"已渲染帧: {frame_idx}/{len(frames)}")

    out.release()
    print(f"结果已保存至: {output_path}")


def main():
    # 1. 检查 GPU
    if torch.cuda.is_available():
        gpus_to_use = range(torch.cuda.device_count())
        print(f"使用 GPU: {gpus_to_use}")
    else:
        print("警告: 未检测到 GPU，运行速度将非常慢。")
        gpus_to_use = []

    # 2. 构建预测器
    print("正在构建 SAM 3 预测器...")
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    # 3. 加载视频帧
    video_frames = load_video_frames(VIDEO_PATH)

    # 4. 开启推理会话
    print("开启推理会话...")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=VIDEO_PATH,
        )
    )
    session_id = response["session_id"]

    try:
        # 5. 添加文本提示 (在第 0 帧)
        print(f"应用文本提示: '{PROMPT_TEXT}' ...")
        _ = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=PROMPT_TEXT,
            )
        )

        # 6. 视频传播 (Propagate)
        print("正在进行视频传播推理 (这可能需要一些时间)...")
        outputs_per_frame = propagate_in_video(predictor, session_id)

        # 7. 可视化并保存
        save_result_video(video_frames, outputs_per_frame, OUTPUT_VIDEO_PATH, FPS)

    finally:
        # 8. 清理资源
        print("清理会话资源...")
        predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        predictor.shutdown()
        print("完成。")


if __name__ == "__main__":
    main()
