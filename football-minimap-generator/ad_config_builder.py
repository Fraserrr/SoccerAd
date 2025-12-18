import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# === 字体设置 ===
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# === 配置区域 (与 visualizer.py 保持完全一致的 Yards 单位) ===
FIELD_LENGTH = 114.83
FIELD_WIDTH = 74.37

# 定义广告牌配置
AD_CONFIGS = [
    # === 上边线 (Top / 远端) ===
    {"name": "中国平安", "side": "top", "start": 0.00, "end": 0.06},
    {"name": "足球中国", "side": "top", "start": 0.06, "end": 0.11},
    {"name": "ChinaFootball", "side": "top", "start": 0.11, "end": 0.16},
    {"name": "福特宝球迷中心", "side": "top", "start": 0.16, "end": 0.22},
    {"name": "央视体育", "side": "top", "start": 0.22, "end": 0.27},
    {"name": "蒙牛", "side": "top", "start": 0.27, "end": 0.33},
    {"name": "Nike", "side": "top", "start": 0.33, "end": 0.38},

    # 中圈附近
    {"name": "中国平安", "side": "top", "start": 0.38, "end": 0.44},
    {"name": "AFC QUALIFIERS", "side": "top", "start": 0.44, "end": 0.56},
    {"name": "中国平安", "side": "top", "start": 0.56, "end": 0.62},

    {"name": "怡宝", "side": "top", "start": 0.62, "end": 0.67},
    {"name": "鱼跃", "side": "top", "start": 0.67, "end": 0.72},
    {"name": "铜梁文旅", "side": "top", "start": 0.72, "end": 0.77},
    {"name": "小红书", "side": "top", "start": 0.77, "end": 0.83},
    {"name": "足球中国", "side": "top", "start": 0.83, "end": 0.88},
    {"name": "Nike", "side": "top", "start": 0.88, "end": 0.94},
    {"name": "中国平安", "side": "top", "start": 0.94, "end": 1.00},

    # === 左边线 (Left / 左侧球门后) ===
    {"name": "福特宝球迷中心", "side": "left", "start": 0.00, "end": 0.10},
    {"name": "Nike", "side": "left", "start": 0.10, "end": 0.20},
    {"name": "怡宝", "side": "left", "start": 0.20, "end": 0.30},
    {"name": "鱼跃", "side": "left", "start": 0.30, "end": 0.40},
    {"name": "铜梁文旅", "side": "left", "start": 0.40, "end": 0.50},
    {"name": "中国平安", "side": "left", "start": 0.50, "end": 0.60},
    {"name": "ChinaFootball", "side": "left", "start": 0.60, "end": 0.70},
    {"name": "足球中国", "side": "left", "start": 0.70, "end": 0.80},
    {"name": "RESPECT", "side": "left", "start": 0.80, "end": 0.90},
    {"name": "左侧未定义广告", "side": "left", "start": 0.90, "end": 1.00},

    # === 右边线 (Right / 右侧球门后) ===
    {"name": "RESPECT", "side": "right", "start": 0.00, "end": 0.10},
    {"name": "足球中国", "side": "right", "start": 0.10, "end": 0.20},
    {"name": "福特宝球迷中心", "side": "right", "start": 0.20, "end": 0.30},
    {"name": "ChinaFootball", "side": "right", "start": 0.30, "end": 0.40},
    {"name": "蒙牛", "side": "right", "start": 0.40, "end": 0.50},
    {"name": "中国平安", "side": "right", "start": 0.50, "end": 0.60},
    {"name": "Nike", "side": "right", "start": 0.60, "end": 0.70},
    {"name": "福特宝球迷中心", "side": "right", "start": 0.70, "end": 0.80},
    {"name": "ChinaFootball", "side": "right", "start": 0.80, "end": 0.90},
    {"name": "右侧未定义广告", "side": "right", "start": 0.90, "end": 1.00},

    # 下边线
    {"name": "底边左侧未定义广告", "side": "bottom", "start": 0.0, "end": 0.5},
    {"name": "底边右侧未定义广告", "side": "bottom", "start": 0.5, "end": 1.0},
]


def generate_ad_segments():
    """
    坐标系以【左上角】为 (0,0)。
    X轴: 0 -> FIELD_LENGTH
    Y轴: 0 -> FIELD_WIDTH
    这样才能与 visualizer.py 的 Minimap 像素坐标系对齐。
    """

    segments = []

    for ad in AD_CONFIGS:
        if ad['side'] == 'top':
            # 上边线: y = 0
            y = 0.0
            x_start = ad['start'] * FIELD_LENGTH
            x_end = ad['end'] * FIELD_LENGTH
            segments.append({"name": ad['name'], "coords": [(x_start, y), (x_end, y)]})

        elif ad['side'] == 'bottom':
            # 下边线: y = FIELD_WIDTH
            y = FIELD_WIDTH
            x_start = ad['start'] * FIELD_LENGTH
            x_end = ad['end'] * FIELD_LENGTH
            segments.append({"name": ad['name'], "coords": [(x_start, y), (x_end, y)]})

        elif ad['side'] == 'left':
            # 左边线: x = 0
            x = 0.0
            y_start = ad['start'] * FIELD_WIDTH
            y_end = ad['end'] * FIELD_WIDTH
            segments.append({"name": ad['name'], "coords": [(x, y_start), (x, y_end)]})

        elif ad['side'] == 'right':
            # 右边线: x = FIELD_LENGTH
            x = FIELD_LENGTH
            y_start = ad['start'] * FIELD_WIDTH
            y_end = ad['end'] * FIELD_WIDTH
            segments.append({"name": ad['name'], "coords": [(x, y_start), (x, y_end)]})

    return segments


def visualize_and_save(segments):
    fig, ax = plt.subplots(figsize=(12, 8))

    # === 画球场背景 ( 0 到 LENGTH) ===
    # (0, 0) 起始
    rect = patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH,
                             linewidth=2, edgecolor='green', facecolor='#eaffea')
    ax.add_patch(rect)

    # 设置显示范围 (稍微留点边距)
    margin = 10
    ax.set_xlim(-margin, FIELD_LENGTH + margin)
    ax.set_ylim(-margin, FIELD_WIDTH + margin)

    # === 强制长宽比 ===
    # 保证生成的图片比例与真实球场(114:74)一致
    ax.set_aspect('equal')

    # === 画中圈 ===
    center_x = FIELD_LENGTH / 2.0
    center_y = FIELD_WIDTH / 2.0
    circle = patches.Circle((center_x, center_y), 9.15, edgecolor='green', facecolor='none')
    ax.add_patch(circle)

    # 画广告牌
    for seg in segments:
        p1, p2 = seg['coords']
        x_vals = [p1[0], p2[0]]
        y_vals = [p1[1], p2[1]]
        ax.plot(x_vals, y_vals, linewidth=4, label=seg['name'])

        # 标注文字
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2

        # 优化文字显示位置
        # 原逻辑: offset = 0 if x1 == x2 else 3
        # 这里的判断 seg['coords'][0][0] == seg['coords'][1][0] 意味着 X 坐标相同，即竖线(左右边)
        is_vertical_line = (seg['coords'][0][0] == seg['coords'][1][0])
        offset = 0 if is_vertical_line else 3

        # 应用偏移 (注意：Y轴现在向下增大，所以 Bottom 需要加，Top 需要减)
        if not is_vertical_line:  # 横线 (Top/Bottom)
            if mid_y < FIELD_WIDTH / 2:  # Top
                mid_y -= offset
            else:  # Bottom
                mid_y += offset
        else:  # 竖线 (Left/Right)
            # 保持 mid_x 不变
            pass

        rotation = 0 if is_vertical_line else 45

        ax.text(mid_x, mid_y, seg['name'], fontsize=8, rotation=rotation, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.title(f"广告牌位置-配置图")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().invert_yaxis()  # 保持 Y 轴向下，符合 CV 图像坐标系习惯

    # 保存配置JSON
    with open('ad_map_config.json', 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print("配置已保存至 ad_map_config.json")

    # 保存预览图
    plt.savefig('ad_map_visualization.png', dpi=150, bbox_inches='tight')
    print("可视化图已保存至 ad_map_visualization.png")


if __name__ == "__main__":
    segs = generate_ad_segments()
    visualize_and_save(segs)