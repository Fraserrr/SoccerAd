import pandas as pd
import re
import os
import datetime

# ================= 配置区域 (User Configuration) =================

# 1. 输入和输出文件路径
INPUT_FILE = 'outputs/ad_logs/more_ad_result_test.csv'  # 你的OCR结果文件
OUTPUT_DIR = 'outputs/analysis_reports'  # 结果保存目录
SUMMARY_FILE = 'sponsor_summary_more.csv'  # 统计报表文件名
TIMELINE_FILE = 'sponsor_timeline_more.csv'  # 时间轴报表文件名

# 2. 视频总时长 (秒)
# 如果设置为 None，脚本将自动使用数据中出现的最大秒数作为视频总时长
# 如果你知道视频确切长度（例如 45分钟 = 2700秒），请在此填入数字，例如: 2700
VIDEO_TOTAL_DURATION_SECONDS = None

# 3. 赞助商匹配规则 (先验知识库)
# 格式: '标准赞助商名称': ['关键词1', '关键词2', ...]
# 逻辑: 只要OCR结果中包含了列表中的任意一个关键词，就认为该赞助商在这一秒出现了
SPONSOR_CONFIG = {
    '小红书': ['小红书', '红书', '小红'],
    '中国平安': ['平安', '理财', '买理', '找平'],
    '鱼跃': ['鱼跃', 'yuwell', '鱼', '跃'],
    '蒙牛': ['蒙牛', '蒙', '牛'],
    '央视体育': ['央视体', '视体育', 'SPORTS'],
    '怡宝': ['怡宝', '怡寳', '怡', '寳'],
    '足球中国': ['足球中', '球中国'],
    '福特宝球迷中心': ['福特', '特宝', '宝球', '球迷', '迷中', '中心'],
    '铜梁文旅': ['铜梁', '铜', '梁', '文旅', '文', '周末到铜梁', '周末']
}


# ================= 核心处理逻辑 (Processing Logic) =================

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def parse_ocr_text(raw_text_line):
    """
    解析OCR结果字符串，去除分数，提取纯文本列表。
    输入示例: "买理财 找平安(0.94); 蒙牛(1.00)"
    输出示例: ["买理财 找平安", "蒙牛"]
    """
    if pd.isna(raw_text_line) or raw_text_line == "":
        return []

    # 按分号分割不同的检测项
    items = raw_text_line.split(';')
    clean_texts = []

    for item in items:
        # 使用正则去除括号及里面的分数，例如 "找平安(0.94)" -> "找平安"
        # (.*?) 非贪婪匹配前面的字符， \(\d+\.\d+\) 匹配 (0.94) 这种格式
        match = re.match(r'(.*?)\(\d+\.\d+\)', item.strip())
        if match:
            text = match.group(1).strip()
            if text:
                clean_texts.append(text)
        else:
            # 如果没有分数格式（异常情况），直接保留原文本
            if item.strip():
                clean_texts.append(item.strip())

    return clean_texts


def match_sponsors(text_list, config):
    """
    将提取的文本列表与赞助商配置进行匹配
    """
    detected_sponsors = set()  # 使用set去重，同一秒内同一赞助商只记一次

    for text in text_list:
        for sponsor_name, keywords in config.items():
            for keyword in keywords:
                # 简单的包含匹配，不区分大小写
                if keyword.lower() in text.lower():
                    detected_sponsors.add(sponsor_name)
                    break  # 命中一个关键词即可确认该赞助商，跳出关键词循环

    return list(detected_sponsors)


def format_seconds(seconds):
    """将秒数转换为 HH:MM:SS 格式"""
    return str(datetime.timedelta(seconds=int(seconds)))


def main():
    global INPUT_FILE

    print(f"🚀 开始分析赞助商数据...")

    # 1. 读取数据
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件 {INPUT_FILE}")
        # 尝试在当前目录查找
        if os.path.exists(os.path.basename(INPUT_FILE)):
            INPUT_FILE = os.path.basename(INPUT_FILE)
            print(f"⚠️ 已切换到当前目录文件: {INPUT_FILE}")
        else:
            return

    df = pd.read_csv(INPUT_FILE)
    print(f"📂 已加载 {len(df)} 行数据")

    # 确定视频总时长
    max_second_in_data = df['second'].max() if not df.empty else 0
    total_duration = VIDEO_TOTAL_DURATION_SECONDS if VIDEO_TOTAL_DURATION_SECONDS else (max_second_in_data + 1)
    print(f"⏱️ 设定视频分析总时长: {total_duration} 秒 ({format_seconds(total_duration)})")

    # 2. 逐秒处理
    timeline_data = []  # 用于存储每一秒的分析结果
    sponsor_counts = {name: 0 for name in SPONSOR_CONFIG.keys()}  # 初始化计数器

    # 这一步是为了填补时间轴上的空缺（如果CSV不是每秒都有数据，视需求而定）
    # 这里我们只处理有识别结果的秒数，如果需要连续时间轴，可以重新索引

    # 对CSV中的每一行（每一秒）进行处理
    for _, row in df.iterrows():
        sec = row['second']
        raw_text = row['text']

        # 解析文本
        texts = parse_ocr_text(raw_text)

        # 匹配赞助商
        visible_sponsors = match_sponsors(texts, SPONSOR_CONFIG)

        # 更新统计
        for sponsor in visible_sponsors:
            sponsor_counts[sponsor] += 1

        # 记录时间轴
        timeline_data.append({
            'second': sec,
            'timestamp': format_seconds(sec),
            'sponsors': ', '.join(visible_sponsors) if visible_sponsors else '[无相关广告]'
        })

    # 3. 生成统计报表 (Summary Report)
    summary_data = []
    for sponsor, count in sponsor_counts.items():
        percentage = (count / total_duration) * 100 if total_duration > 0 else 0
        summary_data.append({
            '赞助商 (Sponsor)': sponsor,
            '出现总时长(秒)': count,
            '出现总时长(时:分:秒)': format_seconds(count),
            '占全片比例 (%)': round(percentage, 2)
        })

    df_summary = pd.DataFrame(summary_data)
    # 按出现时长降序排列
    df_summary = df_summary.sort_values(by='出现总时长(秒)', ascending=False)

    # 4. 生成时间轴报表 (Timeline Report)
    df_timeline = pd.DataFrame(timeline_data)
    # 确保按时间排序
    df_timeline = df_timeline.sort_values(by='second')
    df_timeline = df_timeline[['second', 'timestamp', 'sponsors']]  # 调整列顺序

    # 5. 保存文件
    ensure_dir(os.path.join(OUTPUT_DIR, SUMMARY_FILE))

    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_FILE)
    timeline_path = os.path.join(OUTPUT_DIR, TIMELINE_FILE)

    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    df_timeline.to_csv(timeline_path, index=False, encoding='utf-8-sig')

    print("\n✅ 分析完成！")
    print(f"📊 统计报表已保存至: {summary_path}")
    print(f"🕒 时间轴日志已保存至: {timeline_path}")

    # 打印预览
    print("\n--- 统计概览 ---")
    print(df_summary.to_string(index=False))


if __name__ == '__main__':
    main()