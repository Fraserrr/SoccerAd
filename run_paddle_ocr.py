import pandas as pd
import cv2
import os
import argparse
from tqdm import tqdm
import paddle
from paddleocr import PaddleOCR
import logging
import numpy as np

# 屏蔽无关日志
logging.getLogger("ppocr").setLevel(logging.ERROR)


def preprocess_image(img):
    """
    图像预处理：针对低分辨率和模糊的广告牌图片进行增强
    """
    # 1. 尺寸增强：如果图片高度过小（常见于长条幅截图），放大以提升小字识别率
    h, w = img.shape[:2]
    if h < 128:  # 阈值可调，针对那些只有30-50px高的长条图
        scale_factor = 2.0
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # 2. 锐化处理：增强边缘，解决模糊导致的字母混淆（如 f/t, e/c）
    # 使用标准的锐化卷积核
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    img = cv2.filter2D(src=img, ddepth=-1, kernel=sharpen_kernel)

    return img


def parse_ocr_result(item):
    """
    健壮的解析函数：兼容各种 PaddleOCR 返回格式
    """
    text, score = "", 0.0
    try:
        # 情况 1: 我们在 sliding_window_ocr 手动构造的简单字典
        if isinstance(item, dict):
            text = item.get('text', '')
            score = item.get('score', 0.0)

            # 防御性编程：如果没有取到，尝试旧逻辑
            if not text and 'rec_texts' in item:
                # 这种通常是原始大字典，不应该走到这里，但以防万一
                if len(item['rec_texts']) > 0:
                    text = item['rec_texts'][0]
                    score = item['rec_scores'][0]

        # 情况 2: 标准 List/Tuple 格式 [[bbox], (text, score)]
        elif isinstance(item, (list, tuple)):
            if len(item) >= 2:
                content = item[1]
                if isinstance(content, (list, tuple)) and len(content) >= 2:
                    text = content[0]
                    score = content[1]
                elif isinstance(content, str):
                    text = content
                    score = 1.0
    except Exception as e:
        print(f"Parse error: {e}")
        return "", 0.0

    return text, score


def sliding_window_ocr(ocr_engine, img_rgb):
    """
    滑动窗口切片识别
    针对长宽比过大的图片，切分成多个重叠的片段分别识别，最后汇总结果。
    返回格式统一包装为 [[result1, result2...]]
    """
    if img_rgb is None: return []
    h, w = img_rgb.shape[:2]
    if h == 0 or w == 0: return []

    aspect_ratio = w / float(h)

    # === [参数调整区] ===
    # 1. 切片触发阈值：长宽比超过多少开始切片？(建议 3.0)
    #    调低此值会让更多中等长度的图片也进行切片，提高召回率，但会降低速度。
    SLICE_TRIGGER_RATIO = 3.0

    # 2. 目标切片比例：希望每个小切片的长宽比是多少？(建议 3.0 - 4.0)
    #    调低此值 (如 2.5) 会产生更多、更窄的切片，对严重变形的长图效果更好。
    TARGET_SLICE_RATIO = 3.0

    # 3. 重叠率：切片之间的重叠区域比例 (0.1 - 0.8, 建议 0.5)
    #    0.5 表示重叠一半。重叠越多，边界处的词越不容易被切断，去重逻辑越稳健。
    OVERLAP_RATIO = 0.5
    # ===================

    crops = []

    # 如果长宽比未达到触发值，不切片，直接整图识别
    if aspect_ratio < SLICE_TRIGGER_RATIO:
        crops.append(img_rgb)
    else:
        # 动态计算需要切几份
        # 例如：图片比例 10:1，目标比例 3:1 -> 切 4 份
        num_slices = max(2, int(aspect_ratio / TARGET_SLICE_RATIO) + 1)

        step = w / num_slices
        overlap_width = step * OVERLAP_RATIO

        for i in range(num_slices):
            # 计算包含重叠区的坐标
            start_x = max(0, int(i * step - overlap_width))
            end_x = min(w, int((i + 1) * step + overlap_width))

            if start_x >= end_x: continue

            crop = img_rgb[:, start_x:end_x]
            if crop.size > 0:
                crops.append(crop)

    results_pool = []

    for crop in crops:
        try:
            # 这里的 ocr 调用保持原样
            slice_res = ocr_engine.ocr(crop)
            if not slice_res: continue

            for res_item in slice_res:
                if not res_item: continue

                # 适配 Dict 格式 (新版 PaddleOCR)
                if isinstance(res_item, dict):
                    rec_texts = res_item.get('rec_texts', [])
                    rec_scores = res_item.get('rec_scores', [])
                    for t, s in zip(rec_texts, rec_scores):
                        results_pool.append({'text': t, 'score': s})

                # 适配 List 格式 (旧版 PaddleOCR)
                elif isinstance(res_item, list):
                    results_pool.append(res_item)

        except Exception as e:
            print(f"⚠️ Error on slice: {e}")
            pass

    return [results_pool] if results_pool else []


def filter_contained_texts(df_group):
    """
    过滤逻辑：在同一秒内，如果短词是长词的子串，则丢弃短词。
    例如：存在 '买理财找平安' 和 '找平安'，则删除 '找平安'。
    """
    # 1. 按文本长度降序排列（优先保留长词）
    # 辅助列：text_len
    df_group['text_len'] = df_group['raw_text'].apply(len)
    sorted_df = df_group.sort_values(by='text_len', ascending=False)

    kept_indices = []
    kept_texts = []

    for idx, row in sorted_df.iterrows():
        current_text = row['raw_text']

        # 检查当前词是否是任何“已保留词”的子串
        is_substring = False
        for kept in kept_texts:
            if current_text in kept:
                is_substring = True
                break

        # 如果不是子串，或者是完全相等的词（但因为我们前面已经做了同词取最高分去重，这里通常处理的是不同词），则保留
        if not is_substring:
            kept_indices.append(idx)
            kept_texts.append(current_text)

    return df_group.loc[kept_indices]


def is_garbage(text):
    """
    垃圾字符过滤器
    """
    if not text: return True
    clean_text = text.strip()

    # 1. 基础过滤：去除长度小于2的纯数字/字母 (如 "A", "1")
    if len(clean_text) < 2:
        if clean_text.isascii() and clean_text.isalnum():
            return True

    # === [可选优化] 严格过滤模式 ===
    # 需求：把所有纯英文、纯数字、纯符号的字符串直接删去，只保留包含中文的词。
    # 逻辑：如果整个字符串都是 ASCII 字符 (a-z, 0-9, @, ., etc.)，则视为垃圾。
    # 操作：若要启用，请取消下面两行的注释。

    # if clean_text.isascii():
    #     return True

    # ==============================

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crops_dir', default='outputs/crops_more', help='Directory containing crop images')
    parser.add_argument('--output_csv', default='outputs/ad_logs/more_ad_result_test.csv', help='Final result csv')
    parser.add_argument('--sample_rate', type=int, default=2, help='Process 1 frame every N frames')
    args = parser.parse_args()

    metadata_path = os.path.join(args.crops_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"❌ Error: {metadata_path} not found.")
        return

    df = pd.read_csv(metadata_path)
    # 根据采样率筛选帧
    target_frames = df[df['frame_index'] % args.sample_rate == 0].copy()
    unique_frames = target_frames['frame_index'].unique()
    print(f"📂 Loaded {len(df)} records. Processing {len(target_frames)} crops.")
    print(f"🎯 Sampling Rate: {args.sample_rate}")
    print(f"⚡ Processing {len(target_frames)} crops from {len(unique_frames)} unique frames.")

    # 1. 设置全局设备
    if paddle.device.is_compiled_with_cuda():
        try:
            paddle.set_device('gpu')
            print("🚀 [Step 2] PaddleOCR running on GPU")
        except:
            paddle.set_device('cpu')
    else:
        print("⚠️ PaddleOCR running on CPU")

    # 2. 初始化
    print("📝 Initializing PaddleOCR v3...")

    ocr = PaddleOCR(
        ocr_version='PP-OCRv4',
        use_textline_orientation=True,
        lang='ch',
        text_det_limit_side_len=12000,
        text_det_limit_type='max',
        # 放大检测框，解决艺术字笔画分离问题
        text_det_unclip_ratio=1.8,
        # 降低检测框门槛，提高召回率
        text_det_box_thresh=0.25,
        # 二值化阈值
        text_det_thresh=0.15
    )

    results = []
    print("running OCR...")

    # 3. 循环处理
    for idx, row in tqdm(target_frames.iterrows(), total=len(target_frames)):
        img_path = row['crop_path']
        if not os.path.exists(img_path): continue

        try:
            img = cv2.imread(img_path)
            if img is None: continue

            img_processed = preprocess_image(img)
            if img_processed is None: continue

            img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)

            # 调用 OCR
            ocr_res = sliding_window_ocr(ocr, img_rgb)

            # 校验结果
            if not ocr_res or not isinstance(ocr_res, list):
                continue

            flattened_res = ocr_res[0]
            if not flattened_res:
                continue

            for item in flattened_res:
                # 解析结果
                text, score = parse_ocr_result(item)
                clean_text = str(text).strip()

                if score > 0.35 and len(clean_text) > 0:
                    if not is_garbage(clean_text):
                        results.append({
                            'second': row['second'],
                            'frame_index': row['frame_index'],
                            'raw_text': clean_text,
                            'score': score
                        })

        except Exception as e:
            print(f"Skipping error: {e}")
            pass

    # 4. 保存结果与聚合逻辑
    if results:
        res_df = pd.DataFrame(results)

        # --- 帧间去重（同秒、同词，保留最高分） ---
        # 先按分数降序，这样 drop_duplicates 默认保留第一条（最高分）
        res_df = res_df.sort_values(by='score', ascending=False)
        res_df_dedup = res_df.drop_duplicates(subset=['second', 'raw_text'], keep='first').copy()

        # --- 包含关系过滤,过滤子串（同秒内，删除被长词包含的短词） ---
        # 使用 groupby 对每一秒的数据分别应用 filter_contained_texts
        res_df_filtered = res_df_dedup.groupby('second', group_keys=False).apply(filter_contained_texts)

        # --- 格式化 ---
        res_df_filtered['formatted_text'] = res_df_filtered.apply(
            lambda x: f"{x['raw_text']}({x['score']:.2f})", axis=1
        )

        # --- 聚合输出 ---
        final_df = res_df_filtered.groupby('second')['formatted_text'].apply(
            lambda x: "; ".join(sorted(list(x)))
        ).reset_index()

        # 重命名并保存
        final_df.rename(columns={'formatted_text': 'text'}, inplace=True)

        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        final_df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ Logs saved to {args.output_csv}")
        print(final_df.head(10))
    else:
        print("\n⚠️ No text detected in the selected frames.")


if __name__ == '__main__':
    main()
