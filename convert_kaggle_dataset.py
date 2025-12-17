import json
import base64
import zlib
import io
import cv2
import numpy as np
import os
import glob
import shutil
import random
from tqdm import tqdm

# ================= 配置区域 =================
# Kaggle 数据集路径
# 结构应为:
# KAGGLE_ROOT/
#   ├── images/ (存放 .jpg/.png)
#   └── annotations/ (存放 .json)
KAGGLE_ROOT = "football-banner"

# 输出路径 (将生成适配 DINOv3 的目录)
OUTPUT_ROOT = "input/dataset"

# 训练集占比
TRAIN_RATIO = 0.9

# 想要保留的类别 (如果是全部广告牌，保留所有即可)
# 这里我们将所有非背景物体都视为广告牌
# 如果你想排除某些类别，可以在这里过滤，但通常不需要
target_class_value = 255


# ===========================================

def decode_bitmap(data_string):
    """
    解码 Supervisely 格式的 bitmap 字符串
    Base64 -> Zlib -> Image Bytes -> Numpy Mask
    """
    try:
        # 1. Base64 解码
        compressed_data = base64.b64decode(data_string)
        # 2. Zlib 解压
        decompressed_data = zlib.decompress(compressed_data)
        # 3. 转为 Numpy 字节流
        nparr = np.frombuffer(decompressed_data, np.uint8)
        # 4. 解码为图片 (Supervisely 通常存储为 PNG 格式的二进制流)
        mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        # 处理可能存在的 Alpha 通道
        if len(mask.shape) == 3 and mask.shape[2] == 4:
            mask = mask[:, :, 3]  # 取 Alpha 通道
        elif len(mask.shape) == 3:
            mask = mask[:, :, 0]  # 取第一个通道

        # 二值化，确保非0即为1
        _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
        return mask
    except Exception as e:
        # print(f"Decode error: {e}")
        return None


def process_dataset():
    # 1. 准备目录
    dirs = {
        "train_img": os.path.join(OUTPUT_ROOT, "train_images"),
        "train_mask": os.path.join(OUTPUT_ROOT, "train_masks"),
        "valid_img": os.path.join(OUTPUT_ROOT, "valid_images"),
        "valid_mask": os.path.join(OUTPUT_ROOT, "valid_masks"),
    }

    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 2. 获取所有标注文件
    ann_dir = os.path.join(KAGGLE_ROOT, "annotations")
    img_dir = os.path.join(KAGGLE_ROOT, "images")

    json_files = glob.glob(os.path.join(ann_dir, "*.json"))
    if not json_files:
        print(f"❌ 未找到标注文件，请检查路径: {ann_dir}")
        return

    print(f"🔍 发现 {len(json_files)} 个标注文件，开始处理...")

    # 打乱并划分
    random.seed(42)
    random.shuffle(json_files)
    split_idx = int(len(json_files) * TRAIN_RATIO)
    train_files = json_files[:split_idx]
    valid_files = json_files[split_idx:]

    def process_batch(files, img_dest, mask_dest, mode):
        for json_path in tqdm(files, desc=f"Processing {mode}"):
            try:
                # 读取 JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 获取图像尺寸
                img_h = data['size']['height']
                img_w = data['size']['width']

                # 创建全黑画布 (单通道)
                full_mask = np.zeros((img_h, img_w), dtype=np.uint8)

                # 遍历所有对象
                objects = data.get('objects', [])
                has_objects = False

                for obj in objects:
                    # 检查是否是 bitmap 类型
                    if obj.get('geometryType') != 'bitmap':
                        continue

                    bitmap_data = obj.get('bitmap', {})
                    data_str = bitmap_data.get('data')
                    origin = bitmap_data.get('origin')  # [x, y]

                    if not data_str or not origin:
                        continue

                    # 解码 Mask
                    obj_mask = decode_bitmap(data_str)
                    if obj_mask is None:
                        continue

                    # 粘贴到画布上
                    x, y = origin
                    h_obj, w_obj = obj_mask.shape

                    # 边界检查 (防止贴出画外报错)
                    y1, y2 = y, min(y + h_obj, img_h)
                    x1, x2 = x, min(x + w_obj, img_w)

                    # 截取 object mask 的有效部分 (如果被裁剪)
                    obj_h_valid = y2 - y1
                    obj_w_valid = x2 - x1

                    if obj_h_valid <= 0 or obj_w_valid <= 0:
                        continue

                    # 将对象区域标白 (255)
                    # 使用逻辑或 (OR) 操作，避免重叠区域出问题
                    current_roi = full_mask[y1:y2, x1:x2]
                    obj_roi = obj_mask[0:obj_h_valid, 0:obj_w_valid]

                    # 只要是 mask 的部分，就设为 255
                    full_mask[y1:y2, x1:x2] = np.maximum(current_roi, obj_roi * 255)
                    has_objects = True

                # 寻找对应的原图
                # JSON 文件名通常与图片同名，或者是 图片名.json
                base_name = os.path.basename(json_path)
                # 尝试几种可能的图片扩展名
                image_name_candidates = [
                    base_name.replace('.json', ''),  # 假设 json 是 image.jpg.json
                    os.path.splitext(base_name)[0] + ".jpg",
                    os.path.splitext(base_name)[0] + ".png",
                    os.path.splitext(base_name)[0] + ".jpeg"
                ]

                src_img_path = None
                for name in image_name_candidates:
                    temp_path = os.path.join(img_dir, name)
                    if (os.path.exists(temp_path)):
                        src_img_path = temp_path
                        break

                if src_img_path:
                    # 保存 Mask (PNG 无损)
                    mask_filename = os.path.splitext(os.path.basename(src_img_path))[0] + ".png"
                    cv2.imwrite(os.path.join(mask_dest, mask_filename), full_mask)

                    # 复制原图
                    shutil.copy(src_img_path, os.path.join(img_dest, os.path.basename(src_img_path)))
                else:
                    # print(f"找不到对应的图片: {json_path}")
                    pass

            except Exception as e:
                print(f"Error processing {json_path}: {e}")

    # 执行处理
    process_batch(train_files, dirs["train_img"], dirs["train_mask"], "Train")
    process_batch(valid_files, dirs["valid_img"], dirs["valid_mask"], "Valid")

    print("\n✅ 数据转换完成！")
    print(f"📂 输出目录: {OUTPUT_ROOT}")


if __name__ == "__main__":
    process_dataset()
