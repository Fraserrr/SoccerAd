# 基于 Dinov3 与 PaddleOCR 的足球场广告板识别

Dinov3 参考仓库：

[damdip/SoccerAdBannerSegmentation-Replacement: Deep Learning project about ad banner detection and replacement in soccer matches videos](https://github.com/damdip/SoccerAdBannerSegmentation-Replacement)

### 1 环境配置

**python 3.10**

#### DINOv3 环境

```bash
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt --no-deps
```

可能还有部分包安装不完整：

```bash
pip install huggingface_hub regex requests
```

此时 transformers 和 huggingface-hub 版本可能冲突：

```bash
pip install --upgrade transformers huggingface-hub tokenizers
```

继续安装缺失的包：

```bash
pip install pyparsing cycler python-dateutil kiwisolver albucore pydantic
```

运行可能出现 albucore 相关报错，使用 conda 强制重装：

```bash
conda install -c conda-forge albumentations
```

到这里应该可以正常运行训练和视频推理脚本。

#### Paddle 环境

最好是独立环境（可用 python 3.10），否则容易破坏原来的 Pytorch 环境。

paddle-gpu （需要正确安装 cudnn）

```bash
python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

其他组件：

```bash
pip install paddleocr==3.0.0
pip install protobuf==3.20.2
pip install shapely scikit-image
```

### 2 Dinov3 模型训练

#### 数据集准备

1. 下载 Kaggle 数据集

   [banner segmentation](https://www.kaggle.com/code/caoklong/banner-segmentation/input)

   完整下载，包含 `annotations`, `images` 两个目录，以及 `meta.json` 文件。

2. 数据格式转换

   - 运行脚本 `convert_kaggle_dataset.py`，设置 `KAGGLE_ROOT` 为原始数据集目录
     - 仓库中数据集目录为 `football-banner`

   - DINOv3 需要的数据集被输出到 `input/dataset`
   
   > 原始数据集将广告分为8个类别，分别对应8个品牌，标注形式都是 bitmap；
   >
   > 在处理数据时统一为一个类别，根据每张图对应的 annotation，生成 DINO 所需的黑底白色 Mask 图片（png），舍弃原始的类别，把所有类别都映射成一类白色 Mask 即可。

#### 模型参数

模型在 `model.py` 中定义

DINOv3 small: `"facebook/dinov3-vits16-pretrain-lvd1689m"`

#### 训练参数

```bash
python train.py --lr 0.0001 --batch 2 --imgsz 896 896 --epochs 16 --scheduler --scheduler-epochs 10 13
```

显存 4GB 训练约 15 小时。

### 3 Dinov3 视频推理

**注意：推理使用的 imgsz 参数最好与训练时的一致**

由于测试时使用了剪辑视频，所有的视频输入全部处理为了 1080p，30 帧。

#### 直接推理测试

` only_infer_video.py` 直接调用模型进行推理，输出原始的分割结果，可视化为视频。

> 由于训练数据集画面色调均匀，训练出的模型在对转播视频进行推理时，如果广告板反光严重，中心区域无法被识别，只会生成围绕广告板四周的边缘，因此需要使用填充算法对区域进行填补。

#### 边缘内部填充算法

**(1) 凸包算法**

` infer_video.py` 在推理后使用填充算法，把被像素包裹的区域填充满，可视化为视频。

- 填充思路：
  - 横向连接：先使用宽扁的卷积核进行膨胀，将断断续续的边缘在水平方向连成线。
  - 纵向闭合：使用瘦高的卷积核进行闭运算，尝试将广告牌的上下边缘连接起来。
  - 凸包填充：识别连通域的轮廓，计算其凸包（Convex Hull），然后填充凸包内部。这能保证即使只有上下两条边，中间也能被填满，且不会像外接矩形那样把不该覆盖的弯曲部分也覆盖进去。
  - 噪声过滤：过滤掉面积过小的误识别点（例如球员的球鞋）
- 参数设置（`process_mask_to_fill_holes` 函数）：
  - `kernel_h`: 横向膨胀核，宽高 `(30, 5)`
  - `kernel_v`: 纵向闭运算核，宽高 `(5, 35)`
  - 噪声面积阈值：`cv2.contourArea(c) < 1600`

**(2) 优化算法**

为了替代凸包，我们需要一个能尊重物体凹凸形状，同时又能填满内部空隙的算法。

1. 强力纵向闭合：广告牌上下边缘通常是平行的。我们使用一个很高的纵向核进行形态学“闭运算”。这就像是用一把宽刷子，把广告牌的上下两条线“刷”在一起，变成一个实心的块，但不会在左右方向上过度延伸。
2. 适度横向连接：连接断裂的广告牌片段。
3. 多边形拟合：替代凸包。它会沿着轮廓的边缘进行拟合，允许轮廓向内凹陷。这意味着如果广告牌中间有个弯折，它会贴着弯折走，而不是直接拉直线。

`infer_video_pro.py` 实现优化的填充算法并将结果可视化为视频。

参数调整参考：

- 中间仍然有空洞：增大 `kv_height`（纵向核高度）。这通常发生在特写镜头，广告牌占画面很高比例时。
- 广告牌变成了断裂的几截：增大 `kh_width`（横向核宽度）。
- 边缘太锯齿：增大 `epsilon` 的系数（例如改为 `0.01`）。
- 碎片区域过滤阈值推荐：`contourArea(c) < 3000`

#### 原始视频 CROP 脚本

`infer_video_dinov3.py` 执行从原始视频到 CROP 图片的流程。

- 调用 DINOv3 对输入的视频进行推理分
- 对分割结果进行区域填充
- 将每一帧的填充结果区域裁切出来，保存为单独的图片
  - 图片存储在 crops 目录下
  - 每一帧可能有多个广告牌区域因此一帧可能包含多个 crop 图片

### 4 OCR 识别

基于您上传的最终版本脚本（`run_paddle_ocr.py` 和 `analyze_sponsors.py`），我为您整理了最新的技术文档。

#### OCR 推理脚本

`run_paddle_ocr.py`

**脚本作用**

该脚本负责核心的视觉识别任务。它读取抽帧后的裁剪图片（Crops），针对广告牌特有的低分辨率、长宽比极大、模糊等问题进行图像增强和切片处理，调用 PaddleOCR v4 模型提取文字，并进行初步的数据清洗和聚合。

**核心处理流程**

1. **图像预处理 (`preprocess_image`)**：
   - **尺寸增强**：针对高度低于 128px 的图片，强制放大 2 倍，确保密集汉字的笔画清晰。
   - **边缘锐化**：使用卷积核增强文字边缘对比度，解决模糊导致的形近字混淆。
2. **滑动窗口切片**：
   - 针对长宽比 > 3.0 的长条图片，自动启用切片模式。
   - 按设定比例将图片切分为多个重叠片段分别识别，彻底解决模型压缩导致的长图两端文字丢失问题。
3. **结果解析与过滤**：
   - **垃圾过滤**：自动剔除长度小于 2 的纯数字或纯字母干扰项。
   - **去重聚合**：在同一秒内，对识别结果进行包含关系过滤（如保留“买理财找平安”，删除“找平安”），并按置信度排序保留最优结果。

**参数调整说明**

| **参数名**              | **默认值** | **所在位置/函数**    | **作用与调整建议**                                           |
| ----------------------- | ---------- | -------------------- | ------------------------------------------------------------ |
| `--sample_rate`         | `2`        | 命令行参数           | **采样频率**。每 N 帧处理 1 帧。调小（如 1）可提高时间轴精度但速度慢；调大（如 6）速度快但可能降低召回率。 |
| `h < 128`               | `128`      | `preprocess_image`   | **放大触发阈值**。如果广告牌截图普遍更小或更大，可调整此高度阈值以控制哪些图片需要放大。 |
| `SLICE_TRIGGER_RATIO`   | `3.0`      | `sliding_window_ocr` | **切片触发阈值**。图片长宽比超过此值时启用切片。如果发现中等长度（如 2.5倍）的图片两端识别不清，可**调低**此值。 |
| `TARGET_SLICE_RATIO`    | `3.0`      | `sliding_window_ocr` | **目标切片比例**。期望切出的子图长宽比。**调低**（如 2.0）会产生更多、更窄的切片，适合严重变形或弯曲的长广告牌。 |
| `OVERLAP_RATIO`         | `0.5`      | `sliding_window_ocr` | **重叠率**。切片间的重叠比例。默认 0.5（50%）能有效防止文字正好位于切口处被截断。 |
| `text_det_unclip_ratio` | `1.8`      | `main` (OCR初始化)   | **检测框膨胀系数**。调大有助于连接断裂的艺术字；****调小****有助于强行分开粘连在一起的两个词。 |
| `text_det_box_thresh`   | `0.25`     | `main` (OCR初始化)   | **检测框门槛**。决定多“像”文字的区域才会被保留。当前 **0.25** 为高召回模式，旨在识别极模糊文字；若噪点太多可调高至 0.4。 |
| `text_det_thresh`       | `0.15`     | `main` (OCR初始化)   | **二值化阈值**。对像素亮度的敏感度。**0.15** 适合识别颜色极淡或光照不足的文字。 |
| `score > ...`           | `0.35`     | `main` (循环内)      | **置信度过滤**。识别结果的得分门槛。如果结果中依然混有大量乱码，可将此值**调高**至 0.5。 |

#### 结果数据统计脚本 

`analyze_sponsors.py`

**脚本作用**

该脚本是基于“先验知识”的分析工具。它读取 OCR 输出的原始文本日志，结合用户预定义的赞助商列表（包含各种错别字和变体），进行模糊匹配统计，最终输出汇总报表和时间轴日志。

**核心处理流程**

1. **文本清洗**：解析 OCR 结果字符串（如 `蒙牛(0.99)`），去除分数和括号，提取纯文本。
2. **模糊匹配**：遍历 `SPONSOR_CONFIG`，检查提取的文本是否包含配置中的任意关键词（不区分大小写）。
3. **统计聚合**：
   - **按秒记录**：生成 Timeline 报表，展示每一秒出现的赞助商。
   - **汇总计算**：计算每个赞助商出现的总秒数及占全视频时长的百分比。

**参数调整说明**

所有参数均位于脚本开头的 配置区域：

| **参数/变量名**                | **当前配置示例**                 | **作用与调整建议**                                           |
| ------------------------------ | -------------------------------- | ------------------------------------------------------------ |
| `VIDEO_TOTAL_DURATION_SECONDS` | `None`                           | **视频总时长(秒)**。设为 `None` 时自动使用数据中的最大时间戳。 |
| `SPONSOR_CONFIG`               | `{'小红书': ['红书', ...], ...}` | **赞助商匹配规则**。字典 Key 为标准名称，Value 为关键词列表。 **调整策略**： 1. **添加错别字**：根据 OCR 结果日志，将识别错误的词（如 `yuell`）加入对应品牌的列表。 2. **添加简称**：如将 `买理财`、`找平` 加入 `中国平安` 的列表。 3. **避免冲突**：不要使用太短的通用词（如单字“中”），以免误匹配。 |

### 5 快速推理流程

- 确保模型与数据文件都存在
  - 必须有的模型文件：`outputs/kaggle_model_896/best_model_iou.pth`
  - 原始视频放入 `input/videos` 中，最好是 1080p
  - 如果要跳过 Dinov3 推理直接运行 OCR，需要提前保存推理好的 `output/crops` 目录
- 启动 DINOv3 环境
  - 运行 `infer_video_dinov3.py`，推理并裁剪后的图片输出到 crops 目录下
- 启动 Paddle 环境
  - 运行 `run_paddle_ocr.py`，输出识别结果日志
  - 运行 `analyze_sponsors.py`，输出分析结果
- 启动 football-minimap-generator 环境
  - 具体内容还未实现
