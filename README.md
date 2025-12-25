# 基于 Dinov3 与 PaddleOCR 的足球场广告板识别

Dinov3 参考仓库：

[damdip/SoccerAdBannerSegmentation-Replacement: Deep Learning project about ad banner detection and replacement in soccer matches videos](https://github.com/damdip/SoccerAdBannerSegmentation-Replacement)

### 1 环境配置

#### DINOv3 环境

**python 3.10**

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

#### TVCallib 环境

参考 huggingface 仓库文档 https://huggingface.co/spaces/ramseyy10/football-minimap-generator

#### SAM3 环境

参考官方仓库 [facebookresearch/sam3](https://github.com/facebookresearch/sam3)

sam3 需要的库版本很新，需要独立于 tvcalib 和 PaddleOCR，可用的配置方法：

```bash
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3

pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install git+https://github.com/facebookresearch/sam3.git

pip install opencv-python pandas tqdm matplotlib
```

运行时可能还有部分库缺失，根据报错逐个安装即可，其中 windows 系统下是没有 Triton 库的，如果出现报错需要安装：`pip install triton-windows`

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

### 5 视频 TVCalib 推理

本模块利用计算机视觉几何技术，建立摄像机画面与标准足球场平面（Minimap）之间的映射关系。通过 TvCalib 模型提取球场关键点并计算单应性矩阵（Homography），将预先标记好的广告牌世界坐标投影回摄像机图像平面，通过几何相交判定广告牌是否出现在当前直播画面中，从而生成精准的广告曝光时间轴。

#### 场地广告位置标定

使用 `ad_config_builder.py` 脚本定义标准球场尺寸及周边的广告牌分布。该脚本统一使用 **码 (Yards)** 为单位，以球场左上角为坐标原点，生成 JSON 配置文件与可视化预览图，确保逻辑坐标系与渲染坐标系的一致性。

**可调参数表：**

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `FIELD_LENGTH` | float | 114.83 | 球场长度（单位：码），需与 Minimap 渲染尺寸匹配 |
| `FIELD_WIDTH` | float | 74.37 | 球场宽度（单位：码） |
| `AD_CONFIGS` | list | [...] | 广告牌列表。包含 `name` (名称), `side` (top/bottom/left/right), `start`/`end` (0.0-1.0 比例位置) |

#### 推理运行

使用 `video_ad_analyzer.py` 脚本对视频进行抽帧分析。核心逻辑采用 World-to-Image 反向投影方案：利用逆单应性矩阵将广告牌的世界坐标投影至图像坐标系，判断投影线段是否与图像矩形框相交。该方法可以避免透视变换中远端/天空区域导致的坐标发散问题。

**可调参数表：**

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `DEBUG_VISUALIZE` | bool | `True` | 是否开启调试模式，输出包含 Minimap 视锥和广告识别红绿线的图片 |
| `SAMPLE_FPS` | float | 1.0 | 采样率，每秒分析几帧 |
| `OPTIM_STEPS` | int | 100 | TvCalib 优化迭代步数，数值越低速度越快但精度略降 |
| `CONFIG_FILE` | str | `'ad_map_config.json'` | 读取的广告配置文件路径（由上一步生成） |
| `OUTPUT_CSV` | str | `'ad_timeline.csv'` | 最终生成的广告曝光时间轴文件 |

#### 识别失败镜头的区分与标记

使用 `video_ad_view_validate.py` 脚本，推理完成后根据结果剔除掉识别失败的帧，并且做标记。

由于 TvCalib 在特写镜头、长焦视角或缺乏特征的画面中容易产生错误的定标结果（如将特写误判为全景，或生成“卫星视角”式的平面投影），本工程设计了一套基于标准视角特征的白名单校验机制。对每一帧计算出的单应性矩阵进行多维度的几何合理性检查，只有完全符合标准视角几何特征的帧才会被标记为 `VALID`，否则标记具体拒绝原因，确保广告统计数据的准确性。

**核心校验逻辑：**

- **透视纵深梯度 (Perspective Depth Gradient)**：这是区分正确视角与错误“平面投影”的关键特征。系统强制要求图像上部的像素覆盖的物理纵深必须显著大于下部像素（符合“近大远小”的透视原理），剔除 TvCalib 生成的矩形或平行光柱状错误投影。
- **物理尺度约束**：图像底边（Near Plane）在球场平面上覆盖的物理宽度小于 110 码的合理范围，过宽判定为高空错误。
- **姿态锁定**：
  - **底边水平约束**：图像底边在 Minimap 上的投影必须保持大致水平，防止视角严重倾斜或旋转。
  - **梯形扩张性**：视锥形态必须满足“近窄远宽”的梯形特征。
- **摄像机禁区**：反推的摄像机位置必须位于球场边界之外，且摄像机距离需在合理范围内（小于150码）。

**输出：** CSV 日志文件中的 `status` 列会详细记录每一帧的校验结果，便于后续处理，同时 `debug_output` 目录将推理结果可视化，便于分析或调试。

### 6 SAM3 分割广告牌

这部分工程位于 `sam3-ad-banner` 目录下，需要独立环境，参考环境配置部分介绍。

- 运行 `sam3_process.py` ，会对上一步 tvcalib 识别失败的帧图片进行分割
  - 可视化结果位于 `sam3_results` 目录
  - 切割出来的广告牌图片位于 `crops` 目录，可用于后续 OCR
- 用于DEMO：运行 `sam3_video_demo.py`，对示例视频进行推理和可视化（视频推理与图片推理的逻辑不同，环境可能需要补充安装一些包，暂时还没有验证）

### 7 快速推理流程

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
  - 生成场地配置（仅需运行一次，或在修改广告分布后运行）
    - 运行 `ad_config_builder.py`，检查生成的 `ad_map_visualization.png` 确认广告位无误。
  - 执行视频分析并输出结果
    - 运行 `video_ad_view_validate.py`。
    - 查看生成的 `ad_timeline_validate.csv` 获取广告曝光日志，或查看 `debug_output_v2/` 目录下的可视化校验图。
