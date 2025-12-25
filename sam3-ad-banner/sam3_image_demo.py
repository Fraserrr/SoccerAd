import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model
# 建议显式指定 device，否则模型可能在 CPU 上跑，速度很慢
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

model = build_sam3_image_model()
model.to(device) # 移动模型到显卡
processor = Sam3Processor(model)

# Load an image
# 关键修改：添加 .convert("RGB") 防止 PNG 透明通道报错
image = Image.open("test_images/img1.png").convert("RGB")

print("Processing image...")
inference_state = processor.set_image(image)

# Prompt the model with text
print("Prompting with text 'Ad banner'...")
output = processor.set_text_prompt(state=inference_state, prompt="Ad banner")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

print(f"Success! Detected {len(masks)} objects.")
print(f"Scores: {scores}")