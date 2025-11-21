import torch
from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipModel

# 1. 加载 BLIP 模型和处理器
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")

# 是否使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. 读取图像
image_path = "your_image.jpg"  # 替换为你的图像路径
image = Image.open(image_path).convert("RGB")

# 3. 图像预处理
inputs = processor(images=image, return_tensors="pt").to(device)

# 4. 提取图像特征
with torch.no_grad():
    vision_outputs = model.vision_model(**inputs)
    image_embeds = vision_outputs.last_hidden_state  # [1, num_patches + 1, hidden_dim]

# 可选：提取 CLS token 作为图像全局表示
cls_feature = image_embeds[:, 0, :]  # shape: [1, 768] for base model

# 5. 打印输出形状
print("图像所有 patch token + CLS token 特征 shape:", image_embeds.shape)
print("图像 CLS token 全局特征 shape:", cls_feature.shape)
