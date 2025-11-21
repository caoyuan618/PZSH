import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import scipy.io as sio
import torch.nn.functional as F

# ===== 如果你的BLIP代码在本地的 ./BLIP/ 路径下，并且那里包含 models/blip_itm.py 等文件 =====
import sys
sys.path.insert(0, './BLIP')  # 让 Python 找得到本地 BLIP 代码

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# ====== 以下是你本地的 blip_itm 模块，以及其内部提供的 blip_itm 类 ======
from models.blip_itm import blip_itm  # 假设该文件里有这个类



# ========== 参数设置 ==========
dialog_json = "/hdd/wcx/codes/ChatIR/dialog/iterative/data/output/chatir_output_1.json"
image_base_dir = "/hdd/wcx/codes/ChatIR/ChatIR-main/coco/val2017_test/"
output_file = "/hdd/wcx/codes/ChatIR/ChatIR-main/BLIP-main/mat5/blip_features_dialogs_2.mat"
num_dialogs = 0  # 可修改为 0~10，控制提取几轮问答文本

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 模型本地加载 ==========

model = blip_itm(
    pretrained='/hdd/wcx/codes/ChatIR/ChatIR-main/models/chatir_weights.ckpt',    # 你的本地ckpt权重
    med_config='BLIP/configs/med_config.json',    # 本地med_config
    image_size=224,
    vit='base'
)
model = model.to(device).eval()




# 定义图像预处理 transform（和 BLIP_BASELINE 中保持一致）
transform_test = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

def extract_image_feature(image_path):
    """使用本地 BLIP 模型来提取图像特征"""
    raw_image = Image.open(image_path).convert('RGB')
    image_tensor = transform_test(raw_image).unsqueeze(0).to(device)  # batch_size=1
    with torch.no_grad():
        # 1) 视觉编码
        vision_embeds = model.visual_encoder(image_tensor)         # [1, num_tokens, hidden_dim]
        # 2) 只取第0个token的输出做投影
        projection = model.vision_proj(vision_embeds[:, 0, :])     # [1, hidden_dim]
        # 3) 归一化
        projection = F.normalize(projection, dim=-1)               # [1, hidden_dim]
    return projection[0].cpu().numpy()  # 返回 (hidden_dim, ) 的numpy向量

def extract_text_feature(text):
    """使用本地 BLIP 模型来提取文本特征"""
    text_input = model.tokenizer(
        text,
        padding='longest',
        truncation=True,
        max_length=200,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        text_output = model.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            return_dict=True,
            mode='text'
        )
        # 同样只取第0个token (CLS)
        txt_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
        txt_feat = F.normalize(txt_feat, dim=-1)
    return txt_feat[0].cpu().numpy()

# ========== 数据读取 ==========
with open(dialog_json, "r", encoding="utf-8") as f:
    dialog_data = json.load(f)

image_paths = []
texts = []

for item in dialog_data:
    img_path = os.path.join(image_base_dir, item["img"])
    caption = item["caption"]
    image_paths.append(img_path)
    texts.append(caption)

print(image_paths)
print(texts)

assert len(image_paths) == len(texts), "图像和文本数量不一致！"

# ========== 特征提取 ==========
image_features = []
text_features = []

for image_path, text in tqdm(zip(image_paths, texts), total=len(image_paths)):
    try:
        # 提取图像特征
        img_feat = extract_image_feature(image_path)
        # 提取文本特征
        txt_feat = extract_text_feature(text)

        image_features.append(img_feat)
        text_features.append(txt_feat)

    except Exception as e:
        print(f"[!] 跳过错误样本 {image_path}，错误信息：{e}")
        continue

# ========== 保存归一化特征 ==========
sio.savemat(output_file, {
    "image_features": np.vstack(image_features),
    "text_features": np.vstack(text_features),
})
print(f"✅ 特征提取并归一化完成，保存至：{output_file}")