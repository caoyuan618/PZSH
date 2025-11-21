import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# ====== 设置路径并导入 BLIP 模块 ======
# 指定本地 BLIP 代码所在的根目录，并将其加入 Python 搜索路径
blip_root = os.path.abspath("/hdd/zh/Hash/sd_hash/SD14/BLIP_main")
sys.path.insert(0, blip_root)
print(f"sys.path = {sys.path}")

# 导入 BLIP 模型
from models.blip_itm import blip_itm

# ====== 文件路径参数 ======
input_txt_file = "/hdd/zh/Hash/DeepHash-pytorch-master/data/AWA/JPEGImages/train_100_with_caption_catgoryname_AttrVoc_mskimg_SDimg.txt"  
# 输入：每行包含原图、特征、描述、类别、马赛克图、StableDiffusion生成图等信息

output_txt_file = "/hdd/zh/Hash/DeepHash-pytorch-master/data/AWA/JPEGImages/train_100_with_caption_catgoryname_AttrVoc_mskimg_SDimg_blip768F.txt"
# 输出：在原基础上增加一列 BLIP 特征向量

image_base_dir = "/hdd/zh/Hash/DeepHash-pytorch-master/data/AWA/JPEGImages"
# 图像的基准目录，用于拼接Stable Diffusion生成图的完整路径

# ====== 设备设置 ======
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 加载 BLIP 模型 ======
model = blip_itm(
    pretrained='/hdd/zh/Hash/sd_hash/SD14/BLIP_main/models/BLIP_base.pth',  # 预训练权重路径
    med_config='BLIP_main/configs/med_config.json',  # 配置文件路径
    image_size=224,  # 输入图像尺寸
    vit='base'  # 使用的ViT骨干网络版本
)
model = model.to(device).eval()  # 移动到设备上，并设置为评估模式（不进行梯度更新）

# ====== 图像预处理流程 ======
transform_test = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),  # 调整图像尺寸到224x224
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize(  # 使用BLIP默认的均值和标准差进行归一化
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

def extract_image_feature(image_path):
    """
    给定图像路径，提取图像的BLIP视觉特征（经过投影并归一化）
    返回：numpy数组
    """
    raw_image = Image.open(image_path).convert('RGB')  # 打开并转为RGB格式
    image_tensor = transform_test(raw_image).unsqueeze(0).to(device)  # 预处理并添加batch维度
    with torch.no_grad():
        vision_embeds = model.visual_encoder(image_tensor)  # 通过BLIP视觉编码器提取特征
        # projection = model.vision_proj(vision_embeds[:, 0, :])  # 取CLS token特征，并通过线性投影（vision_proj这个方法是将768投影到256维度特征）
        projection = vision_embeds[:, 0, :]     # 直接输出768维度特征。
        projection = F.normalize(projection, dim=-1)  # L2归一化
    return projection[0].cpu().numpy()  # 返回单张图片的特征向量

# ====== 主处理逻辑：逐行处理输入文件，提取特征并扩展每行 ======
with open(input_txt_file, "r", encoding="utf-8") as fin, open(output_txt_file, "w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="Processing"):
        line = line.strip()
        if not line:
            continue  # 跳过空行
        
        parts = line.split('\t')
        if len(parts) < 2:
            continue  # 数据不完整，跳过
        
        sd_image_rel_path = parts[-1]  # 稳定扩散生成图的相对路径是最后一列
        sd_image_abs_path = os.path.join(image_base_dir, sd_image_rel_path)  # 拼接成完整路径

        try:
            features = extract_image_feature(sd_image_abs_path)  # 提取Stable Diffusion图像的特征
            feature_str = " ".join([f"{v:.6f}" for v in features])  # 特征格式化为6位小数，用空格连接
            new_line = line + '\t' + feature_str  # 将特征字符串作为新字段添加
            fout.write(new_line + '\n')  # 写入输出文件
        except Exception as e:
            # 如果遇到图像损坏或路径错误，跳过此样本并打印错误
            print(f"[!] 错误跳过: {sd_image_abs_path}, 原因: {e}")
            continue

print(f"✅ 所有图像特征已成功添加并保存到：{output_txt_file}")
