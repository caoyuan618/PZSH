import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BLIP_MAIN_DIR = os.environ.get("BLIP_MAIN_DIR", os.path.join(PROJECT_DIR, "BLIP_main"))
for candidate in [os.path.dirname(BLIP_MAIN_DIR), BLIP_MAIN_DIR]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from BLIP_main.models import blip_itm


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def extract_feature(model, transform, image_path, device):
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        vision_embeds = model.visual_encoder(image)
        feature = F.normalize(vision_embeds[:, 0, :], dim=-1)
    return feature[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Extract BLIP features for PZSH training lists")
    parser.add_argument("--input_list", required=True)
    parser.add_argument("--output_list", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--image_field", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = blip_itm.blip_itm(
        pretrained=os.path.join(BLIP_MAIN_DIR, "models", "BLIP_base.pth"),
        med_config=os.path.join(BLIP_MAIN_DIR, "configs", "med_config.json"),
        image_size=224,
        vit="base",
    ).to(device).eval()
    transform = build_transform()

    with open(args.input_list, "r", encoding="utf-8") as fin, open(args.output_list, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Extracting BLIP features"):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            image_path = parts[args.image_field]
            if not os.path.isabs(image_path):
                image_path = os.path.join(args.image_root, image_path)
            feature = extract_feature(model, transform, image_path, device)
            feature_text = " ".join(f"{value:.6f}" for value in feature)
            fout.write(line + "\t" + feature_text + "\n")


if __name__ == "__main__":
    main()
