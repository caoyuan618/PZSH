import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_BLIP_DIR = os.path.join(PROJECT_DIR, "BLIP_main")
EXTERNAL_BLIP_DIR = os.environ.get("BLIP_MAIN_DIR")

for path in [LOCAL_BLIP_DIR, EXTERNAL_BLIP_DIR]:
    if path and os.path.isdir(path):
        for candidate in [os.path.dirname(path), path]:
            if candidate not in sys.path:
                sys.path.insert(0, candidate)

try:
    from BLIP_main.models import blip_itm
except ModuleNotFoundError as exc:
    if exc.name == "BLIP_main":
        raise ModuleNotFoundError(
            "BLIP_main is required. Place it under the project root or set "
            "BLIP_MAIN_DIR to an external BLIP source directory."
        ) from exc
    raise


class PZSHEncoder(nn.Module):
    def __init__(self, config, bit):
        super().__init__()
        self.blip = blip_itm.blip_itm(
            pretrained=config["blip_pretrained_pth"],
            med_config=config["blip_med_config"],
            image_size=224,
            vit=config["blip_vit_mode"],
        )
        self.hash_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, bit),
        )

    def forward(self, images):
        vision_embeds = self.blip.visual_encoder(images)
        cls_feature = vision_embeds[:, 0, :]
        aligned_feature = F.normalize(cls_feature, dim=-1)
        hash_code = torch.tanh(self.hash_head(cls_feature))
        return aligned_feature, hash_code


class PZSH(nn.Module):
    def __init__(self, config, bit):
        super().__init__()
        self.momentum = config["momentum"]
        self.encoder_q = PZSHEncoder(config, bit)
        self.encoder_k = PZSHEncoder(config, bit)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    def forward(self, images):
        aligned_feature, query_hash = self.encoder_q(images)
        with torch.no_grad():
            self._update_momentum_encoder()
            _, momentum_hash = self.encoder_k(images)
        return aligned_feature, query_hash, momentum_hash
