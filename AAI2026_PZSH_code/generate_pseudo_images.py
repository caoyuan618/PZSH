import argparse
import os
import sys

import cv2
import torch
from pytorch_lightning import seed_everything
from torch import autocast

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
T2I_ADAPTER_DIR = os.environ.get("T2I_ADAPTER_DIR", os.path.join(PROJECT_DIR, "T2I-Adapter-SD"))
if T2I_ADAPTER_DIR not in sys.path:
    sys.path.insert(0, T2I_ADAPTER_DIR)

from ldm.inference_base import diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import ExtraCondition, get_adapter_feature, get_cond_model
from ldm.util import tensor2img


def read_prompt_file(prompt_file):
    pairs = []
    with open(prompt_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            cond_path, prompt = line.split("; ", 1)
            pairs.append((cond_path, prompt))
    return pairs


def main():
    parser = get_base_argument_parser()
    parser.add_argument("--prompt_file", required=True, help="Each line: condition_image_path; prompt")
    parser.add_argument("--which_cond", default="color")
    parser.add_argument("--cond_inp_type", default="color")
    parser.add_argument("--suffix", default="_SDimg")
    opt = parser.parse_args()

    which_cond = opt.which_cond
    pairs = read_prompt_file(opt.prompt_file)
    sd_model, sampler = get_sd_models(opt)
    adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))
    cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond)) if opt.cond_inp_type == "image" else None
    process_cond = getattr(api, f"get_cond_{which_cond}")

    with torch.inference_mode(), sd_model.ema_scope(), autocast("cuda"):
        for sample_idx, (cond_path, prompt) in enumerate(pairs):
            seed_everything(opt.seed + sample_idx)
            cond = process_cond(opt, cond_path, opt.cond_inp_type, cond_model)
            adapter_features, append_to_context = get_adapter_feature(cond, adapter)
            opt.prompt = prompt
            result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)

            cond_dir, cond_filename = os.path.split(cond_path)
            name, ext = os.path.splitext(cond_filename)
            save_path = os.path.join(cond_dir, f"{name}{opt.suffix}{ext}")
            cv2.imwrite(save_path, tensor2img(result))


if __name__ == "__main__":
    main()
