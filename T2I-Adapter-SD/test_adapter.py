import os

import cv2
import torch
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)

torch.set_grad_enabled(False)


def main():
    supported_cond = [e.name for e in ExtraCondition]
    parser = get_base_argument_parser()
    parser.add_argument(
        '--which_cond',
        type=str,
        default='color',  # 默认用颜色adapter
        # required=True,
        choices=supported_cond,
        help='which condition modality you want to test',
    )
    opt = parser.parse_args()
    which_cond = opt.which_cond
    if opt.outdir is None:
        opt.outdir = f'outputs/test-{which_cond}'
    os.makedirs(opt.outdir, exist_ok=True)
    if opt.resize_short_edge is None:
        print(f"you don't specify the resize_shot_edge, so the maximum resolution is set to {opt.max_resolution}")
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # support two test mode: single image test, and batch test (through a txt file)
    if opt.prompt.endswith('.txt'):
        assert opt.prompt.endswith('.txt')
        image_paths = []
        prompts = []
        with open(opt.prompt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_paths.append(line.split('; ')[0])
                prompts.append(line.split('; ')[1])
    else:
        image_paths = [opt.cond_path]
        prompts = [opt.prompt]
    print(image_paths)

    # prepare models
    sd_model, sampler = get_sd_models(opt)
    adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))
    cond_model = None
    if opt.cond_inp_type == 'image':
        cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))

    process_cond_module = getattr(api, f'get_cond_{which_cond}')

    # inference
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):
        for test_idx, (cond_path, prompt) in enumerate(zip(image_paths, prompts)):
            seed_everything(opt.seed)
            for v_idx in range(opt.n_samples):
                # seed_everything(opt.seed+v_idx+test_idx)
                cond = process_cond_module(opt, cond_path, opt.cond_inp_type, cond_model)
 

                # 原来的保存Img代码
                # base_count = len(os.listdir(opt.outdir)) // 2
                # cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_{which_cond}.png'), tensor2img(cond))

                # adapter_features, append_to_context = get_adapter_feature(cond, adapter)
                # opt.prompt = prompt
                # result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)
                # cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_result.png'), tensor2img(result))


                adapter_features, append_to_context = get_adapter_feature(cond, adapter)
                opt.prompt = prompt
                result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)

                # 提取颜色图原始文件名和目录
                cond_dir, cond_filename = os.path.split(cond_path)
                filename_wo_ext, ext = os.path.splitext(cond_filename)
                new_filename = f"{filename_wo_ext}_SDimg{ext}"
                save_path = os.path.join(cond_dir, new_filename)

                # 保存生成结果到颜色图旁边
                cv2.imwrite(save_path, tensor2img(result))
                print(f"✅ Saved to: {save_path}")


if __name__ == '__main__':
    main()


# # when input color image
# python test_adapter.py --which_cond color --cond_path examples/color/color_0002.png --cond_inp_type color --prompt "A photo of scenery" --sd_ckpt models/v1-5-pruned-emaonly.ckpt --resize_short_edge 512 --cond_tau 1.0 --cond_weight 1.0 --n_samples 2 --adapter_ckpt models/t2iadapter_color_sd14v1.pth --scale 9

# # when input non-color image, obtain the color image is also straightforward, just bicubic downsample to low res and nearst upsample normal res
# python test_adapter.py --which_cond color --cond_path examples/sketch/scenery.jpg --cond_inp_type image --prompt "A photo of scenery" --sd_ckpt models/v1-5-pruned-emaonly.ckpt --resize_short_edge 512 --cond_tau 1.0 --cond_weight 1.0 --n_samples 2 --adapter_ckpt models/t2iadapter_color_sd14v1.pth --scale 9