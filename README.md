# PZSH: Proxy Zero-Shot Hashing with Multimodal Fusion via Stable Diffusion

## Project Introduction

PZSH is a zero-shot hashing image retrieval framework based on Stable Diffusion and dual-branch contrastive learning.       By generating semantically rich pseudo-images through multimodal fusion, the framework addresses issues such as insufficient modal alignment and semantic drift in traditional zero-shot hashing methods, achieving superior retrieval performance on unseen categories.

Core Advantages:

- Multimodal fusion generation based on Stable Diffusion, deeply integrating visual, semantic, and label information
- Dual-branch BLIP encoder contrastive learning to enhance fine-grained semantic alignment
- Zero-image scenario data augmentation strategy to alleviate the cold-start problem
- Proxy hashing loss function to improve the discriminability of binary codes

Envoirment Installation Command:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Two mainstream zero-shot retrieval datasets are supported, which should be placed in the following structure:

### 1.       AWA2 (Animals with Attributes 2)

- Dataset Path: `dataset/AWA/`
- Contains 50 animal categories (40 seen classes, 10 unseen classes)
- Prerequisites: JPEGImages folder, class attribute vectors, training/test/database split files (.txt)

### 2.       CUB (Caltech-UCSD Birds-200-2011)

- Dataset Path: `dataset/CUB/`
- Contains 200 bird categories (150 seen classes, 50 unseen classes)
- Prerequisites: Image folder, 312-dimensional attribute vectors, split files (.txt)

Data set download link: https://pan.baidu.com/s/1JBlRiE9wF6bELNLSlRL4tg?pwd=6pxg

Extraction code: 6pxg


## Core Code File Description

| File Name                     | Function Description                                         |
| ----------------------------- | ------------------------------------------------------------ |
| `PSZH_main.py`                | Main training file, including BLIP hash network definition, loss functions, and training/validation workflow |
| `get_persudo_img.py`          | Text-to-image generation tool, synthesizing pseudo-images based on Stable Diffusion |
| `tools_with_category_name.py` | Data loading tool, supporting dataset reading with category names and BLIP features |
| `tools.py`                    | Basic utility functions, including image preprocessing, hash code calculation, and retrieval accuracy evaluation |

## Quick Start

### 1.       Generate Pseudo-Images (Zero-Image Augmentation)

```bash
python get_persudo_img.py --prompt "A photo of a <label>" --outdir outputs/txt2img-samples --ddim_steps 50 --scale 7.5
```

- `--prompt`: Text prompt template, replace `<label>` with category name
- `--outdir`: Path to save generated images
- `--ddim_steps`: Number of diffusion model sampling steps
- `--scale`: Guidance scale coefficient

### 2.       Model Training

```bash
python PSZH_main.py --dataset AWA --caption 1 --TGI 1 --freeze_img 1 --freeze_txt 1 --save_path save/PZSH_AWA
```

- `--dataset`: Select dataset (AWA/CUB)
- `--caption`: Whether to use text descriptions (1=use, 0=not use)
- `--TGI`: Whether to enable zero-image augmentation (1=enable, 0=disable)
- `--freeze_img`/`--freeze_txt`: Whether to freeze image/text encoder
- `--save_path`: Path to save models and results

### 3.       Key Parameter Configuration

Core configurations are defined in the `get_config` function, with main parameters:

- `bit_list`: Hash code length (default 64, supports 24/48/64/128)
- `batch_size`: Batch size (default 32)
- `epoch`: Number of training epochs (default 25)
- `lr`: Learning rate (default 1e-5)
- `temperature`: Contrastive loss temperature parameter (default 0.07)

## Result Explanation

### Evaluation Metrics

- Main Metric: Mean Average Precision (mAP@all)
- Supports performance evaluation for different hash code lengths (24/48/64/128 bits)

### Result Saving

After training, the following are automatically saved:

- Hash codes of the optimal model (training set/test set)
- Training logs (including loss curves, mAP changes)
- Model weight files

### Expected Performance

On the AWA2 dataset (128 bits): mAP up to 0.6477;       On the CUB dataset (128 bits): mAP up to 0.2775, outperforming mainstream baseline models.

## Code Repository

The project code is open-source: https://github.com/caoyuan618/PZSH

## Notes

1. Ensure the Stable Diffusion model weights (model.ckpt："models\ldm\stable-diffusion-v1\model.ckpt") are placed in the specified path. And Place the BLIP pre-trained model at "\BLIP_main\models\BLIP_base.pth"
2. Dataset split files must be formatted strictly (image path\tlabel\tfeature)
3. Check GPU memory before training (≥32GB recommended)