# PZSH

Official reproduction code for **Proxy Zero-Shot Hashing with Multimodal
Fusion via Stable Diffusion (PZSH)**.

## Structure

- `train.py`: training and retrieval evaluation.
- `model.py`: BLIP-based PZSH encoder and momentum branch.
- `loss.py`: feature alignment loss and proxy hash loss.
- `A_utils/tools_with_category_name.py`: dataset loading and mAP evaluation.
- `generate_pseudo_images.py`: optional T2I-Adapter pseudo-image generation.
- `extract_blip_features.py`: offline BLIP feature extraction for generated images.

## Data Preparation

PZSH uses pseudo-images generated offline by Stable Diffusion/T2I-Adapter. The
training loader expects the final field of each training line to be the
normalized 768-dimensional BLIP feature of the generated pseudo-image.

Training list format:

```text
image_relative_path<TAB>one_hot_label<TAB>...<TAB>synthetic_blip_feature_768
```

Test and database list format:

```text
image_relative_path<TAB>one_hot_label
```

The default paths are defined in `A_utils/tools_with_category_name.py`.

## Dependencies

Install the training dependencies:

```bash
pip install -r requirements.txt
```

BLIP is required for training and feature extraction. Place `BLIP_main` under
this project or set:

```bash
export BLIP_MAIN_DIR=/path/to/BLIP_main
```

T2I-Adapter is only required for `generate_pseudo_images.py`. Place
`T2I-Adapter-SD` under this project or set:

```bash
export T2I_ADAPTER_DIR=/path/to/T2I-Adapter-SD
```

## Training

```bash
python train.py --dataset AWA --bits 24 48 64 128 --save_path outputs/PZSH
```

For CUB:

```bash
python train.py --dataset CUB --bits 24 48 64 128 --save_path outputs/PZSH
```

## Offline Feature Extraction

```bash
python extract_blip_features.py \
  --input_list dataset/AWA/JPEGImages/train_with_generated_images.txt \
  --output_list dataset/AWA/JPEGImages/train_with_generated_blip768F.txt \
  --image_root dataset/AWA/JPEGImages \
  --image_field -1
```

The input list should contain the generated pseudo-image path in the selected
`--image_field`; the output appends the 768-dimensional BLIP feature.

## Pseudo-Image Generation

Prepare a prompt file where each line is:

```text
condition_image_path; A photo of a <attribute> <class_name>
```

Then run:

```bash
python generate_pseudo_images.py --prompt_file prompts.txt
```

Model checkpoint paths are controlled by the T2I-Adapter argument parser.
