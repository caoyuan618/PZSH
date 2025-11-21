import os
from PIL import Image

def pixelate_image(input_path, output_path, pixel_size=8):
    img = Image.open(input_path).convert("RGB")
    width, height = img.size
    small_img = img.resize((pixel_size, pixel_size), resample=Image.BILINEAR)
    result = small_img.resize((width, height), Image.NEAREST)
    result.save(output_path)

def process_awa_images(root_dir, pixel_size=8):
    image_exts = ('.jpg', '.jpeg', '.png')

    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if not filename.lower().endswith(image_exts):
                continue

            input_path = os.path.join(class_path, filename)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_msk{ext}"
            output_path = os.path.join(class_path, output_filename)

            try:
                pixelate_image(input_path, output_path, pixel_size=pixel_size)
                print(f"Processed: {output_path}")
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

# 设置你的图像根目录
root_image_dir = "/hdd/zh/Hash/DeepHash-pytorch-master/data/CUB/CUB-last50_is_txt2img/images"
process_awa_images(root_image_dir, pixel_size=8)
