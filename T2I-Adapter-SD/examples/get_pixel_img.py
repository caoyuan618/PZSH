from PIL import Image

def pixelate_image(input_path, output_path, pixel_size=8):
    # 打开图像并转换为 RGB 模式
    img = Image.open(input_path).convert("RGB")
    
    # 原图尺寸
    width, height = img.size
    
    # 缩小图像（变成 pixel_size x pixel_size）
    small_img = img.resize((pixel_size, pixel_size), resample=Image.BILINEAR)
    
    # 再放大回原图尺寸（用最近邻插值，不做平滑）
    result = small_img.resize((width, height), Image.NEAREST)
    
    # 保存图像
    result.save(output_path)
    result.show()

# 示例调用
pixelate_image(
    input_path="/hdd/zh/Hash/DeepHash-pytorch-master/data/AWA/JPEGImages/02.grizzly+bear/grizzly+bear_10001.jpg",
    output_path="/hdd/zh/Hash/sd_hash/SD14/T2I-Adapter-SD/examples/grizzly+bear_10001_8x8.jpg",
    pixel_size=8
)
