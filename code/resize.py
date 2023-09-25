import os
from PIL import Image
# 读取文件夹下所有图片
img_folder = "/home/sh/sh_code/yolov5_GIoU_lq/runs/detect_525/yolov5s_3SK_FPN_SIoU_Ghost/"
img_list = []
for filename in os.listdir(img_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_list.append(os.path.join(img_folder, filename))
# 获取所有图片的宽高，取最大值作为新图像的宽高
max_width, max_height = 0, 0
for img_file in img_list:
    with Image.open(img_file) as img:
        width, height = img.size
        max_width = max(max_width, width)
        max_height = max(max_height, height)
# 为每一张图片创建一个新的图片，并将原始图像复制进去
for img_file in img_list:
    with Image.open(img_file) as img:
        width, height = img.size
        new_img = Image.new(mode=img.mode, size=(max_width, max_height))
        new_img.paste(img, box=((max_width-width)//2, (max_height-height)//2))
        new_file = os.path.splitext(img_file)[0] + "_new.png"
        new_img.save(new_file)