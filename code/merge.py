
import os
from PIL import Image,ImageDraw,ImageFont
# 需要对比的图像路径
src_path=[
    '/home/sh/sh_code/yolov5_GIoU_lq/runs/detect_525/yolov5s_3SK_FPN_SIoU_Ghost/',
    '/home/sh/sh_code/yolov5_GIoU_lq/runs/detect_525/yolov5s/'
    ]
# 结果存储路径
dest_path='/home/sh/sh_code/yolov5_GIoU_lq/runs/detect_525/dest/'

# 是否在图像上显示模型名称
TEXT=True
TEXT_SIZE=10

def process(src_path,dest_path):
    """
    @param src_path:源文件路径列表
    @param dest_path:目标文件路径
    """
    # 获取第一个路径
    first_dir=src_path[0]
    size=len(src_path)
    img_lsit=get_img_list(first_dir)
    for img in img_lsit:
        print('正在处理',img)
        # 从图像列表里取出一个图像名   获取第一个源文件中该文件的尺寸
        img_name=first_dir+img
        img_obj=Image.open(img_name)
        width=img_obj.width
        height=img_obj.height
        mode=img_obj.mode
        # 创建一个空白图像
        result=Image.new(mode,(width*size,height))
        # 遍历所有的文件夹，打开每个文件夹下的该图像，合并
        for index,dir in enumerate(src_path):
            # 获取文件夹名称
            dir_name=dir.split('/')[-2]

            img_to_new=dir+img
            img_obj_draw=Image.open(img_to_new)

            # TEXT文字开关打开，再处理标注文字
            if TEXT:
                # 创建一个空图片，用来标注文件夹名，高度是原始图片+文字图片高度
                text_img=Image.new(mode,size=(width,height+TEXT_SIZE))
                text_img_draw=ImageDraw.Draw(text_img)
                text_img_draw.text(xy=(0,0),text=dir_name,fill=(255,255,255))
                text_img.paste(img_obj_draw,box=(0,TEXT_SIZE))
                img_obj_draw=text_img

            result.paste(img_obj_draw,box=(index*width,0))
        result.save(dest_path+img)
        print('保存到',dest_path+img)


def get_img_list(path):
    return os.listdir(path)


if __name__ == '__main__':
    process(src_path,dest_path)