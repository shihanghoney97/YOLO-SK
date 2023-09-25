import torch
print("开始执行第一步 model加载")
# Model
model = torch.hub.load(repo_or_dir='/admin006/wsh/yolov5',model='yolov5s',source='local')  # or yolov5m, yolov5l, yolov5x, custom
print("第二步定义图像的URL")
# Images
img = '/admin006/wsh/zidane原始.jpg'  # or file, Path, PIL, OpenCV, numpy, list
print("第三步将图像输入到模型进行检测 Inference")
# Inference
results = model(img)
print("第四步 输出结果  这里使用save方法")
# Results
results.save("/admin006/wsh")  # or .show(), .save(), .crop(), .pandas(), etc.
