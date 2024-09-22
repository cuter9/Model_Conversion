import torch
import os
import sys

model = torch.nn.ModuleList()
sys.path.insert(0, os.path.join(os.environ["HOME"], "Downloads/yolo/yolo_v7/"))
path_weight = os.path.join(os.environ["HOME"], "Downloads/yolo/yolo_v7/yolo_v7-tiny.pt")
ckpt = torch.load(path_weight, map_location=torch.device('cpu'))  # load
model = model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())[0]  # FP32 model
labels = model.names
stride = model.stride
modules = model.named_modules()