#!/bin/bash
cd $HOME/Downloads
if [ ! -d "./yolo" ]; then
  mkdir yolo
fi
cd yolo

if [ ! -d "./yolo_v7" ]; then
  git clone https://github.com/WongKinYiu/yolov7
  mv yolov7 yolo_v7
  cd yolo_v7
#
# -- Download tiny weights
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
  mv yolov7-tiny.pt yolo_v7-tiny.pt
# -- Download regular weights
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
  mv yolov7.pt yolo_v7.pt
  cd ../
fi
# -- install onnx-simplifier not listed in general yolov7 requirements.txt
sudo pip3 install onnxsim seaborn coremltools

cd yolo_v7

#
# -- Pytorch Yolov7 -> ONNX with grid, EfficientNMS plugin and dynamic batch size
# python3 export.py --weights ./yolo_v7.pt --grid --end2end --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.01 --img-size 640 640
# -- Pytorch Yolov7 -> ONNX with grid, EfficientNMS plugin and static batch size
if [ ! -f "./yolo_v7.onnx" ]; then
  python3 export.py --weights ./yolo_v7.pt --grid --end2end  --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.01 --img-size 640 640
fi
# shellcheck disable=SC2016
if [ ! -d "$HOME/Data_Repo/Model_Conversion/yolo_v7/ONNX_Model/Repo" ]; then
  mkdir -p "$HOME/Data_Repo/Model_Conversion/yolo_v7/ONNX_Model/Repo"
fi
cp ./yolo_v7.onnx $HOME/Data_Repo/Model_Conversion/yolo_v7/ONNX_Model/Repo/

if [ ! -f "./yolo_v7-tiny.onnx" ]; then
  python3 export.py --weights ./yolo_v7-tiny.pt --grid --end2end  --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.01 --img-size 640 640
fi
# shellcheck disable=SC2016
if [ ! -d "$HOME/Data_Repo/Model_Conversion/yolo_v7-tiny/ONNX_Model/Repo" ]; then
  mkdir -p "$HOME/Data_Repo/Model_Conversion/yolo_v7-tiny/ONNX_Model/Repo"
fi
cp ./yolo_v7-tiny.onnx $HOME/Data_Repo/Model_Conversion/yolo_v7-tiny/ONNX_Model/Repo/
