#!/bin/bash

dir_patches="$HOME/Model_Conversion/YOLO_Work_Space/yolov7_patches"

cd $HOME/Downloads
if [ ! -d "./yolo" ]; then
  mkdir yolo
fi
cd ./yolo

if [ ! -d "./yolov7" ]; then
  git clone https://github.com/WongKinYiu/yolov7
fi
cd ./yolov7

cp "$dir_patches/export_2.py" ./
cp "$dir_patches/export_2_trt.sh" ./
cp "$dir_patches/experimental_2.py" ./models/
cp "$dir_patches/add_nms_2.py" ./utils/

if [ ! -f "yolov7-tiny.pt" ]; then
  # -- Download tiny weights
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
fi

if [ ! -f "yolov7.pt" ]; then
# -- Download regular weights
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
fi


# -- install onnx-simplifier not listed in general yolov7 requirements.txt
# sudo pip3 install onnxsim seaborn coremltools scikit-learn

#
# -- Pytorch Yolov7 -> ONNX with grid, EfficientNMS plugin and dynamic batch size
# python3 export.py --weights ./yolov7.pt --grid --end2end --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.01 --img-size 640 640
# -- Pytorch Yolov7 -> ONNX with grid, EfficientNMS plugin and static batch size
if [ ! -f "yolov7.onnx" ]; then
  python3 export_2.py --weights ./yolov7.pt --grid --end2end --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.25 --img-size 640 640
  # python3 export_2.py --weights ./yolov7.pt --grid --simplify --include-nms
  # python3 export_2.py --weights ./yolov7.pt --grid --end2end  --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img_size 640 640
fi
# shellcheck disable=SC2016
if [ ! -d "$HOME/Data_Repo/Model_Conversion/yolov7/ONNX_Model/Repo" ]; then
  mkdir -p "$HOME/Data_Repo/Model_Conversion/yolov7/ONNX_Model/Repo"
fi
cp ./yolov7.onnx $HOME/Data_Repo/Model_Conversion/yolov7/ONNX_Model/Repo/

if [ ! -f "yolov7-tiny.onnx" ]; then
  python3 export_2.py --weights ./yolov7-tiny.pt --grid --end2end --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.25 --img-size 640 640
  # python3 export_2.py --weights ./yolov7-tiny.pt --grid --simplify --include-nms
  # python3 export_2.py --weights ./yolov7-tiny.pt --grid --end2end  --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img_size 640 640
fi
# shellcheck disable=SC2016
if [ ! -d "$HOME/Data_Repo/Model_Conversion/yolov7-tiny/ONNX_Model/Repo" ]; then
  mkdir -p "$HOME/Data_Repo/Model_Conversion/yolov7-tiny/ONNX_Model/Repo"
fi
cp ./yolov7-tiny.onnx $HOME/Data_Repo/Model_Conversion/yolov7-tiny/ONNX_Model/Repo/
