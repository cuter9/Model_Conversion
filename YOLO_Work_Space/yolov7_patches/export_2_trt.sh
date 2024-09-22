#!/bin/bash

clear
MODEL_NAME="yolov7-tiny"
path_weight_file='./'$MODEL_NAME'.pt'
path_onnx_model='./'$MODEL_NAME'.onnx'
path_trt_engine='./'$MODEL_NAME'_fp16.engine'
export PATH=$PATH:/usr/src/tensorrt/bin/
python3 export_2.py --grid --simplify --include-nms --weights=$path_weight_file 2>&1 | tee log/onnx_export.txt
trtexec --onnx=$path_onnx_model --saveEngine=$path_trt_engine --fp16 --workspace=800 --verbose 2>&1 | tee log/trt_export.txt

export DISPLAY=:10.0
python3 infer.py --trt-engine=$path_trt_engine --conf-thres=0.3 --source=inference/images/test.jpg

