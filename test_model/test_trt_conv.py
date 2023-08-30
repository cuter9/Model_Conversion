import os
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageColor
import numpy as np

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
import test_infer_data

PATH_TRT_MODEL_from_ONNX = "/home/cuterbot/Model_Conversion/Onnx_Work_Space/Model_Rep/ssd_mobilenet_v2_coco.engine"
PATH_TRT_MODEL_from_UFF = "/home/cuterbot/Model_Conversion/model_rep/ssd_mobilenet_v2_coco.engine"
WORK = os.getcwd()
WINDOW_NAME = 'TrtSsdModelTest'
INPUT_HW = (300, 300)


def verify_trt_model(path_model, model_type):
    if not os.path.exists(os.path.join(WORK, "000000088462.jpg")):
        str_pic_path = "cd $WORK; wget -q http://images.cocodataset.org/val2017/000000088462.jpg"
        os.system(str_pic_path)
    img = Image.open("000000088462.jpg")
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    # img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    # img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
    # print(img_data.shape)
    img_handle = cv2.imread("000000088462.jpg")

    trt_ssd = TrtSSD(path_model, INPUT_HW)
    test_op = True
    trt_ssd.detect(img_handle, model_type, test_op, conf_th=0.3)


if __name__ == '__main__':
    verify_trt_model(PATH_TRT_MODEL_from_ONNX, "onnx")
    verify_trt_model(PATH_TRT_MODEL_from_UFF, "uff")
