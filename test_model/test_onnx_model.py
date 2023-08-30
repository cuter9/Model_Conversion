import os
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageColor
import numpy as np

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

PATH_ONNX_MODEL = "/home/cuterbot/Model_Conversion/Onnx_Work_Space/Model_Rep/ssd_mobilenet_v2_coco.engine"
WORK = os.getcwd()
WINDOW_NAME = 'TrtSsdModelTest'
INPUT_HW = (300, 300)


def verify_trt_model(path_model):
    if not os.path.exists(os.path.join(WORK, "000000088462.jpg")):
        str_pic_path = "cd $WORK; wget -q http://images.cocodataset.org/val2017/000000088462.jpg"
        os.system(str_pic_path)
    img = Image.open("000000088462.jpg")
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
    print(img_data.shape)
    img_handle = cv2.imread("000000088462.jpg")

    cls_dict = get_cls_dict("coco")
    trt_ssd = TrtSSD(path_model, INPUT_HW)

    vis = BBoxVisualization(cls_dict)
    boxes, confs, clss = trt_ssd.detect(img_handle, conf_th=0.3)
    img_handle = vis.draw_bboxes(img_handle, boxes, confs, clss)
    cv2.imshow(WINDOW_NAME, img_handle)
    key = cv2.waitKey(1)
    full_scrn = False
    if key == 27:  # ESC key: quit program
        sys.exit()
    elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
        full_scrn = not full_scrn
        set_display(WINDOW_NAME, full_scrn)


if __name__ == '__main__':
    verify_trt_model(PATH_ONNX_MODEL)
