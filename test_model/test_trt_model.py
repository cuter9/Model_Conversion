import os
import subprocess
import sys
import wget
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageColor

import cv2

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

# test SSD model with cuda through pycuda.driver
# install pycuda : https://pypi.org/project/pycuda/2024.1/#description
# sudo python3 setup.py bdist_wheel; sudo pip3 install pycuda-2024.1-cp38-cp38-linux_aarch64.whl
WORK = os.getcwd()

DATA_REPO_DIR = os.path.join(os.environ["HOME"], "Data_Repo/Model_Conversion/SSD_mobilenet")
# DATA_REPO_DIR_FPN = os.path.join(os.environ["HOME"], "Data_Repo/Model_Conversion/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8")
DATA_REPO_DIR_FPN = os.path.join(os.environ["HOME"], "Data_Repo/Model_Conversion/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8")
# DATA_REPO_DIR_FPN = os.path.join(os.environ["HOME"], "Data_Repo/Model_Conversion/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8")
# DATA_REPO_DIR_FPN = os.path.join(os.environ["HOME"], "Data_Repo/Model_Conversion/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8")

TEST_DIR = os.path.join(DATA_REPO_DIR, "Test Data")

if os.path.isdir(TEST_DIR):
    subprocess.call(['rm', '-r', TEST_DIR])
subprocess.call(['mkdir', '-p', TEST_DIR])

FPN = True
# os.makedirs(TEST_DIR, exist_ok=True)

# PATH_TRT_MODEL_from_ONNX = "/home/cuterbot/Model_Conversion/SSD_Work_Space/ONNX_Model/Repo/ssd_mobilenet_v2_coco.engine"
# PATH_TRT_MODEL_from_UFF = "/home/cuterbot/Model_Conversion/SSD_Work_Space/UFF_Model/Repo/ssd_mobilenet_v2_coco.engine"
# engine_name = "ssd_mobilenet_v2_coco.engine"
if FPN:
    # engine_name = "ssd_mobilenet_v2_fpnlite_320x320_coco17.engine"
    engine_name = "ssd_mobilenet_v2_fpnlite_640x640_coco17.engine"
    # engine_name = "ssd_mobilenet_v1_fpn_640x640_coco17.engine"
    # engine_name = "ssd_resnet50_v1_fpn_640x640_coco17.engine"
    PATH_TRT_MODEL_from_ONNX = os.path.join(DATA_REPO_DIR_FPN, "ONNX_Model/Repo", engine_name)
else:
    engine_name = "ssd_mobilenet_v2_320x320_coco17_tpu-8_tf_v2.engine"
    PATH_TRT_MODEL_from_ONNX = os.path.join(DATA_REPO_DIR, "ONNX_Model/Repo", engine_name)

PATH_TRT_MODEL_from_UFF = os.path.join(DATA_REPO_DIR, "UFF_Model/Repo/", engine_name)

WINDOW_NAME = 'TrtSsdModelTest'

if FPN:
    # INPUT_HW = (320, 320)   # "ssd_mobilenet_v2_fpn_320x320_coco.engine"
    INPUT_HW = (640, 640)   # "ssd_mobilenet_v2_fpn_640x640_coco.engine"
else:
    INPUT_HW = (300, 300)   # "ssd_mobilenet_v2_coco.engine"


def verify_trt_model(path_model, model_type):
    test_img = os.path.join(TEST_DIR, "test.jpg")
    if not os.path.exists(test_img):
        wget.download("http://images.cocodataset.org/val2017/000000088462.jpg", out=test_img)
    img = Image.open(test_img)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    # img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    # img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
    # print(img_data.shape)
    img_handle = cv2.imread(test_img)

    cls_dict = get_cls_dict("coco")
    trt_ssd = TrtSSD(path_model, INPUT_HW)

    vis = BBoxVisualization(cls_dict)
    test_op = False

    # boxes: [[x_min_object_box, y_min_object_box, x_max_object_box, y_max_object_box], []] for draw
    boxes, confs, clss = trt_ssd.detect(img_handle, model_type, test_op, fpn=FPN, conf_th=0.3)
    img_handle = vis.draw_bboxes(img_handle, boxes, confs, clss)
    cv2.imshow(WINDOW_NAME, img_handle)
    while True:
        key = cv2.waitKey(1)
        full_scrn = False
        if key == 27:  # ESC key: quit program
            sys.exit()
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


if __name__ == '__main__':
    verify_trt_model(PATH_TRT_MODEL_from_ONNX, "onnx")
    # verify_trt_model(PATH_TRT_MODEL_from_UFF, "uff")
