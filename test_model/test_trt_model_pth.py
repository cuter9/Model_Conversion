import os
import sys
import wget
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageColor

import cv2

from utils.ssd_classes import get_cls_dict
from utils.tensorrt_model import TRTModel
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

# test SSD model with cuda through pytorch.cuda
WORK = os.getcwd()
DATA_REPO_DIR = os.path.join("../../", "Data_Repo/Model_Conversion/SSD_mobilenet")
TEST_DIR = os.path.join(DATA_REPO_DIR, "Test Data")
os.makedirs(TEST_DIR, exist_ok=True)

# PATH_TRT_MODEL_from_ONNX = "/home/cuterbot/Model_Conversion/SSD_Work_Space/ONNX_Model/Repo/ssd_mobilenet_v2_coco.engine"
# PATH_TRT_MODEL_from_UFF = "/home/cuterbot/Model_Conversion/SSD_Work_Space/UFF_Model/Repo/ssd_mobilenet_v2_coco.engine"
PATH_TRT_MODEL_from_ONNX = os.path.join(DATA_REPO_DIR, "ONNX_Model/Repo/ssd_mobilenet_v2_coco.engine")
PATH_TRT_MODEL_from_UFF = os.path.join(DATA_REPO_DIR, "UFF_Model/Repo/ssd_mobilenet_v2_coco.engine")

WINDOW_NAME = 'TrtSsdModelTest'


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
    img_handle = cv2.imread("000000088462.jpg")

    cls_dict = get_cls_dict("coco")
    # trt_ssd = TrtSSD(path_model, INPUT_HW)
    trt_ssd = TRTModel(path_model)

    vis = BBoxVisualization(cls_dict)
    test_op = False
    boxes, confs, clss = trt_ssd(img_handle)
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
