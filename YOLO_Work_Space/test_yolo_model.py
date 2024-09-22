import argparse
import os

import cv2
import tensorrt as trt
import torch
import numpy as np
from collections import OrderedDict, namedtuple
import sys

import wget


class TRT_engine():
    def __init__(self, weight) -> None:
        self.imgsz = [640, 640]
        self.weight = weight
        self.device = torch.device('cuda:0')
        self.init_engine()

    def init_engine(self):
        # Infer TensorRT Engine
        self.Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.weight, 'rb') as self.f, trt.Runtime(self.logger) as self.runtime:
            self.model = self.runtime.deserialize_cuda_engine(self.f.read())
        self.bindings = OrderedDict()
        self.fp16 = False
        for index in range(self.model.num_bindings):
            self.name = self.model.get_binding_name(index)
            self.dtype = trt.nptype(self.model.get_binding_dtype(index))
            self.shape = tuple(self.model.get_binding_shape(index))
            self.data = torch.from_numpy(np.empty(self.shape, dtype=np.dtype(self.dtype))).to(self.device)
            self.bindings[self.name] = self.Binding(self.name, self.dtype, self.shape, self.data,
                                                    int(self.data.data_ptr()))
            if self.model.binding_is_input(index) and self.dtype == np.float16:
                self.fp16 = True
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.model.create_execution_context()

    def letterbox(self, im, color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.imgsz
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        self.r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            self.r = min(self.r, 1.0)
        # Compute padding
        new_unpad = int(round(shape[1] * self.r)), int(round(shape[0] * self.r))
        self.dw, self.dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            self.dw, self.dh = np.mod(self.dw, stride), np.mod(self.dh, stride)  # wh padding
        self.dw /= 2  # divide padding into 2 sides
        self.dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        self.img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return self.img, self.r, self.dw, self.dh

    def preprocess(self, image):
        self.img, self.r, self.dw, self.dh = self.letterbox(image)
        self.img = self.img.transpose((2, 0, 1))
        self.img = self.img / 255.0
        self.img = np.expand_dims(self.img, 0)
        self.img = np.ascontiguousarray(self.img)
        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.float()
        return self.img

    def predict(self, img, threshold):
        img = self.preprocess(img)
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data[0].tolist()
        boxes = self.bindings['det_boxes'].data[0].tolist()
        scores = self.bindings['det_scores'].data[0].tolist()
        classes = self.bindings['det_classes'].data[0].tolist()
        top_scores = sorted(scores, reverse=True)[0:10]
        print("Scores - %s \n" % str(top_scores))

        '''
        raw_boxes = self.bindings['boxes'].data[0].tolist()
        raw_scores = self.bindings['scores'].data[0].tolist()
        top_raw_conf = [c[0:20] for c in sorted(raw_boxes, reverse=True)[0:10]]
        top_raw_scores = [s[0:20][0:2] for s in sorted(raw_scores, reverse=True)[0:10]]
        print("Raw_Scores - %s " % str(top_raw_scores))


        raw_classes = self.bindings['classes'].data[0].tolist()
        raw_boxes = self.bindings['boxes'].data[0].tolist()
        top_raw_classes = [c[0:20] for c in sorted(raw_classes, reverse=True)[0:10]]
        top_raw_boxes = [s[0:20][0:2] for s in sorted(raw_boxes, reverse=True)[0:10]]
        print("Raw_Scores - %s " % str(top_raw_boxes))
        '''

        # ys.stdout.write("Scores - ")
        # sys.stdout.write(str(scores))
        num = int(nums[0])
        new_bboxes = []
        for i in range(num):
            if scores[i] < threshold:
                continue
            xmin = (boxes[i][0] - self.dw) / self.r
            ymin = (boxes[i][1] - self.dh) / self.r
            xmax = (boxes[i][2] - self.dw) / self.r
            ymax = (boxes[i][3] - self.dh) / self.r
            new_bboxes.append([classes[i], scores[i], xmin, ymin, xmax, ymax])
        print("No detected object --  %s " % str(len(new_bboxes)))
        # sys.stdout.write())
        # sys.stdout.write("No detected object --  ")
        # sys.stdout.write(str(len(new_bboxes)))
        return new_bboxes


def visualize(img, bbox_array):
    for temp in bbox_array:
        xmin = int(temp[2])
        ymin = int(temp[3])
        xmax = int(temp[4])
        ymax = int(temp[5])
        clas = int(temp[0])
        score = temp[1]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (105, 237, 249), 2)
        img = cv2.putText(img, "class:" + str(clas) + " " + str(round(score, 2)), (xmin, int(ymin) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 1)
    return img


def main(opt):
    trt_engine = TRT_engine(opt.trt_engine)
    img = cv2.imread(opt.source)
    results = trt_engine.predict(img, opt.conf_thres)
    img = visualize(img, results)
    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trt-engine', type=str, default='yolov7-tiny.engine', help='model.engine path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)

    test_img = os.path.join(os.environ['HOME'], "Downloads/yolo/yolov7/inference/images/test.jpg")
    if not os.path.exists(test_img):
        wget.download("http://images.cocodataset.org/val2017/000000088462.jpg", out=test_img)
    opt.source = test_img

    # opt.trt_engine = os.path.join(os.environ['HOME'], "Data_Repo/Model_Conversion/yolov7-tiny/ONNX_Model/Repo/yolov7-tiny.engine")
    opt.trt_engine = os.path.join(os.environ['HOME'], "Data_Repo/Model_Conversion/yolov7/ONNX_Model/Repo/yolov7.engine")

    opt.conf_thres = 0.5
    # opt.save_txt = True
    # opt.save_conf = True

    main(opt)
