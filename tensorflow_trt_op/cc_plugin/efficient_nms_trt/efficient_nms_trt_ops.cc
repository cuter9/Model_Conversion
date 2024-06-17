// kernel_TRT.cc

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("efficientNMS_TRT")
    //.Attr("name: string = 'nms'")
    .Attr("score_threshold: float")
    .Attr("iou_threshold: float")
    .Attr("max_detections_per_class: int")
    .Attr("max_output_boxes: int")
    .Attr("background_class: int")
    .Attr("score_activation: int")
    .Attr("class_agnostic: int")
    .Attr("box_coding: int")
    .Attr("dtype: type = DT_FLOAT")
    .Input("priorbox_concat: dtype")
    .Input("boxloc_concat: dtype")
    .Input("boxconf_concat: dtype")
    .Output("nms: dtype");