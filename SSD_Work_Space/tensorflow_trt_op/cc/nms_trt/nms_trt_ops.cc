// kernel_TRT.cc

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("NMS_TRT")
    //.Attr("name: string = 'nms'")
    .Attr("backgroundLabelId: int")
    .Attr("codeType: int")
    .Attr("confidenceThreshold: float")
    .Attr("confSigmoid: int")
    .Attr("dtype: type = DT_FLOAT")
    .Attr("inputOrder: list(int)")
    .Attr("isNormalized: int")
    .Attr("keepTopK: int")
    .Attr("nmsThreshold: float")
    .Attr("numClasses: int")
    // .Attr("scoreConverter: string")
    .Attr("shareLocation: int")
    .Attr("topK: int")
    .Attr("varianceEncodedInTarget: int")
    .Attr("scoreBits: int")
    .Attr("isBatchAgnostic: int")
    .Input("priorbox_concat: dtype")
    .Input("squeeze: dtype")
    .Input("boxconf_concat: dtype")
    .Output("nms: dtype");