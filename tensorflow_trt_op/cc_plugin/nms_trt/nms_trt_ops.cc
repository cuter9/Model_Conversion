// kernel_TRT.cc

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("NMS_TRT")
    //.Attr("name: string = 'nms'")
    .Attr("backgroundLabelId_u_int: int")
    .Attr("codeType_u_int: int")
    .Attr("confidenceThreshold_u_float: float")
    .Attr("confSigmoid_u_int: int")
    .Attr("dtype: type = DT_FLOAT")
    .Attr("inputOrder_u_ilist: list(int)")
    .Attr("isNormalized_u_int: int")
    .Attr("keepTopK_u_int: int")
    .Attr("nmsThreshold_u_float: float")
    .Attr("numClasses_u_int: int")
    .Attr("scoreConverter_u_str: string")
    .Attr("shareLocation_u_int: int")
    .Attr("topK_u_int: int")
    .Attr("varianceEncodedInTarget_u_int: int")
    .Input("priorbox_concat: dtype")
    .Input("squeeze: dtype")
    .Input("boxconf_concat: dtype")
    .Output("nms: dtype");