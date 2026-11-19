#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ConcatV2")
    .Attr("name: string = 'priorbox_concat'")
    .Attr("axis: int")
    .Attr("dtype: type = DT_FLOAT")
    .Input("priorbox: float32")
    .Output("priorbox_concat: float32");