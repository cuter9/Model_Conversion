#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op.h"


using namespace tensorflow;

REGISTER_OP("Concat_TRT")
    //.Attr("name: string = 'priorbox_concat'")
    .Attr("axis: int")
    .Attr("dtype: type = DT_FLOAT")
    .Attr("T: list(dtype)")
    .Input("priorbox: T")
    .Output("priorbox_concat: dtype");