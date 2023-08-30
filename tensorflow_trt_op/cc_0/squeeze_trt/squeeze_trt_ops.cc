#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"


using namespace tensorflow;

REGISTER_OP("Squeeze_TRT")
    //.Attr("name: string = 'squeeze'")
    .Attr("axis: int")
    .Attr("dtype: type = DT_FLOAT")
    .Input("boxloc_concat: dtype")
    .Output("squeeze: dtype");