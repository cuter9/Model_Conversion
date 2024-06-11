// flattern_concat_trt_op.cc

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("FlattenConcat_TRT")
    //.Attr("name: string = 'boxloc_concat'")
    .Attr("axis: int") 
    .Attr("ignoreBatch: int")
    .Attr("dtype: type = DT_FLOAT")
    // .Attr("T: list(type)")
    .Input("input_1: dtype")
    .Input("input_2: dtype")
    .Input("input_3: dtype")
    .Input("input_4: dtype")
    .Input("input_5: dtype")
    .Output("output: dtype");
