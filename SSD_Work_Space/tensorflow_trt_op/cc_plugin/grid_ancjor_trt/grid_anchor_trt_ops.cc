// kernel_TRT.cc

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("GridAnchor_TRT")
    //.Attr("name: string = 'priorbox'")
    .Attr("minSize_u_float: float = 0.2")
    .Attr("maxSize_u_float: float = 0.95")
    .Attr("aspectRatios_u_flist: list(float) = [1, 2, 0.5, 3, 0.333333333]")
    .Attr("variance_u_flist: list(float) = [0.1, 0.1, 0.2, 0.2]")
    .Attr("featureMapShapes_u_ilist: list(int) = [19, 10, 5, 3, 2, 1]")
    .Attr("numLayers_u_int: int = 6")
    .Attr("dtype: type = DT_FLOAT")
    .Output("priorbox: numLayers_u_int * dtype");