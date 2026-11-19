// kernel_TRT.cc

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("GridAnchorRect_TRT")
    //.Attr("name: string = 'priorbox'")
    .Attr("minSize: float = 0.2")
    .Attr("maxSize: float = 0.95")
    .Attr("aspectRatios: list(float) = [1, 2, 0.5, 3, 0.333333333]")
    .Attr("variance: list(float) =  [0.1, 0.1, 0.2, 0.2]")
    .Attr("featureMapShapes: list(int) = [19, 19, 10, 10, 5, 5, 3, 3, 2, 2, 1, 1]")
    .Attr("numLayers: int = 6")
    .Attr("dtype: type = DT_FLOAT")
    .Output("priorbox: numLayers * dtype");