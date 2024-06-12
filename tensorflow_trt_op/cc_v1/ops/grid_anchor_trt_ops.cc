// kernel_TRT.cc

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("GridAnchor_TRT")
    .Attr("name: string = 'priorbox'")
    .Attr("minSize_u_float: float")
    .Attr("maxSize_u_float: float")
    .Attr("aspectRatios_u_flist: list(float)=[]")
    .Attr("variance_u_flist: list(float) = []")
    .Attr("featureMapShapes_u_ilist: list(int)=[]")
    .Attr("numLayers_u_int: int")    
    .Attr("dtype: type = DT_FLOAT")
    // .Input("input: int32")
    .Output("priorbox: float32");