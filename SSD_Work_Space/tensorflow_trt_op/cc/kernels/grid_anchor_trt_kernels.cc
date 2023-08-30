#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class GridAnchor_TRT : public OpKernel {
 public:
  explicit GridAnchor_TRT(OpKernelConstruction* context) : OpKernel(context) {

  }
};