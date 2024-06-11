#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ConcatV2 : public OpKernel {
 public:
  explicit ConcatV2(OpKernelConstruction* context) : OpKernel(context) {
  }
};