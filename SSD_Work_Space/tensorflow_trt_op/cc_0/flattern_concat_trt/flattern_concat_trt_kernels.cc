#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class FlattenConcat_TRT : public OpKernel {
	public:
		explicit FlattenConcat_TRT(OpKernelConstruction* context) : OpKernel(context) {
  }
};