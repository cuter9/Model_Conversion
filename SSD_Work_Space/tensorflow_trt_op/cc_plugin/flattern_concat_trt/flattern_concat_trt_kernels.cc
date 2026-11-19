#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class FlattenConcat_TRTOP : public OpKernel {
	public:
		explicit FlattenConcat_TRTOP(OpKernelConstruction* context) : OpKernel(context) {
  }
};