#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class Concat_TRTOP : public OpKernel {
	public:
		explicit Concat_TRTOP(OpKernelConstruction* context) : OpKernel(context) {
	}
};