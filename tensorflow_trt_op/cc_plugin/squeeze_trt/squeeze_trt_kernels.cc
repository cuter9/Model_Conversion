#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

class Squeeze_TRTOP : public OpKernel {
	public:
		explicit Squeeze_TRTOP(OpKernelConstruction* context) : OpKernel(context) {
	}
};