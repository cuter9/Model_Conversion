#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

class Squeeze_TRT : public OpKernel {
	public:
		explicit Squeeze_TRT(OpKernelConstruction* context) : OpKernel(context) {
	}
};