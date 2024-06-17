#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class efficientNMS_TRT : public OpKernel {
	public:
		explicit efficientNMS_TRT(OpKernelConstruction* context) : OpKernel(context) {
	}
};