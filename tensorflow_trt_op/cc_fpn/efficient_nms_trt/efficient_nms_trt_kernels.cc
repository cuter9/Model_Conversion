#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class EfficientNMS_TRTOP : public OpKernel {
	public:
		explicit EfficientNMS_TRTOP(OpKernelConstruction* context) : OpKernel(context) {
	}
};