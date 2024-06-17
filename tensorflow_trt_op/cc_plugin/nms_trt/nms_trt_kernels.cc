#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class NMS_TRTOP : public OpKernel {
	public:
		explicit NMS_TRTOP(OpKernelConstruction* context) : OpKernel(context) {
	}
};