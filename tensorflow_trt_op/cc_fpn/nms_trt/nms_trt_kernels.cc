#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class NMS_TRT : public OpKernel {
	public:
		explicit NMS_TRT(OpKernelConstruction* context) : OpKernel(context) {}
		void Compute(OpKernelContext* context) override {}
};