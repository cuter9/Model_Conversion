#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <list>

using namespace tensorflow;
using namespace std;

class GridAnchor_TRT : public OpKernel {
    public:
        explicit GridAnchor_TRT(OpKernelConstruction* context) : OpKernel(context) {}
        void Compute(OpKernelContext* context) override {}
};
