#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <list>

using namespace tensorflow;
using namespace std;

class GridAnchorRect_TRT : public OpKernel {
    public:
        explicit GridAnchorRect_TRT(OpKernelConstruction* context) : OpKernel(context) {
        }    
};
