#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <list>

using namespace tensorflow;
using namespace std;

class GridAnchor_TRTOP : public OpKernel {
    public:
    explicit GridAnchor_TRTOP(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("numLayers_u_int", &numLayers_u_int_));
        OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
        cout << numLayers_u_int_ << endl;

        for (int i = 1; i<=numLayers_u_int_;  i++) {
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, tsh, tmp_var));
            cout << i << ", ";
            priorbox.push_back(tmp_var); 
        }    
    }
    void Compute(OpKernelContext* context) override {
        // Create an output (priorbox) tensor
        // Tensor* priorbox = NULL;
        // auto output_flat = output->template flat<float>();
        }

private: 
    int numLayers_u_int_;
    int dtype_;
    // Tensor* priorbox_n = NULL;
    Tensor* tmp_var = nullptr;
    const TensorShape &tsh = TensorShape({});
    list<Tensor*> priorbox;

};

        // Get the index of the value to preserve
        // Check that preserve_index is positive
        // OP_REQUIRES(context, numLayers_u_int_ < 0, errors::InvalidArgument("Need numLayers_u_int_ < 0, got ", numLayers_u_int_));
        // cout << "No of Layers : " << endl;
        // cout << numLayers_u_int_ << endl;
        // tsh = TensorShape({});
        // Tensor* priorbox = NULL;
        // int i = 0;
        // priorbox.clear;
        //
        //}
        // auto output_flat = output->template flat<float>();


