import onnx
import numpy as np
from onnx.backend.test.case.node import expect
import onnxruntime as rt

x1 = np.array(
    [
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 5, 5) input tensor
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ],
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 5, 5) input tensor
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ]
        ]
    ]
).astype(np.float32)

x = onnx.helper.make_tensor(name="x",
                            data_type=onnx.TensorProto.FLOAT,
                            dims=[1, 2, 5, 5],
                            vals=x1)

W1 = np.array(
    [
        [
            [
                [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ]
    ]
).astype(np.float32)

W = onnx.helper.make_tensor(name="W",
                            data_type=onnx.TensorProto.FLOAT,
                            dims=[1, 2, 3, 3],
                            vals=W1)

# Convolution with padding
node_with_padding = onnx.helper.make_node(
    "Conv",
    inputs=["x", "W"],
    outputs=["y"],
    kernel_shape=[2, 3, 3],
    # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
    pads=[1, 1, 1, 1],
)

# y_with_padding = onnx.helper.make_tensor(name="y_with_padding",
#                                         data_type=onnx.TensorProto.FLOAT,
#                                         dims=[1, 1, 5, 5],
#                                         vals=np.zeros([1, 1, 5, 5]))
y_with_padding = onnx.helper.make_tensor_value_info(name="y_with_padding",
                                                    elem_type=onnx.TensorProto.FLOAT,
                                                    shape=[1, 1, 5, 5])
y_with_padding_1 = np.array(
    [
        [
            [
                [12.0, 21.0, 27.0, 33.0, 24.0],  # (1, 1, 5, 5) output tensor
                [33.0, 54.0, 63.0, 72.0, 51.0],
                [63.0, 99.0, 108.0, 117.0, 81.0],
                [93.0, 144.0, 153.0, 162.0, 111.0],
                [72.0, 111.0, 117.0, 123.0, 84.0],
            ]
        ]
    ]
).astype(np.float32)

onnx_graph = onnx.helper.make_graph(
    nodes=[node_with_padding],
    name="node_with_padding",
    inputs=[],
    outputs=[y_with_padding])


name_onnx_model = "node_with_padding"
onnx_model = onnx.helper.make_model(onnx_graph)
path_onnx_model = "test_node_with_padding"
with open(path_onnx_model, "wb") as f:
    f.write(onnx_model.SerializeToString())

sess = rt.InferenceSession(path_onnx_model)
result = sess.run()

expect(
    node_with_padding,
    inputs=[x, W],
    outputs=[y_with_padding],
    name="test_basic_conv_with_padding",
)
print(node_with_padding)
