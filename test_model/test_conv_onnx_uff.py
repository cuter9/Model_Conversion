import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import onnx
import numpy as np
import tensorflow as tf
from onnx.backend.test.case.node import expect
import onnxruntime as rt

WORK = os.getcwd()
SHAPE = (300, 300)
CHANNELS = 3
DIMS = [1, 3, 300, 300]


def get_input_data():
    if not os.path.exists(os.path.join(WORK, "000000088462.jpg")):
        str_pic_path = "cd $WORK; wget -q http://images.cocodataset.org/val2017/000000088462.jpg"
        os.system(str_pic_path)
    img = Image.open("000000088462.jpg")
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    img = cv2.imread("000000088462.jpg")
    img = cv2.resize(img, SHAPE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img *= (2.0 / 255.0)
    img -= 1.0

    return img


def model_param():
    w = np.ones((32, CHANNELS, 3, 3)).astype(np.float32)
    return w


# create onnx model with conv
def onnx_model():
    img = get_input_data()
    # x_onnx = onnx.helper.make_tensor(name="x_onnx",
    #                                 data_type=onnx.TensorProto.FLOAT,
    #                                 dims=[1, CHANNELS, SHAPE[0], SHAPE[1]],
    #                                 vals=img)

    x_onnx = onnx.helper.make_tensor_value_info(name="x_onnx",
                                                elem_type=onnx.TensorProto.FLOAT,
                                                shape=[1, CHANNELS, SHAPE[0], SHAPE[1]])
    w1 = model_param()
    w_onnx = onnx.numpy_helper.from_array(w1, name='w_onnx')

    y_onnx = onnx.helper.make_tensor_value_info(name="y_onnx",
                                                elem_type=onnx.TensorProto.FLOAT,
                                                shape=[1, 1, 5, 5])

    # Convolution with padding
    node_onnx = onnx.helper.make_node(
        "Conv",
        inputs=["x_onnx", "w_onnx"],
        outputs=["y_onnx"],
        auto_pad="SAME_LOWER",
        kernel_shape=list(w1.shape),
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        # pads=[1, 1, 1, 1],
    )

    onnx_graph = onnx.helper.make_graph(
        nodes=[node_onnx],
        name="node_onnx",
        inputs=[x_onnx],
        outputs=[y_onnx],
        initializer=[w_onnx]
    )

    onnx_model = onnx.helper.make_model(onnx_graph)

    onnx.checker.check_model(onnx_model)
    # print(onnx_model)
    path_onnx_model = "test_node_with_padding.onnx"
    with open(path_onnx_model, "wb") as f:
        f.write(onnx_model.SerializeToString())

    return path_onnx_model


class tf_model(tf.Module):
    def __int__(self, name=None):
        super().__init__(name=name)

    @tf.function
    def __call__(self, x):
        data_in = np.expand_dims(get_input_data(), axis=0)
        self.x_in = tf.constant(data_in, dtype=tf.float32)
        # self.x_in = tf.Variable(data_in, name='input')
        # self.x = self.x_in
        kernel_in = model_param()
        self.w = tf.constant(kernel_in, dtype=tf.float32)
        # self.w = tf.Variable(kernel_in, name='weight')
        y = tf.nn.conv2d(self.x, self.w, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
        # y = tf.matmul(self.x, self.w)
        return y


class tf_model_simple(tf.Module):
    def __int__(self, name=None):
        super().__init__(name=name)

    @tf.function
    def __call__(self, x):
        # self.x = tf.constant([[2.0, 2.0, 2.0]])
        self.w = tf.Variable(tf.random.normal([3, 2]), name='w')
        self.b = tf.Variable(tf.zeros([2]), name='b')
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


# sess = tf.compat.v1.Session()
# print(sess.run(y))

def test_model():
    pth_model = WORK + "/conv_model"
    data_in = np.expand_dims(get_input_data(), axis=0)
    tm = tf_model(name="conv_model")
    tf.saved_model.save(tm, pth_model)
    x_in = tf.constant(data_in, dtype=tf.float32)
    # sess = tf.compat.v1.Session()
    # print(sess.run(tm(x_in)))


def test_model_simple():
    pth_model = WORK + "/simple_model"
    tm = tf_model_simple(name="simple_model")
    tf.saved_model.save(tm, pth_model)
    # sess = tf.compat.v1.Session()
    # print(sess.run(tm(tf.constant([[2.0, 2.0, 2.0]]))))


# sess = rt.InferenceSession(onnx_model())
# result = sess.run()

if __name__ == "__main__":
    onnx_model()
    test_model_simple()

# expect(
#    node_with_padding,
#    inputs=[x, W],
#    outputs=[y_with_padding],
#    name="test_basic_conv_with_padding",
# )
# print(node_with_padding)
