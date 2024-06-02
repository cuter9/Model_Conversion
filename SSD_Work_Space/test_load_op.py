import tensorflow as tf
from tf_onnx import load_customer_op
import os

ol_path = os.path.join(os.environ['HOME'], 'Model_Conversion/tensorflow_trt_op/python3/ops/set/')
a = load_customer_op(ol_path)
# a = load_customer_op('tensorflow_trt_op/python3/ops/set/')
# a = tf.load_op_library("/home/cuterbot/Model_Conversion/tensorflow_trt_op/python3/ops/set/_concat_trt_ops.so")
dir(a)