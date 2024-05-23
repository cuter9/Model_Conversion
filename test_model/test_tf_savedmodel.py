import tensorflow as tf
from google.protobuf.message import Message
from tensorflow.python.saved_model.loader_impl import parse_saved_model
class ExampleModel(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
  def capture_fn(self, x):
    if not hasattr(self, 'weight'):
      self.weight = tf.Variable(5.0, name='weight')
    self.weight.assign_add(x * self.weight)
    return self.weight

  @tf.function
  def polymorphic_fn(self, x):
    return tf.constant(3.0) * x

model = ExampleModel()
model.polymorphic_fn(tf.constant(4.0))
model.polymorphic_fn(tf.constant([1.0, 2.0, 3.0]))
tf.saved_model.save(
    model, "/home/cuterbot/temp/example-model", signatures={'capture_fn': model.capture_fn})

saved_model = tf.saved_model.load('/home/cuterbot/temp/example-model')
g_serving = saved_model.signatures["capture_fn"]
g = g_serving.graph

# MetaGraphDef : https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/compat/v1/MetaGraphDef
# meta_graph : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/load.py#L1019
# parse saved_model : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/loader_impl.py#L46
# model_msg = tf.compat.v1.MetaGraphDef()
mpb = parse_saved_model('/home/cuterbot/temp/example-model')
# mpb = parse_saved_model("/home/cuterbot/Data_Repo/Model_Conversion/SSD_mobilenet/TF_Model/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")
meta_graph = mpb.meta_graphs[0]
# with tf.io.gfile.GFile('/home/cuterbot/temp/example-model/saved_model.pb', 'rb') as f:
#  model_msg.MergeFromString(f.read())
  # text_format.Parse(f.read(), model_def)
g_def = g.as_graph_def()

