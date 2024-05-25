# Example of https://blog.tensorflow.org/2021/03/a-tour-of-savedmodel-signatures.html
import subprocess

import tensorflow as tf
# from google.protobuf.message import Message
from tensorflow.python.saved_model.loader_impl import parse_saved_model
import os
class ExampleModel(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
  def capture_fn(self, x):
    if not hasattr(self, 'weight_0'):
      self.weight_0 = tf.Variable(5.0, name='weight_0')
    if not hasattr(self, 'weight_1'):
      self.weight_1 = tf.Variable(5.0, name='weight_1')
    if not hasattr(self, 'weight_2'):
      self.weight_2 = tf.Variable(5.0, name='weight_2')

    self.weight_0.assign_add(x * self.weight_0)
    self.weight_1.assign_add(x * self.weight_1)
    self.weight_2.assign_add(x * self.weight_2)
    return self.weight_1

  @tf.function
  def polymorphic_fn(self, x):
    return tf.constant(3.0) * x

model = ExampleModel()
model.polymorphic_fn(tf.constant(4.0))
model.polymorphic_fn(tf.constant([1.0, 2.0, 3.0]))
tf.saved_model.save(
    model, "/home/cuterbot/temp/example-model", signatures={'capture_fn': model.capture_fn})

saved_model_1 = tf.saved_model.load('/home/cuterbot/temp/example-model')
g1_serving = saved_model_1.signatures["capture_fn"]
g1 = g1_serving.graph

saved_model_2 = tf.saved_model.load('/home/cuterbot/Data_Repo/Model_Conversion/SSD_mobilenet/TF_Model/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model')
g2_serving = saved_model_2.signatures["serving_default"]
g2 = g2_serving.graph
# https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk : A SavedModel contains one or more model variants (technically, v1.MetaGraphDefs), identified by their tag-sets.
# MetaGraphDef : https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/compat/v1/MetaGraphDef
# meta_graph : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/load.py#L1019
# parse saved_model : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/loader_impl.py#L46
# model_msg = tf.compat.v1.MetaGraphDef()
mpb_1 = parse_saved_model('/home/cuterbot/temp/example-model')
mpb_2 = parse_saved_model("/home/cuterbot/Data_Repo/Model_Conversion/SSD_mobilenet/TF_Model/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")
meta_graph_1 = mpb_1.meta_graphs[0]
meta_graph_2 = mpb_2.meta_graphs[0]
list_fname_1 = [s.signature.name for s in meta_graph_1.graph_def.library.function]
list_fname_2 = [s.signature.name for s in meta_graph_2.graph_def.library.function]

gdf_2 = meta_graph_2.graph_def
cfs_2 = meta_graph_2.object_graph_def.concrete_functions
cfs_2_name = list(cfs_2.keys())
sig_2 = [s for s in cfs_2_name if 'wrapper' in s.split('_')]
cfs_2_sig = cfs_2[sig_2[0]]
gdf_2_sig = [g for g in gdf_2.library.function if 'wrapper' in g.signature.name.split('_')]
# with tf.io.gfile.GFile('/home/cuterbot/temp/example-model/saved_model.pb', 'rb') as f:
#  model_msg.MergeFromString(f.read())
  # text_format.Parse(f.read(), model_def)
tmp_tbdir_s = os.path.join("/home/cuterbot/temp/", "tf_board_data_s")  # for storing static graph
if os.path.isdir(tmp_tbdir_s):
  subprocess.call(['rm', '-r', tmp_tbdir_s])
subprocess.call(['mkdir', '-p', tmp_tbdir_s])

writer_s = tf.summary.create_file_writer(tmp_tbdir_s)
with writer_s.as_default():
  #    tf.summary.graph(graph_def_spc)
  tf.summary.graph(meta_graph_1.graph_def)

g_def = g.as_graph_def()

