import tensorflow as tf
from tf_onnx import load_customer_op
import os


PATH_FROZEN_MODEL = "/home/cuterbot/Model_Conversion/Onnx_Work_Space/Model_Rep/ssd_mobilenet_v2_coco_2018_03_29_4_onnx_conv.pb"

WORK = os.getcwd()
os.chdir("../")
load_customer_op()
os.chdir(WORK)

graph_def = tf.GraphDef()

with open(PATH_FROZEN_MODEL, 'rb') as f:
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.graph_util.import_graph_def(graph_def, name='')

new_model = tf.GraphDef()

with tf.Session(graph=graph) as sess:
    nd = [n for n in sess.graph_def.node if n.op == 'FusedBatchNormV3']
    for n in nd:
        new_attr_value = tf.AttrValue()
        new_attr_value.b = 1
        # new_attr = tf.compat.v1.NodeDef.AttrEntry(key="is_training", value=new_attr_value)
        node_new = tf.NodeDef(name=n.name,
                              op=n.op,
                              attr={'is_training': new_attr_value})
        print("----", n.name, "\n", n.attr)
        # n_attr.MergeFrom(new_attr)
        n.MergeFrom(node_new)
        print("----", n.name, "\n", n.attr)
        # nn = new_model.node.add()
        # nn.op = n.op
        # nn.name = n.name
        # nn.attr["is_training"] = 1
        # for i in n.input:
        #    nn.input.extend([i])
        # nn.CopyFrom(n)
