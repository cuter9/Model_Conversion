# onnx version : 1.10.0
# onnxruntime : 1.10.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import onnx
import tf2onnx

import numpy as np
from PIL import Image, ImageDraw, ImageColor
import math
import matplotlib.pyplot as plt
# from tensorflow.python.framework import graph_io
import graphsurgeon as gs
import tensorflow as tf

# ROOT = os.getcwd()
WORK = os.getcwd()
# WORK = os.path.join(ROOT, "work")
# MODEL = "ssd_mobilenet_v1_coco_2018_01_28"
MODEL = "ssd_mobilenet_v2_coco_2018_03_29"
FROZEN_MODEL = "ssd_mobilenet_v2_coco_2018_03_29_frozen"
REV_MODEL = "ssd_mobilenet_v2_coco_2018_03_29_rev"

MODEL_REP_DIR = "model_rep"

# os.makedirs(WORK, exist_ok=True)

# force tf2onnx to cpu
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['MODEL'] = MODEL
os.environ['FROZEN_MODEL'] = FROZEN_MODEL
os.environ['REV_MODEL'] = REV_MODEL
os.environ['WORK'] = WORK
os.environ['MODEL_REP_DIR'] = MODEL_REP_DIR


def convtf2onnx():
    str_show_tf_model = "saved_model_cli show --dir $WORK/$MODEL/saved_model/ " \
                        "--tag_set serve  " \
                        "--signature_def serving_default"
    os.system(str_show_tf_model)

    # opset = 10 --> 11
    str_tf2onnx = "python3 -m tf2onnx.convert " \
                  "--opset 13 " \
                  "--saved-model $WORK/$MODEL/saved_model " \
                  "--output $WORK/$MODEL_REP_DIR/$MODEL.onnx " \
                  "--verbose"

    str_tf2onnx_1 = "python3 -m tf2onnx.convert " \
                    "--graphdef $WORK/$MODEL/frozen_inference_graph.pb --output $WORK/$MODEL_REP_DIR/$FROZEN_MODEL.onnx " \
                    "--opset 13 " \
                    "--inputs image_tensor:0 " \
                    "--outputs num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0 "

    str_tf2onnx_2 = "python3 -m tf2onnx.convert " \
                    "--graphdef $WORK/$REV_MODEL.pb " \
                    "--output $WORK/$MODEL_REP_DIR/$REV_MODEL.onnx " \
                    "--opset 13 " \
                    "--inputs input:0 " \
                    "--outputs nms:0 "

    os.system(str_tf2onnx_1)
    # path_onnx_model = os.path.join(WORK, MODEL_REP_DIR, MODEL + ".onnx")
    path_onnx_model = os.path.join(WORK, MODEL_REP_DIR, FROZEN_MODEL + ".onnx")
    # path_onnx_model = os.path.join(WORK, MODEL_REP_DIR, REV_MODEL + ".onnx")
    return path_onnx_model


def convtf2onnx_py(type_model, path_graph_pb, path_onnx_model, path_onnx_model_1,
                   input_name, output_name):
    # ref : https://blog.nowcoder.net/n/73d6765be1bc420589cd7bdb9495c796?from=nowcoder_improve
    # from tensorflow.python.platform import gfile
    from tensorflow.compat.v1 import gfile
    from tensorflow.python.platform import resource_loader
    from tensorflow.tools import graph_transforms
    from SSD_Work_Space.Utils.ssd_utils import load_config, tf_saved2frozen
    import subprocess

    # type_model = 3
    if type_model == 1:
        # tf save_model_graph can not be used to onnx conversion, must be exported to frozen graph
        # GRAPH_PB_PATH = WORK + "/" + MODEL + "/saved_model/saved_model.pb"
        path_onnx_model = os.path.join(WORK, MODEL + '_1.onnx')
        config_path = os.path.join(WORK, MODEL, "pipeline.config")
        config = load_config(config_path)
        checkpoint_path = os.path.join(WORK, MODEL, "model.ckpt")
        tmp_dir = os.path.join(WORK, "model_4_onnx_conv")
        if os.path.isdir(tmp_dir):
            subprocess.call(['rm', '-r', tmp_dir])
        subprocess.call(['mkdir', '-p', tmp_dir])
        tf_saved2frozen(config, checkpoint_path, tmp_dir)
        GRAPH_PB_PATH = os.path.join(tmp_dir, "frozen_inference_graph.pb")
        # the node/tensor name should prefix with "import",
        # if using tf.import_graph_def() function for tf2onnx.tfonnx.process_tf_graph() conversion
        input_names = ['import/image_tensor:0']
        output_names = ["import/num_detections:0",
                        "import/detection_boxes:0",
                        "import/detection_scores:0",
                        "import/detection_classes:0"
                        ]

    elif type_model == 2:
        # use directly the frozen_inference_graph.pb for onnx conversion
        GRAPH_PB_PATH = os.path.join(WORK, MODEL, "frozen_inference_graph.pb")
        path_onnx_model = os.path.join(WORK, MODEL_REP_DIR, FROZEN_MODEL + '_1.onnx')
        # the node/tensor name should prefix with "import",
        # if using tf.import_graph_def() function for tf2onnx.tfonnx.process_tf_graph() conversion
        input_names = ['import/image_tensor:0']
        output_names = ["import/num_detections:0",
                        "import/detection_boxes:0",
                        "import/detection_scores:0",
                        "import/detection_classes:0"
                        ]

    elif type_model == 3:
        # this function is used for convert tf model with customer op,
        # which should be registered (in directory ~/tensorflow_trt_op) before converting
        # load_customer_op()
        # GRAPH_PB_PATH = os.path.join(work_dir, model_rep_dir, model_pb)
        # path_onnx_model = os.path.join(work_dir, model_rep_dir, path_onnx_model)
        input_names = ["import/input:0"]
        output_names = ["import/nms:0"]
        # input_names = ["input:0"]
        # output_names = ["nms:0"]

        # with tf.compat.v1.Session() as sess:
        print("---- load graph and start convert tf frozen model to onnx")
        with tf.io.gfile.GFile(path_graph_pb, 'rb') as f, tf.io.gfile.GFile(path_graph_pb, 'rb') as f1:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph_def_1 = tf.compat.v1.GraphDef()
            graph_def_1.ParseFromString(f1.read())

            modify_node_attr(graph_def)  # set FusedBatchNormV3 node attribute is_training = true
            # *** tf2onnx.convert.from_graph_def is also applicable to the Graph with custom op
            # *** NHWC is the native image format used in the tf SSD model,
            #     thus the image format should be in NHWC format when doing inference.
            # *** OR, use inputs_as_nchw=input_name if the image format for inference is NCHW format.
            model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(graph_def,
                                                                                  inputs_as_nchw=input_name,
                                                                                  input_names=input_name,
                                                                                  output_names=output_name)

        with tf.Graph().as_default() as graph:
            # *** The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            # and 'import' will be prefixed in op/nodes because using tf.compat.v1.import_graph_def()***
            # ref. source : https://github.com/onnx/tensorflow-onnx/tree/main/tf2onnx
            # ref. https://github.com/onnx/tensorflow-onnx/blob/main/examples/custom_op_via_python.py
            # ref. https://www.tensorflow.org/api_docs/python/tf/graph_util/import_graph_def
            # *** Error unregistered op :
            #       1. check the data type of the op input/output, which may result in op unregistered
            #       2. the i/o of tf op might not compatible with the custom op, replace the tf op with a custom op
            tf.compat.v1.import_graph_def(graph_def_1)
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(graph,
                                                         inputs_as_nchw=input_name,
                                                         input_names=input_names,
                                                         output_names=output_names)

            # print("---- convert onnx graph to onnx model proto ----")
            name_onnx_model = "ssd_mobilenet_v2_coco"
            model_proto_1 = onnx_graph.make_model(name_onnx_model)

    # original_stdout = sys.stdout
    # with open('tmp_onnx_model.txt', 'w') as f:
    #    sys.stdout = f  # Change the standard output to the file we created.
    #    print(model_proto)
    #    sys.stdout = original_stdout  # Reset the standard output to its original value

    print('----save onnx file to : ', path_onnx_model)
    with open(path_onnx_model, "wb") as f, open(path_onnx_model_1, "wb") as f1:
        f.write(model_proto.SerializeToString())
        f1.write(model_proto_1.SerializeToString())
    return model_proto, model_proto_1


def load_customer_op(op_path):
    # dir_path_op = './"tensorflow_trt_op/python3/ops/set'  # the directory stored the custom op library
    # dir_path_op_1 = 'tensorflow_trt_op/python3/ops/set'
    dir_path_op = "./" + op_path  # the directory stored the custom op library
    dir_path_op_1 = op_path
    ops_list = os.listdir(dir_path_op)
    print(ops_list)
    tf_op = []
    for ol in ops_list:
        print('--- load custom op library : ', os.path.join(dir_path_op, ol))
        try:
            tf_op.append(tf.load_op_library(os.path.join(dir_path_op, ol)))
            # tf.load_op_library(os.path.join(dir_path_op, ol))
        except:
            tf_op.append(tf.load_op_library(os.path.join(dir_path_op_1, ol)))
            # tf.load_op_library(os.path.join(dir_path_op_1, ol))
    return


def modify_node_attr(graph_def):
    nd = [n for n in graph_def.node if n.op == 'FusedBatchNormV3']
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


def draw_detection(draw, d, c):
    """Draw box and label for 1 detection."""
    coco_classes = {
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
    }
    width, height = draw.im.size
    # the box is relative to the image size so we multiply with height and width to get pixels.
    top = d[0] * height
    left = d[1] * width
    bottom = d[2] * height
    right = d[3] * width
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
    right = min(width, np.floor(right + 0.5).astype('int32'))
    label = coco_classes[c]
    label_size = draw.textsize(label)
    if top - label_size[1] >= 0:
        text_origin = tuple(np.array([left, top - label_size[1]]))
    else:
        text_origin = tuple(np.array([left, top + 1]))
    color = ImageColor.getrgb("red")
    thickness = 0
    draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color)
    draw.text(text_origin, label, fill=color)  # , font=font)


def verify_onnx_model(path_model):
    import onnxruntime as rt

    if not os.path.isfile(os.path.join(WORK, "000000088462.jpg")):
        str_pic_path = "cd $WORK; wget -q http://images.cocodataset.org/val2017/000000088462.jpg"
        os.system(str_pic_path)
    img = Image.open("000000088462.jpg")
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
    print(img_data.shape)

    sess = rt.InferenceSession(path_model)
    # sess = rt.InferenceSession(os.path.join(WORK, MODEL + "_1.onnx"))
    # sess = rt.InferenceSession(os.path.join(WORK, MODEL + ".onnx"))
    # sess = rt.InferenceSession(os.path.join(WORK, FROZEN_MODEL + ".onnx"))
    # sess = rt.InferenceSession(os.path.join(WORK, FROZEN_MODEL + "_1.onnx"))
    # sess = rt.InferenceSession(os.path.join(WORK, "ssd_mobilenet_v2_coco_2018_03_29_rev.onnx"))
    # sess = rt.InferenceSession(os.path.join(WORK, "ssd_mobilenet_v2_coco_2018_03_29_rev_1.onnx"))

    # we want the outputs in this order
    # outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]
    outputs = [on.name for on in sess.get_outputs()]
    inputs = sess.get_inputs()[0].name
    # result = sess.run(outputs, {"image_tensor:0": img_data})
    result = sess.run(outputs, {inputs: img_data})

    num_detections, detection_boxes, detection_scores, detection_classes = result
    # there are 8 detections

    print(num_detections)
    print(detection_classes)

    for output in sess.get_outputs():
        print(output.name)
    print([on.name for on in sess.get_outputs()])
    for input in sess.get_inputs():
        print(input.name)

    batch_size = num_detections.shape[0]
    draw = ImageDraw.Draw(img)
    for batch in range(0, batch_size):
        for detection in range(0, int(num_detections[batch])):
            c = detection_classes[batch][detection]
            d = detection_boxes[batch][detection]
            draw_detection(draw, d, c)

    plt.figure(figsize=(80, 40))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def main():
    # tf_trt_grapf()
    work_space = {"work_dir": WORK,
                  "model_name": MODEL,
                  "model_rep_dir": MODEL_REP_DIR,
                  "model_pb": REV_MODEL + ".pb",
                  "path_onnx_model": REV_MODEL + "_1.onnx"}
    type_model = 3
    path_graph_pb = os.path.join(WORK, MODEL_REP_DIR, REV_MODEL + ".pb")
    path_onnx_model = os.path.join(WORK, MODEL_REP_DIR, REV_MODEL + "_1.onnx")
    convtf2onnx_py(type_model, path_graph_pb, path_onnx_model)
    # path_onnx_model = convtf2onnx()
    # path_onnx_model = os.path.join(WORK, MODEL_REP_DIR, MODEL + "_1.onnx")
    # path_onnx_model = os.path.join(WORK, MODEL_REP_DIR, FROZEN_MODEL + "_1.onnx")
    # path_onnx_model = os.path.join(WORK, MODEL_REP_DIR, REV_MODEL + "_1.onnx")
    # gs_onnx(path_onnx_model)
    if type_model != 3:
        verify_onnx_model(path_onnx_model)


if __name__ == "__main__":
    main()
