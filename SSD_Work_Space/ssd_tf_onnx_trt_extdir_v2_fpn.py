# This TF model conversion is for TF version 2.X
import numpy as np
import os
import subprocess
import sys
import faulthandler
import tensorrt as trt
import onnx
import onnx_graphsurgeon as gs_onnx

from tf_onnx_v2 import convtf2onnx, load_customer_op
from Utils.ssd_utils_v2 import load_config, tf_saved2frozen, get_feature_map_shape, download_model

# from tf_graphsurgeon_v2 import tf_graphsurgeon as tf_gs
from tf_graphsurgeon_v2_fpn import tf_ssd_fpn_graphsurgeon as tf_gs

faulthandler.enable()

TRT_INPUT_NAME = 'input'
TRT_OUTPUT_NAME = 'nms'
# FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'   # for TF v1
FROZEN_GRAPH_NAME = 'saved_model.pb'  # for TF v2

# TF v1 model
# MODEL_NAME = "ssd_mobilenet_v1_coco_2018_01_28"
# MODEL_NAME = "ssd_mobilenet_v2_coco_2018_03_29"

# Object Detection Model in TF V2 Model Zoo :
# https://github.com/tensorflow/models/tree/master/research/object_detection
# MODEL_NAME = "ssd_mobilenet_v2_320x320_coco17_tpu-8"      # use ssd_tf_onnx_trt_extdir_v2.py to convert model
# MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
# MODEL_NAME = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"
MODEL_NAME = "ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8"
# MODEL_NAME = "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"
# MODEL_NAME = "ssd_resnet152_v1_fpn_640x640_coco17_tpu-8"

# MODEL_TRT = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
# MODEL_TRT = "ssd_mobilenet_v2_fpnlite_320x320_coco17"
# MODEL_TRT = "ssd_mobilenet_v2_fpnlite_640x640_coco17"
MODEL_TRT = "ssd_mobilenet_v1_fpn_640x640_coco17"
# # MODEL_TRT = "ssd_resnet50_v1_fpn_640x640_coco17"
# MODEL_TRT = "ssd_resnet152_v1_fpn_640x640_coco17"

WORK = os.getcwd()

# DATA_REPO_DIR = os.path.join("../../", "Data_Repo/Model_Conversion/SSD_mobilenet")
# DATA_REPO_DIR = os.path.join(os.environ['HOME'], "Data_Repo/Model_Conversion/SSD_mobilenet")
DATA_REPO_DIR = os.path.join(os.environ['HOME'], "Data_Repo/Model_Conversion", MODEL_NAME)
os.makedirs(DATA_REPO_DIR, exist_ok=True)

ONNX_WORK_SPACE = os.path.join(DATA_REPO_DIR, "ONNX_Model")
os.makedirs(ONNX_WORK_SPACE, exist_ok=True)

MODEL_REPO_DIR = os.path.join(ONNX_WORK_SPACE, "Repo")
os.makedirs(MODEL_REPO_DIR, exist_ok=True)

TMP_PB_GRAPH_NAME = MODEL_NAME + "_4_onnx_conv.pb"

TF_MODEL_DIR = os.path.join(DATA_REPO_DIR, "TF_Model")
os.makedirs(TF_MODEL_DIR, exist_ok=True)

TMP_MODEL = os.path.join(TF_MODEL_DIR, "Exported_Model", "saved_model")  # for TF v2
os.makedirs(TMP_MODEL, exist_ok=True)

# TF_CUSTOM_OP = os.path.join(WORK, "tensorflow_trt_op/python3/ops/set")  # the path stores the trt custom op for tf parsing
TF_CUSTOM_OP = os.path.join(WORK,
                            "tensorflow_trt_op/python3/ops_fpn/set")  # the path stores the trt custom op for tf parsing

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['MODEL_NAME'] = MODEL_NAME
os.environ['WORK'] = WORK
os.environ['ONNX_WORK_SPACE'] = ONNX_WORK_SPACE
os.environ['MODEL_REPO_DIR'] = MODEL_REPO_DIR


def ssd_pipeline_to_onnx(checkpoint_path, config_path,
                         path_graph_pb, path_onnx_model, tmp_dir=TMP_MODEL):
    print('---- start converting tf ssd to onnx model ----')
    print('---- start graphsurgeon tf ssd for onnx model conversion----')

    # the following use tf.export to export and saved the tf saved model in a temporary directory
    # config = load_config(config_path)
    # frozen_graph_path = os.path.join(tmp_dir, FROZEN_GRAPH_NAME)
    # if not os.path.exists(frozen_graph_path):  # check frozen_graph_path is existed
    #    tf_saved2frozen(config, checkpoint_path,
    #                    tmp_dir)  # export saved model to frozen graph, tmp_dir : frozen graph path

    # load and process the tf saved model directly instated of storing the (frozen) pb model  file in temporary directory method as above
    # surge the TF model Graph for ONNX model conversion
    path_tf_model = os.path.join(TF_MODEL_DIR, MODEL_NAME)  # the tensorflow model downloaded from TF 2.0 model zoo

    input_names, output_names, path_tf_custom_op = tf_gs(path_tf_model=path_tf_model,
                                                         input_name=TRT_INPUT_NAME, output_name=TRT_OUTPUT_NAME,
                                                         onnx_work_dir=ONNX_WORK_SPACE,
                                                         path_graph_pb=path_graph_pb,
                                                         # the path stores the model graph def pb file
                                                         path_tf_custom_op=TF_CUSTOM_OP)  # the path stores the TF custom op for TF to ONNX model graph conversion

    # load custom ops need for conversion from tf model to onnx model when "parsing with tf backend" is needed
    # the custom ops can be constructed by the makefile in dir /tensorflow_trt_op
    # ref: https://www.tensorflow.org/guide/create_op

    ssd_fpn_op = load_customer_op(path_tf_custom_op)

    print('---- start onnx conversion with surged tf model ----')

    path_onnx_model_0 = path_onnx_model.split(".")[0] + "_0.onnx"
    # path_onnx_model_1_0 = path_onnx_model_1.split(".")[0] + "_0.onnx"
    # path_onnx_model_0:    file name for model which nodes are prefixed without "import/"; but
    #                       the BatchNormalization op need too many input with additional op
    # path_onnx_model_1_0:  file name for model which nodes  prefixed with "import/"; and
    #                       additional transport ops are inserted for transforming the NHWC (tf native format)
    #                       to NCHW format for nodes input, thus this ONNX model is not good for performance
    # onnx_model_proto_0 : model of path_onnx_model_0
    # onnx_model_proto_1_0 : model of path_onnx_model_0_1
    onnx_model_proto_0 = convtf2onnx(path_graph_pb=path_graph_pb,
                                     path_onnx_model=path_onnx_model_0,
                                     input_names=input_names,
                                     output_names=output_names)

    print("---- modify the attributes of onnx model for ONNX parsing. ---- \n")
    # onnx_model_proto = onnx.load(path_onnx_model, format='protobuf')
    rev_onnx_attr(onnx_model_proto_0)
    onnx.save_model(onnx_model_proto_0, path_onnx_model)
    onnx_graph = gs_onnx.import_onnx(onnx_model_proto_0)

    '''
    nd_bn = [n for n in onnx_graph.nodes if n.op == "BatchNormalization"]
    '''

    # print("---- modify the attributes of onnx model 1 for ONNX parsing.  ----\n")
    # onnx_model_proto_1 = onnx.load(path_onnx_model_1, format='protobuf')
    # rev_onnx_attr(onnx_model_proto_1_0)
    # onnx.save_model(onnx_model_proto_1_0, path_onnx_model_1)

    '''
    onnx_graph_1 = gs_onnx.import_onnx(onnx_model_proto_1_0)    
    nd_bn_1 = [n for n in onnx_graph_1.nodes if n.op == "BatchNormalization"]
    
    # replace the input data of BatchNormalization op in onnx_model_proto_0
    # with onnx_model_proto_0_1 parameters;
    # and clean up the onnx_model_proto_0.
    for n0 in nd_bn:
        for n1 in nd_bn_1:
            if n0.name == n1.name.split("/", 1)[-1]:
                n0.attrs = retrieve_attrs(n1)

                n0.inputs[1] = n1.inputs[1]
                n0.inputs[2] = n1.inputs[2]
                n0.inputs[3] = n1.inputs[3]
                n0.inputs[4] = n1.inputs[4]
    '''
    onnx_graph.cleanup().toposort()
    onnx_model_proto_new = gs_onnx.export_onnx(onnx_graph)
    onnx.save_model(onnx_model_proto_new, path_onnx_model)

    return


def rev_onnx_attr(onnx_model_proto):
    # The op Reshape needs the attribute "allowzero" for passing the ONNX parsing for trt
    nd_reshape = [nd for nd in onnx_model_proto.graph.node if nd.op_type == "Reshape"]
    for nd in nd_reshape:
        n_attr_proto = onnx.helper.make_attribute("allowzero", 0)
        nd.attribute.append(n_attr_proto)

    # delete the attribute "dtype" of the custom op in onnx model when convert from tf model,
    # it is not needed in onnx parser for trt
    '''
    nd_box = [nd for nd in onnx_model_proto.graph.node
              if nd.name in ["priorbox",
                             "priorbox_concat", "priorbox_concat_0", "priorbox_concat_1",
                             "boxconf_concat", "boxloc_concat", "nms"]]
    '''
    nd_box = [nd for nd in onnx_model_proto.graph.node
              if nd.name in ["priorbox_concat", "boxconf_concat", "boxloc_concat", "nms"]]
    for nd in nd_box:
        nd_attr = [(iattr, attr) for iattr, attr in enumerate(nd.attribute) if attr.name == "dtype"][0]
        nd.attribute.pop(nd_attr[0])

    return


def redef_onnx_node_4_trt_plugin(path_onnx_model, path_onnx_model_new):
    # ref : https://developer.nvidia.com/blog/estimating-depth-beyond-2d-using-custom-layers-on-tensorrt-and-onnx-models/

    print("---- surge the onnx model by replacing the some nodes with TRT plugin and ops. ----")
    onnx_model_proto = onnx.load(path_onnx_model, format='protobuf')
    onnx_graph = gs_onnx.import_onnx(onnx_model_proto)

    '''
    nodes_reshape_conf = [nd for nd in onnx_graph.nodes
                          if nd.op == "Reshape" and nd.name.split("/")[1] == "WeightSharedConvolutionalClassHead"]
    for nd in nodes_reshape_conf:
        nd.inputs[1].values = np.array([1, -1, 1, 91])

    nodes_reshape_loc = [nd for nd in onnx_graph.nodes
                          if nd.op == "Reshape" and nd.name.split("/")[1] == "WeightSharedConvolutionalBoxHead"]
    for nd in nodes_reshape_loc:
        nd.inputs[1].values = np.array([1, -1, 1, 4])
    '''

    '''
    # insert transpose after BoxPredictor_5 boxconf
#    node_to_boxconf_5_trnsp = [nd for nd in onnx_graph.nodes
#                               if nd.name == "BoxPredictor_5/ClassPredictor/BiasAdd"][0]
    node_to_boxconf_5_trnsp = [nd for nd in onnx_graph.nodes
                               if nd.name == "BoxPredictor/ConvolutionalClassHead_5/ClassPredictor/BiasAdd"][0]
    input_boxconf_5_trnsp = node_to_boxconf_5_trnsp.outputs[0]
    output_boxconf_5_trnsp = gs_onnx.Variable(name="boxconf_5_trnsp_out", dtype=np.float32)
    node_boxconf_5_trnsp = gs_onnx.Node(op="Transpose", name="boxconf_5_trnsp",
                                        attrs={"perm": [0, 2, 3, 1]},
                                        inputs=[input_boxconf_5_trnsp],
                                        outputs=[output_boxconf_5_trnsp]
                                        )
    onnx_graph.nodes.append(node_boxconf_5_trnsp)
#    node_reshape_boxconf_5 = [nd for nd in onnx_graph.nodes if nd.name == "BoxPredictor_5/Reshape_1"][0]
    node_reshape_boxconf_5 = [nd for nd in onnx_graph.nodes if nd.name == "BoxPredictor/ConvolutionalClassHead_5/Reshape"][0]
    node_reshape_boxconf_5.inputs[0] = node_boxconf_5_trnsp.outputs[0]

    # insert transpose after BoxPredictor_5 boxloc
#    node_to_boxloc_5_trnsp = [nd for nd in onnx_graph.nodes
#                              if nd.name == "BoxPredictor_5/BoxEncodingPredictor/BiasAdd"][0]
    node_to_boxloc_5_trnsp = [nd for nd in onnx_graph.nodes
                              if nd.name == "BoxPredictor/ConvolutionalBoxHead_5/BoxEncodingPredictor/BiasAdd"][0]
    input_boxloc_5_trnsp = node_to_boxloc_5_trnsp.outputs[0]
    output_boxloc_5_trnsp = gs_onnx.Variable(name="boxloc_5_trnsp_out", dtype=np.float32)
    node_boxloc_5_trnsp = gs_onnx.Node(op="Transpose", name="boxloc_5_trnsp",
                                       attrs={"perm": [0, 2, 3, 1]},
                                       inputs=[input_boxloc_5_trnsp],
                                       outputs=[output_boxloc_5_trnsp]
                                       )
    onnx_graph.nodes.append(node_boxloc_5_trnsp)
    # node_reshape_boxloc_5 = [nd for nd in onnx_graph.nodes if nd.name == "BoxPredictor_5/Reshape"][0]
    node_reshape_boxloc_5 = [nd for nd in onnx_graph.nodes if nd.name == "BoxPredictor/ConvolutionalBoxHead_5/Reshape"][0]
    node_reshape_boxloc_5.inputs[0] = node_boxloc_5_trnsp.outputs[0]
    '''

    '''
    # node_GridAnchor_TRT_0, add dummy input
    node_GridAnchor_TRT_0 = [nd for nd in onnx_graph.nodes if nd.name == "priorbox_0"][0]
    input_GridAnchor_TRT_0 = gs_onnx.Constant(name="priorbox_0_in:", values=np.ones(1))
    node_GridAnchor_TRT_0.inputs = [input_GridAnchor_TRT_0]

    # node_GridAnchor_TRT_1, add dummy input
    node_GridAnchor_TRT_1 = [nd for nd in onnx_graph.nodes if nd.name == "priorbox_1"][0]
    input_GridAnchor_TRT_1 = gs_onnx.Constant(name="priorbox_1_in:", values=np.ones(1))
    node_GridAnchor_TRT_1.inputs = [input_GridAnchor_TRT_1]

    node_priorbox_concat_0 = [nd for nd in onnx_graph.nodes if nd.name == "priorbox_concat_0"][0]
    node_priorbox_concat_0.op = "Concat"
    node_priorbox_concat_0.attrs.pop("N", None)
    node_priorbox_concat_0.inputs = node_GridAnchor_TRT_0.outputs
    node_priorbox_concat_0.outputs[0].dtype = np.float32

    # reshape priorbox_cocate_0 output from [1, 1, -1, 1] --> [1, 1, -1, 4], and then concatenate with priorbox_cocate_1
    nd_name_0 = 'priorbox_concat_0_reshape'
    out_name_0 = nd_name_0 + '_output'
    output_0 = gs_onnx.Variable(name=out_name_0, dtype=np.float32)
    shape_0 = gs_onnx.Constant(name="'priorbox_concat_0_shape", values=np.array([1, 2, -1, 4]))
    pc_rs_0 = gs_onnx.Node(op="Reshape", name=nd_name_0, attrs={"allowzero": 0},
                      inputs=[node_priorbox_concat_0.outputs[0], shape_0],
                      outputs=[output_0])
    onnx_graph.nodes.append(pc_rs_0)

    node_priorbox_concat_1 = [nd for nd in onnx_graph.nodes if nd.name == "priorbox_concat_1"][0]
    node_priorbox_concat_1.op = "Concat"
    node_priorbox_concat_1.attrs.pop("N", None)
    node_priorbox_concat_1.inputs = node_GridAnchor_TRT_1.outputs
    node_priorbox_concat_1.outputs[0].dtype = np.float32

    # reshape priorbox_cocate_1 output from [1, 2, -1, 1] --> [1, 2, -1, 4], and then concatenate with priorbox_cocate_0
    nd_name_1 = 'priorbox_concat_1_reshape'
    out_name_1 = nd_name_1 + '_output'
    output_1 = gs_onnx.Variable(name=out_name_1, dtype=np.float32)
    shape_1 = gs_onnx.Constant(name="priorbox_concat_1_shape", values=np.array([1, 2, -1, 4]))
    pc_rs_1 = gs_onnx.Node(op="Reshape", name=nd_name_1, attrs={"allowzero": 0},
                           inputs=[node_priorbox_concat_1.outputs[0], shape_1],
                           outputs=[output_1])
    onnx_graph.nodes.append(pc_rs_1)
    '''

    '''
    priorbox_concat_p = []
    for n in range(len(node_GridAnchor_TRT_0.outputs)):
        nd_name = "priorbox_concat_p_" + str(n)
        out_name = nd_name + '_output'
        output_pc = gs_onnx.Variable(name=out_name, dtype=np.float32)
        pc = gs_onnx.Node(op="Concat", name=nd_name, attrs={"axis": 2},
                          inputs=[node_GridAnchor_TRT_0.outputs[n], node_GridAnchor_TRT_1.outputs[n]],
                          outputs=[output_pc])
        priorbox_concat_p.append(pc)
        onnx_graph.nodes.append(pc)
    '''

    # node_priorbox_concat, modify op name and add explicitly the inputs from node_GridAnchor_TRT
    # node_priorbox_concat = [nd for nd in onnx_graph.nodes if nd.name == "priorbox_concat"][0]

    '''
    node_priorbox_concat.op = "Concat"
    node_priorbox_concat.attrs.pop("N", None)
    node_priorbox_concat.attrs["axis"] = 3
    node_priorbox_concat.inputs = [pc_rs_0.outputs[0], pc_rs_1.outputs[0]]
    # node_priorbox_concat.inputs = [node_priorbox_concat_0.outputs[0], node_priorbox_concat_1.outputs[0]]
    node_priorbox_concat.outputs[0].dtype = np.float32
    '''

    # reshape node_priorbox_concat from [1, 1, -1, 4] back to [1, 1, -1, 1]
    '''
    nd_name_ga_trt = 'priorbox_concat_reshape'
    out_name_ga_trt = nd_name_ga_trt + '_output'

    output_priorbox_concat_shape = gs_onnx.Variable(name="priorbox_concat_shape_output", dtype=np.float32)
    node_priorbox_concat_reshape = gs_onnx.Node(op="Reshape", name="priorbox_concat_shape", attrs={"allowzero": 0},
                                                inputs=[node_priorbox_concat.outputs[0], (1, 2, -1, 1)],
                                                outputs=[output_priorbox_concat_shape])
    onnx_graph.nodes.append(node_priorbox_concat_reshape)
    '''
    # reshape confidence of prediction class of boxes
    node_boxconf_concat = [nd for nd in onnx_graph.nodes if nd.name == "concat_1"][0]
    node_boxconf_concat.outputs[0].dtype = np.float32
    for o in list(node_boxconf_concat.inputs):
        o.dtype = np.float32

    output_boxcon_concat_reshape = gs_onnx.Variable(name="boxcon_concat_reshape_output", dtype=np.float32)
    boxcon_shape = gs_onnx.Constant(values=np.array([1, -1, 1, 1]), name="boxcon_shape")
    node_boxcon_concat_reshape = gs_onnx.Node(op="Reshape", name="boxcon_concat_reshape", attrs={"allowzero": 0},
                                              inputs=[node_boxconf_concat.outputs[0], boxcon_shape],
                                              outputs=[output_boxcon_concat_reshape])
    onnx_graph.nodes.append(node_boxcon_concat_reshape)

    # reshape confidence of prediction location of boxes
    node_boxloc_concat = [nd for nd in onnx_graph.nodes if nd.name == "concat"][0]
    node_boxloc_concat.outputs[0].dtype = np.float32
    for o in list(node_boxloc_concat.inputs):
        o.dtype = np.float32

    output_boxloc_concat_reshape = gs_onnx.Variable(name="boxloc_concat_reshape_output", dtype=np.float32)
    boxloc_shape = gs_onnx.Constant(values=np.array([1, -1, 1, 1]), name="boxloc_shape")
    node_boxloc_concat_reshape = gs_onnx.Node(op="Reshape", name="boxloc_concat_reshape", attrs={"allowzero": 0},
                                              inputs=[node_boxloc_concat.outputs[0], boxloc_shape],
                                              outputs=[output_boxloc_concat_reshape])
    onnx_graph.nodes.append(node_boxloc_concat_reshape)

    # node_boxloc_concat = [nd for nd in onnx_graph.nodes if nd.name == "concat"][0]
    # node_boxconf_concat = [nd for nd in onnx_graph.nodes if nd.name == "concat_1"][0]

    '''
    # squeeze, modify op name
    node_squeeze = [nd for nd in onnx_graph.nodes if nd.name == "squeeze"][0]
    node_squeeze.op = "Squeeze"
    '''
    '''
    # node_NMS_TRT, reconnect
    node_NMS_TRT = [nd for nd in onnx_graph.nodes if nd.name == "nms"][0]
    node_NMS_TRT.op = ""
    output_nms_0 = gs_onnx.Variable(name="nms:0", dtype=np.float32)
    output_nms_1 = gs_onnx.Variable(name="nms:1", dtype=np.float32)
    node_NMS_TRT.inputs = [node_priorbox_concat.outputs[0],
                           node_boxloc_concat.outputs[0],
                           node_boxconf_concat.outputs[0]]
    # node_NMS_TRT.inputs = [node_boxloc_concat.outputs[0],
    #                       node_priorbox_concat.outputs[0],
    #                       node_boxconf_concat.outputs[0]]    
    node_NMS_TRT.outputs = [output_nms_0, output_nms_1]
    onnx_graph.outputs = node_NMS_TRT.outputs
    '''

    node_NMS_TRT = [nd for nd in onnx_graph.nodes if nd.name == "nms"][0]
    # node_NMS_TRT.op = "EfficientNMS_TRT"
    # output_nms_0 = gs_onnx.Variable(name="num_detections", dtype=np.float32)
    # output_nms_1 = gs_onnx.Variable(name="detection_boxes", dtype=np.float32)
    # output_nms_2 = gs_onnx.Variable(name="detection_scores", dtype=np.float32)
    # output_nms_3 = gs_onnx.Variable(name="detection_classes", dtype=np.float32)

    node_NMS_TRT.inputs[0] = node_boxloc_concat_reshape.outputs[0]
    node_NMS_TRT.inputs[2] = node_boxcon_concat_reshape.outputs[0]
    #                       node_priorbox_concat.outputs[0]]

    # node_NMS_TRT.outputs = [output_nms_0, output_nms_1, output_nms_2, output_nms_3]
    onnx_graph.outputs = node_NMS_TRT.outputs
    # onnx_graph.outputs.append(node_boxcon_concat_reshape.outputs[0])
    # onnx_graph.outputs.append(node_boxloc_concat_reshape.outputs[0])

    # onnx_graph.outputs = [output_boxloc_concat]
    # onnx_graph.outputs = [output_priorbox]
    # onnx_graph.outputs = [output_priorbox_concat]
    # onnx_graph.outputs = [output_boxconf_concat]

    # onnx_graph.outputs = [node_NMS_TRT.outputs[0], node_NMS_TRT.outputs[0]]
    # onnx_graph.outputs.extend(node_priorbox_concat.outputs)
    # onnx_graph.outputs.extend(node_boxloc_concat.outputs)
    # onnx_graph.outputs.extend(node_boxconf_concat.outputs)
    # onnx_graph.outputs.extend(node_boxloc_concat.inputs)
    # onnx_graph.outputs.extend(node_boxconf_concat.inputs)

    '''
    # *** use "SAME_UPPER" because it is the same padding operation as the "same" used in uff conv op
    for nd_conv in onnx_graph.nodes:
        if nd_conv.op == "Conv":
            # nd_conv.attrs["pads"] = []  # [0, 0, 1, 1] --> [1, 1, 0, 0]
            nd_conv.attrs["auto_pad"] = "SAME_UPPER"  # "SAME_LOWER"; "SAME_UPPER" is same as the "same" in uff conv op
    '''

    test_conv = False  # test the convolution op only, all the other ops and nodes will be removed
    if test_conv:
        nd = [n for n in onnx_graph.nodes
              if 4 >= len(n.name.split("/")) >= 3 and n.name.split("/")[2] == "Conv"]

        output_conv_tranp = gs_onnx.Variable(name="nms:1", dtype=np.float32)
        node_conv_tranp = gs_onnx.Node(op="Transpose", name="conv_trnsp",
                                       attrs={"perm": [0, 2, 3, 1]},
                                       inputs=[input_boxloc_5_trnsp],
                                       outputs=[output_conv_tranp]
                                       )
        nd_out = []
        for n in nd:
            if n.op == "Conv":  # add transport op to the convolution node
                output_conv_transp = gs_onnx.Variable(name=n.outputs[0].name + "_new", dtype=np.float32)
                input_conv_transp = n.outputs[0]
                n.outputs[0].dtype = np.float32
                node_conv_transp = gs_onnx.Node(op="Transpose", name="conv_trnsp",
                                                attrs={"perm": [0, 2, 3, 1]},
                                                inputs=[input_conv_transp],
                                                outputs=[output_conv_transp]
                                                )
                onnx_graph.nodes.append(node_conv_transp)
                nd_out.append(output_conv_transp)
            else:
                n.outputs[0].dtype = np.float32
                nd_out.append(n.outputs[0])
        onnx_graph.outputs.extend(nd_out)

        # ----the follow script should be commented if not testing convolution
        onnx_graph.outputs = node_conv_tranp.outputs
        # onnx_graph.outputs = [input_conv_tranp]
        # ----

    onnx_graph.cleanup().toposort()
    # onnx_graph.toposort()

    onnx_model_proto_new = gs_onnx.export_onnx(onnx_graph)
    onnx.save_model(onnx_model_proto_new, path_onnx_model_new)
    return


def retrieve_attrs(node):
    """
    Gather the required attributes for the GroupNorm plugin from the subgraph.
    Args:
        nodes:  node in the graph.
    """
    attrs = {}
    for k, v in node.attrs.items():
        attrs[k] = v
    return attrs


def onnx_support_check(parser, path_onnx_model):
    with open(path_onnx_model, "rb") as so:
        serialized_model = so.read()
    model_support = parser.supports_model(serialized_model)
    if not model_support[0]:
        b_supported = False
        print("the tensorrt unsupported op listed as followings: \n")
        for m in model_support[1]:
            print(m, "\n")
    else:
        b_supported = True
        print("the onnx model is supported in tensorrt! \n")
    return b_supported, serialized_model, model_support


def ssd_onnx_to_engine(path_onnx_model,
                       fp16_mode=True,
                       max_batch_size=1,
                       max_workspace_size=1 << 30,  # 26
                       min_find_iterations=2,
                       average_find_iterations=1,
                       strict_type_constraints=False,
                       log_level=trt.ILogger.INFO):
    # create the tensorrt engine
    from contextlib import redirect_stderr
    with open('ssd_conversion.log', 'w') as stderr, redirect_stderr(stderr):
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Logger(log_level) as logger, \
                trt.Builder(logger) as builder, \
                builder.create_network(explicit_batch) as network, \
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, logger) as parser:
            # init built in plugins
            # trt.init_libnvinfer_plugins(logger, '')

            # load jetbot plugins
            # load_plugins()

            # b_supported, serialized_model, model_support = onnx_support_check(parser, path_onnx_model)
            # if not b_supported:
            #    return

            config.set_flag(trt.BuilderFlag.FP16)
            # config.set_flag(trt.BuilderFlag.INT8)
            # builder.fp16_mode = fp16_mode
            # config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            # builder.strict_type_constraints = strict_type_constraints
            config.max_workspace_size = max_workspace_size
            # builder.max_workspace_size = max_workspace_size
            config.min_timing_iterations = min_find_iterations
            # builder.min_find_iterations = min_find_iterations
            config.avg_timing_iterations = average_find_iterations
            # builder.average_find_iterations = average_find_iterations

            builder.max_batch_size = max_batch_size

            print("\n\n------ trt starts parsing onnx model")
            # onnx.checker.check_model(path_onnx_model)
            # gm = onnx.load(path_onnx_model)
            # unsupport_node = parser.supports_model(gm.SerializeToString())
            # print(" --- the following node in onnx model is not supported by trt onnx parser: \n", unsupport_node)

            parser.parse_from_file(path_onnx_model)

            print("\n\n----- start trt engine construction -----")
            engine = builder.build_engine(network, config)
            # engine = builder.build_cuda_engine(network, config)

    return engine


if __name__ == '__main__':
    download_model(MODEL_NAME, TF_MODEL_DIR)
    # checkpoint_path = os.path.join(TF_MODEL_DIR, MODEL_NAME, "model.ckpt")    # the ckpt path of TF v1
    checkpoint_path = os.path.join(TF_MODEL_DIR, MODEL_NAME,
                                   "checkpoint")  # the ckpt director of TF v2, refer to tf.train.checkpoint_management.py
    config_path = os.path.join(TF_MODEL_DIR, MODEL_NAME, "pipeline.config")
    output_engine = os.path.join(MODEL_REPO_DIR, MODEL_TRT + ".engine")

    path_graph_pb = os.path.join(MODEL_REPO_DIR, TMP_PB_GRAPH_NAME)
    path_onnx_model = os.path.join(MODEL_REPO_DIR, MODEL_NAME + ".onnx")
    # path_onnx_model_1 = os.path.join(MODEL_REPO_DIR, MODEL_NAME + "_1.onnx")
    ssd_pipeline_to_onnx(checkpoint_path, config_path,
                         path_graph_pb, path_onnx_model, tmp_dir=TMP_MODEL)

    # path_onnx_model = os.path.join(MODEL_REPO_DIR, MODEL_NAME + ".onnx")
    # onnx.save_model(onnx_model_proto, path_onnx_model)
    path_onnx_model_new = os.path.join(MODEL_REPO_DIR, MODEL_NAME + "_new.onnx")
    redef_onnx_node_4_trt_plugin(path_onnx_model, path_onnx_model_new)  # redefine the trt plugin nodes

    #     path_onnx_model_new_1 = os.path.join(MODEL_REPO_DIR, MODEL_NAME + "_1_new.onnx")
    #    redef_onnx_node_4_trt_plugin_1(path_onnx_model_1, path_onnx_model_new_1)  # redefine the trt plugin nodes
    # path_onnx_model = path_onnx_model_new
    # onnx.save_model(onnx_model_proto_new, path_onnx_model_new)

    print("----- Start build engine of the surged onnx model -----")
    engine = ssd_onnx_to_engine(path_onnx_model=path_onnx_model_new,
                                fp16_mode=True,
                                max_batch_size=1,
                                max_workspace_size=1 << 30,  # 26
                                min_find_iterations=2,
                                average_find_iterations=1,
                                strict_type_constraints=False,
                                log_level=trt.Logger.VERBOSE)

    buf = engine.serialize()
    with open(output_engine, 'wb') as f:
        f.write(buf)
        f.close()
    # verify_trt_model(output_engine)
