import numpy as np
import os
import subprocess
import sys
import faulthandler
import tensorrt as trt
import onnx
import onnx_graphsurgeon as gs_onnx

from tf_onnx import convtf2onnx, load_customer_op
from Utils.ssd_utils_v2 import load_config, tf_saved2frozen, get_feature_map_shape, download_model

faulthandler.enable()

TRT_INPUT_NAME = 'input'
TRT_OUTPUT_NAME = 'nms'
# FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'   # for TF v1
FROZEN_GRAPH_NAME = 'saved_model.pb'        # for TF v2

# MODEL_NAME = "ssd_mobilenet_v1_coco_2018_01_28"
# MODEL_NAME = "ssd_mobilenet_v2_coco_2018_03_29"     # tf v1 model
MODEL_NAME = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
MODEL_TRT = "ssd_mobilenet_v2_coco_tf_v2"

WORK = os.getcwd()

# DATA_REPO_DIR = os.path.join("../../", "Data_Repo/Model_Conversion//SSD_mobilenet")
DATA_REPO_DIR = os.path.join(os.environ['HOME'], "Data_Repo/Model_Conversion//SSD_mobilenet")
os.makedirs(DATA_REPO_DIR, exist_ok=True)

ONNX_WORK_SPACE = os.path.join(DATA_REPO_DIR, "ONNX_Model")
os.makedirs(ONNX_WORK_SPACE, exist_ok=True)

MODEL_REPO_DIR = os.path.join(ONNX_WORK_SPACE, "Repo")
os.makedirs(MODEL_REPO_DIR, exist_ok=True)

TMP_PB_GRAPH_NAME = MODEL_NAME + "_4_onnx_conv.pb"

TF_MODEL_DIR = os.path.join(DATA_REPO_DIR, "TF_Model")
os.makedirs(TF_MODEL_DIR, exist_ok=True)

TMP_MODEL = os.path.join(TF_MODEL_DIR, "Exported_Model", "saved_model")     # for TF v2
os.makedirs(TMP_MODEL, exist_ok=True)

TF_CUSTOM_OP = "tensorflow_trt_op/python3/ops/set"  # the path stores the trt custom op for tf parsing

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['MODEL_NAME'] = MODEL_NAME
os.environ['WORK'] = WORK
os.environ['ONNX_WORK_SPACE'] = ONNX_WORK_SPACE
os.environ['MODEL_REPO_DIR'] = MODEL_REPO_DIR


def ssd_pipeline_to_onnx(checkpoint_path, config_path,
                         path_graph_pb, path_onnx_model, path_onnx_model_1, tmp_dir=TMP_MODEL):
    import graphsurgeon as gs
    import tensorflow as tf

    print('---- start converting tf ssd to onnx model ----')
    print('---- start graphsurgeon tf ssd for onnx model conversion----')

    config = load_config(config_path)
    frozen_graph_path = os.path.join(tmp_dir, FROZEN_GRAPH_NAME)
    if not os.path.exists(frozen_graph_path):  # check frozen_graph_path is existed
        tf_saved2frozen(config, checkpoint_path,
                        tmp_dir)  # export saved model to frozen graph, tmp_dir : frozen graph path

    # get input shape
    channels = 3
    height = config.model.ssd.image_resizer.fixed_shape_resizer.height
    width = config.model.ssd.image_resizer.fixed_shape_resizer.width

    tmp_tbdir_s = os.path.join(ONNX_WORK_SPACE, "tf_board_data_s")  # for storing static graph
    if os.path.isdir(tmp_tbdir_s):
        subprocess.call(['rm', '-r', tmp_tbdir_s])
    subprocess.call(['mkdir', '-p', tmp_tbdir_s])

    tmp_tbdir_d = os.path.join(ONNX_WORK_SPACE, "tf_board_data_d")  # for storing dynamic graph
    if os.path.isdir(tmp_tbdir_d):
        subprocess.call(['rm', '-r', tmp_tbdir_d])
    subprocess.call(['mkdir', '-p', tmp_tbdir_d])

    _static_graph = tf.saved_model.load(tmp_dir)
    g = _static_graph.signatures["serving_default"].graph   # creat tf Graph objects
    static_graph = gs.StaticGraph(g)
    # static_graph = gs.StaticGraph(frozen_graph_path)
    # static_graph.write_tensorboard(tmp_tbdir_s)       # TensorRT use TF v1 to write graph which can not be used in TF v2
    # https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/summary/graph
    writer_s = tf.summary.create_file_writer(tmp_tbdir_s)
    with writer_s.as_default():
        tf.summary.graph(g)

    dynamic_graph = gs.DynamicGraph(g)

    spc = dynamic_graph.find_nodes_by_op('StatefulPartitionedCall')
    fspc = spc.f
    fspc.name = "StatefulPartitionedCall"

    # forward all identity nodes
    all_identity_nodes = dynamic_graph.find_nodes_by_op("Identity")
    dynamic_graph.forward_inputs(all_identity_nodes)

    # create input plugin
    input_plugin = gs.create_node(
        name=TRT_INPUT_NAME,
        op="Placeholder",
        dtype=tf.float32,
        # dtype=tf.uint8,
        shape=[1, height, width, channels])

    # create anchor box generator
    anchor_generator_config = config.model.ssd.anchor_generator.ssd_anchor_generator
    box_coder_config = config.model.ssd.box_coder.faster_rcnn_box_coder
    priorbox_plugin = gs.create_node(
        name="priorbox",
        op="GridAnchor_TRT",
        minSize=anchor_generator_config.min_scale,  # minSize
        maxSize=anchor_generator_config.max_scale,
        aspectRatios=list(anchor_generator_config.aspect_ratios),
        variance=[
            1.0 / box_coder_config.y_scale,
            1.0 / box_coder_config.x_scale,
            1.0 / box_coder_config.height_scale,
            1.0 / box_coder_config.width_scale
        ],
        featureMapShapes=get_feature_map_shape(config),
        # featureMapShapes=[1, 2, 3, 5, 10, 19],
        numLayers=config.model.ssd.anchor_generator.ssd_anchor_generator.num_layers)

    # create nms plugin
    nms_config = config.model.ssd.post_processing.batch_non_max_suppression
    nms_plugin = gs.create_node(
        name=TRT_OUTPUT_NAME,
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=nms_config.score_threshold,
        nmsThreshold=nms_config.iou_threshold,
        topK=nms_config.max_detections_per_class,
        keepTopK=nms_config.max_total_detections,
        numClasses=config.model.ssd.num_classes + 1,  # add background
        inputOrder=[1, 2, 0],  # [1, 2, 0]
        confSigmoid=1,
        isNormalized=1,
        # scoreConverter="SIGMOID",
        scoreBits=16,
        isBatchAgnostic=1,
        codeType=3)

    # tf built in op Concat is not suitable for onnx conversion,
    # thus use a custom op instead and replace later
    priorbox_concat_plugin = gs.create_node(
        "priorbox_concat", op="Concat_TRT", dtype=tf.float32, axis=2)

    squeeze_plugin = gs.create_node(
        "squeeze", op="Squeeze_TRT", axis=2, inputs=["boxloc_concat"])

    boxloc_concat_plugin = gs.create_node(
        "boxloc_concat",
        op="FlattenConcat_TRT",
        # dtype=tf.float32,
        axis=1,  # Currently only axis = 1 is supported.
        ignoreBatch=0  # Currently only ignoreBatch = false is supported.
    )

    boxconf_concat_plugin = gs.create_node(
        "boxconf_concat",
        op="FlattenConcat_TRT",
        # dtype=tf.float32,
        axis=1,
        ignoreBatch=0
    )

    # transform (map) tf namespace to trt namespace --> tf namespace : trt namespace
    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": priorbox_plugin,
        "Postprocessor": nms_plugin,
        "Preprocessor": input_plugin,
        "Cast": input_plugin,
        "ToFloat": input_plugin,
        "image_tensor": input_plugin,
        "Concatenate": priorbox_concat_plugin,
        "Squeeze": squeeze_plugin,
        "concat": boxloc_concat_plugin,
        "concat_1": boxconf_concat_plugin
    }

    dynamic_graph.collapse_namespaces(namespace_plugin_map)

    namespace_remove = {
        "Cast",
        "ToFloat",
        "Preprocessor"
    }

    dynamic_graph.remove(
        dynamic_graph.find_nodes_by_path(namespace_remove), remove_exclusive_dependencies=False)

    # fix name and draw out the graph input from the input to the NMS_TRT node (output node)
    for n in range(len(dynamic_graph.find_nodes_by_op('NMS_TRT'))):
        for i, name in enumerate(
                dynamic_graph.find_nodes_by_op('NMS_TRT')[n].input):
            if TRT_INPUT_NAME in name:
                dynamic_graph.find_nodes_by_op('NMS_TRT')[n].input.pop(i)

    # remove all inputs to the node GridAnchor_TRT which needs no input
    for n in range(len(dynamic_graph.find_nodes_by_op('GridAnchor_TRT'))):
        for nk in range(len(dynamic_graph.find_nodes_by_op('GridAnchor_TRT')[n].input)):
            dynamic_graph.find_nodes_by_op('GridAnchor_TRT')[n].input.pop(nk)

    dynamic_graph.remove(
        dynamic_graph.graph_outputs, remove_exclusive_dependencies=False)

    print('---- the graphsurgeon tf ssd completed ----', '\n',
          '---- store the surged tf model to ', path_graph_pb, 'for onnx conversion ---- \n')
    # dynamic_graph.write_tensorboard(tmp_tbdir_d)
    # dynamic_graph.write(path_graph_pb)  # store the surged tf model
    writer_d = tf.summary.create_file_writer(tmp_tbdir_d)
    with writer_d.as_default():
        tf.summary.graph(dynamic_graph.as_graph_def())

    print('---- start onnx conversion with surged tf model ----')

    input_name = [TRT_INPUT_NAME + ":0"]
    output_name = [TRT_OUTPUT_NAME + ":0"]
    # input_name = [TRT_INPUT_NAME]
    # output_name = [TRT_OUTPUT_NAME]

    # ##load custom ops need for conversion from tf model to onnx model when parsing with tf backend
    # the custom ops can be constructed by the makefile in dir /tensorflow_trt_op
    load_customer_op(TF_CUSTOM_OP)

    path_onnx_model_0 = path_onnx_model.split(".")[0] + "_0.onnx"
    path_onnx_model_1_0 = path_onnx_model_1.split(".")[0] + "_0.onnx"
    # path_onnx_model_0:    file name for model which nodes are prefixed without "import/"; but
    #                       the BatchNormalization op need too many input with additional op
    # path_onnx_model_1_0:  file name for model which nodes  prefixed with "import/"; and
    #                       additional transport ops are inserted for transforming the NHWC (tf native format)
    #                       to NCHW format for nodes input, thus this ONNX model is not good for performance
    # onnx_model_proto_0 : model of path_onnx_model_0
    # onnx_model_proto_1_0 : model of path_onnx_model_0_1
    onnx_model_proto_0, onnx_model_proto_1_0 = convtf2onnx(path_graph_pb=path_graph_pb,
                                                           path_onnx_model=path_onnx_model_0,
                                                           path_onnx_model_1=path_onnx_model_1_0,
                                                           input_name=input_name,
                                                           output_name=output_name)

    print("---- modify the attributes of onnx model for ONNX parsing. ---- \n")
    # onnx_model_proto = onnx.load(path_onnx_model, format='protobuf')
    rev_onnx_attr(onnx_model_proto_0)
    onnx.save_model(onnx_model_proto_0, path_onnx_model)

    onnx_graph = gs_onnx.import_onnx(onnx_model_proto_0)
    nd_bn = [n for n in onnx_graph.nodes if n.op == "BatchNormalization"]

    print("---- modify the attributes of onnx model 1 for ONNX parsing.  ----\n")
    # onnx_model_proto_1 = onnx.load(path_onnx_model_1, format='protobuf')
    rev_onnx_attr(onnx_model_proto_1_0)
    onnx.save_model(onnx_model_proto_1_0, path_onnx_model_1)

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
    nd_box = [nd for nd in onnx_model_proto.graph.node
              if nd.name in ["priorbox", "priorbox_concat", "boxconf_concat", "boxloc_concat", "squeeze", "nms"]]
    for nd in nd_box:
        nd_attr = [(iattr, attr) for iattr, attr in enumerate(nd.attribute) if attr.name == "dtype"][0]
        nd.attribute.pop(nd_attr[0])

    return


def redef_onnx_node_4_trt_plugin(path_onnx_model, path_onnx_model_new):
    # ref : https://developer.nvidia.com/blog/estimating-depth-beyond-2d-using-custom-layers-on-tensorrt-and-onnx-models/

    print("---- surge the onnx model by replacing the some nodes with TRT plugin and ops. ----")
    onnx_model_proto = onnx.load(path_onnx_model, format='protobuf')
    onnx_graph = gs_onnx.import_onnx(onnx_model_proto)

    nodes_reshape_conf = [nd for nd in onnx_graph.nodes
                          if nd.op == "Reshape" and nd.name.split("/", 1)[1] == "Reshape_1"]
    for nd in nodes_reshape_conf:
        nd.inputs[1].values = np.array([1, -1, 1, 91])

    # insert transpose after BoxPredictor_5 boxconf
    node_to_boxconf_5_trnsp = [nd for nd in onnx_graph.nodes
                               if nd.name == "BoxPredictor_5/ClassPredictor/BiasAdd"][0]
    input_boxconf_5_trnsp = node_to_boxconf_5_trnsp.outputs[0]
    output_boxconf_5_trnsp = gs_onnx.Variable(name="boxconf_5_trnsp_out", dtype=np.float32)
    node_boxconf_5_trnsp = gs_onnx.Node(op="Transpose", name="boxconf_5_trnsp",
                                        attrs={"perm": [0, 2, 3, 1]},
                                        inputs=[input_boxconf_5_trnsp],
                                        outputs=[output_boxconf_5_trnsp]
                                        )
    onnx_graph.nodes.append(node_boxconf_5_trnsp)
    node_reshape_boxconf_5 = [nd for nd in onnx_graph.nodes if nd.name == "BoxPredictor_5/Reshape_1"][0]
    node_reshape_boxconf_5.inputs[0] = node_boxconf_5_trnsp.outputs[0]

    # insert transpose after BoxPredictor_5 boxloc
    node_to_boxloc_5_trnsp = [nd for nd in onnx_graph.nodes
                              if nd.name == "BoxPredictor_5/BoxEncodingPredictor/BiasAdd"][0]
    input_boxloc_5_trnsp = node_to_boxloc_5_trnsp.outputs[0]
    output_boxloc_5_trnsp = gs_onnx.Variable(name="boxloc_5_trnsp_out", dtype=np.float32)
    node_boxloc_5_trnsp = gs_onnx.Node(op="Transpose", name="boxloc_5_trnsp",
                                       attrs={"perm": [0, 2, 3, 1]},
                                       inputs=[input_boxloc_5_trnsp],
                                       outputs=[output_boxloc_5_trnsp]
                                       )
    onnx_graph.nodes.append(node_boxloc_5_trnsp)
    node_reshape_boxloc_5 = [nd for nd in onnx_graph.nodes if nd.name == "BoxPredictor_5/Reshape"][0]
    node_reshape_boxloc_5.inputs[0] = node_boxloc_5_trnsp.outputs[0]

    # node_GridAnchor_TRT, add dummy input
    node_GridAnchor_TRT = [nd for nd in onnx_graph.nodes if nd.name == "priorbox"][0]
    input_GridAnchor_TRT = gs_onnx.Constant(name="priorbox_in:", values=np.ones(1))
    node_GridAnchor_TRT.inputs = [input_GridAnchor_TRT]

    # node_priorbox_concat, modify op name and add explicitly the inputs from node_GridAnchor_TRT
    node_priorbox_concat = [nd for nd in onnx_graph.nodes if nd.name == "priorbox_concat"][0]
    node_priorbox_concat.op = "Concat"
    node_priorbox_concat.inputs = node_GridAnchor_TRT.outputs
    node_priorbox_concat.outputs[0].dtype = np.float32

    node_boxconf_concat = [nd for nd in onnx_graph.nodes if nd.name == "boxconf_concat"][0]
    node_boxconf_concat.outputs[0].dtype = np.float32
    for o in list(node_boxconf_concat.inputs):
        o.dtype = np.float32

    node_boxloc_concat = [nd for nd in onnx_graph.nodes if nd.name == "boxloc_concat"][0]
    node_boxloc_concat.outputs[0].dtype = np.float32
    for o in list(node_boxloc_concat.inputs):
        o.dtype = np.float32

    # squeeze, modify op name
    node_squeeze = [nd for nd in onnx_graph.nodes if nd.name == "squeeze"][0]
    node_squeeze.op = "Squeeze"

    # node_NMS_TRT, reconnect
    node_NMS_TRT = [nd for nd in onnx_graph.nodes if nd.name == "nms"][0]
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

    # *** use "SAME_UPPER" because it is the same padding operation as the "same" used in uff conv op
    for nd_conv in onnx_graph.nodes:
        if nd_conv.op == "Conv":
            nd_conv.attrs["pads"] = []  # [0, 0, 1, 1] --> [1, 1, 0, 0]
            nd_conv.attrs["auto_pad"] = "SAME_UPPER"  # "SAME_LOWER"; "SAME_UPPER" is same as the "same" in uff conv op

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


def redef_onnx_node_4_trt_plugin_1(path_onnx_model, path_onnx_model_new):
    # ref : https://developer.nvidia.com/blog/estimating-depth-beyond-2d-using-custom-layers-on-tensorrt-and-onnx-models/

    print("---- surge the onnx model by replacing the some nodes with TRT plugin and ops. ----")
    onnx_model_proto = onnx.load(path_onnx_model, format='protobuf')
    onnx_graph = gs_onnx.import_onnx(onnx_model_proto)

    nodes_reshape_conf = [nd for nd in onnx_graph.nodes
                          if nd.name in ["import/BoxPredictor_0/Reshape_1",
                                         "import/BoxPredictor_1/Reshape_1",
                                         "import/BoxPredictor_2/Reshape_1",
                                         "import/BoxPredictor_3/Reshape_1",
                                         "import/BoxPredictor_4/Reshape_1",
                                         "import/BoxPredictor_5/Reshape_1"]]
    for nd in nodes_reshape_conf:
        nd.inputs[1].values = np.array([1, -1, 1, 91])

    # insert transpose after BoxPredictor_5 boxconf
    node_to_boxconf_5_trnsp = [nd for nd in onnx_graph.nodes
                               if nd.name == "import/BoxPredictor_5/ClassPredictor/BiasAdd"][0]
    input_boxconf_5_trnsp = node_to_boxconf_5_trnsp.outputs[0]
    output_boxconf_5_trnsp = gs_onnx.Variable(name="boxconf_5_trnsp_out", dtype=np.float32)
    node_boxconf_5_trnsp = gs_onnx.Node(op="Transpose", name="boxconf_5_trnsp",
                                        attrs={"perm": [0, 2, 3, 1]},
                                        inputs=[input_boxconf_5_trnsp],
                                        outputs=[output_boxconf_5_trnsp]
                                        )
    onnx_graph.nodes.append(node_boxconf_5_trnsp)
    node_reshape_boxconf_5 = [nd for nd in onnx_graph.nodes if nd.name == "import/BoxPredictor_5/Reshape_1"][0]
    node_reshape_boxconf_5.inputs[0] = node_boxconf_5_trnsp.outputs[0]

    # insert transpose after BoxPredictor_5 boxloc
    node_to_boxloc_5_trnsp = [nd for nd in onnx_graph.nodes
                              if nd.name == "import/BoxPredictor_5/BoxEncodingPredictor/BiasAdd"][0]
    input_boxloc_5_trnsp = node_to_boxloc_5_trnsp.outputs[0]
    output_boxloc_5_trnsp = gs_onnx.Variable(name="boxloc_5_trnsp_out", dtype=np.float32)
    node_boxloc_5_trnsp = gs_onnx.Node(op="Transpose", name="boxloc_5_trnsp",
                                       attrs={"perm": [0, 2, 3, 1]},
                                       inputs=[input_boxloc_5_trnsp],
                                       outputs=[output_boxloc_5_trnsp]
                                       )
    onnx_graph.nodes.append(node_boxloc_5_trnsp)
    node_reshape_boxloc_5 = [nd for nd in onnx_graph.nodes if nd.name == "import/BoxPredictor_5/Reshape"][0]
    node_reshape_boxloc_5.inputs[0] = node_boxloc_5_trnsp.outputs[0]

    # node_GridAnchor_TRT, add dummy input
    node_GridAnchor_TRT = [nd for nd in onnx_graph.nodes if nd.name == "import/priorbox"][0]
    input_GridAnchor_TRT = gs_onnx.Constant(name="priorbox_in:", values=np.ones(1))
    node_GridAnchor_TRT.inputs = [input_GridAnchor_TRT]

    # node_priorbox_concat, modify op name and add explicitly the inputs from node_GridAnchor_TRT
    node_priorbox_concat = [nd for nd in onnx_graph.nodes if nd.name == "import/priorbox_concat"][0]
    node_priorbox_concat.op = "Concat"
    node_priorbox_concat.inputs = node_GridAnchor_TRT.outputs
    node_priorbox_concat.inputs.reverse()

    node_boxconf_concat = [nd for nd in onnx_graph.nodes if nd.name == "import/boxconf_concat"][0]
    # node_boxconf_concat.inputs.reverse()
    # node_boxconf_concat.attrs["axis"] = 1

    node_boxloc_concat = [nd for nd in onnx_graph.nodes if nd.name == "import/boxloc_concat"][0]
    # node_boxloc_concat.inputs.reverse()
    # node_boxloc_concat.attrs["axis"] = 1

    # squeeze, modify op name
    node_squeeze = [nd for nd in onnx_graph.nodes if nd.name == "import/squeeze"][0]
    node_squeeze.op = "Squeeze"

    # node_NMS_TRT, reconnect
    node_NMS_TRT = [nd for nd in onnx_graph.nodes if nd.name == "import/nms"][0]
    output_nms_0 = gs_onnx.Variable(name="nms:0", dtype=np.float32)
    output_nms_1 = gs_onnx.Variable(name="nms:1", dtype=np.float32)
    node_NMS_TRT.inputs = [node_priorbox_concat.outputs[0],
                           node_boxloc_concat.outputs[0],
                           node_boxconf_concat.outputs[0]]
    # node_NMS_TRT.inputs = [node_priorbox_concat.outputs[0],
    #                       node_boxconf_concat.outputs[0],
    #                       node_boxloc_concat.outputs[0]]
    # node_NMS_TRT.inputs = [node_boxloc_concat.outputs[0],
    #                       node_priorbox_concat.outputs[0],
    #                       node_boxconf_concat.outputs[0]]

    node_NMS_TRT.outputs = [output_nms_0, output_nms_1]
    # onnx_graph.outputs = [output_boxloc_concat]
    # onnx_graph.outputs = [output_priorbox]
    # onnx_graph.outputs = [output_priorbox_concat]
    # onnx_graph.outputs = [output_boxconf_concat]
    onnx_graph.outputs = node_NMS_TRT.outputs

    onnx_graph.cleanup().toposort()
    # onnx_graph.toposort()

    # graph = gs.Graph(nodes=[node_GridAnchor_TRT], inputs=[input_one, input_sec], outputs=[output],
    #                 name="FlattenConcat_test")
    # path_onnx_model_new = os.path.join(MODEL_REPO_DIR, MODEL_NAME + "_new.onnx")
    # onnx.save_model(gs_onnx.export_onnx(onnx_graph), path_onnx_model_new)
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
    checkpoint_path = os.path.join(TF_MODEL_DIR, MODEL_NAME, "checkpoint")      # the ckpt director of TF v2, refer to tf.train.checkpoint_management.py
    config_path = os.path.join(TF_MODEL_DIR, MODEL_NAME, "pipeline.config")
    output_engine = os.path.join(MODEL_REPO_DIR, MODEL_TRT + ".engine")

    path_graph_pb = os.path.join(MODEL_REPO_DIR, TMP_PB_GRAPH_NAME)
    path_onnx_model = os.path.join(MODEL_REPO_DIR, MODEL_NAME + ".onnx")
    path_onnx_model_1 = os.path.join(MODEL_REPO_DIR, MODEL_NAME + "_1.onnx")
    ssd_pipeline_to_onnx(checkpoint_path, config_path,
                         path_graph_pb, path_onnx_model, path_onnx_model_1, tmp_dir=TMP_MODEL)

    # path_onnx_model = os.path.join(MODEL_REPO_DIR, MODEL_NAME + ".onnx")
    # onnx.save_model(onnx_model_proto, path_onnx_model)
    path_onnx_model_new = os.path.join(MODEL_REPO_DIR, MODEL_NAME + "_new.onnx")
    redef_onnx_node_4_trt_plugin(path_onnx_model, path_onnx_model_new)  # redefine the trt plugin nodes

    path_onnx_model_new_1 = os.path.join(MODEL_REPO_DIR, MODEL_NAME + "_1_new.onnx")
    redef_onnx_node_4_trt_plugin_1(path_onnx_model_1, path_onnx_model_new_1)  # redefine the trt plugin nodes
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
