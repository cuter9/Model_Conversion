import os
import subprocess
import tensorflow as tf
import graphsurgeon as gs
from tensorflow.python.framework import function_def_to_graph as f2g
from Utils.ssd_utils_v2 import get_feature_map_shape
from tf_onnx import load_customer_op

def tf_graphsurgeon(config, input_name=None, output_name=None,
                    onnx_work_dir=None, tmp_dir=None, path_graph_pb=None, path_tf_custom_op=None):
    # get input shape
    channels = 3
    height = config.model.ssd.image_resizer.fixed_shape_resizer.height
    width = config.model.ssd.image_resizer.fixed_shape_resizer.width

    tmp_tbdir_s = os.path.join(onnx_work_dir, "tf_board_data_s")  # for storing static graph
    if os.path.isdir(tmp_tbdir_s):
        subprocess.call(['rm', '-r', tmp_tbdir_s])
    subprocess.call(['mkdir', '-p', tmp_tbdir_s])

    tmp_tbdir_d = os.path.join(onnx_work_dir, "tf_board_data_d")  # for storing dynamic graph
    if os.path.isdir(tmp_tbdir_d):
        subprocess.call(['rm', '-r', tmp_tbdir_d])
    subprocess.call(['mkdir', '-p', tmp_tbdir_d])

    saved_model = tf.saved_model.load(tmp_dir)
    g_serving = saved_model.signatures["serving_default"]  # creat tf Graph objects
    g = g_serving.graph
    g_def = g.as_graph_def()
    g_def_lib_func = g_def.library.function
    static_graph = gs.StaticGraph(g_def)
    # static_graph.lib_func = g_def_lib_func
    # static_graph = gs.StaticGraph(frozen_graph_path)
    # static_graph.write_tensorboard(tmp_tbdir_s)       # TensorRT use TF v1 to write graph which can not be used in TF v2
    # https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/summary/graph
    # writer_s = tf.summary.create_file_writer(tmp_tbdir_s)
    # with writer_s.as_default():
    #    tf.summary.graph(g_def)

    dynamic_graph = gs.DynamicGraph(g_def)

    # Convert a FunctionDef used in the TF v2 ssd model to a GraphDefTF
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/function_def_to_graph.py
    spc = dynamic_graph.find_nodes_by_op('StatefulPartitionedCall')
    spc_func_name = spc[0].attr['f'].func.name
    func_spc = [f for f in dynamic_graph._internal_graphdef.library.function if f.signature.name == spc_func_name][0]  # []
    graph_def_spc = f2g.function_def_to_graph_def(func_spc)[0]  # [0] : GraphDef; [1] : list of nodes

    # find the nodes corresponding to resource variables
    v_rsc_name = [v.name.split("/") for v in g.variables]
    nd_v = []
    for vn in v_rsc_name:
        nd_v.append([n for n in graph_def_spc.node if vn[-2] in n.name.split('/') and n.op != 'ReadVariableOp'])

    dynamic_graph_spc = gs.DynamicGraph(graph_def_spc)

    writer_s = tf.summary.create_file_writer(tmp_tbdir_s)
    with writer_s.as_default():
        tf.summary.graph(graph_def_spc)

    all_noop_nodes = dynamic_graph_spc.find_nodes_by_op("NoOp")
    all_resources_nodes = dynamic_graph_spc.find_nodes_by_op("ReadVariableOp")
    name_nd_rsc = [n.name for n in all_resources_nodes]
    # n_idx = 0
    map_input = []
    for nk, nd in dynamic_graph_spc.node_map.items():
        input_name = [(idx, nd_in) for idx, nd_in in enumerate(nd.input) if nd_in.split(":")[0] in name_nd_rsc]
        if input_name:
            _map_input = []
            for in_n in input_name:
                nd.input[in_n[0]] = [rnd.input[0] for rnd in all_resources_nodes if rnd.name == in_n[1].split(":")[:-1][0]][
                    0]
                _map_input.append([in_n[1], nd.input[in_n[0]]])
            # print(n_idx, _map_input)
            map_input.append(_map_input)
        # else:
        # print(n_idx, [])
        # n_idx += 1

    # dynamic_graph_spc.forward_inputs(all_resources_nodes)
    dynamic_graph_spc.remove(all_resources_nodes, remove_exclusive_dependencies=False)

    writer_d = tf.summary.create_file_writer(tmp_tbdir_d)
    with writer_d.as_default():
        tf.summary.graph(dynamic_graph_spc.as_graph_def())

    # forward all identity nodes
    all_identity_nodes = dynamic_graph.find_nodes_by_op("Identity")
    dynamic_graph.forward_inputs(all_identity_nodes)

    # create input plugin
    input_plugin = gs.create_node(
        name=input_name,
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
        name=output_name,
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
            if input_name in name:
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

    input_name = [input_name + ":0"]
    output_name = [output_name + ":0"]
    # input_name = [input_name]
    # output_name = [output_name]

    # ##load custom ops need for conversion from tf model to onnx model when parsing with tf backend
    # the custom ops can be constructed by the makefile in dir /tensorflow_trt_op
    load_customer_op(path_tf_custom_op)

    return input_name, output_name