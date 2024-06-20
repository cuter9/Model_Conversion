# This is for graph resurgence of TF SSD_FPN model conversion of TF version 2.X
import os
import subprocess

import numpy as np
import tensorflow as tf
import graphsurgeon as gs
# from tensorflow.python.framework import function_def_to_graph as f2g
from Utils.ssd_utils_v2 import get_feature_map_shape, get_feature_map_shape_fpn, load_config
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def tf_ssd_fpn_graphsurgeon(path_tf_model=None, input_name=None, output_name=None,
                            onnx_work_dir=None, path_graph_pb=None, path_tf_custom_op=None):
    config = load_config(os.path.join(path_tf_model, "pipeline.config"))
    ssd_config_path = os.path.join(onnx_work_dir, "ssd_config.txt")  # for storing static graph
    if os.path.isfile(ssd_config_path):
        subprocess.call(['rm', ssd_config_path])
    # subprocess.call(['mkdir', '-p', ssd_config_dir])
    with open(ssd_config_path, 'w') as f:
        print(config, file=f)

    # get input shape
    channels = 3
    height = config.model.ssd.image_resizer.fixed_shape_resizer.height
    width = config.model.ssd.image_resizer.fixed_shape_resizer.width

    saved_model = tf.saved_model.load(os.path.join(path_tf_model, 'saved_model'))
    g_sig = saved_model.signatures["serving_default"]  # creat tf Graph objects
    g_def = g_sig.graph.as_graph_def()

    tmp_tbdir_s_0 = os.path.join(onnx_work_dir, "tf_board_data_s_0")  # for storing static graph before frozen
    if os.path.isdir(tmp_tbdir_s_0):
        subprocess.call(['rm', '-r', tmp_tbdir_s_0])
    subprocess.call(['mkdir', '-p', tmp_tbdir_s_0])

    writer_s_0 = tf.summary.create_file_writer(tmp_tbdir_s_0)
    with writer_s_0.as_default():
        #    tf.summary.graph(graph_def_spc)
        tf.summary.graph(g_def)

    # https://blog.tensorflow.org/2021/03/a-tour-of-savedmodel-signatures.html
    # 1. In TF Version>2.0,model is excuted with "graph function" with specified "signature", rather than graph definition as in TF V1.X,
    # 2. In TF Version>2.0, the concrete function using tf.function and wrapped with specified "signature" is used to model training and inference.
    # 3. In TF Version>2.0, to speed up the inference, the signature concrete functions are excuted through StatefulPartitionedCall operation
    #    with resources variables is used to fully leverage the devices for parallel computing.
    # 4. Thus, before it can be converted to ONNX and TRT model, the StatefulPartitionedCall operation and
    #    its control dependencies shall be removed, and
    #    the related model resources variables for concrete function shall be "frozen to constant".
    # https://medium.com/@sebastingarcaacosta/how-to-export-a-tensorflow-2-x-keras-model-to-a-frozen-and-optimized-graph-39740846d9eb
    g_frozen = convert_variables_to_constants_v2(g_sig)  # freeze resources variables to constant
    g_frozen_def = g_frozen.graph.as_graph_def()
    # static_graph = gs.StaticGraph(g_frozen_def)

    tmp_tbdir_s_1 = os.path.join(onnx_work_dir, "tf_board_data_s_1")  # for storing static graph after frozen
    if os.path.isdir(tmp_tbdir_s_1):
        subprocess.call(['rm', '-r', tmp_tbdir_s_1])
    subprocess.call(['mkdir', '-p', tmp_tbdir_s_1])

    writer_s = tf.summary.create_file_writer(tmp_tbdir_s_1)
    with writer_s.as_default():
        #    tf.summary.graph(graph_def_spc)
        tf.summary.graph(g_frozen_def)

    for nd in g_frozen_def.node:
        if nd.name.split('/')[
            0] == 'StatefulPartitionedCall':  # remove the namespace used for 'StatefulPartitionedCall' operation
            nd.name = '/'.join(nd.name.split('/')[1:])
        for ndi in range(len(nd.input)):
            if nd.input[ndi].split('/')[
                0] == 'StatefulPartitionedCall':  # remove the namespace 'StatefulPartitionedCall' of node input name            \
                nd.input[ndi] = '/'.join(nd.input[ndi].split('/')[1:])
        ni = list(nd.input)
        for n in ni:
            # if ni.split('/')[0] == '^StatefulPartitionedCall':
            if list(n)[0] == '^':  # remove the control dependence input (which has "^" in the name string)
                nd.input.remove(n)

    '''
    s = []
    for n in g_freezen_gdef.node:
        if 'StatefulPartitionedCall' in n.name.split('/'):
            n.name = '/'.join(n.name.split('/')[1:])
 
        for ni in range(len(n.input)):
            if 'StatefulPartitionedCall' in n.input[ni].split('/'):
                n.input[ni] = '/'.join(n.input[ni].split('/'))

        # for no in n.output:
        #    if 'StatefulPartitionedCall' in no.split('/'):
        #        no = '/'.join(no.split('/')[1:])

        s.append(n)
    '''
    '''
    g = g_serving.graph
    g_def = g.as_graph_def()

    g_def_lib_func = g_def.library.function
    static_graph = gs.StaticGraph(g_def)
    '''
    # static_graph.lib_func = g_def_lib_func
    # static_graph = gs.StaticGraph(frozen_graph_path)
    # static_graph.write_tensorboard(tmp_tbdir_s)       # TensorRT use TF v1 to write graph which can not be used in TF v2
    # https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/summary/graph
    # writer_s = tf.summary.create_file_writer(tmp_tbdir_s)
    # with writer_s.as_default():
    #    tf.summary.graph(g_def)

    '''
    dynamic_graph = gs.DynamicGraph(g_freezen_gdef)
    
    # https://blog.tensorflow.org/2021/03/a-tour-of-savedmodel-signatures.html
    # https://github.com/tensorflow/tensorflow/blob/255a314badfe538f7ddaa6345ef774755973143d/tensorflow/python/saved_model/load.py#L332
    g_cap_map_list = list(g.captures)
    g_cap_shape = [b[0]._handle_data.shape_and_type[0].shape for b in g_cap_map_list]
    var_shape = [v.handle._handle_data.shape_and_type[0].shape for v in g.variables]
    assert g_cap_shape == var_shape, 'captures is not the same as variables'

    # Convert a FunctionDef used in the TF v2 ssd model to a GraphDefTF
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/function_def_to_graph.py
    spc = dynamic_graph.find_nodes_by_op('StatefulPartitionedCall')
    spc_func_name = spc[0].attr['f'].func.name
    # f_def_spc_d_graph = [f for f in dynamic_graph._internal_graphdef.library.function if f.signature.name == spc_func_name][0]  # []
    f_def_spc = [f for f in g_def.library.function if f.signature.name == spc_func_name][0]
    graph_def_spc = f2g.function_def_to_graph_def(f_def_spc)[0]  # [0] : GraphDef; [1] : list of nodes

    # find the nodes corresponding to resource variables
    v_rsc_name = [v.name.split("/") for v in g.variables]
    nd_vsc = []
    for vn in v_rsc_name:
        nd_vsc.append([n for n in graph_def_spc.node if vn[-2] in n.name.split('/') and n.op != 'ReadVariableOp'])

    dynamic_graph_spc = gs.DynamicGraph(graph_def_spc)
    '''
    tmp_tbdir_s = os.path.join(onnx_work_dir, "tf_board_data_s")  # for storing static graph after frozen
    if os.path.isdir(tmp_tbdir_s):
        subprocess.call(['rm', '-r', tmp_tbdir_s])
    subprocess.call(['mkdir', '-p', tmp_tbdir_s])

    writer_s = tf.summary.create_file_writer(tmp_tbdir_s)
    with writer_s.as_default():
        #    tf.summary.graph(graph_def_spc)
        tf.summary.graph(g_frozen_def)

    '''
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
    # dynamic_graph_spc.remove(all_resources_nodes, remove_exclusive_dependencies=False)
    
    writer_d = tf.summary.create_file_writer(tmp_tbdir_d)
    with writer_d.as_default():
        tf.summary.graph(dynamic_graph_spc.as_graph_def())
    '''

    dynamic_graph = gs.DynamicGraph(g_frozen_def)

    # forward all identity nodes
    all_identity_nodes = dynamic_graph.find_nodes_by_op("Identity")
    dynamic_graph.forward_inputs(all_identity_nodes)

    # f_nd = dynamic_graph.find_nodes_by_path("Func")
    all_noop_nodes = dynamic_graph.find_nodes_by_op("NoOp")
    dynamic_graph.remove(all_noop_nodes, remove_exclusive_dependencies=False)

    # all_assert_nodes = dynamic_graph.find_nodes_by_op("Assert")
    # dynamic_graph.remove(all_assert_nodes, remove_exclusive_dependencies=False)
    node_map = dynamic_graph.node_map
    '''
    # remove the reshape operation for original predict box and class
    all_reshape_nodes = []
    for key, node in node_map.items():
        if 'WeightSharedConvolutionalBoxHead' in key.split('/') and node.op == "Reshape":
            ni = list(node.input)
            for n in ni:
                if n == '/'.join([key, 'shape']):
                    node.input.remove(n)
                    node.input.remove(n)        # remove the input
                    dynamic_graph.remove(dynamic_graph.find_nodes_by_path(n))   # remove the node of input
            all_reshape_nodes.append(node)

        if 'WeightSharedConvolutionalClassHead' in key.split('/') and node.op == "Reshape" :
            ni = list(node.input)
            for n in ni:
                if n == '/'.join([key, 'shape']):
                    node.input.remove(n)        # remove the input
                    dynamic_graph.remove(dynamic_graph.find_nodes_by_path(n))       # remove the node of input
            all_reshape_nodes.append(node)
    '''
    # dynamic_graph.forward_inputs(all_reshape_nodes)

    tmp_tbdir_d_0 = os.path.join(onnx_work_dir,
                                 "tf_board_data_d_0")  # for storing dynamic graph before insert TRT plugins
    if os.path.isdir(tmp_tbdir_d_0):
        subprocess.call(['rm', '-r', tmp_tbdir_d_0])
    subprocess.call(['mkdir', '-p', tmp_tbdir_d_0])

    writer_d_0 = tf.summary.create_file_writer(tmp_tbdir_d_0)
    with writer_d_0.as_default():
        tf.summary.graph(dynamic_graph.as_graph_def())

    '''
    tmp_tbdir_d = os.path.join(onnx_work_dir, "tf_board_data_d")  # for storing dynamic graph
    if os.path.isdir(tmp_tbdir_d):
        subprocess.call(['rm', '-r', tmp_tbdir_d])
    subprocess.call(['mkdir', '-p', tmp_tbdir_d])

    writer_d = tf.summary.create_file_writer(tmp_tbdir_d)
    with writer_d.as_default():
        tf.summary.graph(dynamic_graph.as_graph_def())
    '''

    # create input plugin
    input_plugin = gs.create_node(
        name=input_name,
        op="Placeholder",
        dtype=tf.float32,
        # dtype=tf.uint8,
        shape=[1, height, width, channels])

    '''
    # create anchor box generator

    anchor_generator_config = config.model.ssd.anchor_generator.multiscale_anchor_generator
    box_coder_config = config.model.ssd.box_coder.faster_rcnn_box_coder

    num_layers = anchor_generator_config.max_level - anchor_generator_config.min_level + 1

    anchor_scale = anchor_generator_config.anchor_scale
    # anchor_scale = 2
    scales_per_octave = anchor_generator_config.scales_per_octave
    min_size_0 = anchor_scale * 2**anchor_generator_config.min_level / height  # add additional layer but ignor it in later operation
    max_size_0 = anchor_scale * 2**anchor_generator_config.max_level / height
    spo = [2 ** (float(scale) / scales_per_octave) for scale in range(scales_per_octave)]
    min_size = [min_size_0 * s  for s in spo]
    max_size = [max_size_0 * s  for s in spo]

    aspect_ratios = list(anchor_generator_config.aspect_ratios)[1:]     # ignor aspect_ratios = 1 which will be auto gen by TRT
    # aspect_ratios.reverse()
    feature_map_shapes = get_feature_map_shape_fpn(config)

    # use 2 GridAnchor_TRT to generate bounding boxes grid for each scale in octatve layer for all layer
    # create GridAnchor for scale 0 of each octave
    priorbox_plugin_0 = gs.create_node(
        name="priorbox_0",
        op="GridAnchor_TRT",
#        minSize=anchor_generator_config.min_scale,  # minSize
#        maxSize=anchor_generator_config.max_scale,
        minSize=min_size[0],  # minSize
        maxSize=max_size[0],
#        aspectRatios=list(anchor_generator_config.aspect_ratios),
        aspectRatios=aspect_ratios,
        variance=[
            1.0 / box_coder_config.y_scale,
            1.0 / box_coder_config.x_scale,
            1.0 / box_coder_config.height_scale,
            1.0 / box_coder_config.width_scale
        ],
        featureMapShapes=feature_map_shapes,     #[80, 40, 20, 10, 5, 3]
        # featureMapShapes=[1, 2, 3, 5, 10, 19],
        numLayers=num_layers)
#        numLayers = config.model.ssd.anchor_generator.ssd_anchor_generator.num_layers)

    # create GridAnchor for scale 1 of each octave
    priorbox_plugin_1 = gs.create_node(
        name="priorbox_1",
        op="GridAnchor_TRT",
        #        minSize=anchor_generator_config.min_scale,  # minSize
        #        maxSize=anchor_generator_config.max_scale,
        minSize=min_size[1],  # minSize with 2nd scales of octave
        maxSize=max_size[1],
        #        aspectRatios=list(anchor_generator_config.aspect_ratios),
        aspectRatios=aspect_ratios,
        variance=[
            1.0 / box_coder_config.y_scale,
            1.0 / box_coder_config.x_scale,
            1.0 / box_coder_config.height_scale,
            1.0 / box_coder_config.width_scale
        ],
        featureMapShapes=feature_map_shapes,  # [80, 40, 20, 10, 5, 3]
        # featureMapShapes=[1, 2, 3, 5, 10, 19],
        numLayers=num_layers)

    dynamic_graph.append(priorbox_plugin_0)
    dynamic_graph.append(priorbox_plugin_1)
    '''

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
        numClasses=config.model.ssd.num_classes + 1,  # add background class
        inputOrder=[0, 2, 1],  # [1, 2, 0]
        #        inputOrder=[0, 1, 2],  # [1, 2, 0]
        confSigmoid=1,
        isNormalized=1,
        # scoreConverter="SIGMOID",
        scoreBits=16,
        isBatchAgnostic=1,
        codeType=1)  # box CodeTypeSSD : 1 = CORNER, https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/api/c_api/_nv_infer_plugin_utils_8h_source.html
    # codeType = 3)  # box CodeTypeSSD : 3 = TF_CENTER, https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/api/c_api/_nv_infer_plugin_utils_8h_source.html

    '''
    
    nms_config = config.model.ssd.post_processing.batch_non_max_suppression
    nms_plugin = gs.create_node(
        name=output_name,
        op="EfficientNMS_TRT",
        background_class=0,     # The label ID for the background
        score_threshold=nms_config.score_threshold, # The scalar threshold for score (low scoring boxes are removed).
        iou_threshold=nms_config.iou_threshold,     # The scalar threshold for IOU
        max_detections_per_class=nms_config.max_detections_per_class,
        max_output_boxes=nms_config.max_total_detections,
        class_agnostic=0,      # Set to true to do class-independent NMS
        score_activation=1,        # Set to true to apply sigmoid activation
        box_coding=0)     #  0 = BoxCorner, 1 = BoxCenterSize, https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/api/c_api/_nv_infer_plugin_utils_8h_source.html
    '''
    # dynamic_graph.append(nms_plugin)

    '''
    # tf built in op Concat is not suitable for onnx conversion,
    # thus use a custom op instead and replace later
    priorbox_concat_plugin = gs.create_node(
        "priorbox_concat", op="Concat_TRT", N=2, dtype=tf.float32, axis=2)
    priorbox_concat_plugin.input.extend(["priorbox_concat_0", "priorbox_concat_1"])

    priorbox_concat_plugin_0 = gs.create_node(
        "priorbox_concat_0", op="Concat_TRT", N=1, dtype=tf.float32, axis=2)
    priorbox_concat_plugin_0.input.extend(["priorbox_0"])

    priorbox_concat_plugin_1 = gs.create_node(
        "priorbox_concat_1", op="Concat_TRT", N=1, dtype=tf.float32, axis=2)
    priorbox_concat_plugin_1.input.extend(["priorbox_1"])
    '''
    # using Constant node instead of using TRT AnchorGrid generator
    # which is not compatible with the Anchor Box used in FPN model
    grid_anchor_list, grid_anchor_tensor = grid_anchor_gen(config)
    priorbox_concat_plugin = gs.create_node(
        name="priorbox_concat",
        op="Const",
        value=grid_anchor_tensor,
        dtype=tf.float32
    )
    # there is a bug of trt graphsurgeon API missing dtype in tensor conversion
    priorbox_concat_plugin.attr["value"].tensor.dtype = tf.as_dtype(grid_anchor_tensor.dtype).as_datatype_enum

    # squeeze_plugin = gs.create_node(
    #    "squeeze", op="Squeeze_TRT", axis=2, inputs=["boxloc_concat"])

    # dynamic_graph.append(priorbox_concat_plugin)
    # dynamic_graph.append(priorbox_concat_plugin_0)
    # dynamic_graph.append(priorbox_concat_plugin_1)
    '''
    boxloc_concat_plugin = gs.create_node(
        name="boxloc_concat",
        op="Concat",
        # dtype=tf.float32,
        axis=1,  # Currently only axis = 1 is supported.
        ignoreBatch=0  # Currently only ignoreBatch = false is supported.
    )

    boxconf_concat_plugin = gs.create_node(
        name="boxconf_concat",
        op="Cancat",
        # dtype=tf.float32,
        axis=1,
        ignoreBatch=0
    )
    '''
    # nms_plugin.input.extend(["boxloc_concat", "boxconf_concat", "priorbox_concat"])
    # nms_plugin.input.extend(["concat", "concat_1", "priorbox_concat"])
    # dynamic_graph.append(nms_plugin)

    # create output plugin
    output_plugin = gs.create_node(
        name="output",
        op="Identity",
        dtype=tf.float32,
        # dtype=tf.uint8,
        shape=[1, 1, nms_config.max_total_detections, 7])

    # output_plugin.input.extend([output_name])
    # dynamic_graph.append(output_plugin)

    # transform (map) tf namespace to trt namespace --> tf namespace : trt namespace
    # the collapse_namespaces method will append a new node, thus append the node will duplicate
    namespace_plugin_map = {
        "MultiscaleGridAnchorGenerator": priorbox_concat_plugin,
        "Postprocessor": nms_plugin,
        "Preprocessor": input_plugin,
        "Cast": input_plugin,
        "input_tensor": input_plugin,
        "image_tensor": input_plugin,
        "Concatenate": priorbox_concat_plugin,
        #        "Squeeze": squeeze_plugin,
        #        "concat": boxloc_concat_plugin,
        #        "concat_1": boxconf_concat_plugin
    }

    dynamic_graph.collapse_namespaces(namespace_plugin_map)

    namespace_remove = {
        "Cast",
        "add",
        "add/y",
        "Cast_1",
        #        "ToFloat",
        "Identity",
        "Identity_1",
        "Identity_2",
        "Identity_3",
        "Identity_4",
        "Identity_5",
        "Identity_6",
        "Identity_7",
        "Preprocessor",
        "MultiscaleGridAnchorGenerator"
    }

    dynamic_graph.remove(
        dynamic_graph.find_nodes_by_path(namespace_remove), remove_exclusive_dependencies=False)

    # fix name and draw out the graph input from the input to the NMS_TRT node (output node)
    for n in range(len(dynamic_graph.find_nodes_by_op('NMS_TRT'))):
        for i, name in enumerate(
                dynamic_graph.find_nodes_by_op('NMS_TRT')[n].input):
            if input_name in name:
                dynamic_graph.find_nodes_by_op('NMS_TRT')[n].input.pop(i)
    '''
    # fix name and draw out the graph input from the input to the NMS_TRT node (output node)
    for n in range(len(dynamic_graph.find_nodes_by_op('EfficientNMS_TRT'))):
        for i, name in enumerate(
                dynamic_graph.find_nodes_by_op('EfficientNMS_TRT')[n].input):
            if input_name in name:
                dynamic_graph.find_nodes_by_op('EfficientNMS_TRT')[n].input.pop(i)
    '''
    # remove all inputs to the node GridAnchor_TRT which needs no input
    # for n in range(len(dynamic_graph.find_nodes_by_op('GridAnchor_TRT'))):
    #    nd_priorbox = dynamic_graph.find_nodes_by_op('GridAnchor_TRT')[0]
    #    # nd_priorbox.input.clear()
    #    lst_input = list(nd_priorbox.input)
    #    for ni in lst_input:
    #        nd_priorbox.input.remove(ni)

    nd_cant = dynamic_graph.find_nodes_by_name('priorbox_concat')[0]
    nd_cand_input = list(nd_cant.input)
    for n in nd_cand_input:
        if "MultiscaleGridAnchorGenerator" in n.split("/"):
            nd_cant.input.remove(n)

    nd_mgag = []
    for key, node in node_map.items():
        if 'MultiscaleGridAnchorGenerator' in key.split('/'):
            nd_mgag.append(node)
            if "strided" in key.split('/')[-1].split('_')[
                0]:  # remove the unknown_X const node in MultiscaleGridAnchorGenerator
                ni = node.input
                for n in ni:
                    if n.split("_")[0] == "unknown":
                        dynamic_graph.remove(dynamic_graph.find_nodes_by_name(n))

    dynamic_graph.remove(nd_mgag)

    # dynamic_graph.remove(
    #    dynamic_graph.graph_outputs, remove_exclusive_dependencies=False)

    '''
    nd_outputs = dynamic_graph.node_outputs
    for nd_name, nd in nd_outputs.items():
        for n in nd:
            if n.name.split('/')[0] == 'StatefulPartitionedCall':
                n.name = '/'.join(n.name.split('/')[1:])
            for ni in range(len(n.input)):
                if n.input[ni].split('/')[0] == 'StatefulPartitionedCall':
                    n.input[ni] = '/'.join(n.input[ni].split('/')[1:])
    
    for n in nd_outputs['nms']:
        n.name = '/'.join(n.name.split('/')[1:])
        for ni in range(len(n.input)):
            if n.input[ni].split('/')[0] == 'StatefulPartitionedCall':
                if dynamic_graph.find_nodes_by_name(n.input[ni]):
                    _n = dynamic_graph.find_nodes_by_name(n.input[ni])[0]
                    _n.name = '/'.join(n.input[ni].split('/')[1:])
                n.input[ni] = '/'.join(n.input[ni].split('/')[1:])
    '''
    print('---- the graphsurgeon tf ssd completed ----', '\n',
          '---- store the surged tf model to ', path_graph_pb, 'for onnx conversion ---- \n')

    # dynamic_graph.write_tensorboard(tmp_tbdir_d)
    # dynamic_graph.write(path_graph_pb)  # store the surged tf model
    tmp_tbdir_d = os.path.join(onnx_work_dir, "tf_board_data_d")  # for storing dynamic graph
    if os.path.isdir(tmp_tbdir_d):
        subprocess.call(['rm', '-r', tmp_tbdir_d])
    subprocess.call(['mkdir', '-p', tmp_tbdir_d])

    writer_d = tf.summary.create_file_writer(tmp_tbdir_d)
    with writer_d.as_default():
        tf.summary.graph(dynamic_graph.as_graph_def())

    print('---- start onnx conversion with surged tf model ----')

    input_name = [input_name + ":0"]
    output_name = [output_name + ":0", output_name + ":1", output_name + ":2", output_name + ":3"]
    # input_name = [input_name]
    # output_name = [output_name]

    dynamic_graph.write(path_graph_pb)  # store the surged tf model

    # ##load custom ops need for conversion from tf model to onnx model when parsing with tf backend
    # the custom ops can be constructed by the makefile in dir /tensorflow_trt_op
    # load_customer_op(path_tf_custom_op)

    return input_name, output_name, path_tf_custom_op


def grid_anchor_gen_effnms(config):  # for anchor box used in TRT efficientNMP plugin
    from object_detection.anchor_generators import multiscale_grid_anchor_generator

    anchor_generator_config = config.model.ssd.anchor_generator.multiscale_anchor_generator
    # box_coder_config = config.model.ssd.box_coder.faster_rcnn_box_coder
    num_layers = anchor_generator_config.max_level - anchor_generator_config.min_level + 1
    min_level = anchor_generator_config.min_level
    max_level = anchor_generator_config.max_level
    anchor_scale = anchor_generator_config.anchor_scale
    scales_per_octave = anchor_generator_config.scales_per_octave
    aspect_ratios = anchor_generator_config.aspect_ratios
    grid_gen = multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator(min_level,
                                                                              max_level,
                                                                              anchor_scale,
                                                                              aspect_ratios,
                                                                              scales_per_octave)

    feature_map_shapes = [(h, h) for h in get_feature_map_shape_fpn(config)]
    height = config.model.ssd.image_resizer.fixed_shape_resizer.height
    width = config.model.ssd.image_resizer.fixed_shape_resizer.width
    grid_anchor_list = grid_gen._generate(feature_map_shapes, im_height=height, im_width=width)

    grid_anchor_list_reshape = []
    for g in grid_anchor_list:
        # grid_anchor_list_reshape.append(np.reshape(g.data['boxes'].numpy(), newshape=[1, -1, 1]))
        b = tf.convert_to_tensor(np.expand_dims(g.data['boxes'].numpy(), axis=0))
        grid_anchor_list_reshape.append(b)

    grid_anchor_tensor_0 = np.concatenate(grid_anchor_list_reshape, 1)
    grid_anchor_tensor = grid_anchor_tensor_0.astype(np.float32)
    return grid_anchor_list, grid_anchor_tensor

# FPN model use Multiscale Grid Anchor,
# thus we leverage the multiscale_grid_anchor_generator in TF object_detection.anchor_generators to
# generate the anchor boxes for inference
# https://github.com/tensorflow/models/blob/master/research/object_detection/anchor_generators/multiscale_grid_anchor_generator.py#L30
# the FPN model predicted target has no variance, so the variance should used to adjust the predict boxes is TRT NMS plugin
def grid_anchor_gen(config):  # for anchor box used in TRT NMP plugin anchor boxes and their box variance
    from object_detection.anchor_generators import multiscale_grid_anchor_generator

    anchor_generator_config = config.model.ssd.anchor_generator.multiscale_anchor_generator
    # num_layers = anchor_generator_config.max_level - anchor_generator_config.min_level + 1
    min_level = anchor_generator_config.min_level
    max_level = anchor_generator_config.max_level
    anchor_scale = anchor_generator_config.anchor_scale
    scales_per_octave = anchor_generator_config.scales_per_octave
    aspect_ratios = anchor_generator_config.aspect_ratios
    # create anchor grid generator
    grid_gen = multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator(min_level,
                                                                              max_level,
                                                                              anchor_scale,
                                                                              aspect_ratios,
                                                                              scales_per_octave)

    feature_map_shapes = [(h, h) for h in get_feature_map_shape_fpn(config)]    # for each layer: eg. [40*40, 20*20, 10*10, 5*5, 3*3]
    height = config.model.ssd.image_resizer.fixed_shape_resizer.height
    width = config.model.ssd.image_resizer.fixed_shape_resizer.width

    # generate grid anchor boxes scale using generator grid_gen(), which will give a list of grid anchors boxes for each feature layer
    grid_anchor_list = grid_gen._generate(feature_map_shapes, im_height=height, im_width=width)

    #
    grid_anchor_list_reshape = []
    for g in grid_anchor_list:
        b = tf.convert_to_tensor(
            np.expand_dims(g.data['boxes'].numpy(), axis=0))  # g.data['boxes'] has shape = [no of grid, 4], e.g no of grid = 40*40 for first feat layer
        grid_anchor_list_reshape.append(b)
    # grid_anchor_list_reshape has expaned shape, eg. [[1, 40*40, 4], [1, 20*20, 4], [1, 10*10, 4], 1, 5*5, 4], [1, 3*3, 4]]
    # anchor grid shape should has shape [1, 2, numPriors * 4, 1]
    grid_anchor_tensor_0 = np.concatenate(grid_anchor_list_reshape, axis=1)     # [1, (40*40 + 20*20 + 10*10 + 5*5 + 3*3), 4]
    gar = np.reshape(grid_anchor_tensor_0, newshape=[1, -1, 1])     # [1, (40*40 + 20*20 + 10*10 + 5*5 + 3*3) * 4, 1]

    # boxes variance generation has the same shape as grid_anchor_tensor_0
    box_coder_config = config.model.ssd.box_coder.faster_rcnn_box_coder
    variance = np.asarray([[
        1.0 / box_coder_config.y_scale,
        1.0 / box_coder_config.x_scale,
        1.0 / box_coder_config.height_scale,
        1.0 / box_coder_config.width_scale
    ]])
    v = np.expand_dims(np.repeat(variance, grid_anchor_tensor_0.shape[1], axis=0), axis=0)
    var = np.reshape(v, newshape=[1, -1, 1])    # reshape to the shape as gar

    grid_anchor_tensor_1 = np.expand_dims(np.vstack((gar, var)), axis=0)    # stack grid anchor boxes and their variance
    grid_anchor_tensor = np.copy(grid_anchor_tensor_1.astype(np.float32))
    return grid_anchor_list, grid_anchor_tensor


def grid_anchor_gen_1(config):
    anchor_generator_config = config.model.ssd.anchor_generator.multiscale_anchor_generator
    box_coder_config = config.model.ssd.box_coder.faster_rcnn_box_coder
    num_layers = anchor_generator_config.max_level - anchor_generator_config.min_level + 1
    anchor_scale = anchor_generator_config.anchor_scale
    # anchor_scale = 2
    scales_per_octave = anchor_generator_config.scales_per_octave
    spo = [2 ** (float(scale) / scales_per_octave) for scale in range(scales_per_octave)]
    aspect_ratios = list(anchor_generator_config.aspect_ratios)
    # aspect_ratios.reverse()
    feature_map_shapes = get_feature_map_shape_fpn(config)
    grid_size = [2 ** bs for bs in range(anchor_generator_config.min_level, anchor_generator_config.max_level + 1)]
    base_anchor_size = anchor_scale * np.asarray(grid_size)
    anchor_height = np.asarray(
        [[np.asarray(spo) * bar * np.sqrt(ar) for ar in aspect_ratios] for bar in base_anchor_size])
    anchor_width = np.asarray(
        [[np.asarray(spo) * bar / np.sqrt(ar) for ar in aspect_ratios] for bar in base_anchor_size])
    anchor_center_x = [
        np.asarray([np.ones(len(spo) * len(aspect_ratios)) * (g + 0.5) * np.asarray(gsize) for g in range(fms)]) for
        fms, gsize in zip(feature_map_shapes, grid_size)]
    anchor_center_y = [
        np.asarray([np.ones(len(spo) * len(aspect_ratios)) * (g + 0.5) * np.asarray(gsize) for g in range(fms)]) for
        fms, gsize in zip(feature_map_shapes, grid_size)]
    return [anchor_height, anchor_width, anchor_center_x, anchor_center_y]
