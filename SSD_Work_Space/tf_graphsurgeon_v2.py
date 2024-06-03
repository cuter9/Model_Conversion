# This is for graph resurgence of TF model conversion of TF version 2.X
import os
import subprocess
import tensorflow as tf
import graphsurgeon as gs
# from tensorflow.python.framework import function_def_to_graph as f2g
from Utils.ssd_utils_v2 import get_feature_map_shape, load_config
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def tf_graphsurgeon(path_tf_model=None, input_name=None, output_name=None,
                    onnx_work_dir=None, path_graph_pb=None, path_tf_custom_op=None):

    config = load_config(os.path.join(path_tf_model, "pipeline.config"))
    ssd_config_path = os.path.join(onnx_work_dir, "ssd_config.txt")  # for storing static graph
    if os.path.isfile(ssd_config_path):
        subprocess.call(['rm', ssd_config_path])
    # subprocess.call(['mkdir', '-p', ssd_config_dir])
    with open(ssd_config_path, 'w') as f:
        print(config, file = f)

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
    g_frozen = convert_variables_to_constants_v2(g_sig)     # freeze resources variables to constant
    g_frozen_def = g_frozen.graph.as_graph_def()
    # static_graph = gs.StaticGraph(g_frozen_def)

    for nd in g_frozen_def.node:
        if nd.name.split('/')[0] == 'StatefulPartitionedCall':  # remove the namespace used for 'StatefulPartitionedCall' operation
            nd.name = '/'.join(nd.name.split('/')[1:])
        for ndi in range(len(nd.input)):
            if nd.input[ndi].split('/')[0] == 'StatefulPartitionedCall':   # remove the namespace 'StatefulPartitionedCall' of node input name            \
                nd.input[ndi] = '/'.join(nd.input[ndi].split('/')[1:])
        for ni in nd.input:
            # if ni.split('/')[0] == '^StatefulPartitionedCall':
            if list(ni)[0] == '^':  # remove the control dependence input
                nd.input.remove(ni)

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

    all_assert_nodes = dynamic_graph.find_nodes_by_op("Assert")
    dynamic_graph.remove(all_assert_nodes, remove_exclusive_dependencies=False)

    tmp_tbdir_d_0 = os.path.join(onnx_work_dir, "tf_board_data_d_0")  # for storing dynamic graph before insert TRT plugins
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
        codeType=3)     # box CodeTypeSSD : TF_CENTER, https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/api/c_api/_nv_infer_plugin_utils_8h_source.html

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

    # create output plugin
    output_plugin = gs.create_node(
        name="output",
        op="Identity",
        dtype=tf.float32,
        # dtype=tf.uint8,
        shape=[1, 1, nms_config.max_total_detections, 7])

    dynamic_graph.append(output_plugin)

    # transform (map) tf namespace to trt namespace --> tf namespace : trt namespace
    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": priorbox_plugin,
        "Postprocessor": nms_plugin,
        "Identity_1": output_plugin,
        "Identity_2": output_plugin,
        "Identity_3": output_plugin,
        "Identity_4": output_plugin,
        "Identity_5": output_plugin,
        "Identity_6": output_plugin,
        "Identity_7": output_plugin,
        "Preprocessor": input_plugin,
        "Cast": input_plugin,
        "input_tensor": input_plugin,
        "image_tensor": input_plugin,
        "Concatenate": priorbox_concat_plugin,
        "Squeeze": squeeze_plugin,
        "concat": boxloc_concat_plugin,
        "concat_1": boxconf_concat_plugin
    }

    dynamic_graph.collapse_namespaces(namespace_plugin_map, exclude_nodes=[output_plugin])

    namespace_remove = {
        "Cast",
        "add",
        "Cast_1",
        "ToFloat",
        "Identity",
        "Preprocessor",
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
    output_name = [output_name + ":0"]
    # input_name = [input_name]
    # output_name = [output_name]

    dynamic_graph.write(path_graph_pb)  # store the surged tf model

    # ##load custom ops need for conversion from tf model to onnx model when parsing with tf backend
    # the custom ops can be constructed by the makefile in dir /tensorflow_trt_op
    # load_customer_op(path_tf_custom_op)

    return input_name, output_name, path_tf_custom_op