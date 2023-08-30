import ctypes
import numpy as np
import os
import subprocess
import tensorrt as trt
from Utils.ssd_utils import load_config, tf_saved2frozen, get_feature_map_shape, download_model
import tf2onnx

MODEL_TRT = "ssd_mobilenet_v2_coco"
MODEL_NAME = "ssd_mobilenet_v2_coco_2018_03_29"
# MODEL_NAME = "ssd_inception_v2_coco_2017_11_17"

TRT_INPUT_NAME = 'input'
TRT_OUTPUT_NAME = 'nms'
FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'

WORK = os.getcwd()

DATA_REPO_DIR = os.path.join("../../", "Data_Repo/Model_Conversion/SSD_mobilenet")
os.makedirs(DATA_REPO_DIR, exist_ok=True)

UFF_WORK_SPACE = os.path.join(DATA_REPO_DIR, "UFF_Model")
os.makedirs(UFF_WORK_SPACE, exist_ok=True)

MODEL_REP_DIR = os.path.join(UFF_WORK_SPACE, "Repo")
os.makedirs(MODEL_REP_DIR, exist_ok=True)

TF_MODEL_DIR = os.path.join(DATA_REPO_DIR, "TF_Model")
os.makedirs(TF_MODEL_DIR, exist_ok=True)

TMP_MODEL = os.path.join(TF_MODEL_DIR, "Exported_Model")
os.makedirs(TMP_MODEL, exist_ok=True)


def ssd_pipeline_to_uff(checkpoint_path, config_path, tmp_dir='Exported_Model'):
    import graphsurgeon as gs
    import tensorflow as tf
    import uff

    print('---- start converting tf ssd model to uff ssd model ----')
    # TODO(@jwelsh): Implement by extending model builders with
    # TensorRT plugin stubs.  Currently, this method uses pattern
    # matching which is a bit hacky and subject to fail when TF
    # object detection API exporter changes.  We should add object
    # detection as submodule to avoid versioning incompatibilities.

    config = load_config(config_path)
    frozen_graph_path = os.path.join(tmp_dir, FROZEN_GRAPH_NAME)

    # get input shape
    channels = 3
    height = config.model.ssd.image_resizer.fixed_shape_resizer.height
    width = config.model.ssd.image_resizer.fixed_shape_resizer.width

    tmp_tbdir_s = os.path.join(UFF_WORK_SPACE, "tf_board_data_s")  # for storing static graph in UFF_WORK_SPACE
    if os.path.isdir(tmp_tbdir_s):
        subprocess.call(['rm', '-r', tmp_tbdir_s])
    subprocess.call(['mkdir', '-p', tmp_tbdir_s])

    tmp_tbdir_d = os.path.join(UFF_WORK_SPACE, "tf_board_data_d")  # for storing dynamic graph in UFF_WORK_SPACE
    if os.path.isdir(tmp_tbdir_d):
        subprocess.call(['rm', '-r', tmp_tbdir_d])
    subprocess.call(['mkdir', '-p', tmp_tbdir_d])

    if not os.path.exists(frozen_graph_path):  # check "frozen_inference_graph.pb" (frozen_graph_path) is existed
        tf_saved2frozen(config, checkpoint_path,
                        tmp_dir)  # export saved model to frozen graph, in path tmp_dir + FROZEN_GRAPH_NAME

    static_graph = gs.StaticGraph(frozen_graph_path)
    static_graph.write_tensorboard(tmp_tbdir_s)

    dynamic_graph_uff = gs.DynamicGraph(frozen_graph_path)

    # remove all assert nodes
    # all_assert_nodes = dynamic_graph.find_nodes_by_op("Assert")
    # dynamic_graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    # forward all identity nodes
    all_identity_nodes = dynamic_graph_uff.find_nodes_by_op("Identity")
    dynamic_graph_uff.forward_inputs(all_identity_nodes)

    # create input plugin
    input_plugin = gs.create_plugin_node(
        name=TRT_INPUT_NAME,
        op="Placeholder",
        dtype=tf.float32,
        # dtype=tf.uint8,
        shape=[1, height, width, channels])

    # create anchor box generator
    anchor_generator_config = config.model.ssd.anchor_generator.ssd_anchor_generator
    box_coder_config = config.model.ssd.box_coder.faster_rcnn_box_coder
    priorbox_plugin = gs.create_plugin_node(
        name="priorbox",
        op="GridAnchor_TRT",
        minSize=anchor_generator_config.min_scale,
        maxSize=anchor_generator_config.max_scale,
        aspectRatios=list(anchor_generator_config.aspect_ratios),
        variance=[
            1.0 / box_coder_config.y_scale,
            1.0 / box_coder_config.x_scale,
            1.0 / box_coder_config.height_scale,
            1.0 / box_coder_config.width_scale
        ],
        featureMapShapes=get_feature_map_shape(config),
        numLayers=config.model.ssd.anchor_generator.ssd_anchor_generator.num_layers)

    # create nms plugin
    nms_config = config.model.ssd.post_processing.batch_non_max_suppression
    nms_plugin = gs.create_plugin_node(
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
        inputOrder=[1, 2, 0],
        confSigmoid=1,
        isNormalized=1,
        # scoreConverter="SIGMOID",
        scoreBits=16,
        isBatchAgnostic=1,
        codeType=3)

    priorbox_concat_plugin = gs.create_node(
        "priorbox_concat", op="ConcatV2", dtype=tf.float32, axis=2)

    boxloc_concat_plugin = gs.create_plugin_node(
        "boxloc_concat",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
        axis=1,
        ignoreBatch=0
    )

    boxconf_concat_plugin = gs.create_plugin_node(
        "boxconf_concat",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
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
        "concat": boxloc_concat_plugin,
        "concat_1": boxconf_concat_plugin
    }

    dynamic_graph_uff.collapse_namespaces(namespace_plugin_map)
    # dynamic_graph_uff.append(stack_boxconf)
    # dynamic_graph_uff.append(stack_boxloc)

    # fix name
    for i, name in enumerate(
            dynamic_graph_uff.find_nodes_by_op('NMS_TRT')[0].input):
        if TRT_INPUT_NAME in name:
            dynamic_graph_uff.find_nodes_by_op('NMS_TRT')[0].input.pop(i)

    for n in range(len(dynamic_graph_uff.find_nodes_by_op('GridAnchor_TRT'))):
        for nk in range(len(dynamic_graph_uff.find_nodes_by_op('GridAnchor_TRT')[n].input)):
            dynamic_graph_uff.find_nodes_by_op('GridAnchor_TRT')[n].input.pop(nk)

    namespace_remove = {
        "Cast",
        "ToFloat",
        "Preprocessor"
    }
    dynamic_graph_uff.remove(
        dynamic_graph_uff.find_nodes_by_path(namespace_remove), remove_exclusive_dependencies=False)

    dynamic_graph_uff.remove(
        dynamic_graph_uff.graph_outputs, remove_exclusive_dependencies=False)

    dynamic_graph_uff.write_tensorboard(tmp_tbdir_d)
    dynamic_graph_uff.write(os.path.join(MODEL_REP_DIR, MODEL_NAME + "_4_uff_conv.pb"))
    path_uff_graph = os.path.join(MODEL_REP_DIR, MODEL_NAME + ".uff")

    g_outputs = [TRT_OUTPUT_NAME]
    print(g_outputs)

    test_conv = False
    if test_conv:
        # ------ the following should be commented if not used for testing convolution and g_outputs for graph output
        nd_remv = [nd_key for nd_key, nd in dynamic_graph_uff.node_map.items()
                   if not (nd_key == "FeatureExtractor/MobilenetV2/Conv/Conv2D"
                           or nd_key == "FeatureExtractor/MobilenetV2/Conv/weights"
                           or nd_key == TRT_INPUT_NAME)]
        dynamic_graph_uff.remove(dynamic_graph_uff.find_nodes_by_path(nd_remv), remove_exclusive_dependencies=False)
        g_conv_outputs = ["FeatureExtractor/MobilenetV2/Conv/Conv2D"]
        g_outputs = g_conv_outputs
        # ------

    uff_buffer = uff.from_tensorflow(dynamic_graph_uff.as_graph_def(),
                                     g_outputs, output_filename=path_uff_graph)

    return uff_buffer


def ssd_uff_to_engine(uff_buffer,
                      fp16_mode=True,
                      max_batch_size=1,
                      max_workspace_size=1 << 30,  # 26
                      min_find_iterations=2,
                      average_find_iterations=1,
                      strict_type_constraints=False,
                      log_level=trt.Logger.INFO):
    print('-------- start TRT engine generation from uff model --------')

    import uff
    # create the tensorrt engine
    with trt.Logger(log_level) as logger, \
            trt.Builder(logger) as builder, \
            builder.create_network() as network, \
            builder.create_builder_config() as config, \
            trt.UffParser() as parser:

        # init built in plugins
        trt.init_libnvinfer_plugins(logger, '')

        # load jetbot plugins
        # load_plugins()

        meta_graph = uff.model.uff_pb2.MetaGraph()
        meta_graph.ParseFromString(uff_buffer)

        input_node = None
        for n in meta_graph.ListFields()[3][1][0].nodes:
            if 'Input' in n.operation:
                input_node = n

        channels = input_node.fields['shape'].i_list.val[3]
        height = input_node.fields['shape'].i_list.val[1]
        width = input_node.fields['shape'].i_list.val[2]

        # parse uff to create network
        parser.register_input(TRT_INPUT_NAME, (channels, height, width))
        parser.register_output(TRT_OUTPUT_NAME)
        parser.parse_buffer(uff_buffer, network)

        # use Class config for TRT ver. >8.0 instead Class builder in <8.0
        config.set_flag(trt.BuilderFlag.FP16)
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
        engine = builder.build_engine(network, config)
        # engine = builder.build_serialized_network(network, config)
        # engine = builder.build_cuda_engine(network, config)

    return engine


if __name__ == '__main__':
    download_model(MODEL_NAME, TF_MODEL_DIR)
    ssd_tmp_model = TMP_MODEL
    ssd_checkpoint_path = os.path.join(TF_MODEL_DIR, MODEL_NAME, "model.ckpt")
    ssd_config_path = os.path.join(TF_MODEL_DIR, MODEL_NAME, "pipeline.config")
    ssd_output_engine = os.path.join(MODEL_REP_DIR, MODEL_TRT + ".engine")

    uff_buffer = ssd_pipeline_to_uff(ssd_checkpoint_path, ssd_config_path, tmp_dir=ssd_tmp_model)

    engine = ssd_uff_to_engine(uff_buffer,
                               fp16_mode=True,
                               max_batch_size=1,
                               max_workspace_size=1 << 30,  # 26
                               min_find_iterations=2,
                               average_find_iterations=1,
                               strict_type_constraints=False,
                               log_level=trt.Logger.INFO)

    buf = engine.serialize()
    with open(ssd_output_engine, 'wb') as f:
        f.write(buf)
