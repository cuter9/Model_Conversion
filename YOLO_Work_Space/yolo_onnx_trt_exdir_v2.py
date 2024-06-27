import tensorrt as trt
import os
import onnx
import onnx_graphsurgeon as gs_onnx

MODEL_NAME = "yolo_v7"
MODEL_TRT = "yolo_v7"
DATA_REPO_DIR = os.path.join(os.environ['HOME'], "Data_Repo/Model_Conversion", MODEL_NAME)
ONNX_WORK_SPACE = os.path.join(DATA_REPO_DIR, "ONNX_Model")
MODEL_REPO_DIR = os.path.join(ONNX_WORK_SPACE, "Repo")

TRT_INPUT_NAME = 'images'
TRT_OUTPUT_NAME = 'nms'

def redef_onnx_node_4_trt_plugin(path_onnx_model, path_onnx_model_4_trt):
    # ref : https://developer.nvidia.com/blog/estimating-depth-beyond-2d-using-custom-layers-on-tensorrt-and-onnx-models/

    print("---- surge the onnx model by replacing the some nodes with TRT plugin and ops. ----")
    onnx_model_proto = onnx.load(path_onnx_model, format='protobuf')
    onnx_graph = gs_onnx.import_onnx(onnx_model_proto)

    node_NMS_TRT = [nd for nd in onnx_graph.nodes if nd.name == "/end2end/EfficientNMS_TRT"][0]
    onnx_graph.outputs = node_NMS_TRT.outputs
    onnx_graph.cleanup().toposort()

    onnx_model_proto_new = gs_onnx.export_onnx(onnx_graph)
    onnx.save_model(onnx_model_proto_new, path_onnx_model_4_trt)


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
    path_onnx_model = os.path.join(MODEL_REPO_DIR, "yolo_v7.onnx")
    output_engine = os.path.join(MODEL_REPO_DIR, MODEL_TRT + ".engine")
    path_onnx_model_4_trt = os.path.join(MODEL_REPO_DIR, MODEL_TRT + "_new.onnx")
    redef_onnx_node_4_trt_plugin(path_onnx_model, path_onnx_model_4_trt)

    print("----- Start build engine of the surged onnx model -----")
    engine = ssd_onnx_to_engine(path_onnx_model=path_onnx_model_4_trt,
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
