import tensorrt as trt
import os
import onnx
import onnx_graphsurgeon as gs_onnx
import subprocess

import wget

MODEL_NAME = "yolov7"
# MODEL_NAME = "yolov7-tiny"
MODEL_TRT = "yolov7"
# MODEL_TRT = "yolov7-tiny"
DATA_REPO_DIR = os.path.join(os.environ['HOME'], "Data_Repo/Model_Conversion", MODEL_NAME)
ONNX_WORK_SPACE = os.path.join(DATA_REPO_DIR, "ONNX_Model")
MODEL_REPO_DIR = os.path.join(ONNX_WORK_SPACE, "Repo")

TRT_INPUT_NAME = 'images'
TRT_OUTPUT_NAME = 'nms'

# -- convert yolov7 to onnx and then run this script to convert onnx to trt model
# ref https://hackmd.io/_oaJhYNqTvyL_h01X1Fdmw?both
# https://hackmd.io/@YungHuiHsu/BJL54lDy3
# ref https://github.com/Monday-Leo/YOLOv7_Tensorrt

subprocess.run("./yolov7_2_onnx.sh")

# mkdir yolo
# cd yolo
# git clone https://github.com/WongKinYiu/yolov7
# cd yolov7
#
# -- Download tiny weights
# wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
# -- Download regular weights
# wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

# -- install onnx-simplifier not listed in general yolov7 requirements.txt
# pip3 install onnxsim
#
# -- Pytorch Yolov7 -> ONNX with grid, EfficientNMS plugin and dynamic batch size
# python export.py --weights ./yolov7.pt --grid --end2end --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.01 --img-size 640 640
# -- Pytorch Yolov7 -> ONNX with grid, EfficientNMS plugin and static batch size
# python export.py --weights ./yolov7.pt --grid --end2end  --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.01 --img-size 640 640
def redef_onnx_node_4_trt_plugin(path_onnx_model, path_onnx_model_4_trt):
    # ref : https://developer.nvidia.com/blog/estimating-depth-beyond-2d-using-custom-layers-on-tensorrt-and-onnx-models/

    print("---- surge the onnx model by replacing the some nodes with TRT plugin and ops. ----")
    onnx_model_proto = onnx.load(path_onnx_model, format='protobuf')
    onnx_graph = gs_onnx.import_onnx(onnx_model_proto)

    # node_NMS_TRT = [nd for nd in onnx_graph.nodes if nd.name == "/end2end/EfficientNMS_TRT"][0]
    node_NMS_TRT = [nd for nd in onnx_graph.nodes if nd.name == "batched_nms"][0]
    # The  score output from EfficientNMS_TRT needs sigmoid function,
    # but this is a bug in the onnx model directly converted from the github source!
    # node_NMS_TRT.attrs["score_activation"] = 1
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
            # unsupported_node = parser.supports_model(gm.SerializeToString())
            # print(" --- the following node in onnx model is not supported by trt onnx parser: \n", unsupport_node)

            parser.parse_from_file(path_onnx_model)

            print("\n\n----- start trt engine construction -----")
            engine = builder.build_engine(network, config)
            # engine = builder.build_cuda_engine(network, config)

    return engine


if __name__ == '__main__':
    path_onnx_model = os.path.join(MODEL_REPO_DIR, MODEL_TRT + ".onnx")
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

    '''
    TEST_DIR = os.path.join(os.environ["HOME"], "Downloads/yolo/yolov7/inference/images")
    test_img = os.path.join(TEST_DIR, "test.jpg")
    if not os.path.exists(test_img):
        wget.download("http://images.cocodataset.org/val2017/000000088462.jpg", out=test_img)

    subprocess.run("python3 ./test_yolo.py --trt_engine %s --source %s" % (output_engine, test_img))
    '''
