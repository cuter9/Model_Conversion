import ctypes
import numpy as np
import os
import subprocess


def download_model(model_name, model_dir):
    # import six.moves.urllib as urllib
    import wget
    import tarfile
    """
    Download Model form TF's Model Zoo
    """
    model_file = model_name + ".tar.gz"
    # Ref. 1 : the path to download the model from Tensorflow web can be found in TF1 Model Zoo as following :
    #           https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
    # Ref. 2 :  TensorFlow 2 Object Detection API tutorial
    #           https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#downloading-the-tensorflow-model-garden
#    download_base = 'http://download.tensorflow.org/models/object_detection/'      # base of tf v1 models
    download_base = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
    # ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz
    # ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
    model_file_path = os.path.join(model_dir, model_file)
    model_dir_path = os.path.join(model_dir, model_name)

    if not os.path.isfile(model_file_path):
        print('{} not found. Downloading it now.'.format(model_file))
        # opener = urllib.request.URLopener()
        # opener.retrieve(download_base + model_file, model_file_path)
        url = download_base + model_file
        wget.download(url, model_file_path)
        # print(url)
    else:
        print('{} found. Proceed.'.format(model_file))
    if not os.path.isdir(model_dir_path):
        print('{} not found. Extract it now.'.format(model_dir_path))
        tar_file = tarfile.open(model_file_path)
        tar_file.extractall(path=model_dir)
        tar_file.close()
    else:
        print('{} found. Proceed.'.format(model_dir_path))


def load_plugins():
    # load the liberary of customed operation plugin for SSD model conversion in TensorRT
    # it is not need for TRT ver>8.0 which will load automatically
    library_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'libssd_tensorrt.so')
    ctypes.CDLL(library_path)


def get_feature_map_shape(config):
    width = config.model.ssd.image_resizer.fixed_shape_resizer.width
    fms = []
    curr = int(np.ceil(width / 16.0))
    for i in range(6):
        fms.append(curr)
        curr = int(np.ceil(curr / 2.0))
    print("---- feature map size array : ", fms)
    return fms


def load_config(config_path):
    from object_detection.protos.pipeline_pb2 import TrainEvalPipelineConfig
    # from object_detection.protos.pipeline_pb2 import TrainAndEvalPipelineConfig
    from google.protobuf.text_format import Merge
    config = TrainEvalPipelineConfig()

    with open(config_path, 'r') as f:
        config_str = f.read()

    lines = config_str.split('\n')
    lines = [line for line in lines if 'batch_norm_trainable' not in line]
    config_str = '\n'.join(lines)

    Merge(config_str, config)

    return config


def tf_saved2frozen(config, checkpoint_path, dir_frozen_graph):
    # model should be exported to frozen graph, then use it for uff conversion using ssd_pipeline_to_uff()
    # config : config in model dir, load in advance with _load_config()
    # checkpoint_path : checkpoint file of th model
    # dir_frozen_graph : directory to store the model exported from the tensorflow model
#    from object_detection import exporter      # the contrib was remove since tf v2
    from object_detection import exporter_lib_v2 as exporter    # use exporter_lib_v2
    import tensorflow as tf

    # tf_config = tf.ConfigProto()
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.visible_device_list = '0'
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9

    if os.path.isdir(dir_frozen_graph):
        subprocess.call(['rm', '-r', dir_frozen_graph])
    subprocess.call(['mkdir', '-p', dir_frozen_graph])

    # export checkpoint and config of the downloaded trained model (from tensorflow) to a frozen graph
    # and saved in directory dir_frozen_graph.
    # ref. 1 : TensorFlow 2 Object Detection API tutorial
    # https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
    # https://blog.csdn.net/yukinorong/article/details/103242940?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-103242940-blog-77033659.235%5Ev32%5Epc_relevant_default_base3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-103242940-blog-77033659.235%5Ev32%5Epc_relevant_default_base3&utm_relevant_index=1
    # with tf.Session(config=tf_config) as tf_sess:
    # with tf.compat.v1.Session(config=tf_config) as tf_sess:
    #    with tf.Graph().as_default() as tf_graph:
    exporter.export_inference_graph(
        'image_tensor',  # the input type of the model to be exported
        config,  # pipelined config file of the downloaded train model to be exported
        checkpoint_path,  # the checkpoint of the download trained model to be exported
        dir_frozen_graph)  # the directory to store the exported model from the downloaded trained model
        # use_side_inputs=True,
        # side_input_shapes=[1/None/None/3])
    return

