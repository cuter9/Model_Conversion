"""ssd.py

This module implements the TrtSSD class.
"""

import ctypes
import os
import subprocess

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import atexit

# pycuda.autoinit is needed for initializing CUDA driver
import pycuda.autoinit

def _preprocess_trt(img, shape=(300, 300), m_type='uff'):
    """Preprocess an image before TRT SSD inferencing."""
    img = cv2.resize(img, shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Shape of camera input data : ", img.shape,
          "Max. and Min. value", np.amax(img), ", ", np.amin(img), "; \nCamera input data : ", img)
    # if m_type == 'uff':
    img = img.transpose((2, 0, 1)).astype(np.float32)
    # else:
    # img = img.astype(np.float32)
    img *= (2.0 / 255.0)
    img -= 1.0
    return img


def _postprocess_trt(img, out, conf_th, output_layout=7):
    """Postprocess TRT SSD output."""
    output = out[0]
    img_h, img_w, _ = img.shape
    boxes, confs, clss = [], [], []
    # # box CodeTypeSSD : TF_CENTER, https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/api/c_api/_nv_infer_plugin_utils_8h_source.html
    # output :  [ ,class ID, confidence score, x_min_object_box, y_min_object_box, x_max_object_box, y_max_object_box,
    #             , ....
    #             ,class ID, confidence score, x_min_object_box, y_min_object_box, x_max_object_box, y_max_object_box,
    #             , ....]
    for prefix in range(0, len(output), output_layout):
        if not output[1] < 0:
            print("---- one detection ---- \n ", output[prefix: prefix + output_layout])
        # index = int(output[prefix+0])
        conf = float(output[prefix + 2])
        if conf < conf_th:
            continue
        x1 = int(output[prefix + 3] * img_w)
        y1 = int(output[prefix + 4] * img_h)
        x2 = int(output[prefix + 5] * img_w)
        y2 = int(output[prefix + 6] * img_h)
        cls = int(output[prefix + 1])
        boxes.append((x1, y1, x2, y2))
        confs.append(conf)
        clss.append(cls)
    return boxes, confs, clss


def _postprocess_fpn_trt(img, output, conf_th):
    """Postprocess TRT SSD output."""
    img_h, img_w, _ = img.shape
    boxes, confs, clss = output[1].tolist(), output[2].tolist(), output[3].tolist()
    boxes_out, confs_out, clss_out = [], [], []
    # # box CodeTypeSSD : TF_CENTER, https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/api/c_api/_nv_infer_plugin_utils_8h_source.html
    # output :  [ ,class ID, confidence score, x_min_object_box, y_min_object_box, x_max_object_box, y_max_object_box,
    #             , ....
    #             ,class ID, confidence score, x_min_object_box, y_min_object_box, x_max_object_box, y_max_object_box,
    #             , ....]
    for d in range(0, len(output[2])):
        # index = int(output[prefix+0])
        conf = float(confs[d])
        if conf < conf_th:
            continue
        y1 = int(boxes[4*d] * img_h)
        x1 = int(boxes[4*d+1] * img_w)
        y2 = int(boxes[4*d+2] * img_h)
        x2 = int(boxes[4*d+3] * img_w)
        cls = int(clss[d])

        bs = ','.join(list(map(str, boxes[4*d:4*d+4])))
        print("---- one detection ---- \n class: %d, confidence: %f, box: [%s] \n "
              % (clss[d], confs[d], bs))
        boxes_out.append((x1, y1, x2, y2))
        confs_out.append(conf)
        clss_out.append(cls)
    return boxes_out, confs_out, clss_out


class TrtSSD(object):
    """TrtSSD class encapsulates things needed to run TRT SSD."""

    def _load_plugins(self):
        if trt.__version__[0] < '7':
            ctypes.CDLL("ssd/libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        # TRTbin = 'ssd/TRT_%s.bin' % self.model
        trt_engine = self.model
        with open(trt_engine, 'rb') as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings = \
            [], [], [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
            print("--- engine binding name ---", binding, "\n",
                  "--- binding shape ---", self.engine.get_binding_shape(binding), "\n"
                                                                                   "--- Host memory locked : ",
                  len(host_mem), "bytes\n",
                  "--- Cuda buffer allocated at: ", cuda_mem, "\n")

        return host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings

    def get_outputs_binding_name(self):
        outputs_binding_name = []
        for binding in self.engine:
            if not self.engine.binding_is_input(binding):
                idx_binding = self.engine.get_binding_index(binding)
                outputs_binding_name.append(self.engine.get_binding_name(idx_binding))

        return outputs_binding_name

    def __init__(self, model, input_shape, cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.input_shape = input_shape
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.runtime = trt.Runtime(self.trt_logger)
        self.engine = self._load_engine()
        self.outputs_binding_name = self.get_outputs_binding_name()
        try:
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.host_inputs, self.host_outputs, self.cuda_inputs, self.cuda_outputs, self.bindings = self._allocate_buffers()
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

        atexit.register(self.destroy) # the destroy function is depreciated


    def __del__(self):
        """Free CUDA memories and context."""
        del self.cuda_outputs
        del self.cuda_inputs
        del self.stream

    def detect(self, img, model_type, test_op, fpn=False, conf_th=0.3):
        """Detect objects in the input image."""
        img_resized = _preprocess_trt(img, self.input_shape, model_type)
        np.copyto(self.host_inputs[0], img_resized.ravel())

        if self.cuda_ctx:
            self.cuda_ctx.push()
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        if model_type == "uff":
            self.context.execute_async(
                batch_size=1,
                bindings=self.bindings,
                stream_handle=self.stream.handle)
        elif model_type == "onnx":
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle)
        for ho, co in zip(self.host_outputs, self.cuda_outputs):
            cuda.memcpy_dtoh_async(ho, co, self.stream)
        # cuda.memcpy_dtoh_async(
        #    self.host_outputs[1], self.cuda_outputs[1], self.stream)
        # cuda.memcpy_dtoh_async(
        #    self.host_outputs[0], self.cuda_outputs[0], self.stream)
        # cuda.memcpy_dtoh_async(
        #    self.host_outputs[2], self.cuda_outputs[2], self.stream)
        # cuda.memcpy_dtoh_async(
        #    self.host_outputs[3], self.cuda_outputs[3], self.stream)
        # cuda.memcpy_dtoh_async(
        #    self.host_outputs[4], self.cuda_outputs[4], self.stream)
        self.stream.synchronize()
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        # output = self.host_outputs[0]     # original model output structure
        # output = self.host_outputs[3]
        # work_dir = os.getcwd()
        # save_dir = os.path.join(work_dir, "/home/cuterbot/Model_Conversion/test_model/saved_data", model_type)
        # subprocess.run(["rm", "-r", save_dir])
        # subprocess.run(["mkdir", save_dir])
        '''
        for ho, name_ho in zip(self.host_outputs, self.outputs_binding_name):
            
            if test_op:
                output = self.host_outputs[0]
            else:
                print(name_ho)
                if name_ho == "nms:0" or name_ho == "nms":
                    output = ho
        '''
        output = self.host_outputs
            # np.save(os.path.join(save_dir, name_ho.replace("/", "_")), ho)
            # np.save(os.path.join(save_dir, "priorbox"), self.host_outputs[0])
            # np.save(os.path.join(save_dir, "boxconf"), self.host_outputs[1])
            # np.save(os.path.join(save_dir, "boxloc"), self.host_outputs[2])
            # np.save(os.path.join(save_dir, "output_0"), self.host_outputs[3])
            # np.save(os.path.join(save_dir, "output_1"), self.host_outputs[4])
            # np.save(os.path.join(save_dir, "onnx", "input"), self.host_inputs[0])
        print("----- output of detection results ----- \n", output)
        if test_op:
            return
        else:
            if fpn:
                return _postprocess_fpn_trt(img, output, conf_th)
            else:
                return _postprocess_trt(img, output, conf_th)

    def destroy(self):
        # self.runtime.destroy() # this function is depreciated
        # self.trt_logger.destroy()
        # self.engine.destroy()
        # self.context.destroy()
        del self.runtime
        del self.engine
        del self.context
