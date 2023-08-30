import torch
import tensorrt as trt
import atexit
import numpy as np
import cv2

# SSD model inference with cuda through pytorch.cuda

INPUT_HW = (300, 300)
LABEL_IDX = 1
CONFIDENCE_IDX = 2
X0_IDX = 3
Y0_IDX = 4
X1_IDX = 5
Y1_IDX = 6


def preprocess_trt(camera_value):
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, INPUT_HW)
    x = x.transpose((2, 0, 1)).astype(np.float32)
    x *= (2.0 / 255.0)
    x -= 1.0
    return x[None, ...]


def postprocess_trt(img, outputs, conf_th=0.3):
    """Postprocess TRT SSD output."""
    img_h, img_w, _ = img.shape

    bboxes = outputs[0]

    # iterate through each image index
    all_detections = []
    for i in range(bboxes.shape[0]):

        detections = []
        # iterate through each bounding box
        for j in range(bboxes.shape[2]):

            bbox = bboxes[i][0][j]
            label = bbox[LABEL_IDX]
            conf = bbox[CONFIDENCE_IDX]

            if not label < 0 and conf >= conf_th:
                print("---- one detection ---- \n ", bbox)

            # last detection if < 0
            else:
                break

            detections.append(dict(
                label=int(label),
                confidence=float(conf),
                bbox=[
                    int(float(bbox[X0_IDX]) * img_w),
                    int(float(bbox[Y0_IDX]) * img_h),
                    int(float(bbox[X1_IDX]) * img_w),
                    int(float(bbox[Y1_IDX]) * img_h)
                ]
            ))
        all_detections.append(detections)

    clss = [det.get("label") for det in all_detections[0]]
    boxes = [det.get("bbox") for det in all_detections[0]]
    confs = [det.get("confidence") for det in all_detections[0]]
    # return all_detections
    return boxes, confs, clss


def torch_dtype_to_trt(dtype):
    if dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError('%s is not supported by tensorrt' % dtype)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def torch_device_to_trt(device):
    if device.type == torch.device('cuda').type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device('cpu').type:
        return trt.TensorLocation.HOST
    else:
        return TypeError('%s is not supported by tensorrt' % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


class TRTModel(object):

    def __init__(self, engine_path, input_names=None, output_names=None, final_shapes=None):

        # load engine
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # self.stream = torch.cuda.Stream()         # no need to set if the default stream is used

        if input_names is None:
            self.input_names = self._trt_input_names()
        else:
            self.input_names = input_names

        if output_names is None:
            self.output_names = self._trt_output_names()
        else:
            self.output_names = output_names

        self.final_shapes = final_shapes

        # destroy at exit
        atexit.register(self.destroy)

    def _input_binding_indices(self):
        return [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]

    def _output_binding_indices(self):
        return [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]

    def _trt_input_names(self):
        return [self.engine.get_binding_name(i) for i in self._input_binding_indices()]

    def _trt_output_names(self):
        return [self.engine.get_binding_name(i) for i in self._output_binding_indices()]

    def create_output_buffers(self, batch_size):
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            if self.final_shapes is not None:
                shape = (batch_size,) + self.final_shapes[i]
            else:
                shape = (batch_size,) + tuple(self.engine.get_binding_shape(idx))
                # shape = tuple(self.engine.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            print("The shape of the output --- ", output_name, "  :  ", output.size())
        return outputs

    def execute(self, *inputs):
        # print("The number of inputs:", len(inputs), ";      The input image size", inputs[0].shape, 
        #      "\nThe max and min of input : " , np.amax(inputs[0]), ";    " , np.amin(inputs[0]))

        resized_inputs = [preprocess_trt(inp) for inp in inputs]

        batch_size = resized_inputs[0].shape[0]
        # batch_size = 1

        bindings = [None] * (len(self.input_names) + len(self.output_names))

        # map input bindings
        inputs_torch = [None] * len(self.input_names)
        for i, name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(name)

            # convert to appropriate format
            inputs_torch[i] = torch.from_numpy(resized_inputs[i])
            torch_dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))

            # To meet the model input format,
            # the input image data has been transform form NHWC to NCHW in input preprocess,
            # Thus torch.contiguous_format should be used in input memory format
            # to keep the gpu memory format as allocated as the transformed input image data format (strides),
            # otherwise the model will be operated with NHWC and will result in incorrect output.
            inputs_torch[i] = inputs_torch[i].to(torch_device_from_trt(self.engine.get_location(idx)), torch_dtype,
                                                 memory_format=torch.contiguous_format)
            inputs_torch[i] = inputs_torch[i].type(torch_dtype)
            # print("shape of input to GPU: ", inputs_torch[i].size(),
            #      "\n -----The input to GPU ---- \n", inputs_torch[i])
            bindings[idx] = int(inputs_torch[i].data_ptr())
            print("The location of the input --- ", name, "  :  ", bindings[idx], " with binding index : ", idx,
                  "  in device : ", torch_device_from_trt(self.engine.get_location(idx)))

        output_buffers = self.create_output_buffers(batch_size)

        # map output bindings
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            bindings[idx] = int(output_buffers[i].data_ptr())
            print("The location of the output --- ", output_name, " :  ", bindings[idx], " with binding index : ", idx,
                  "  in device : ", torch_device_from_trt(self.engine.get_location(idx)))

        # self.context.execute_async(batch_size, bindings, self.stream.cuda_stream)
        self.context.execute(batch_size,
                             bindings)  # streanm is no need if the default stream is used and synchronize automatically
        # self.context.execute_async_v2(bindings, self.stream.cuda_stream)
        # self.context.execute_v2(bindings)

        if self.engine.has_implicit_batch_dimension:
            print("engine is built from uff model")
            outputs = [buffer.cpu().numpy() for buffer in output_buffers]
        else:
            print("engine is built from onnx model")
            outputs = [np.squeeze(buffer.cpu().numpy(), axis=0) for buffer in output_buffers]
            # outputs = [buffer.cpu().numpy() for buffer in output_buffers]

        # self.stream.synchronize()

        # print("shape of output buffer of GPU: ", output_buffers[0].shape, ";  ", output_buffers[1].shape,
        #      "\n -----The output buffer of GPU ---- \n", output_buffers[1],  "\n", output_buffers[0])

        # print("shape of output copied from GPU: ", outputs[0].shape, ";  ", outputs[1].shape, "\n ----Output from GPU---- \n", outputs[0])

        return postprocess_trt(inputs[0], outputs)

    def __call__(self, *inputs):
        # img_resized = preprocess_trt(inputs[0])
        # with torch.cuda.stream(self.stream):

        return self.execute(*inputs)

    def destroy(self):
        self.runtime.destroy()
        self.logger.destroy()
        self.engine.destroy()
        self.context.destroy()
