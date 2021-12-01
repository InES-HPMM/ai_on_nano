#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script can be used to run pytorch trt engines with pycuda and measure exec time and verify the output

Example output on Jetson Nano:

    ----------------- Running TensorRT Pytorch Pycuda ----------------
    Loading model from: torch.engine
    Input files: n01491361_tiger_shark.jpg
    Running 100 measurements...
    Torch TRT mean:0.042910363939073354(s) median:0.040138959884643555(s) std:0.010458157941541342 n:99
    Torch TRT top  5: [['tiger_shark', 8.8338585], ['great_white_shark', 5.916095], ['hammerhead', 3.7348726], ['dugong', 3.01505], ['airship', 2.8707445]]

:Author: **Raphael Zingg zing@zhaw.ch**
:Copyright: **2021 Institute of Embedded Systems (InES) All rights reserved**
"""

import json
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import time
import cv2 as cv2
from torchvision import transforms


def decode_imagenet(preds, idx_file='imagenet_idx.json', top=5):
    # see: https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py

    with open(idx_file) as json_file:
        CLASS_INDEX = json.load(json_file)
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [[CLASS_INDEX[str(i)][1], pred[i]]for i in top_indices]
    return result


class TRTInference:
    def __init__(self, trt_engine_path, trt_engine_datatype, net_type='EfficientNet'):
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)

        # deserialize engine
        with open(trt_engine_path, 'rb') as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        context = engine.create_execution_context()

        # prepare buffer
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        # prepare bindings
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # store
        self.stream = stream
        self.context = context
        self.engine = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.net_type = net_type

    def infer(self, image, result):

        # copy inputs
        np.copyto(self.host_inputs[0], image.ravel())

        # uncomment to print inputs
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)

        # run the trt engine
        self.context.execute_async(
            bindings=self.bindings, stream_handle=self.stream.handle)

        # get outputs
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        result[self.net_type] = self.host_outputs


# settings
engine_path = 'torch.engine'
n_measurements = 100
input_path = 'n01491361_tiger_shark.jpg'

# load model and prep dicts
print('\n----------------- Running TensorRT Pytorch Pycuda ----------------')
print('Loading model from:', engine_path)
print('Input files:', input_path)

trt_engine = TRTInference(
engine_path, trt_engine_datatype=trt.DataType.FLOAT, net_type='seg_process_softmax')
result = {}
out = []
exec_times = []

# measure trt performance
print('Running', n_measurements, 'measurements...')
for i in range(0, n_measurements):
    start_time = time.time()

    # prepare image
    image = cv2.imread(input_path)
    image = cv2.resize(image, dsize=(224, 224))
    preprocess_torch = transforms.Compose([
        transforms.ToTensor(),
        # imagenet mean
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess_torch(image)

    # run inference and check the results
    trt_engine.infer(input_tensor, result)
    exec_times.append(time.time() - start_time)

# remove warm up measurement
exec_times.pop(0)
output = result['seg_process_softmax'][0]

print('Torch TRT mean:' + str(np.mean(exec_times)) + '(s) median:' + str(np.median(exec_times)) + '(s) std:' + str(np.std(exec_times)) + ' n:' + str(i))
print('Torch TRT top  5:', decode_imagenet(output.reshape(1, 1000), top=5))

# remove any context from the top of the context stack
trt_engine.cfx.pop()
exit(0)
