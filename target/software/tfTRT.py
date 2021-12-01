#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script can be used to run tensorflow trt engines with pycuda and measure exec time and verify the output

Example output on Jetson Nano:

    ----------------- Running TensorRT Tensorflow Pycuda ----------------
    Loading model from: tf.engine
    Input file: n01491361_tiger_shark.jpg
    Running 100 measurements...
    Mean:0.046010465332956024(s) median:0.0402679443359375(s) std:0.010749198502712038 n:99
    Top 5: [['tiger_shark', 0.8076849], ['hammerhead', 0.04587668], ['great_white_shark', 0.03532296], ['killer_whale', 0.0027615032], ['sturgeon', 0.0021188608]]

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
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)

        # run the trt engine
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)

        # get outputs
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        result[self.net_type] = self.host_outputs

# settings
engie_path = 'tf.engine'
input_path = 'n01491361_tiger_shark.jpg'
n_measurements = 100

# load model and prep dicts
print('\n----------------- Running TensorRT Tensorflow Pycuda ----------------')
print('Loading model from:', engie_path)
print('Input file:', input_path)

# output of the model is seg_process_softmax
trt_engine = TRTInference(engie_path, trt_engine_datatype=trt.DataType.FLOAT, net_type='seg_process_softmax')
result = {}
exec_times = []

# measure trt performance
print('Running', n_measurements, 'measurements...')
for i in range(0, n_measurements):
    start_time = time.time()

    # prepare image
    image = cv2.imread(input_path)
    image = cv2.resize(image, dsize=(224, 224))
    image = np.moveaxis(image, 2, 0)
    image = np.array(image, dtype='float32').reshape(1, 3, 224, 224)

    # run inference and check the results
    trt_engine.infer(image, result)
    exec_times.append(time.time() - start_time)

# remove warm up measurement
exec_times.pop(0)

# reshape for decode function (imagenet has 1000 classes)
output = result['seg_process_softmax'][0]
print('Mean:' + str(np.mean(exec_times)) + '(s) median:' +
      str(np.median(exec_times)) + '(s) std:' + str(np.std(exec_times)) + ' n:' + str(i))
print('Top 5:', decode_imagenet(output.reshape(1, 1000), top=5))

# remove any context from the top of the context stack
trt_engine.cfx.pop()
exit(0)
