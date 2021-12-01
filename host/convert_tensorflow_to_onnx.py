#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 07:12:27 2021

Store an tensorflow imagenet classifier (EfficientNetB0) as .onnx model and compare outputs to original model

Works with following dockerfile (versions MUST match exaclty):

    FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
    ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
    RUN apt-get update
    RUN apt install -y git gcc libglib2.0-0 ffmpeg libsm6 libxext6
    RUN python3 -m pip install --upgrade pip
    RUN pip install imageio==2.9.0
    RUN pip install numpy==1.19.2
    RUN pip install opencv_python==4.5.2.52
    RUN pip install torchvision==0.10.0
    RUN pip install timm==0.4.12
    RUN pip install onnxruntime
    RUN pip install tensorflow==2.3.1
    RUN pip install tf2onnx==1.8.3

Use it inside the docker container with:
python convert_tensorflow_to_onnx.py

Generated files/models are at: '../target/'

Example output on cpu:
    Converted outputs match original ones.
    Top 5: [['tiger_shark', 0.8076844], ['hammerhead', 0.045876916], ['great_white_shark', 0.035323028], ['killer_whale', 0.0027614888], ['sturgeon', 0.0021188671]]
    Tensorflow inference time mean: 0.17363241403409752 median: 0.17278146743774414 std: 0.010722805990543515 n: 100
    Onnx inference time mean: 0.05326660788885438 median: 0.05391216278076172 std: 0.004484912728677083 n: 100


:Author: **Raphael Zingg zing@zhaw.ch**
:Copyright: **2021 Institute of Embedded Systems (InES) All rights reserved**
"""

import time
import json
import os
import numpy as np
import tensorflow as tf
import tf2onnx
import cv2 as cv
import onnxruntime
from tensorflow.python.ops.gen_image_ops import resize_nearest_neighbor
import tensorflow.keras.backend as backend

def decode_imagenet(preds, idx_file='./data/imagenet_idx.json', top=5):
    # see: https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py

    with open(idx_file) as json_file:
        CLASS_INDEX = json.load(json_file)
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [[CLASS_INDEX[str(i)][1], pred[i]]for i in top_indices]
    return result

# check versions
assert tf.__version__ == '2.4.1'
assert tf2onnx.__version__ == '1.8.4'

# load pretrained model
model = tf.keras.applications.EfficientNetB0(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling='avg',
    classifier_activation="softmax",
)

# fix trt conversion
# - https://github.com/onnx/tensorflow-onnx/issues/293
# - https://github.com/NVIDIA/TensorRT/issues/1061
def monkey_2_patched_UpSampling2D_call(self, inputs):
    _, height, width, _ = inputs.get_shape().as_list()
    if height != None:
        # fixed call in order to avoid trt conversion error
        new_height, new_width = height * 2, width * 2
        inputs = resize_nearest_neighbor(inputs, size=[new_height, new_width], align_corners=False, half_pixel_centers=False)
        inputs.set_shape([None, new_height, new_width, None])
        return inputs
    else:
        # original call
        return backend.resize_images(
        inputs, self.size[0], self.size[1], self.data_format,
        interpolation=self.interpolation)

tf.keras.layers.UpSampling2D.call = monkey_2_patched_UpSampling2D_call

# run tensorflow for 100 times on host to compare to embedded (with image loading)
et_tf = []
for i in range(0, 101):
    start = time.time()
    image = cv.imread('../data/n01491361_tiger_shark.jpg')
    image = cv.resize(image, dsize=(224, 224))
    image = np.array(image, dtype='float32').reshape(1, 224, 224, 3)
    tf_out = model(image).numpy()
    et_tf.append(time.time() - start)
classes = decode_imagenet(model(image).numpy(), idx_file='../data/imagenet_idx.json')

# save as saved model in order to use tfonnx
try:
    os.mkdir('/tmp/tf_model')
    os.mkdir('../target/models')
except OSError as error:
    print(error)
model.save('/tmp/tf_model', include_optimizer=False)

# connvert to onnx
print('Convert to onnx: ../target/models/tf.onnx')
# os.system('python -m tf2onnx.convert  --target tensorrt --inputs input_1:0[1,224,224,3] --saved-model /tmp/tf_model --opset 10 --output ../target/models/tf.onnx --fold_const')
os.system('python -m tf2onnx.convert --target tensorrt --inputs input_1:0[1,224,224,3] --inputs-as-nchw input_1:0 --saved-model /tmp/tf_model --opset 10 --output ../target/models/tf.onnx')

# run onnxruntime for 100 times on host to compare to embedded and original framework
session = onnxruntime.InferenceSession('../target/models/tf.onnx', None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
et_onnx = []
for i in range(0, 101):
    start = time.time()
    image = cv.imread('../data/n01491361_tiger_shark.jpg')
    image = cv.resize(image, dsize=(224, 224))
    image = np.moveaxis(image, 2, 0)
    image = np.array(image, dtype='float32').reshape(1, 3, 224, 224)
    onnx_out = session.run([output_name], {input_name: image})
    et_onnx.append(time.time() - start)

# compare results
np.testing.assert_allclose(tf_out.reshape(1000), np.array(onnx_out).reshape(1000), rtol=1e-03, atol=1e-05)
print('Converted outputs match original ones.')
print('Top 5:', classes)
print('Tensorflow inference time mean:', np.mean(et_tf), 'median:', np.median(et_tf), 'std:', np.std(et_tf), 'n:', i)
print('Onnx inference time mean:', np.mean(et_onnx), 'median:', np.median(et_onnx), 'std:', np.std(et_onnx), 'n:', i)
