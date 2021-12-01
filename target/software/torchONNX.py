#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run onnx model with converted pytorch model with onnxruntime on Jetson Nano

Example output on Jetson Nano:

    Onnx inference time mean: 0.20283411989117614 median: 0.20269513130187988 std: 0.0012351465988597714 n: 100
    Onnx top  5: [['tiger_shark', 8.833855], ['great_white_shark', 5.916094], ['hammerhead', 3.7348738], ['dugong', 3.0150516], ['airship', 2.8707457]]

:Author: **Raphael Zingg zing@zhaw.ch**
:Copyright: **2021 Institute of Embedded Systems (InES) All rights reserved**
"""

import json
import time
import cv2 as cv
import onnxruntime
from torchvision import transforms
import numpy as np


def decode_imagenet(preds, idx_file='imagenet_idx.json', top=5):
    # see: https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py

    with open(idx_file) as json_file:
        CLASS_INDEX = json.load(json_file)
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [[CLASS_INDEX[str(i)][1], pred[i]]for i in top_indices]
    return result

# prepare example image input
preprocess_torch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# run onnxruntime
et_onnx = []
session = onnxruntime.InferenceSession('torch.onnx', None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
for i in range(0, 101):
    start = time.time()

    # load image and pre-process
    image = cv.imread('n01491361_tiger_shark.jpg')
    image = cv.resize(image, dsize=(224, 224))
    input_tensor = preprocess_torch(image)
    input_batch = input_tensor.unsqueeze(0)
    input_tensor = input_batch.numpy()

    # inference
    onnx_out = session.run([output_name], {input_name: input_tensor})
    et_onnx.append(time.time() - start)

# compare results
print('Onnx inference time mean:', np.mean(et_onnx), 'median:', np.median(et_onnx), 'std:', np.std(et_onnx), 'n:', i)
print('Onnx top  5:', decode_imagenet(np.array(onnx_out).reshape(1, 1000), top=5))
