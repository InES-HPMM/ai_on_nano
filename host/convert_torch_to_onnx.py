#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 07:12:27 2021

Store an pytorch imagenet classifier (EfficientNetB0) as .onnx model and compare outputs to original model

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
python convert_torch_to_onnx.py

Generated files/models are at: '../target/'

Example output on cpu:
    Running on: cpu
    Converted outputs match original ones.
    Top 5: [['tiger_shark', 8.833856], ['great_white_shark', 5.9160933], ['hammerhead', 3.7348697], ['dugong', 3.0150492], ['airship', 2.8707442]]
    Torch inference time mean: 0.06793577364175626 median: 0.06620407104492188 std: 0.004579993089959609 n: 100
    Onnx inference time mean: 0.03297300149898718 median: 0.03278017044067383 std: 0.0018422324899045209 n: 100

    On P40 GPU
    Torch inference time mean: 0.022438469499644665 median: 0.016788244247436523 std: 0.03214770053393938 n: 100

:Author: **Raphael Zingg zing@zhaw.ch**
:Copyright: **2021 Institute of Embedded Systems (InES) All rights reserved**
"""

import os
import time
import torch
from timm import create_model
import json
import cv2 as cv
import onnxruntime
import numpy as np
from torchvision import transforms


def decode_imagenet(preds, idx_file='./data/imagenet_idx.json', top=5):
    # see: https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py

    with open(idx_file) as json_file:
        CLASS_INDEX = json.load(json_file)
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [[CLASS_INDEX[str(i)][1], pred[i]]for i in top_indices]
    return result

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# check if we have gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Running on:', device)

# get classifier in eval mode
model = create_model('efficientnet_b0',
                     num_classes=1000,
                     pretrained=True)
model.to(device)
model.eval()

# prepare example image input
preprocess_torch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
et_tf = []
for i in range(0, 101):
    start = time.time()

    # load image and pre-process
    image = cv.imread('../data/n01491361_tiger_shark.jpg')
    image = cv.resize(image, dsize=(224, 224))
    input_tensor = preprocess_torch(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # inference
    torch_out = model(input_batch)
    et_tf.append(time.time() - start)

classes = torch_out.detach().cpu().numpy()
classes = decode_imagenet(classes.reshape(1, 1000), idx_file='../data/imagenet_idx.json')

# convert pytorch to onnx
try:
    os.mkdir('../target/models')
except OSError as error:
    print(error)

# convert pytorch to onnx
torch.onnx.export(model,                     # model being run
                  input_batch,               # model input
                  '../target/models/torch.onnx',    # where to save the model
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],     # the model's input names
                  output_names=['output'])   # the model's output names

# run onnxruntime
et_onnx = []
session = onnxruntime.InferenceSession('../target/models/torch.onnx', None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
for i in range(0, 101):
    start = time.time()

    # load image and pre-process
    image = cv.imread('../data/n01491361_tiger_shark.jpg')
    image = cv.resize(image, dsize=(224, 224))
    input_tensor = preprocess_torch(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    input_tensor = input_batch.cpu().numpy()

    # inference
    onnx_out = session.run([output_name], {input_name: input_tensor})
    et_onnx.append(time.time() - start)

# compare results
np.testing.assert_allclose(to_numpy(torch_out), np.array(onnx_out).reshape(1, 1000), rtol=1e-03, atol=1e-05)
print('Converted outputs match original ones.')
print('Top 5:', classes)
print('Torch inference time mean:', np.mean(et_tf), 'median:', np.median(et_tf), 'std:', np.std(et_tf), 'n:', i)
print('Onnx inference time mean:', np.mean(et_onnx), 'median:', np.median(et_onnx), 'std:', np.std(et_onnx), 'n:', i)
