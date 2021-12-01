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
RUN pip install tensorflow==2.4.1
RUN pip install tf2onnx==1.8.4