# Embedded Artificial Intelligence on Jetson Nano

__The group High Performance Multimedia from the Institute of Embedded Systems
associated with ZHAW School of Engineering proudly presents a reference workflow to run custom,
optimized Tensorflow and Pytorch models as TensorRT Engines on a Jetson Nano. Both the C++ API and the Python API of TensorRT is supported.
Image processing is done with OpenCV 4.4 and everything runs inside Docker containers.__
  
> For recent news check out our [Blog](https://blog.zhaw.ch/high-performance/).

![logo](https://github.zhaw.ch/storage/user/1361/files/466e0a13-c0ae-4ec1-8526-e38b110563c2)


## Jetson Nano Requirements
Steup the Jetson Nano with the default OS from NVIDIA, and connect it to network.  
Requirements:
- [Running OS](
https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write)  
- [Jetpack 4.5](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html)
- SSH
- Cuda 10.2
- TensorRT 7.1.3

## Host Computer Requirements
The following steps are done on a host computer, they are tested on:
- GNU/Linux: Ubuntu 18.04.6 LTS
- Docker: Version 18.06.1-ce, build e68fc7a
- GPU is not required on host

### Cross Compile C++ TensorRT Applications for Jetson Nano
Run the following script to download the sources and build the docker container:  
```
$ cd host
$ ./prepare_cross_compile.sh
```

Run the container and build all examples, plugins, libs, etc with the following command:
```
$ sudo docker run --network host  -v ~/ai_on_nano/host/tmp/TensorRT/docker/jetpack_files/:/tensorrt -v ~/ai_on_nano/host/tmp/TensorRT/:/workspace/TensorRT -it tensorrt-ubuntu-jetpack:latest
# cd TensorRT
# mkdir build
# cd build
# cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out -DBUILD_PLUGINS=OFF -DCMAKE_TOOLCHAIN_FILE=$TRT_SOURCE/cmake/toolchains/cmake_aarch64.toolchain -DCMAKE_CUDA_SEPARABLE_COMPILATION=OFF -DCUDA_VERSION=10.2 -DCUBLASLT_LIB="/usr/lib/aarch64-linux-gnu/libcublasLt.so" -DCUBLAS_LIB="/usr/lib/aarch64-linux-gnu/libcublas.so" -DCUDNN_LIB="/pdk_files/cudnn/lib/libcudnn.so.8"
# make -j 8
```  
After the build is done quite the docker container and proceed with the next step:

### Convert Custom Tensorflow or Pytorch to .onnx

Create the host docker to convert the tensorflow/keras models  
**Adapt the docker image id `ddf619d90da4`**
```
$ sudo docker build . -f ./convert_to_onnx.Dockerfile 
$ sudo docker run --network host -v ~/ai_on_nano/:/workspace -v ~/ai_on_nano/data/:/data -it ddf619d90da4
# cd host
# python convert_tensorflow_to_onnx.py
# python convert_torch_to_onnx.py
```

After the converting is done quit the docker container.  
If you want to use a own, custom model change it in the script `host/convert_tensorflow_to_onnx.py` or `host/convert_torch_to_onnx.py` 
Copy the files to the Nano:  
**Adapt `192.168.188.36` and `host_name`**

```
$ ./copy_to_target.sh 192.168.188.36 host_name
```

## Run TensorRT Engines on Jetson Nano

On the Jetson Nano, build and run the docker-container with:  
**Adapt the docker image id `4a04848cd325`**

```
$ cd app/
$ sudo docker build .
$ sudo docker run -it --rm --net=host --runtime nvidia -v ~/app/:/app 4a04848cd325
```

**Only once:** Inside the application container parse the onnx models to tensorrt engines with: 
```
# cd app
# ./trtexec --onnx=tf.onnx --explicitBatch --workspace=2048 --saveEngine=tf.engine --verbose
# ./trtexec --onnx=torch.onnx --explicitBatch --workspace=2048 --saveEngine=torch.engine --verbose
```
This can take a while, but needs only be done once for every model.  

### Run Tensorflow TensorRT Engines on Jetson Nano
C++
```
# ./tf_trt
```  
Python
```
# python3 ./tfTRT.py
```

### Run Pytorch TensorRT Engines on Jetson Nano
C++
```
# ./torch_trt
```  
Python
```
# python3 ./torchTRT.py
```  
Onnxruntime
```
# python3 ./torchONNX.py
```
