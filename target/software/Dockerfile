# Dockerfile to run:
# - TensorRT c++ and python API inside a docker with opencv support!
# - Onnx models from pytorch and tensorflow inside docker
# - Fixes missing libs etc, tested on Nano with Jetpack 4.5 and cuda 10.2

FROM nvcr.io/nvidia/l4t-base:r32.5.0

RUN apt-get update && apt-get install -y \
    wget \
    p7zip-full \
    python3-pip

# Download pre-build opencv 4.4 can also be build from scratch on a nano with the script 'build_cv.sh'
# see: https://github.com/mdegans/nano_build_opencv
RUN wget -O opencv.7z https://drive.switch.ch/index.php/s/mUQvR13Kr3mkHQY/download
RUN 7z e opencv.7z -o./pdk_files/open_cv
RUN dpkg -x ./pdk_files/open_cv/OpenCV-4.4.0-aarch64-dev.deb ./pdk_files/open_cv
RUN dpkg -x ./pdk_files/open_cv/OpenCV-4.4.0-aarch64-libs.deb ./pdk_files/open_cv
RUN dpkg -x ./pdk_files/open_cv/OpenCV-4.4.0-aarch64-main.deb ./pdk_files/open_cv
RUN dpkg -x ./pdk_files/open_cv/OpenCV-4.4.0-aarch64-python.deb ./pdk_files/open_cv
RUN dpkg -x ./pdk_files/open_cv/OpenCV-4.4.0-aarch64-scripts.deb ./pdk_files/open_cv
RUN dpkg -x ./pdk_files/open_cv/OpenCV-4.4.0-aarch64-licenses.deb ./pdk_files/open_cv
RUN ln -s /pdk_files/open_cv/usr/include/opencv4/opencv2 /usr/include/opencv2
RUN ln -s /pdk_files/open_cv/usr/lib /usr/lib/opencv

# install missing packages and libs, set paths
RUN set -o noclobber
RUN touch /etc/ld.so.conf.d/opencv.conf
RUN echo "/pdk_files/open_cv/usr/lib" > /etc/ld.so.conf.d/opencv.conf
RUN ldconfig -v
RUN apt-get update && apt-get install -y \
    liblapack3 \
    libatlas-base-dev
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.2/targets/aarch64-linux/lib:/usr/local/cuda-10.2/lib64/:./libs

# download missing libnvrtc lib
RUN wget -O libnvrtc.7z https://drive.switch.ch/index.php/s/iztGJ7lpmab5txu/download
RUN 7z e libnvrtc.7z -o./pdk_files/libs/
RUN ln -s /pdk_files/libs/libnvrtc.so.10.2 /usr/lib/aarch64-linux-gnu/libnvrtc.so.10.2

# install python libs: numpy, tensorflow, pycuda, opencv
RUN apt update && apt install -y --fix-missing make g++
RUN apt-get install -y zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install -U pip testresources setuptools==49.6.0 
RUN pip3 install -U Cython
RUN pip3 install -U onnxruntime
RUN pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5
RUN pip3 install -U opencv-python
RUN pip3 install -U pycuda
RUN pip3 install -U torchvision