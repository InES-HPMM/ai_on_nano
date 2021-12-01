#!/bin/bash
# 2021 @author Raphael Zingg / zing@zhaw.ch

echo "-----------------------------------------------------------"
echo "                   Downloading TensorRT                    "
echo "-----------------------------------------------------------"

mkdir tmp
git clone -b master https://github.com/nvidia/TensorRT tmp/TensorRT
cd tmp/TensorRT
git submodule update --init --recursive
export TRT_SOURCE=`pwd`
git checkout release/7.1
cd  parsers/onnx/
git checkout 7.2.1

echo "-----------------------------------------------------------"
echo "                   Replacing Files                         "
echo "-----------------------------------------------------------"
cd ../../../..
cp ./cross_compile_tensorRT_cpp.Dockerfile ./tmp/TensorRT/docker/ubuntu-cross-aarch64.Dockerfile
cp ./cmake/CMakeListsAddPaths.txt ./tmp/TensorRT/CMakeLists.txt
cp ./cmake/CMakeListsOnlyCustom.txt ./tmp/TensorRT/samples/opensource/CMakeLists.txt 
cp ./cmake/CMakeSamplesTemplateWithOpenCV.txt ./tmp/TensorRT/samples/CMakeSamplesTemplate.txt
cp ./patch/sampleEngines.cpp ./tmp/TensorRT/samples/common/sampleEngines.cpp

# Tensorflow TensorRT C++ Application
mkdir ./tmp/TensorRT/samples/opensource/tfTRT
cp ../target/software/tfTRT.cpp ./tmp/TensorRT/samples/opensource/tfTRT/
cp ./cmake/CMakeListsTf.txt ./tmp/TensorRT/samples/opensource/tfTRT/CMakeLists.txt

# Pytorch TensorRT C++ Application
mkdir ./tmp/TensorRT/samples/opensource/torchTRT
cp ../target/software/torchTRT.cpp ./tmp/TensorRT/samples/opensource/torchTRT/
cp ./cmake/CMakeListsTorch.txt ./tmp/TensorRT/samples/opensource/torchTRT/CMakeLists.txt

echo "-----------------------------------------------------------"
echo "                   Downloading Sources                     "
echo "-----------------------------------------------------------"
# Sources are from Nvidia SDK: https://developer.nvidia.com/embedded/jetpack
# However, the jetpack files are missing a lib(needs to be copied from nano) see:
# https://github.com/NVIDIA/TensorRT/issues/1168
FILE=./tmp/jetpack_files.7z
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    wget -O ./tmp/jetpack_files.7z https://drive.switch.ch/index.php/s/kGGlE8dc1y60m8b/download 
    7z e ./tmp/jetpack_files.7z -o./tmp/TensorRT/docker/jetpack_files
fi

# Download pre-build opencv 4.4 can also be build from scratch on a nano with the script target/opencv/build_cv.sh
# see: https://github.com/mdegans/nano_build_opencv
FILE=./tmp/open_cv_aarch64.7z
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    wget -O ./tmp/open_cv_aarch64.7z https://drive.switch.ch/index.php/s/mUQvR13Kr3mkHQY/download
    7z e ./tmp/open_cv_aarch64.7z -o./tmp/TensorRT/docker/open_cv
fi

echo "-----------------------------------------------------------"
echo "                   Building Container                      "
echo "-----------------------------------------------------------"
cd tmp/TensorRT
sudo ./docker/build.sh --file ./docker/ubuntu-cross-aarch64.Dockerfile --tag tensorrt-ubuntu-jetpack --os 18.04 --cuda 10.2

echo "-----------------------------------------------------------"
echo "                            Done                           "
echo "-----------------------------------------------------------"