#!/bin/bash
# 2021 @author Raphael Zingg / zing@zhaw.ch

# copy all files used on the nano into one target
mkdir -p ../target/app
mkdir -p ../target/app/libs

cp ./tmp/TensorRT/build/out/*_trt ../target/app/
cp ./tmp/TensorRT/build/out/lib* ../target/app/libs/
cp ./tmp/TensorRT/build/out/trtexec ../target/app/trtexec
cp ../target/models/*.onnx ../target/app/
cp ../data/* ../target/app
cp ../target/software/* ../target/app
cp ../target/software/Dockerfile ../target/app/Dockerfile

# scp the files to nano
scp -r ../target/app $2@$1:"/home/"$2"/"

if [[ $? > 1 ]]; then
    echo "-----------------------------------------------------------"
    echo "                        Failed                             "
    echo "-----------------------------------------------------------"
else
    echo "-----------------------------------------------------------"
    echo "                           Done                            "
    echo "-----------------------------------------------------------"
fi
