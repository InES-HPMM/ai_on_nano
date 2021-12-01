/** This program can be used to run tensorflow trt engines with the C++ API
 * 
 * It measures the inference time and verify the output. It is based on the NVIDIA onnx mnist example
 * See: https://github.com/NVIDIA/TensorRT/tree/master/samples/sampleOnnxMNIST
 * 
 * Example output on Jetson Nano:
 * 
 * -------Running TensorRT Tensorflow C++-------
 * Loading model from:tf.engine
 * Input file:n01491361_tiger_shark.jpg
 * 
 * 
 * Measurement results of 100 runs:
 * 
 * Mean               = 0.0405859 (s)
 * Median             = 0.038 (s)
 * Standard Deviation = 0.00643069
 * Classification Results:
 * 
 * Top:1 index:3 value:0.807685
 * Top:2 index:4 value:0.0458767
 * Top:3 index:2 value:0.035323
 * Top:4 index:148 value:0.0027615
 * Top:5 index:394 value:0.00211886
 *
 * :Note: imagenet index to label can be found in data/imagenet_idx.json. Tiger shark has index 3
 * :Author: **Raphael Zingg zing@zhaw.ch**
 * :Copyright: **2021 Institute of Embedded Systems (InES) All rights reserved**
 **/

#include "NvInfer.h"
#include "algorithm"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>

// opencv includes
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudawarping.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;
template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

typedef struct Statistics
{
    double mean;
    double median;
    double stdev;
} Statistics;

Statistics getMeanMedianStd(const std::vector<double> &i_vector)
{
    Statistics stats;
    std::vector<double> vector(i_vector);
    std::sort(vector.begin(), vector.end());
    double sumD = std::accumulate(vector.begin(), vector.end(), 0.0);
    stats.mean = sumD / double(vector.size());
    std::vector<double> Diff(vector.size());
    std::transform(vector.begin(), vector.end(), Diff.begin(), std::bind2nd(std::minus<double>(), stats.mean));
    stats.stdev = std::inner_product(Diff.begin(), Diff.end(), Diff.begin(), 0.0);
    stats.median = vector[std::floor(double(vector.size()) / 2.0)];
    stats.stdev = std::sqrt(stats.stdev / vector.size());
    return stats;
}

class Classifier
{
public:
    /**
     * Init the tensorRT model, memory and runtime
     *
     * @param error_path path to error classifier
     * @param input_size image input size of error classifier (eg 128)
     *
     * @return True if the init was successfull
     */
    bool init(std::string error_path, int input_size);

    /**
     * Run the tensorRT model on input image
     *
     * @param image_path path of the current image (.jpg or raw is possible)
     *
     * @return cv::Mat with the 1000 output classes (no softmax)
     */
    cv::Mat infer(std::string image_path);

    /**
     * Print top-n index and corresponding values
     *
     * @param preds cv::Mat from infer
     * @param top print top-n values
     */
    void post_process(cv::Mat preds, int top);

private:
    // variables set in init()
    std::shared_ptr<nvinfer1::ICudaEngine> trt_engine;
    SampleUniquePtr<nvinfer1::IExecutionContext> trt_context;
    int mInputIdx, m_output_idx;
    int m_input_size;

    // create cuda memory for classification
    cv::Mat trt_output;
    cv::cuda::GpuMat gpu_output;
    void *trt_buffer[2]; // one input one output

    // prase tensorRT models to engine and context
    std::shared_ptr<nvinfer1::ICudaEngine> parse_models(std::string engine_path);
};

void Classifier::post_process(cv::Mat preds, int top)
{
    std::vector<float> all_preds;

    // store all float values to perform some sort/math on them
    for (int i = 0; i < preds.rows; i++)
        all_preds.push_back(preds.at<float>(i));

    // get index and top n
    std::vector<size_t> idx(all_preds.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](size_t A, size_t B)
                      { return all_preds[A] > all_preds[B]; });
    std::cout << "Classification Results:\n"
              << std::endl;
    for (int i = 0; i < top; i++)
        std::cout << "Top:" << i + 1 << " index:" << idx[i] << " value:" << all_preds[idx[i]] << std::endl;
}

bool Classifier::init(std::string error_path, int input_size)
{

    // input size of models
    m_input_size = input_size;

    // parse the models
    trt_engine = Classifier::parse_models(error_path);
    if (trt_engine == NULL)
        return false;

    // create context
    trt_context = SampleUniquePtr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    if (trt_context == NULL)
        return false;

    // get binding idx of imagenet classifier
    mInputIdx = trt_engine->getBindingIndex("input_1:0");
    m_output_idx = trt_engine->getBindingIndex("predictions");

    // init cuda memory for input and output
    cudaMalloc(&trt_buffer[0], m_input_size * m_input_size * 3 * sizeof(float));
    gpu_output = cv::cuda::GpuMat(1000 * 1, 1, CV_32F);
    return true;
}

std::shared_ptr<nvinfer1::ICudaEngine> Classifier::parse_models(std::string engine_path)
{
    // init variables
    ICudaEngine *engine_ptr;
    std::ifstream planFile(engine_path);
    std::stringstream plan_buffer;

    // get engine and context from converted model
    plan_buffer << planFile.rdbuf();
    std::string plan = plan_buffer.str();
    IRuntime *runtime = createInferRuntime(sample::gLogger.getTRTLogger());
    engine_ptr = runtime->deserializeCudaEngine((void *)plan.data(), plan.size(), nullptr);
    if (!engine_ptr)
        return NULL;

    // create context and engine
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(engine_ptr, samplesCommon::InferDeleter());
    if (!engine)
        return NULL;
    else
        return engine;
}

cv::Mat Classifier::infer(std::string input_path)
{
    // read image (BGR)
    cv::Mat input_image = cv::imread(input_path);
    cv::resize(input_image, input_image, cv::Size(m_input_size, m_input_size));

    // preprocess image for pytorch tensorRT
    input_image.convertTo(input_image, CV_32FC3);

    // upload scaled image to gpu
    cv::cuda::GpuMat resized_rgb(input_image);

    // get pointers to preallocated image planes
    std::vector<cv::cuda::GpuMat> input_channels{
        cv::cuda::GpuMat(m_input_size, m_input_size, CV_32F, (float *)trt_buffer[mInputIdx]),
        cv::cuda::GpuMat(m_input_size, m_input_size, CV_32F, (float *)trt_buffer[mInputIdx] + m_input_size * m_input_size),
        cv::cuda::GpuMat(m_input_size, m_input_size, CV_32F, (float *)trt_buffer[mInputIdx] + m_input_size * m_input_size * 2)};

    // split color planess HWC - CHW
    cv::cuda::split(resized_rgb, input_channels);

    trt_buffer[m_output_idx] = gpu_output.data;

    // inference classifier
    if (trt_context->executeV2(trt_buffer) == false)
    {
        std::cout << "infer failed" << std::endl;
        return trt_output;
    }

    // download result of classifier
    cudaDeviceSynchronize();
    gpu_output.download(trt_output);

    // return predictions to driver (no softmax)
    return trt_output;
}

int main(int argc, char *argv[])
{

    // init variables
    std::string model_path("tf.engine");
    std::string image_path("n01491361_tiger_shark.jpg");
    std::vector<double> all_measurements;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    cv::Mat res;

    // print some info
    std::cout << "-------Running TensorRT Tensorflow C++-------" << std::endl;
    std::cout << "Loading model from:" << model_path << std::endl;
    std::cout << "Input file:" << image_path << std::endl;

    // init tensorRT
    Classifier *classifier = new Classifier();
    if (classifier->init(model_path, 224) == false)
    {
        std::cout << "init failed" << std::endl;
        return -1;
    }

    // run inference 100 times
    for (size_t i = 0; i < 100; i++)
    {
        t1 = std::chrono::high_resolution_clock::now();
        res = classifier->infer(image_path);
        ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1);
        all_measurements.push_back(float(ms_int.count()) / 1000.0);
    }

    // calc results, remove first measurement it is the 'warm up' phase of tensorRT
    all_measurements.erase(all_measurements.begin());
    std::cout << "\n\nMeasurement results of 100 runs:\n"
              << std::endl;
    Statistics stats = getMeanMedianStd(all_measurements);
    std::cout << "Mean               = " << stats.mean << " (s)\n"
              << "Median             = " << stats.median << " (s)\n"
              << "Standard Deviation = " << stats.stdev << "\n";

    // print top 5
    classifier->post_process(res, 5);
    return 0;
}
