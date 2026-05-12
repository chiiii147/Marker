#include "TensorRTInfer.h"
#include <fstream>
#include <iostream>


void Logger::log(Severity severity, const char* msg) noexcept
{
    if(severity <= Severity::kWARNING)
        std::cout << "TensorRT" << msg << std::endl;
}

bool TensorRTInfer::build()
{
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(mLogger));
    if(!builder) return false;
    
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    if(!network) return false;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, mLogger));
    if(!parser) return false;

    auto constructed = constructNetwork(builder, network, config, parser);
    if(!constructed) return false;

    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if(!plan) return false;

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(mLogger));
    if(!mRuntime) return false;

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()));
    if(!mEngine) return false;

    //auto profileStream = 

    if(network -> getNbInputs() != 1)
        return false;

    if(network -> getNbOutputs() != 1)
        return false;

    for(int i = 0; i < mEngine -> getNbIOTensors(); i++)
    {
        auto const name = mEngine -> getIOTensorName(i);
        auto mode = mEngine -> getTensorIOMode(name);

        if(mode == nvinfer1::TensorIOMode::kINPUT)
            mInputDims = mEngine -> getTensorShape(name);

        else if(mode == nvinfer1::TensorIOMode::kOUTPUT)
            mOutputDims = mEngine -> getTensorShape(name);
    }

    if(!AllocateMemory())
    return false;

    return true;
}

bool TensorRTInfer::constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
                                     std::unique_ptr<nvinfer1::INetworkDefinition>& network,
                                     std::unique_ptr<nvinfer1::IBuilderConfig>& config, 
                                     std::unique_ptr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(mOnnxpath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if(!parsed) return false;

    return true;
}

bool TensorRTInfer::infer()
{
    
    if(cudaMemcpyAsync(mDeviceInput, mHostInput, mInputSize, cudaMemcpyHostToDevice, mStream) != cudaSuccess)
        return false;
        
    bool status = mContext -> enqueueV3(mStream);
    if(!status) return false;

    if(cudaMemcpyAsync(mHostOutput, mDeviceOutput, mOutputSize, cudaMemcpyDeviceToHost, mStream) != cudaSuccess)
        return false;

    cudaStreamSynchronize(mStream);

    return true;
}

bool TensorRTInfer::preProcess(const cv::Mat& image)
{
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputW, inputH));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255);

    std::vector<cv::Mat> chw(3);
    for(int i = 0; i < 3; i++)
        chw[i] = cv::Mat(inputH, inputW, CV_32F, static_cast<float*>(mHostInput) + i * inputH * inputW);

    cv::split(resized, chw);

    return true;
}

bool TensorRTInfer::postProcess(std::vector<Detection>& detections)
{
    float* output = static_cast<float*>(mHostOutput);

    const int numDetections = 300;
    const int elementsPerDetection = 6;

    const float confThreshold = 0.75f;

    detections.clear();

    for (int i = 0; i < numDetections; i++)
    {
        int idx = i * elementsPerDetection;

        float x1 = output[idx + 0];
        float y1 = output[idx + 1];
        float x2 = output[idx + 2];
        float y2 = output[idx + 3];

        float score = output[idx + 4];
        int classId = static_cast<int>(output[idx + 5]);

        if (score < confThreshold)
        {
            continue;
        }

        Detection det;

        det.x1 = x1;
        det.y1 = y1;
        det.x2 = x2;
        det.y2 = y2;

        det.confidence = score;
        det.classId = classId;

        detections.push_back(det);
    }

    return true;
}

bool TensorRTInfer::AllocateMemory()
{
    mContext = std::shared_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if(!mContext) return false;

    if(cudaStreamCreate(&mStream) != cudaSuccess)
    return false;

    //Number of IO tensors is the number of input and output tensors for the network from which the engine was built.
    //The names of the IO tensors can be discovered by calling getIOTensorName(i) for i in 0 to getNbIOTensors()-1.
    for(int i = 0; i < mEngine -> getNbIOTensors(); i++)
    {
        auto const name = mEngine -> getIOTensorName(i);
        auto mode = mEngine -> getTensorIOMode(name);
        auto shapes = mEngine -> getTensorShape(name);

        size_t elem = 1;
        for(int j = 0; j < shapes.nbDims; j++)
            elem = elem * shapes.d[j];

        if(mode == nvinfer1::TensorIOMode::kINPUT)
        {
            mInputSize = elem * sizeof(float);
            if(cudaMalloc(&mDeviceInput, mInputSize) != cudaSuccess)
            return false;

            mHostInput = new float[mInputSize / sizeof(float)];
            mContext -> setTensorAddress(name, mDeviceInput);
        }
        else
        {
            mOutputSize = elem * sizeof(float);
            if(cudaMalloc(&mDeviceOutput, mOutputSize) != cudaSuccess)
            return false;

            mHostOutput = new float[mOutputSize / sizeof(float)];
            mContext -> setTensorAddress(name, mDeviceOutput);
        }
    }
    
    return true;
}

TensorRTInfer::~TensorRTInfer()
{
    if(mDeviceInput)
        cudaFree(mDeviceInput);
    if(mDeviceOutput)
        cudaFree(mDeviceOutput);

    if(mHostInput)
        delete[] static_cast<float*>(mHostInput);
    if(mHostOutput)
        delete[] static_cast<float*>(mHostOutput);

    if(mStream)
        cudaStreamDestroy(mStream);
}