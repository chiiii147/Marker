#ifndef TENSORRT_INFER_H
#define TENSORRT_INFER_H

#include <NvInfer.h>
#include <NvOnnxParse.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <iostream>

class OnnxMarker
{
    public:
    Onnxmarker(const std::string& Onnxpath):
    mOnnxpath(Onnxpath),
    mRuntime(nullptr),
    mEngine(nullptr),
    mContext(nullptr)
    {}
    bool build()
    bool infer()

    private:
    std::string mOnnxpath;
    std::shared_ptr<nvinfer1::IRuntime> mRuntime;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext;

    Logger mLogger;

    bool constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
                      std::unique_ptr<nvinfer1::INetworkDefinition>& network,
                      std::unique_ptr<nvinfer1::IBuilderConfig>& config,
                      std::unique_ptr<nvinfer1::IParser>& parser)
};

class Logger: public nvinfer1::ILogger
{
    public:
    void log(Serverity severity, const char* msg) noexcept overrite;
}
