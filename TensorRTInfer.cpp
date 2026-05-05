#include <TensorRTInfer.h
#include <fstream.h>
#include <iostream.h>


void TensorRTInfer::log(Severity severity, const char* msg) noexcept
{
    if(severity <= Severity::kWARNING)
    std::cout << "TensorRT" << msg << std::endl;
}

bool TensorRTInfer::build()
{
    auto builder = std::unique_ptr<nvinfer1_IBulder>(nvinfer1::createInferBuilder(mLogger));
    if(!builder) return false;
    
    auto network = std::unique_ptr<nvinfer1_INetworkDefination>(nvinfer1::builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinationCreationFlag::kSTRONGLY_TYPED)));
    if(!network) return false;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, mLogger));
    if(!parser) return false;

    auto constructed = constructNetwork(builder, network, config, parser);
    if(!constructed) return false;

    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if(!plan) return false;

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(mLogger));
    if(!mRuntime) return false;

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()));
    if(!mEngine) return false;

    return true;
}

bool TensorRTInfer::constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder, std::unique_ptr<nvinfer1::INetworkDefinition>& network,
                                     std::unique_ptr<nvinfer1::IBuilderConfig>& config, std::unique_ptr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(mOnnxpath.onnxFileName, mOnnxpath.dataDirs).c_str(),static_cast<int>(mLogger);
    if(!parsed) return false;

}

