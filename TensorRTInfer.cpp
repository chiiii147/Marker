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
    if(!builder)
    {
        return false;
    }
    
    auto network = std::unique_ptr<nvinfer1_INetworkDefination>(nvinfer1::builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinationCreationFlag::kSTRONGLY_TYPED)));
    if(!network)
    {
        return false;
    }
}