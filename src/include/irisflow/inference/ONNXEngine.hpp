#pragma once

#include "irisflow/inference/IInferenceEngine.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>

namespace irisflow {
namespace inference {

class ONNXEngine : public IInferenceEngine {
public:
    ONNXEngine();
    ~ONNXEngine() override;

    bool loadModel(const std::string& modelPath) override;
    bool infer(const cv::Mat& frame, const cv::Rect& roi, core::DetectionResult& result) override;

private:
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::MemoryInfo m_memoryInfo{nullptr};
    
    std::vector<const char*> m_inputNodeNames;
    std::vector<const char*> m_outputNodeNames;
    
    cv::Mat preprocess(const cv::Mat& frame, const cv::Rect& roi);
    void postprocess(const float* outputData, size_t outputSize, core::DetectionResult& result);
};

} // namespace inference
} // namespace irisflow
