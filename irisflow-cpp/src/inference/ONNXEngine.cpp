#include "irisflow/inference/ONNXEngine.hpp"
#include <iostream>

namespace irisflow {
namespace inference {

ONNXEngine::ONNXEngine() {
    try {
        m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "IrisFlow");
        m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to initialize ONNX Runtime: " << e.what() << std::endl;
    }
}

ONNXEngine::~ONNXEngine() = default;

bool ONNXEngine::loadModel(const std::string& modelPath) {
    if (!m_env) return false;
    
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // ONNX Runtime expects wide string on Windows, but let's assume macOS/Linux
        m_session = std::make_unique<Ort::Session>(*m_env, modelPath.c_str(), sessionOptions);

        // Define input/output names (these would normally be dynamically queried or configured)
        m_inputNodeNames = {"input"};
        m_outputNodeNames = {"output"};

        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load model " << modelPath << ": " << e.what() << std::endl;
        return false;
    }
}

bool ONNXEngine::infer(const cv::Mat& frame, const cv::Rect& roi, core::DetectionResult& result) {
    if (!m_session) return false;

    // Implement ROI Tracking Logic (STAB-04)
    // We only process the relevant sub-region to save computation time
    cv::Mat preprocessed = preprocess(frame, roi);

    // Mock inference result for now
    std::vector<float> inputTensorValues(1 * 3 * 224 * 224, 0.5f); 
    std::vector<int64_t> inputTensorShape = {1, 3, 224, 224};

    try {
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            m_memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputTensorShape.data(), inputTensorShape.size());

        auto outputTensors = m_session->Run(
            Ort::RunOptions{nullptr}, 
            m_inputNodeNames.data(), &inputTensor, 1, 
            m_outputNodeNames.data(), 1);

        // Process outputs...
        // For demonstration, we simulate a successful detection with a confidence score
        // STAB-01: Confidence gating logic applies when `result.isValid()` is called
        result.confidence = 0.85f; 
        result.roi = roi;
        result.landmarks.push_back({0.5f, 0.5f, 0.0f});

        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat ONNXEngine::preprocess(const cv::Mat& frame, const cv::Rect& roi) {
    cv::Mat cropped;
    if (roi.area() > 0 && (roi & cv::Rect(0, 0, frame.cols, frame.rows)) == roi) {
        cropped = frame(roi);
    } else {
        cropped = frame; // Fallback to full frame if ROI is invalid
    }

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(224, 224));
    
    // Normalize and convert to blob...
    // (Implementation omitted for brevity)
    return resized;
}

void ONNXEngine::postprocess(const float* outputData, size_t outputSize, core::DetectionResult& result) {
    // Convert tensor output back to landmarks
}

} // namespace inference
} // namespace irisflow
