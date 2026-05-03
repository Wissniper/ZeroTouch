#pragma once

#include "irisflow/core/Types.hpp"
#include <opencv2/opencv.hpp>
#include <string>

namespace irisflow {
namespace inference {

class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;

    // Load the model from file
    virtual bool loadModel(const std::string& modelPath) = 0;
    
    // Run inference on a specific Region of Interest (ROI)
    // Updates the DetectionResult which includes the confidence score
    virtual bool infer(const cv::Mat& frame, const cv::Rect& roi, core::DetectionResult& result) = 0;
};

} // namespace inference
} // namespace irisflow
