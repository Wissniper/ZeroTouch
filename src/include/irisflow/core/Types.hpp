#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace irisflow {
namespace core {

struct Landmark {
    float x;
    float y;
    float z;
};

struct DetectionResult {
    float confidence;
    cv::Rect roi;
    std::vector<Landmark> landmarks;
    
    // True if detection passed the confidence gate
    bool isValid() const {
        return confidence >= 0.7f;
    }
};

} // namespace core
} // namespace irisflow
