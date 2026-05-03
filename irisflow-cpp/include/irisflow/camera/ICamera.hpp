#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

namespace irisflow {
namespace camera {

class ICamera {
public:
    virtual ~ICamera() = default;
    
    // Initialize the camera
    virtual bool start() = 0;
    
    // Stop the camera
    virtual void stop() = 0;
    
    // Get the next frame (blocking or non-blocking depending on implementation)
    // Returns true if a frame was successfully retrieved
    virtual bool getFrame(cv::Mat& frame) = 0;
};

} // namespace camera
} // namespace irisflow
