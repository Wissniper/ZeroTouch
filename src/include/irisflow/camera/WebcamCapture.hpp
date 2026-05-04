#pragma once

#include "irisflow/camera/ICamera.hpp"
#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace irisflow {
namespace camera {

class WebcamCapture : public ICamera {
public:
    WebcamCapture(int deviceId = 0);
    ~WebcamCapture() override;

    bool start() override;
    void stop() override;
    bool getFrame(cv::Mat& frame) override;

private:
    void captureLoop();

    int m_deviceId;
    cv::VideoCapture m_capture;
    
    std::atomic<bool> m_running{false};
    std::thread m_captureThread;
    
    cv::Mat m_latestFrame;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_hasNewFrame{false};
};

} // namespace camera
} // namespace irisflow
