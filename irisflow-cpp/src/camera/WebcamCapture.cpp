#include "irisflow/camera/WebcamCapture.hpp"
#include <iostream>

namespace irisflow {
namespace camera {

WebcamCapture::WebcamCapture(int deviceId) : m_deviceId(deviceId) {}

WebcamCapture::~WebcamCapture() {
    stop();
}

bool WebcamCapture::start() {
    if (m_running) {
        return true;
    }

    m_capture.open(m_deviceId, cv::CAP_ANY);
    if (!m_capture.isOpened()) {
        std::cerr << "Failed to open camera device " << m_deviceId << std::endl;
        return false;
    }
    
    // Request 60 FPS
    m_capture.set(cv::CAP_PROP_FPS, 60);

    m_running = true;
    m_captureThread = std::thread(&WebcamCapture::captureLoop, this);
    
    return true;
}

void WebcamCapture::stop() {
    if (!m_running) {
        return;
    }

    m_running = false;
    if (m_captureThread.joinable()) {
        m_captureThread.join();
    }

    if (m_capture.isOpened()) {
        m_capture.release();
    }
}

bool WebcamCapture::getFrame(cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(m_mutex);
    // Wait until there is a new frame or we are stopped
    m_cv.wait(lock, [this]() { return m_hasNewFrame || !m_running; });

    if (!m_running || m_latestFrame.empty()) {
        return false;
    }

    // Shallow copy the frame header
    frame = m_latestFrame;
    m_hasNewFrame = false;
    
    return true;
}

void WebcamCapture::captureLoop() {
    cv::Mat frame;
    while (m_running) {
        if (m_capture.read(frame)) {
            std::lock_guard<std::mutex> lock(m_mutex);
            // Shallow copy to minimize overhead
            m_latestFrame = frame;
            m_hasNewFrame = true;
            m_cv.notify_one();
        } else {
            std::cerr << "Failed to read frame from camera" << std::endl;
            // Optionally sleep or handle error gracefully
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

} // namespace camera
} // namespace irisflow
