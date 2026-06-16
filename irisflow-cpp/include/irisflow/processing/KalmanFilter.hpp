#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

namespace irisflow {
namespace processing {

class KalmanFilter2D {
public:
    KalmanFilter2D();
    ~KalmanFilter2D() = default;

    // Initialize with first measurement
    void init(float x, float y);

    // Predict and update with new measurement
    cv::Point2f update(float x, float y);

    // Predict next state without measurement (e.g. during occlusion)
    cv::Point2f predict();

private:
    int m_stateSize{4}; // [x, y, v_x, v_y]
    int m_measSize{2};  // [x, y]
    int m_contrSize{0};
    cv::KalmanFilter m_kf;
    bool m_initialized{false};
};

} // namespace processing
} // namespace irisflow
