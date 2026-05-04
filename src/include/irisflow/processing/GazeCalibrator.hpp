#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace irisflow {
namespace processing {

class GazeCalibrator {
public:
    GazeCalibrator() = default;
    ~GazeCalibrator() = default;

    // GAZE-02: 9-point Homography Calibration
    void addCalibrationPoint(const cv::Point2f& gazePt, const cv::Point2f& screenPt);
    bool computeHomography();
    
    // Map gaze point to screen coordinates
    cv::Point2f mapToScreen(const cv::Point2f& gazePt) const;

    // GAZE-03: Head-Pose Compensation
    // Decouple head movement from intentional gaze
    cv::Point2f compensateHeadPose(const cv::Point2f& rawGaze, float headYaw, float headPitch) const;

private:
    std::vector<cv::Point2f> m_gazePoints;
    std::vector<cv::Point2f> m_screenPoints;
    cv::Mat m_homography;
    
    const float HEAD_COMP_SCALE = 0.012f;
};

} // namespace processing
} // namespace irisflow
