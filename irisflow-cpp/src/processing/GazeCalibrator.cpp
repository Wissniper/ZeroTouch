#include "irisflow/processing/GazeCalibrator.hpp"

namespace irisflow {
namespace processing {

void GazeCalibrator::addCalibrationPoint(const cv::Point2f& gazePt, const cv::Point2f& screenPt) {
    m_gazePoints.push_back(gazePt);
    m_screenPoints.push_back(screenPt);
}

bool GazeCalibrator::computeHomography() {
    if (m_gazePoints.size() < 4) {
        return false; // Need at least 4 points for homography
    }
    
    // GAZE-02: RANSAC outlier rejection
    m_homography = cv::findHomography(m_gazePoints, m_screenPoints, cv::RANSAC, 5.0);
    return !m_homography.empty();
}

cv::Point2f GazeCalibrator::mapToScreen(const cv::Point2f& gazePt) const {
    if (m_homography.empty()) {
        return gazePt; // Uncalibrated fallback
    }
    
    std::vector<cv::Point2f> src = { gazePt };
    std::vector<cv::Point2f> dst;
    
    cv::perspectiveTransform(src, dst, m_homography);
    return dst.empty() ? gazePt : dst[0];
}

cv::Point2f GazeCalibrator::compensateHeadPose(const cv::Point2f& rawGaze, float headYaw, float headPitch) const {
    // GAZE-03: Head-Pose Compensation
    // Simple linear compensation, similar to Python prototype
    float compX = rawGaze.x - (headYaw * HEAD_COMP_SCALE);
    float compY = rawGaze.y - (headPitch * HEAD_COMP_SCALE);
    
    return cv::Point2f(compX, compY);
}

} // namespace processing
} // namespace irisflow
