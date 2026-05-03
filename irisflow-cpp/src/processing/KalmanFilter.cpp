#include "irisflow/processing/KalmanFilter.hpp"

namespace irisflow {
namespace processing {

KalmanFilter2D::KalmanFilter2D() 
    : m_kf(m_stateSize, m_measSize, m_contrSize, CV_32F) 
{
    // Transition State Matrix A
    // [1 0 dT 0]
    // [0 1 0 dT]
    // [0 0 1  0]
    // [0 0 0  1]
    cv::setIdentity(m_kf.transitionMatrix);
    m_kf.transitionMatrix.at<float>(0, 2) = 1.0f; // dT
    m_kf.transitionMatrix.at<float>(1, 3) = 1.0f; // dT

    // Measurement Matrix H
    // [1 0 0 0]
    // [0 1 0 0]
    m_kf.measurementMatrix = cv::Mat::zeros(m_measSize, m_stateSize, CV_32F);
    m_kf.measurementMatrix.at<float>(0, 0) = 1.0f;
    m_kf.measurementMatrix.at<float>(1, 1) = 1.0f;

    // Process Noise Covariance Matrix Q
    cv::setIdentity(m_kf.processNoiseCov, cv::Scalar::all(1e-4));
    
    // Measurement Noise Covariance Matrix R
    cv::setIdentity(m_kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    
    // Error Covariance Matrix P
    cv::setIdentity(m_kf.errorCovPost, cv::Scalar::all(0.1));
}

void KalmanFilter2D::init(float x, float y) {
    m_kf.statePre.at<float>(0) = x;
    m_kf.statePre.at<float>(1) = y;
    m_kf.statePre.at<float>(2) = 0.0f;
    m_kf.statePre.at<float>(3) = 0.0f;

    m_kf.statePost.at<float>(0) = x;
    m_kf.statePost.at<float>(1) = y;
    m_kf.statePost.at<float>(2) = 0.0f;
    m_kf.statePost.at<float>(3) = 0.0f;

    m_initialized = true;
}

cv::Point2f KalmanFilter2D::predict() {
    if (!m_initialized) return cv::Point2f(0.0f, 0.0f);
    
    cv::Mat prediction = m_kf.predict();
    return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
}

cv::Point2f KalmanFilter2D::update(float x, float y) {
    if (!m_initialized) {
        init(x, y);
        return cv::Point2f(x, y);
    }

    // First predict
    m_kf.predict();

    // Then update with measurement
    cv::Mat measurement = cv::Mat::zeros(m_measSize, 1, CV_32F);
    measurement.at<float>(0) = x;
    measurement.at<float>(1) = y;
    
    cv::Mat estimated = m_kf.correct(measurement);
    
    return cv::Point2f(estimated.at<float>(0), estimated.at<float>(1));
}

} // namespace processing
} // namespace irisflow
