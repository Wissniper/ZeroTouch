#include "irisflow/camera/WebcamCapture.hpp"
#include "irisflow/inference/ONNXEngine.hpp"
#include "irisflow/processing/KalmanFilter.hpp"
#include "irisflow/processing/GazeCalibrator.hpp"
#include "irisflow/control/GestureController.hpp"
#include "irisflow/control/OSEventInjector.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "IrisFlow C++ Rewrite Initialized" << std::endl;
    
    // Initialize camera
    irisflow::camera::WebcamCapture camera(0);
    if (!camera.start()) {
        std::cerr << "Failed to start camera capture." << std::endl;
        return -1;
    }

    // Initialize inference engine
    irisflow::inference::ONNXEngine inferenceEngine;
    if (!inferenceEngine.loadModel("models/dummy.onnx")) {
        std::cerr << "Warning: Could not load model. Continuing without inference." << std::endl;
    }

    // Initialize processors & controllers
    irisflow::processing::KalmanFilter2D gazeFilter;
    irisflow::processing::GazeCalibrator calibrator;
    irisflow::control::GestureController gestureController;
    irisflow::control::OSEventInjector osInjector;
    
    // Simulate some calibration points for testing
    calibrator.addCalibrationPoint(cv::Point2f(0.3f, 0.3f), cv::Point2f(100.f, 100.f));
    calibrator.addCalibrationPoint(cv::Point2f(0.7f, 0.3f), cv::Point2f(1820.f, 100.f));
    calibrator.addCalibrationPoint(cv::Point2f(0.3f, 0.7f), cv::Point2f(100.f, 980.f));
    calibrator.addCalibrationPoint(cv::Point2f(0.7f, 0.7f), cv::Point2f(1820.f, 980.f));
    calibrator.computeHomography();

    std::cout << "Camera capture started. Reading frames..." << std::endl;
    cv::Mat frame;
    int frameCount = 0;
    
    auto startTime = std::chrono::steady_clock::now();
    
    // Process frames (demo loop)
    while (frameCount < 60) {
        if (camera.getFrame(frame)) {
            frameCount++;
            
            cv::Rect roi(frame.cols / 4, frame.rows / 4, frame.cols / 2, frame.rows / 2);
            irisflow::core::DetectionResult result;
            
            if (inferenceEngine.infer(frame, roi, result)) {
                if (result.isValid() && !result.landmarks.empty()) {
                    // Get raw gaze from model
                    cv::Point2f rawGaze(result.landmarks[0].x, result.landmarks[0].y);
                    
                    // GAZE-03: Head Pose Compensation
                    cv::Point2f compGaze = calibrator.compensateHeadPose(rawGaze, 0.1f, -0.05f);
                    
                    // STAB-03: Kalman Filter Stabilization
                    cv::Point2f smoothedGaze = gazeFilter.update(compGaze.x, compGaze.y);
                    
                    // GAZE-02: Homography Mapping
                    cv::Point2f screenGaze = calibrator.mapToScreen(smoothedGaze);

                    // CORE-03: OS Cursor Move
                    osInjector.moveCursor(screenGaze);

                    // GEST-02 & STAB-05: Hand Gesture Detection
                    irisflow::control::GestureType rawGesture = gestureController.detectHandGesture(result);
                    
                    // STAB-02: Temporal Filtering
                    irisflow::control::GestureType finalGesture = gestureController.filterGesture(rawGesture);

                    // CORE-03: OS Gesture Injection
                    if (finalGesture != irisflow::control::GestureType::NONE) {
                        osInjector.executeGesture(finalGesture);
                    }
                }
            }
        }
    }
    
    auto endTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    
    std::cout << "Captured " << frameCount << " frames in " << elapsed.count() << " seconds." << std::endl;
    std::cout << "Average FPS: " << frameCount / elapsed.count() << std::endl;
    
    camera.stop();
    return 0;
}
