#include "irisflow/camera/WebcamCapture.hpp"
#include "irisflow/processing/KalmanFilter.hpp"
#include "irisflow/processing/GazeCalibrator.hpp"
#include "irisflow/control/GestureController.hpp"
#include "irisflow/control/OSEventInjector.hpp"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "IrisFlow starting..." << std::endl;

    irisflow::camera::WebcamCapture camera(0);
    if (!camera.start()) {
        std::cerr << "Failed to open camera." << std::endl;
        return 1;
    }

    irisflow::processing::KalmanFilter2D gazeFilter;
    irisflow::processing::GazeCalibrator calibrator;
    irisflow::control::GestureController gestureController;
    irisflow::control::OSEventInjector osInjector;

    calibrator.addCalibrationPoint(cv::Point2f(0.1f, 0.1f), cv::Point2f(0.f,    0.f));
    calibrator.addCalibrationPoint(cv::Point2f(0.9f, 0.1f), cv::Point2f(1920.f, 0.f));
    calibrator.addCalibrationPoint(cv::Point2f(0.1f, 0.9f), cv::Point2f(0.f,    1080.f));
    calibrator.addCalibrationPoint(cv::Point2f(0.9f, 0.9f), cv::Point2f(1920.f, 1080.f));
    calibrator.computeHomography();

    cv::Mat frame;
    int frameCount = 0;
    const int TARGET_FRAMES = 60;

    auto startTime = std::chrono::steady_clock::now();

    while (frameCount < TARGET_FRAMES) {
        if (!camera.getFrame(frame)) {
            continue;
        }
        frameCount++;

        // Use the normalised frame centre as a synthetic gaze point.
        // Replace this with real landmark output once a model is wired in.
        cv::Point2f rawGaze(
            static_cast<float>(frame.cols) / 2.0f / static_cast<float>(frame.cols),
            static_cast<float>(frame.rows) / 2.0f / static_cast<float>(frame.rows)
        );

        cv::Point2f smoothed  = gazeFilter.update(rawGaze.x, rawGaze.y);
        cv::Point2f screenPt  = calibrator.mapToScreen(smoothed);

        osInjector.moveCursor(screenPt);

        // Wink detection requires real per-eye blink scores from a model.
        // Gesture detection requires real landmark data from a model.
        // Both are stubbed here until inference is added.
    }

    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - startTime).count();

    std::cout << "Processed " << frameCount << " frames in "
              << elapsed << "s  ("
              << frameCount / elapsed << " fps)" << std::endl;

    camera.stop();
    return 0;
}
