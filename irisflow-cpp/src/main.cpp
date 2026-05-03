#include "irisflow/camera/WebcamCapture.hpp"
#include "irisflow/inference/ONNXEngine.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "IrisFlow C++ Rewrite Initialized" << std::endl;
    
    // Initialize camera
    irisflow::camera::WebcamCapture camera(0); // device 0
    if (!camera.start()) {
        std::cerr << "Failed to start camera capture." << std::endl;
        return -1;
    }

    // Initialize inference engine
    irisflow::inference::ONNXEngine inferenceEngine;
    if (!inferenceEngine.loadModel("models/dummy.onnx")) {
        std::cerr << "Warning: Could not load model. Continuing without inference." << std::endl;
    }

    std::cout << "Camera capture started. Reading frames..." << std::endl;
    cv::Mat frame;
    int frameCount = 0;
    
    auto startTime = std::chrono::steady_clock::now();
    
    // Process frames (demo loop, e.g., 60 frames)
    while (frameCount < 60) {
        if (camera.getFrame(frame)) {
            frameCount++;
            
            // STAB-04: ROI Tracking
            // We use a fixed ROI for demonstration, normally tracked between frames
            cv::Rect roi(frame.cols / 4, frame.rows / 4, frame.cols / 2, frame.rows / 2);
            irisflow::core::DetectionResult result;
            
            // GAZE-01: ONNX Inference
            if (inferenceEngine.infer(frame, roi, result)) {
                // STAB-01: Confidence Gating
                if (result.isValid()) {
                    // std::cout << "Valid detection with confidence: " << result.confidence << std::endl;
                } else {
                    // std::cout << "Detection ignored due to low confidence." << std::endl;
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
