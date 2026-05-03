#include "irisflow/camera/WebcamCapture.hpp"
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

    std::cout << "Camera capture started. Reading frames..." << std::endl;
    cv::Mat frame;
    int frameCount = 0;
    
    auto startTime = std::chrono::steady_clock::now();
    
    // Process frames (demo loop, e.g., 60 frames)
    while (frameCount < 60) {
        if (camera.getFrame(frame)) {
            frameCount++;
        }
    }
    
    auto endTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    
    std::cout << "Captured " << frameCount << " frames in " << elapsed.count() << " seconds." << std::endl;
    std::cout << "Average FPS: " << frameCount / elapsed.count() << std::endl;
    
    camera.stop();
    return 0;
}
