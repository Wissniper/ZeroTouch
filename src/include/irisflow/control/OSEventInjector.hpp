#pragma once

#include <opencv2/opencv.hpp>
#include "irisflow/control/GestureController.hpp"

namespace irisflow {
namespace control {

class OSEventInjector {
public:
    OSEventInjector();
    ~OSEventInjector();

    // CORE-03: Move cursor to screen coordinates
    void moveCursor(const cv::Point2f& screenPt);

    // CORE-03: Execute action based on gesture
    void executeGesture(GestureType gesture);

private:
#ifdef __APPLE__
    // macOS specific state
#elif __linux__
    // Linux specific state
#endif
};

} // namespace control
} // namespace irisflow
