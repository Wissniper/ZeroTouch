#include "irisflow/control/GestureController.hpp"
#include <algorithm>
#include <cmath>

namespace irisflow {
namespace control {

GestureController::GestureController() {
    // Initialize history with NONE
    for (size_t i = 0; i < HISTORY_SIZE; ++i) {
        m_gestureHistory.push_back(GestureType::NONE);
    }
}

GestureType GestureController::detectWink(float leftBlinkScore, float rightBlinkScore) {
    // GEST-01: Wink Detection based on differential blink scores
    float diff = std::abs(leftBlinkScore - rightBlinkScore);
    
    if (diff > WINK_THRESHOLD) {
        if (leftBlinkScore > rightBlinkScore) {
            return GestureType::WINK_LEFT;
        } else {
            return GestureType::WINK_RIGHT;
        }
    }
    
    return GestureType::NONE;
}

GestureType GestureController::detectHandGesture(const core::DetectionResult& handDetection) {
    if (!handDetection.isValid() || handDetection.landmarks.empty()) {
        return GestureType::NONE;
    }

    // STAB-05: Normalized Finger Counting
    // Here we'd compute distances relative to bounding box or palm size
    // to make the heuristic scale invariant.
    // For demonstration, we return a mock SCROLL gesture.
    
    float boundingBoxWidth = static_cast<float>(handDetection.roi.width);
    if (boundingBoxWidth == 0) return GestureType::NONE;

    // Simulate counting extended fingers (e.g. index finger extended -> SCROLL)
    int extendedFingers = 1; // Simulated
    
    switch (extendedFingers) {
        case 1: return GestureType::SCROLL;
        case 2: return GestureType::ZOOM;
        case 3: return GestureType::DRAG;
        case 4: return GestureType::SWITCH_DESKTOP;
        default: return GestureType::NONE;
    }
}

GestureType GestureController::filterGesture(GestureType rawGesture) {
    // STAB-02: Temporal Filtering using majority voting
    m_gestureHistory.pop_front();
    m_gestureHistory.push_back(rawGesture);

    std::map<GestureType, int> counts;
    for (auto g : m_gestureHistory) {
        counts[g]++;
    }

    GestureType majorityGesture = GestureType::NONE;
    int maxCount = 0;
    
    for (const auto& pair : counts) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            majorityGesture = pair.first;
        }
    }

    // Require strict majority (e.g., > 50%)
    if (maxCount > HISTORY_SIZE / 2) {
        return majorityGesture;
    }

    return GestureType::NONE;
}

} // namespace control
} // namespace irisflow
