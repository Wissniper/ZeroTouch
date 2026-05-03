#pragma once

#include "irisflow/core/Types.hpp"
#include <deque>
#include <map>

namespace irisflow {
namespace control {

enum class GestureType {
    NONE,
    WINK_LEFT,
    WINK_RIGHT,
    SCROLL,
    ZOOM,
    DRAG,
    SWITCH_DESKTOP
};

class GestureController {
public:
    GestureController();
    ~GestureController() = default;

    // GEST-01: Wink Detection (Differential blink score)
    GestureType detectWink(float leftBlinkScore, float rightBlinkScore);

    // GEST-02: Multi-finger gesture recognition
    // STAB-05: Normalized Finger Counting
    GestureType detectHandGesture(const core::DetectionResult& handDetection);

    // STAB-02: Temporal Filtering: 3-5 frame majority voting
    GestureType filterGesture(GestureType rawGesture);

private:
    // Temporal filtering state
    std::deque<GestureType> m_gestureHistory;
    const size_t HISTORY_SIZE = 5;

    // Wink detection thresholds
    const float WINK_THRESHOLD = 0.06f;

    // Cooldown state to prevent double-triggers
    std::map<GestureType, uint64_t> m_lastTriggerTime;
    const uint64_t COOLDOWN_MS = 500;
};

} // namespace control
} // namespace irisflow
