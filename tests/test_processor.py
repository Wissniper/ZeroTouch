"""Tests for the One-Euro filter and GazeProcessor.

Validates mathematical properties:
- First sample pass-through (bootstrap)
- Smoothing reduces jitter (variance shrinks)
- Low-latency tracking of fast ramps
- Duplicate timestamps are handled safely
- Reset clears state
"""

import math
import pytest
from src.core.processor import _OneEuroFilter, GazeProcessor


# ── _OneEuroFilter unit tests ────────────────────────────────────────


class TestOneEuroFilter:
    def test_first_sample_passthrough(self):
        """First input should be returned unchanged (bootstrap)."""
        f = _OneEuroFilter(min_cutoff=1.0, beta=0.05)
        assert f.filter(42.0, t=0.0) == 42.0

    def test_smoothing_reduces_jitter(self):
        """Feeding a noisy signal should produce output with lower variance."""
        f = _OneEuroFilter(min_cutoff=1.0, beta=0.0)  # beta=0 → pure smoothing

        # Noisy signal: constant 100 with +/-5 jitter
        import random
        random.seed(42)
        raw = [100 + random.uniform(-5, 5) for _ in range(100)]

        smoothed = []
        for i, v in enumerate(raw):
            smoothed.append(f.filter(v, t=i * 0.033))  # ~30 FPS

        raw_var = sum((x - 100) ** 2 for x in raw) / len(raw)
        smooth_var = sum((x - 100) ** 2 for x in smoothed) / len(smoothed)

        assert smooth_var < raw_var, "Smoothed variance should be less than raw"

    def test_tracks_fast_ramp(self):
        """With high beta, filter should closely follow a fast ramp."""
        f = _OneEuroFilter(min_cutoff=1.0, beta=10.0)  # high beta → responsive

        # Ramp from 0 to 100 over 30 samples
        for i in range(30):
            val = i * (100 / 29)
            out = f.filter(val, t=i * 0.033)

        # After the ramp, output should be close to the final value
        assert abs(out - 100) < 5.0, f"Expected ~100, got {out}"

    def test_constant_signal_converges(self):
        """A constant input should converge to that value exactly."""
        f = _OneEuroFilter(min_cutoff=1.0, beta=0.05)
        for i in range(50):
            out = f.filter(50.0, t=i * 0.033)
        assert abs(out - 50.0) < 0.01

    def test_duplicate_timestamp_returns_cached(self):
        """If dt ≈ 0, should return the previously filtered value, not crash."""
        f = _OneEuroFilter()
        f.filter(10.0, t=1.0)
        result = f.filter(20.0, t=1.0)  # same timestamp
        assert result == 10.0  # should return cached x_hat

    def test_reset_clears_state(self):
        """After reset, the next sample should act as bootstrap again."""
        f = _OneEuroFilter()
        f.filter(100.0, t=0.0)
        f.filter(200.0, t=0.033)
        f.reset()

        # After reset, first sample should pass through
        assert f.filter(42.0, t=1.0) == 42.0

    def test_alpha_range(self):
        """Alpha should always be in (0, 1]."""
        f = _OneEuroFilter()
        for freq in [1, 10, 30, 60, 120]:
            for cutoff in [0.1, 0.5, 1.0, 5.0, 50.0]:
                a = f._alpha(freq, cutoff)
                assert 0 < a <= 1.0, f"alpha={a} out of range for freq={freq}, cutoff={cutoff}"


# ── GazeProcessor tests ─────────────────────────────────────────────


class TestGazeProcessor:
    def test_process_returns_tuple(self):
        gp = GazeProcessor(min_cutoff=1.0, beta=0.05)
        sx, sy = gp.process(100.0, 200.0, t=0.0)
        assert isinstance(sx, float)
        assert isinstance(sy, float)

    def test_axes_independent(self):
        """X and Y filters should operate independently."""
        gp = GazeProcessor(min_cutoff=1.0, beta=0.0)

        # Feed constant X, ramping Y
        for i in range(30):
            sx, sy = gp.process(50.0, float(i * 10), t=i * 0.033)

        assert abs(sx - 50.0) < 1.0, "X should stay near 50"
        assert sy > 200, "Y should have tracked the ramp"

    def test_reset(self):
        gp = GazeProcessor()
        gp.process(100.0, 200.0, t=0.0)
        gp.reset()
        sx, sy = gp.process(0.0, 0.0, t=1.0)
        # After reset, should bootstrap to new values
        assert sx == 0.0
        assert sy == 0.0

    def test_high_frequency_stability(self):
        """Filter should remain stable even at very high sampling rates."""
        gp = GazeProcessor(min_cutoff=1.0, beta=0.05)
        prev = None
        for i in range(1000):
            sx, sy = gp.process(500.0, 500.0, t=i * 0.001)  # 1000 Hz
            if prev is not None:
                assert abs(sx - prev[0]) < 50, "X shouldn't oscillate"
            prev = (sx, sy)
