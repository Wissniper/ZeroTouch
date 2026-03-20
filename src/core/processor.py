import time
import math


class _OneEuroFilter:
    """
    One-Euro Filter for a single scalar signal.
    Automatically reduces lag during fast movements and heavy-smooths at rest.

    Parameters
    ----------
    min_cutoff : float
        Minimum low-pass cutoff frequency (Hz). Lower = smoother at rest but more lag.
        Typical range 0.5 – 2.0. Start with 1.0.
    beta : float
        Speed coefficient. Higher = less lag during fast movements, more jitter.
        Typical range 0.0 – 0.5. Start with 0.05.
    d_cutoff : float
        Cutoff for the derivative low-pass filter. Usually leave at 1.0.
    """

    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        """
        x_hat = filtered signal, Estimated Value.
                Hat convention: In statistics and filter theory, a "hat" (circumflex or $\hat{}$) above a variable indicates that it is an estimate of the true (often unobservable or noisy) value.
        dx_hat = filtered derivative
        prev_raw = previous raw input (for derivative estimation)
        prev_t = timestamp of previous sample
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_hat = None      # filtered value
        self._dx_hat = 0.0      # filtered derivative
        self._prev_raw = None   # previous raw input
        self._prev_t = None

    def _alpha(self, freq, cutoff):
        """
        Compute EMA alpha from sampling frequency and cutoff frequency.
        
        te = The time between samples (1/freq).
        tau = The time constant of the filter (1/(2π*cutoff)).
        alpha = 1 / (1 + tau/te) ensures that the filter's response adapts to the sampling rate and cutoff frequency.
        """

        te = 1.0 / freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t=None):
        if t is None:
            t = time.time()

        # Bootstrap on first sample
        if self._prev_t is None:
            self._prev_t = t
            self._prev_raw = x
            self._x_hat = x
            return x

        dt = t - self._prev_t
        if dt <= 1e-6:
            return self._x_hat
        freq = 1.0 / dt
        self._prev_t = t

        # 1. Estimate raw derivative
        dx = (x - self._prev_raw) / dt
        self._prev_raw = x

        # 2. Smooth the derivative (fixed cutoff)
        a_d = self._alpha(freq, self.d_cutoff)
        self._dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat

        # 3. Adaptive signal cutoff: higher speed → higher cutoff → less lag
        cutoff = self.min_cutoff + self.beta * abs(self._dx_hat)

        # 4. Filter the signal
        a = self._alpha(freq, cutoff)
        self._x_hat = a * x + (1.0 - a) * self._x_hat

        return self._x_hat


class GazeProcessor:
    def __init__(self, min_cutoff=0.5, beta=0.05, velocity_scale=2.0):
        """
        Smooths gaze coordinates with per-axis One-Euro filters.

        Parameters
        ----------
        min_cutoff : float
            Base smoothing (lower = smoother but more lag). Default 1.0.
        beta : float
            Responsiveness during fast movement. Default 0.05.
        velocity_scale : float
            Reserved for future velocity-based cursor acceleration.
        """
        self.velocity_scale = velocity_scale
        self._fx = _OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._fy = _OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def process(self, raw_x, raw_y):
        t = time.time()
        return self._fx.filter(raw_x, t), self._fy.filter(raw_y, t)
