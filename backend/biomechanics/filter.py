class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) filter.

    Smooths noisy real-time measurements while
    preserving responsiveness.

    If the input measurement becomes None,
    the filtered output also becomes None.
    """

    def __init__(self, alpha=0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                "alpha must be between 0 and 1."
            )

        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        """
        Add a new measurement.

        Returns:
            Smoothed value, or None when the
            current measurement is unavailable.
        """

        # Landmark/joint measurement unavailable.
        if new_value is None:
            self.value = None
            return None

        # First valid measurement.
        if self.value is None:
            self.value = new_value
            return self.value

        # EMA smoothing.
        self.value = (
            self.alpha * new_value
            + (1.0 - self.alpha) * self.value
        )

        return self.value

    def reset(self):
        """
        Reset the filter state.
        """

        self.value = None