"""Timer utility for measuring elapsed time."""

import time


class Timer:
    """Context manager for measuring elapsed time.

    Uses time.perf_counter() for high-resolution timing.

    Example:
        >>> with Timer() as t:
        ...     result = some_function()
        >>> print(f"Took {t.elapsed_ms:.2f}ms")

    """

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop the timer and calculate elapsed time in milliseconds."""
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
