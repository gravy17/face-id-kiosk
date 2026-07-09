"""
app/utils/timing.py
Simple utilities for measuring execution time consistently across the app.
"""
import time
from contextlib import contextmanager
from typing import Generator


class Timer:
    """
    Simple wall-clock timer.

    Usage:
        t = Timer()
        t.start()
        ...
        ms = t.elapsed_ms()
    """

    def __init__(self) -> None:
        self._start: float | None = None

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def elapsed_ms(self) -> int:
        if self._start is None:
            return 0
        return int((time.perf_counter() - self._start) * 1000)


@contextmanager
def timed() -> Generator[Timer, None, None]:
    """
    Context manager that yields a running Timer.

    Usage:
        with timed() as t:
            do_work()
        print(t.elapsed_ms())
    """
    t = Timer().start()
    yield t
