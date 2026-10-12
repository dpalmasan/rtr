from __future__ import annotations

import sys
import time
from typing import TextIO


def _clock(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class ProgressBar:
    """Dependency-free progress bar with elapsed time and ETA, redrawn in place on one line."""

    def __init__(self, total: int, prefix: str = "", width: int = 30, stream: TextIO = sys.stderr, min_interval: float = 0.5) -> None:
        self.total = max(1, total)
        self.prefix = prefix
        self.width = width
        self.stream = stream
        self.min_interval = min_interval
        self.done = 0
        self._started = time.monotonic()
        self._last_draw = float("-inf")
        self._draw()

    def update(self, steps: int = 1) -> None:
        self.done = min(self.total, self.done + steps)
        now = time.monotonic()
        if self.done >= self.total or now - self._last_draw >= self.min_interval:
            self._draw()

    def _draw(self) -> None:
        self._last_draw = time.monotonic()
        elapsed = self._last_draw - self._started
        fraction = self.done / self.total
        filled = int(self.width * fraction)
        eta = elapsed * (self.total - self.done) / self.done if self.done else 0.0
        eta_text = _clock(eta) if self.done else "--:--"
        line = f"\r{self.prefix} [{'#' * filled}{'.' * (self.width - filled)}] {self.done}/{self.total} {fraction:6.1%} elapsed {_clock(elapsed)} eta {eta_text}"
        self.stream.write(line)
        if self.done >= self.total:
            self.stream.write("\n")
        self.stream.flush()
