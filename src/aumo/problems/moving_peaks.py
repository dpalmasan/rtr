"""Moving Peaks world (Branke's cone peaks, maximization) for studying peak coverage by a team.

Time runs in rounds. Every ``period`` rounds the world *may* change (with ``change_probability``):
each peak shifts by ``move_severity`` (direction correlated with its last shift by ``correlation``),
and its height and width drift. A peak is *visible* when no other cone covers its top, i.e. it is
a local optimum; hidden peaks cannot be found and do not count towards coverage.

``measure`` returns, for a set of agent positions:
  coverage   fraction of visible peaks with at least one agent within ``radius`` (diversity),
  occupancy  fraction of agents within ``radius`` of some visible peak (time spent on peaks),
  error      tallest visible peak minus the best agent's current value (classic tracking error).

With ``change_frequency`` > 0 the world instead changes every ``change_frequency`` evaluations made
through ``evaluate`` (the standard protocol; ``maybe_change`` is then a no-op), and ``offline_error``
is the standard offline error: the mean over all evaluations of the global optimum minus the best
value evaluated since the last change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PeaksConfig:
    dim: int = 2
    peaks: int = 10
    low: float = 0.0
    high: float = 100.0
    height: tuple[float, float] = (30.0, 70.0)
    width: tuple[float, float] = (1.0, 12.0)
    move_severity: float = 1.0
    height_severity: float = 7.0
    width_severity: float = 1.0
    correlation: float = 0.0
    period: int = 20
    change_probability: float = 0.5
    change_frequency: int = 0
    # Starting height of every peak (Branke's standard 50 in scenario 2); None = uniform in ``height``.
    initial_height: float | None = None


def scenario2(dim: int = 5, peaks: int = 10, change_frequency: int = 5000, **overrides) -> "PeaksConfig":
    """Branke's Moving Peaks scenario 2: cone peaks, heights [30, 70] starting at 50, widths [1, 12],
    shift 1, height severity 7, width severity 1, correlation 0, a change every 5000 evaluations."""
    return PeaksConfig(dim=dim, peaks=peaks, change_frequency=change_frequency, initial_height=50.0, **overrides)


@dataclass(frozen=True, slots=True)
class PeaksMeasure:
    coverage: float
    occupancy: float
    error: float


class MovingPeaksWorld:
    def __init__(self, config: PeaksConfig, seed: int) -> None:
        c = config
        self.config = c
        self.rng = np.random.default_rng([seed, 3])
        self.centers = self.rng.uniform(c.low, c.high, (c.peaks, c.dim))
        self.heights = self.rng.uniform(*c.height, c.peaks)
        if c.initial_height is not None:
            self.heights[:] = c.initial_height
        self.widths = self.rng.uniform(*c.width, c.peaks)
        self.shifts = np.zeros((c.peaks, c.dim))
        self.changes = 0
        self.evaluations = 0
        self.best_since_change = -np.inf
        self.offline_sum = 0.0

    def values(self, x: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(x[:, None, :] - self.centers[None, :, :], axis=2)
        return np.max(self.heights - self.widths * distance, axis=1)

    def maybe_change(self, round_: int) -> bool:
        c = self.config
        if c.change_frequency or round_ == 0 or round_ % c.period or self.rng.random() >= c.change_probability:
            return False
        self._change()
        return True

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Counted evaluation: triggers evaluation-based changes and accumulates the offline error."""
        freq = self.config.change_frequency
        out = np.empty(len(x))
        start = 0
        while start < len(x):
            if freq and self.evaluations and self.evaluations % freq == 0 and self._changed_at != self.evaluations:
                self._change()
                self._changed_at = self.evaluations
                self.best_since_change = -np.inf
            take = len(x) - start if not freq else min(len(x) - start, freq - self.evaluations % freq)
            values = self.values(x[start : start + take])
            running = np.maximum.accumulate(np.concatenate(([self.best_since_change], values)))[1:]
            self.offline_sum += float(np.sum(self.heights.max() - running))
            self.best_since_change = running[-1]
            self.evaluations += take
            out[start : start + take] = values
            start += take
        return out

    @property
    def offline_error(self) -> float:
        return self.offline_sum / max(self.evaluations, 1)

    _changed_at = -1

    def _change(self) -> None:
        c = self.config
        direction = self.rng.normal(size=self.centers.shape)
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        combined = (1.0 - c.correlation) * direction + c.correlation * self.shifts
        combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12)
        self.shifts = c.move_severity * combined
        moved = self.centers + self.shifts
        moved = np.where(moved < c.low, 2 * c.low - moved, moved)
        moved = np.where(moved > c.high, 2 * c.high - moved, moved)
        self.centers = np.clip(moved, c.low, c.high)
        self.heights = np.clip(self.heights + c.height_severity * self.rng.normal(size=c.peaks), *c.height)
        self.widths = np.clip(self.widths + c.width_severity * self.rng.normal(size=c.peaks), *c.width)
        self.changes += 1

    def visible(self) -> np.ndarray:
        return self.values(self.centers) >= self.heights - 1e-9

    def measure(self, x: np.ndarray, radius: float) -> PeaksMeasure:
        visible = self.visible()
        near = np.linalg.norm(x[:, None, :] - self.centers[None, visible, :], axis=2) <= radius
        return PeaksMeasure(
            coverage=float(near.any(axis=0).mean()),
            occupancy=float(near.any(axis=1).mean()),
            error=float(self.heights[visible].max() - self.values(x).max()),
        )
