"""Holm's step-down adjustment (shared by the scorers)."""
from __future__ import annotations


def holm(p_values: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted
