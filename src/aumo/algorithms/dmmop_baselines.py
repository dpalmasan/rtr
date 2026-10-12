"""Baselines for the DMMOP benchmark (src/aumo/problems/dmmop.py).

``run_de``: the organisers' reference DE (DE.m): DE/rand/1/bin, N = 100, F = 0.5, CR = 0.5, reflective
bounds, re-evaluation of the population when told of a change. One deliberate fix: DE.m compares
every trial with ``fits(num)`` (a single individual's fitness); here selection is one-to-one, as in DE.
"""
from __future__ import annotations

import numpy as np

from aumo.problems.dmmop import DMMOPWorld, _boundary_check


def run_de(world: DMMOPWorld, seed: int, n: int = 100, f_scale: float = 0.5, cr: float = 0.5) -> DMMOPWorld:
    rng = np.random.default_rng([seed, 23])
    d = world.dim
    pop = rng.uniform(world.low, world.high, (n, d))
    world.reporter = lambda: pop.copy()
    fits = world.evaluate(pop)
    seen = world.changes
    while not world.exhausted:
        picks = np.array([rng.choice(n - 1, 3, replace=False) for _ in range(n)])
        picks += picks >= np.arange(n)[:, None]
        mutant = pop[picks[:, 0]] - f_scale * (pop[picks[:, 1]] - pop[picks[:, 2]])
        cross = rng.random((n, d)) < cr
        cross[np.arange(n), rng.integers(d, size=n)] |= ~cross.any(axis=1)
        trial = _boundary_check(np.where(cross, mutant, pop), world.lower, world.upper)
        trial_fits = world.evaluate(trial)
        if world.changes != seen:
            seen = world.changes
            fits = world.evaluate(pop)
            continue
        better = trial_fits > fits
        pop[better] = trial[better]
        fits[better] = trial_fits[better]
    world.finish()
    return world
