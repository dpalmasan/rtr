"""mQSO: multi-quantum-swarm optimisation with exclusion and anti-convergence (Blackwell & Branke 2006).

``swarms`` swarms of ``neutral`` PSO particles (constriction chi = 0.7298, c1 = c2 = 2.05) and
``quantum`` particles resampled each iteration uniformly in a ball of radius ``r_cloud`` around the
swarm attractor. Exclusion: when two attractors are closer than r_excl = span / (2 * swarms ** (1 / d)),
the swarm with the worse attractor is reinitialised. Anti-convergence: when every swarm has converged
(all neutral particles within r_conv of the attractor), the worst swarm is reinitialised. Change
detection re-evaluates the best attractor every iteration; on a change all personal bests are
re-evaluated. Coverage is measured on the swarm attractors (what the swarms hold).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from aumo.problems.moving_peaks import MovingPeaksWorld, PeaksConfig

CHI, C1, C2 = 0.729843788, 2.05, 2.05


@dataclass(frozen=True, slots=True)
class MQSOConfig:
    swarms: int = 10
    neutral: int = 5
    quantum: int = 5
    r_cloud: float = 0.5
    exclusion: float | None = None
    convergence: float | None = None
    radius: float = 1.0


def _ball(rng: np.random.Generator, shape: tuple[int, int], dim: int, radius: float) -> np.ndarray:
    direction = rng.normal(size=(*shape, dim))
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    return direction * radius * rng.random((*shape, 1)) ** (1.0 / dim)


def run_mqso(world_config, seed: int, budget: int | None = None, config: MQSOConfig | None = None) -> dict[str, float]:
    """``world_config``: a PeaksConfig (Moving Peaks, change detection by re-evaluating the best
    attractor) or a world object such as DMMOPWorld (told of changes; reports all personal bests).
    Distances are in units of 1/100 of the domain span."""
    cfg = config or MQSOConfig()
    informed = not isinstance(world_config, PeaksConfig)
    if informed:
        world = world_config
        wc = SimpleNamespace(low=world.low, high=world.high, dim=world.dim, change_frequency=0)
        budget = world.budget if budget is None else budget
    else:
        wc = world_config
        world = MovingPeaksWorld(wc, seed)
    unit = (wc.high - wc.low) / 100.0
    r_cloud = cfg.r_cloud * unit
    radius = cfg.radius * unit
    rng = np.random.default_rng([seed, 17])
    m, k, d = cfg.swarms, cfg.neutral, wc.dim
    span = wc.high - wc.low
    r_excl = cfg.exclusion if cfg.exclusion is not None else span / (2 * m ** (1 / d))
    r_conv = cfg.convergence if cfg.convergence is not None else r_excl

    pos = rng.uniform(wc.low, wc.high, (m, k, d))
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    if informed:
        world.reporter = lambda: pbest.reshape(-1, d).copy()
    seen = world.changes
    pfit = world.evaluate(pos.reshape(-1, d)).reshape(m, k)
    gpos = pbest[np.arange(m), pfit.argmax(axis=1)].copy()
    gfit = pfit.max(axis=1)

    def reinit(s: int) -> None:
        pos[s] = rng.uniform(wc.low, wc.high, (k, d))
        vel[s] = 0.0
        pbest[s] = pos[s]
        pfit[s] = world.evaluate(pos[s])
        gpos[s] = pbest[s, pfit[s].argmax()]
        gfit[s] = pfit[s].max()

    coverage, memory_coverage, occupancy = [], [], []
    while world.evaluations < budget:
        if informed:
            changed = world.changes != seen
            seen = world.changes
        else:
            best = int(gfit.argmax())
            changed = world.evaluate(gpos[best : best + 1])[0] != gfit[best]
        if changed:
            pfit[:] = world.evaluate(pbest.reshape(-1, d)).reshape(m, k)
            gpos[:] = pbest[np.arange(m), pfit.argmax(axis=1)]
            gfit[:] = pfit.max(axis=1)

        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = CHI * (vel + C1 * r1 * (pbest - pos) + C2 * r2 * (gpos[:, None, :] - pos))
        moved = pos + vel
        clipped = (moved < wc.low) | (moved > wc.high)
        pos = np.clip(moved, wc.low, wc.high)
        vel[clipped] = 0.0
        f = world.evaluate(pos.reshape(-1, d)).reshape(m, k)
        better = f > pfit
        pbest[better] = pos[better]
        pfit[better] = f[better]
        top = pfit.argmax(axis=1)
        improved = pfit[np.arange(m), top] > gfit
        gpos[improved] = pbest[np.arange(m), top][improved]
        gfit[improved] = pfit[np.arange(m), top][improved]

        if cfg.quantum:
            cloud = np.clip(gpos[:, None, :] + _ball(rng, (m, cfg.quantum), d, r_cloud), wc.low, wc.high)
            fq = world.evaluate(cloud.reshape(-1, d)).reshape(m, cfg.quantum)
            qbest = fq.argmax(axis=1)
            won = fq[np.arange(m), qbest] > gfit
            gpos[won] = cloud[np.arange(m), qbest][won]
            gfit[won] = fq[np.arange(m), qbest][won]

        for a in range(m):
            for b in range(a + 1, m):
                if np.linalg.norm(gpos[a] - gpos[b]) < r_excl:
                    reinit(a if gfit[a] < gfit[b] else b)
        spread = np.linalg.norm(pos - gpos[:, None, :], axis=2).max(axis=1)
        if np.all(spread < r_conv):
            reinit(int(gfit.argmin()))

        if world.evaluations >= (wc.change_frequency or 0):
            held = world.measure(gpos, radius)
            coverage.append(held.coverage)
            occupancy.append(held.occupancy)
            memory_coverage.append(world.measure(pbest.reshape(-1, d), radius).coverage)
    if informed:
        world.finish()
    return {"offline_error": getattr(world, "offline_error", float("nan")), "coverage": float(np.mean(coverage)), "occupancy": float(np.mean(occupancy)),
            "memory_coverage": float(np.mean(memory_coverage)), "evaluations": world.evaluations, "changes": world.changes}
