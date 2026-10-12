"""HmSO: hibernating multi-swarm optimisation (Kamosi, Hashemi and Meybodi, 2010), re-implemented
from its published description (no reference code available here).

A parent swarm (``parent`` particles, PSO with w = 0.729844, c1 = c2 = 1.49618) searches the whole
space. When the parent's best improves outside every child swarm's radius ``r``, a child swarm of
``child`` particles is created there: the parent particles within ``r`` of that best join it (and are
replaced by random particles in the parent), and the rest are drawn uniformly in a ball of radius ``r``
around the best. Parent particles that enter a child swarm's radius are re-initialised. When two
child swarms' bests are closer than ``r_excl`` the worse swarm is removed. A child swarm hibernates
(spends no evaluations) once all its particles are within ``r_conv`` of its best and its best is not
the overall best; on an environment change every swarm wakes up, re-evaluates its memory and is
re-diversified around its best within ``r_change``. Distances are in units of 1/100 of the domain
span (Moving Peaks units).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from aumo.problems.moving_peaks import MovingPeaksWorld, PeaksConfig

W, C1, C2 = 0.729844, 1.49618, 1.49618


@dataclass(frozen=True, slots=True)
class HmSOConfig:
    parent: int = 5
    child: int = 10
    r: float = 30.0
    r_excl: float = 30.0
    r_conv: float = 1.0
    r_change: float = 1.0
    radius: float = 1.0


def _ball(rng: np.random.Generator, n: int, dim: int, radius: float) -> np.ndarray:
    direction = rng.normal(size=(n, dim))
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    return direction * radius * rng.random((n, 1)) ** (1.0 / dim)


class _Swarm:
    def __init__(self, pos: np.ndarray, fit: np.ndarray) -> None:
        self.pos, self.vel = pos.copy(), np.zeros_like(pos)
        self.pbest, self.pfit = pos.copy(), fit.copy()
        self.asleep = False

    @property
    def best(self) -> tuple[np.ndarray, float]:
        k = int(self.pfit.argmax())
        return self.pbest[k], float(self.pfit[k])


def run_hmso(world_config, seed: int, budget: int | None = None, config: HmSOConfig | None = None) -> dict[str, float]:
    cfg = config or HmSOConfig()
    informed = not isinstance(world_config, PeaksConfig)
    if informed:
        world = world_config
        wc = SimpleNamespace(low=world.low, high=world.high, dim=world.dim, change_frequency=0)
        budget = world.budget if budget is None else budget
    else:
        wc = world_config
        world = MovingPeaksWorld(wc, seed)
    unit = (wc.high - wc.low) / 100.0
    r, r_excl, r_conv, r_change, radius = (v * unit for v in (cfg.r, cfg.r_excl, cfg.r_conv, cfg.r_change, cfg.radius))
    rng = np.random.default_rng([seed, 29])
    d = wc.dim
    clip = lambda a: np.clip(a, wc.low, wc.high)
    parent = _Swarm(rng.uniform(wc.low, wc.high, (cfg.parent, d)), np.zeros(cfg.parent))
    children: list[_Swarm] = []

    def memory() -> np.ndarray:
        return np.concatenate([parent.pbest] + [c.pbest for c in children])

    if informed:
        world.reporter = memory
    parent.pfit = world.evaluate(parent.pos)
    seen = world.changes
    coverage, occupancy = [], []

    def pso_step(s: _Swarm, gbest: np.ndarray) -> np.ndarray:
        r1, r2 = rng.random(s.pos.shape), rng.random(s.pos.shape)
        s.vel = W * s.vel + C1 * r1 * (s.pbest - s.pos) + C2 * r2 * (gbest - s.pos)
        moved = s.pos + s.vel
        s.vel[(moved < wc.low) | (moved > wc.high)] = 0.0
        s.pos = clip(moved)
        f = world.evaluate(s.pos)
        better = f > s.pfit
        s.pbest[better], s.pfit[better] = s.pos[better], f[better]
        return f

    while world.evaluations < budget:
        if informed:
            changed = world.changes != seen
            seen = world.changes
        else:
            bests = [parent.best] + [c.best for c in children]
            k = int(np.argmax([b[1] for b in bests]))
            changed = world.evaluate(bests[k][0][None, :])[0] != bests[k][1]
        if changed:
            parent.pfit = world.evaluate(parent.pbest)
            for c in children:
                c.asleep = False
                center = c.best[0]
                c.pfit = world.evaluate(c.pbest)
                center = c.best[0]
                c.pos = clip(center + _ball(rng, len(c.pos), d, r_change))
                c.vel[:] = 0.0

        # parent swarm
        old_best = parent.best[1]
        pso_step(parent, parent.best[0])
        pb, pf = parent.best
        if pf > old_best and all(np.linalg.norm(c.best[0] - pb) > r for c in children):
            near = np.linalg.norm(parent.pos - pb, axis=1) <= r
            members = np.concatenate([[pb], parent.pos[near]])[: cfg.child]
            extra = clip(pb + _ball(rng, cfg.child - len(members), d, r))
            pos = np.concatenate([members, extra])
            fit = np.concatenate([[pf], world.evaluate(pos[1:])])
            children.append(_Swarm(pos, fit))
            parent.pos[near] = rng.uniform(wc.low, wc.high, (int(near.sum()), d))
            parent.pbest[near] = parent.pos[near]
            parent.pfit[near] = world.evaluate(parent.pos[near]) if near.any() else parent.pfit[near]
            # forget the parent's best so the next improvement must come from elsewhere
            k = int(parent.pfit.argmax())
            parent.pos[k] = parent.pbest[k] = rng.uniform(wc.low, wc.high, d)
            parent.pfit[k] = world.evaluate(parent.pos[k][None, :])[0]

        # child swarms
        overall = max([c.best[1] for c in children], default=-np.inf)
        for c in children:
            if c.asleep:
                continue
            pso_step(c, c.best[0])
            center, value = c.best
            if np.all(np.linalg.norm(c.pos - center, axis=1) < r_conv) and value < overall:
                c.asleep = True

        # exclusion between child swarms, and between the parent and the children
        keep = []
        for c in sorted(children, key=lambda c: -c.best[1]):
            if all(np.linalg.norm(c.best[0] - k.best[0]) >= r_excl for k in keep):
                keep.append(c)
        children = keep
        for c in children:
            inside = np.linalg.norm(parent.pos - c.best[0], axis=1) < r
            if inside.any():
                parent.pos[inside] = rng.uniform(wc.low, wc.high, (int(inside.sum()), d))
                parent.vel[inside] = 0.0
                parent.pbest[inside] = parent.pos[inside]
                parent.pfit[inside] = world.evaluate(parent.pos[inside])

        if world.evaluations >= (wc.change_frequency or 0) and children:
            held = world.measure(np.array([c.best[0] for c in children]), radius)
            coverage.append(held.coverage)
            occupancy.append(held.occupancy)
    if informed:
        world.finish()
    return {"offline_error": getattr(world, "offline_error", float("nan")),
            "coverage": float(np.mean(coverage)) if coverage else float("nan"),
            "occupancy": float(np.mean(occupancy)) if occupancy else float("nan"),
            "evaluations": world.evaluations, "changes": world.changes, "swarms": len(children)}
