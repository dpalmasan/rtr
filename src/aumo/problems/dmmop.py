"""Python port of the CEC 2022/2023/2024 benchmark "Seeking Multiple Optima in Dynamic Environments".

Ported from the organisers' MATLAB code (MiLabHITSZ/dmmops: problem/DMMOP.m, dynamic_change.m,
get_info.m, Sub_f/*.m, boundary_check.m) with the same rules, constants and data files
(data/dmmop/*.mat). Reference: W. Luo, X. Lin, C. Li, S. Yang, Y. Shi, "Benchmark Functions for CEC 2022
Competition on Seeking Multiple Optima in Dynamic Environments", arXiv:2201.00523.

The one deliberate difference: the problem's own random stream (peak repositioning, rotations, C7/C8
peak counts) is numpy's, seeded with ``dynamics_seed`` (0 like the original), not MATLAB's mt19937ar,
so the dynamics follow the same rules and distributions but are not bit-identical to the MATLAB
instances. Methods compared head to head must all run on this port. With ``instances=DIR`` the world
instead replays the official environments dumped from MATLAB by scripts/matlab/dump_dmmop_instances.m
(DIR/P01.mat ... P24.mat), and checks every optimum value against the dump.

Protocol: 24 problems, maximisation on [-5, 5]^D, 60 environments of 5000 * D evaluations each. The
environment changes when its evaluations are used up, and the algorithm is told (``changes`` counts
the changes). At every change (and at the end) the world asks the algorithm for its current
population through ``reporter`` and records it. The peak ratio counts, per environment, the global
optima that have a reported solution within 0.05 whose value is within 1e-3 / 1e-4 / 1e-5 of the optimum.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.io as sio

DATA = Path(__file__).resolve().parents[3] / "data" / "dmmop"
FUNCTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 6, 7, 8]
CHANGES = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 1, 1, 1, 1]
DIMS = [5] * 16 + [10] * 8
ACCURACY = (1e-3, 1e-4, 1e-5)
FOUND_DISTANCE = 0.05


def _boundary_check(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    length = ub - lb
    x = np.where(x < lb, lb + np.fmod(lb - x, length), x)
    return np.where(x > ub, ub - np.fmod(x - ub, length), x)


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=1)


def _rastrigin(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10, axis=1)


def _griewank(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    return np.sum(x**2 / 4000, axis=1) - np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))), axis=1) + 1


_K = np.arange(21)
_C1, _C2 = 0.5**_K, 2 * np.pi * 3.0**_K


def _weierstrass(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    t = np.sum(_C1 * np.cos(_C2 * (x[:, :, None] + 0.5)), axis=2)
    return np.sum(t, axis=1) - d * np.sum(_C1 * np.cos(_C2 * 0.5))


def _f8f2(x: np.ndarray) -> np.ndarray:
    f2 = 100 * (x[:, 0] ** 2 - x[:, 1]) ** 2 + (1 - x[:, 0]) ** 2
    return 1 + f2**2 / 4000 - np.cos(f2)


def _ef8f2(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    total = np.zeros(len(x))
    for i in range(d - 1):
        total += _f8f2(x[:, [i, i + 1]] + 1)
    return total + _f8f2(x[:, [d - 1, 0]] + 1)


_SUBS = {
    5: ([_griewank, _griewank, _weierstrass, _weierstrass, _sphere, _sphere], [1] * 6, [1, 1, 8, 8, 1 / 5, 1 / 5]),
    6: ([_rastrigin, _rastrigin, _weierstrass, _weierstrass, _griewank, _griewank, _sphere, _sphere], [1] * 8,
        [1, 1, 10, 10, 1 / 10, 1 / 10, 1 / 7, 1 / 7]),
    7: ([_ef8f2, _ef8f2, _weierstrass, _weierstrass, _griewank, _griewank], [1, 1, 2, 2, 2, 2], [1 / 4, 1 / 10, 2, 1, 2, 5]),
    8: ([_rastrigin, _rastrigin, _ef8f2, _ef8f2, _weierstrass, _weierstrass, _griewank, _griewank], [1, 1, 1, 1, 1, 2, 2, 2],
        [4, 1, 4, 1, 1 / 10, 1 / 5, 1 / 10, 1 / 40]),
}


def _dynamic_change(rng: np.random.Generator, u: np.ndarray, change_type: int, u_min: float, u_max: float,
                    severity: float, count: int, phi: np.ndarray) -> np.ndarray:
    alpha, alpha_max, chaos, period, noise = 0.04, 0.1, 3.67, 12, 0.8
    u = np.atleast_1d(np.asarray(u, dtype=float))
    span = u_max - u_min
    if change_type == 1:
        r = 2 * rng.random(len(u)) - 1
        return np.clip(u + alpha * span * r * severity, u_min, u_max)
    if change_type == 2:
        r = 2 * rng.random(len(u)) - 1
        return np.clip(u + span * (alpha * np.sign(r) + (alpha_max - alpha) * r) * severity, u_min, u_max)
    if change_type == 3:
        return np.clip(u + rng.standard_normal(len(u)) * severity, u_min, u_max)
    if change_type == 4:
        v = (u - u_min) / span
        v = chaos * v * (1 - v)
        return np.clip(u_min + v * span, u_min, u_max)
    if change_type in (5, 6):
        out = u_min + span * (np.sin(2 * np.pi * count / period + phi[: len(u)]) + 1) / 2
        if change_type == 6:
            out = np.clip(out + rng.standard_normal(len(u)) * noise, u_min, u_max)
        return out
    raise ValueError(f"change_type {change_type}")


@dataclass(frozen=True, slots=True)
class DMMOPMeasure:
    coverage: float
    occupancy: float
    error: float


class DMMOPWorld:
    """One DMMOP problem (1..24). ``evaluate`` is the counted, environment-aware evaluation."""

    def __init__(self, problem: int, dynamics_seed: int = 0, max_env: int = 60, instances: str | Path | None = None) -> None:
        if not 1 <= problem <= 24:
            raise ValueError("DMMOP problem must be 1..24")
        self.problem = problem
        self.fun, self.change_type, self.dim = FUNCTIONS[problem - 1], CHANGES[problem - 1], DIMS[problem - 1]
        self.low, self.high = -5.0, 5.0
        self.lower, self.upper = np.full(self.dim, -5.0), np.full(self.dim, 5.0)
        self.freq = 5000 * self.dim
        self.max_env = max_env
        self.budget = self.max_env * self.freq
        self.rng = np.random.default_rng(dynamics_seed)
        self.phi = sio.loadmat(DATA / "phi.mat")["phi"].ravel()
        self.env = 0
        self.changes = 0
        self.evaluations = 0
        self.reporter: Callable[[], np.ndarray] | None = None
        self.records: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        self._pending = False
        self._recorded_final = False
        self.rs: dict[str, np.ndarray] = {}
        self.gn_max, self.ln, self.dpeaks = 4, 0, 0.1
        self._initialize()
        self._replay = None
        if instances is not None:
            self._replay = sio.loadmat(Path(instances) / f"P{problem:02d}.mat")
            if self._replay["X"].shape[0] < self.max_env:
                raise ValueError(f"instance file has {self._replay['X'].shape[0]} environments, need {self.max_env}")
            self._apply(0)

    def _apply(self, e: int) -> None:
        """Set environment ``e`` from the replayed (official) instance and check its optimum values."""
        r = self._replay
        self.x = np.array(r["X"][e], dtype=float)
        if self.fun <= 4:
            n = len(self.x)
            self.h, self.w = np.array(r["H"][e, :n], dtype=float), np.array(r["W"][e, :n], dtype=float)
        else:
            self.M = [np.array(r["M"][e, i], dtype=float) for i in range(self.gn_max)]
        self.gn = int(r["GN"].ravel()[e])
        sel = np.asarray(r["SEL"][e]).ravel()
        self.selected = (sel[sel > 0] - 1).astype(int)
        self._norm_key = None
        self._set_optima()
        expected = np.asarray(r["OF"][e]).ravel()[: self.gn]
        if not np.allclose(self.of, expected, rtol=0, atol=1e-6):
            raise ValueError(f"P{self.problem} env {e}: optimum values {self.of} differ from the dump {expected}")

    # --- construction ------------------------------------------------------------------------
    def _initialize(self) -> None:
        d = self.dim
        if self.fun <= 4:
            h_global = 75.0
            if self.fun == 1:
                self.ln = 4
                self.h = np.array([h_global] * self.gn_max + [62.5889, 66.2317, 68.0795, 66.5350])
                self.w = np.array([7.9560, 2.0729, 4.0635, 7.0157, 11.5326, 11.6138, 2.7337, 11.6765])
                self.x = sio.loadmat(DATA / "F1X.mat")["data"][: self.gn_max + self.ln, :d].astype(float)
            elif self.fun == 2:
                self.h, self.w = np.full(self.gn_max, h_global), np.full(self.gn_max, 12.0)
                self.x = np.repeat(np.array([[-3.0], [-2.0], [2.0], [3.0]]), d, axis=1)
            elif self.fun == 3:
                self.h, self.w = np.full(self.gn_max, h_global), np.full(self.gn_max, 5.0)
                self.x = np.repeat(np.array([[-3.0], [-2.0], [0.0], [4.0]]) + 0.5, d, axis=1)
            else:
                self.h, self.w = np.full(self.gn_max, h_global), np.full(self.gn_max, 5.0)
                self.x = np.repeat(np.array([[-3.0], [-1.0], [1.0], [3.0]]), d, axis=1)
            self.x = self._set_min_distance(self.x)
            self.x_initial = self.x.copy()
            self.x_angle = 0.0
            if self.change_type in (5, 6):
                self.h = np.concatenate([np.full(self.gn_max, h_global),
                                         self._change(self.h[self.gn_max:], 30.0, 70.0, 7.0)]) if self.ln else self.h
                self.w = self._change(self.w, 1.0, 12.0, 1.0)
                self.x, self.x_angle = self._change_matrix(self.x, self.x_initial, self.lower, self.upper, self.x_angle, "x")
            self.x = self._set_min_distance(self.x)
        else:
            subs, sigma, lam = _SUBS[self.fun]
            self.sub_fun, self.sigma, self.lam = subs, np.array(sigma, float), np.array(lam, float)
            self.gn_max = len(subs)
            if self.fun in (5, 6):
                self.M = [np.eye(d) for _ in range(self.gn_max)]
            else:
                cells = sio.loadmat(DATA / f"CF{self.fun - 4}_M_D{d}.mat")["M"]
                self.M = [np.array(cells[i, 0], dtype=float) for i in range(self.gn_max)]
            optima = sio.loadmat(DATA / "optima.mat")["o"]
            self.x = optima[::-1][: self.gn_max, :d].astype(float)
            self.x = self._set_min_distance(self.x)
            self.x_initial = self.x.copy()
            self.x_angle = 0.0
            self.M_initial = [m.copy() for m in self.M]
            self.M_angle = np.zeros(self.gn_max)
            if self.change_type in (5, 6):
                self.x, self.x_angle = self._change_matrix(self.x, self.x_initial, self.lower, self.upper, self.x_angle, "x")
                for i in range(self.gn_max):
                    self.M[i], self.M_angle[i] = self._change_matrix(self.M[i], self.M_initial[i], -np.ones(d), np.ones(d),
                                                                     self.M_angle[i], f"M{i}")
            self.x = self._set_min_distance(self.x)
        if self.change_type in (7, 8):
            self.gn = int(self.rng.integers(1, self.gn_max)) + 1
            chosen = np.sort(self.rng.permutation(self.gn_max)[: self.gn])
            self.selected = np.concatenate([chosen, np.arange(self.gn_max, self.gn_max + self.ln)]).astype(int)
            self.is_add = self.change_type == 7 and self.gn == 2
        else:
            self.selected = np.arange(self.gn_max + self.ln)
            self.gn = self.gn_max
            self.is_add = False
        self._set_optima()

    def _set_optima(self) -> None:
        self.o = self.x[self.selected[: self.gn]].copy()
        self.of = self._core(self.o)

    def _change(self, u: np.ndarray, low: float, high: float, severity: float) -> np.ndarray:
        return _dynamic_change(self.rng, u, self.change_type, low, high, severity, self.env, self.phi)

    def _set_min_distance(self, pos: np.ndarray) -> np.ndarray:
        pos = pos.copy()
        for i in range(1, len(pos)):
            while not np.all(np.linalg.norm(pos[:i] - pos[i], axis=1) > self.dpeaks):
                v = self.rng.random(self.dim) - 0.5
                pos[i] = _boundary_check((pos[i] + v * self.dpeaks / np.linalg.norm(v))[None, :], self.lower, self.upper)[0]
        return pos

    def _change_matrix(self, m: np.ndarray, m_initial: np.ndarray, lb: np.ndarray, ub: np.ndarray, old_theta: float,
                       name: str) -> tuple[np.ndarray, float]:
        d = m.shape[1]
        l = d - 1 if d % 2 else d
        r = self.rng.permutation(d)[:l]
        if self.change_type in (5, 6):
            phi_min, phi_max = 0.0, np.pi / 6
            m = m_initial
            if self.env >= 12:
                r = self.rs[name][self.env % 12]
        else:
            phi_min, phi_max = -np.pi, np.pi
        if name not in self.rs:
            self.rs[name] = np.zeros((self.max_env + 1, l), dtype=int)
        self.rs[name][self.env] = r
        if self.change_type in (1, 2, 3):
            old_theta = 0.0
        theta = float(_dynamic_change(self.rng, np.array([old_theta]), self.change_type, phi_min, phi_max, 1.0, self.env, self.phi)[0])
        rotation = np.eye(d)
        for i in range(l // 2):
            a, b = r[2 * i], r[2 * i + 1]
            rotation[a, a] = rotation[b, b] = np.cos(theta)
            rotation[a, b], rotation[b, a] = -np.sin(theta), np.sin(theta)
        return _boundary_check(m @ rotation, lb, ub), theta

    # --- fitness -----------------------------------------------------------------------------
    def _core(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        sel = self.selected
        if self.fun <= 4:
            dist = np.linalg.norm(x[:, None, :] - self.x[sel][None, :, :], axis=2)
            return np.max(self.h[sel] - self.w[sel] * dist, axis=1)
        return self._composition(x, sel[: self.gn])

    def _composition(self, x: np.ndarray, sel: np.ndarray) -> np.ndarray:
        n, d = x.shape
        o = self.x[sel]
        key = (self.env, tuple(sel))
        if getattr(self, "_norm_key", None) != key:
            # 2000 / f_i(upper / lambda_i @ M_i): constant within an environment.
            self._norm = np.array([2000.0 / self.sub_fun[i]((self.upper / self.lam[i])[None, :] @ self.M[i])[0] for i in sel])
            groups: dict = {}
            for j, i in enumerate(sel):
                groups.setdefault(self.sub_fun[i], []).append((j, i))
            self._groups = [(fun, np.array([j for j, _ in members]), np.stack([self.M[i] / self.lam[i] for _, i in members]))
                            for fun, members in groups.items()]
            self._norm_key = key
        weight = np.exp(-np.sum((x[:, None, :] - o[None, :, :]) ** 2, axis=2) / (2 * d * self.sigma[sel] ** 2))
        top = weight.max(axis=1, keepdims=True)
        weight = np.where(weight == top, weight, weight * (1 - top**10))
        if np.all(weight.sum(axis=1) == 0):
            weight = weight + 1
        weight = weight / weight.sum(axis=1, keepdims=True)
        fits = np.empty((n, len(sel)))
        for fun, cols, mats in self._groups:
            # ((x - o_j) / lambda_j) @ M_j == (x - o_j) @ (M_j / lambda_j), stacked over the group.
            z = np.einsum("nkd,kde->nke", x[:, None, :] - o[cols][None, :, :], mats)
            fits[:, cols] = fun(z.reshape(-1, d)).reshape(n, len(cols))
        return -np.sum(weight * fits * self._norm, axis=1)

    def values(self, x: np.ndarray) -> np.ndarray:
        """Uncounted fitness in the current environment."""
        return self._core(x)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Counted fitness. A batch that crosses the end of an environment is split: the rest is
        evaluated in the next one (after the reporter's population has been recorded)."""
        x = np.atleast_2d(x)
        out = np.full(len(x), -np.inf)
        start = 0
        while start < len(x) and self.evaluations < self.budget:
            if self._pending:
                self._record()
                self._advance()
            take = min(len(x) - start, self.freq - self.evaluations % self.freq)
            out[start : start + take] = self._core(x[start : start + take])
            self.evaluations += take
            start += take
            if self.evaluations % self.freq == 0:
                self._pending = True
        return out

    def finish(self) -> None:
        """Record the final environment (call once the budget is used up)."""
        if self._pending and not self._recorded_final:
            self._record()
            self._recorded_final = True

    @property
    def exhausted(self) -> bool:
        return self.evaluations >= self.budget

    def _record(self) -> None:
        pop = np.atleast_2d(self.reporter()) if self.reporter is not None else np.empty((0, self.dim))
        self.records.append((pop.copy(), self._core(pop) if len(pop) else np.empty(0), self.o.copy(), float(self.of[0])))
        self._pending = False

    def _advance(self) -> None:
        self.env += 1
        self.changes += 1
        if self._replay is not None:
            self._apply(self.env)
            return
        if self.change_type in (7, 8):
            if self.change_type == 7:
                self.gn += 1 if self.is_add else -1
                if self.is_add and self.gn == self.gn_max:
                    self.is_add = False
                elif not self.is_add and self.gn == 2:
                    self.is_add = True
                chosen = np.arange(self.gn)
            else:
                self.gn = int(self.rng.integers(1, self.gn_max)) + 1
                chosen = np.sort(self.rng.permutation(self.gn_max)[: self.gn])
            self.selected = np.concatenate([chosen, np.arange(self.gn_max, self.gn_max + self.ln)]).astype(int)
            mode, self.change_type = self.change_type, 1
            self._change_parameters()
            self.change_type = mode
        else:
            self._change_parameters()
        self._set_optima()

    def _change_parameters(self) -> None:
        d = self.dim
        if self.fun <= 4:
            if self.ln:
                self.h[self.gn_max :] = self._change(self.h[self.gn_max :], 30.0, 70.0, 7.0)
            self.w = self._change(self.w, 1.0, 12.0, 1.0)
        else:
            for i in range(self.gn_max):
                self.M[i], self.M_angle[i] = self._change_matrix(self.M[i], self.M_initial[i], -np.ones(d), np.ones(d),
                                                                 self.M_angle[i], f"M{i}")
        self.x, self.x_angle = self._change_matrix(self.x, self.x_initial, self.lower, self.upper, self.x_angle, "x")
        self.x = self._set_min_distance(self.x)

    # --- metric ------------------------------------------------------------------------------
    def peak_counts(self) -> tuple[np.ndarray, int]:
        """Found optima summed over environments at each accuracy level, and total optima."""
        found = np.zeros(len(ACCURACY))
        total = 0
        for pop, fits, o, of in self.records:
            total += len(o)
            for j, eps in enumerate(ACCURACY):
                chosen = pop[(of - fits) < eps] if len(pop) else pop
                if len(chosen):
                    found[j] += np.sum(np.linalg.norm(o[:, None, :] - chosen[None, :, :], axis=2).min(axis=1) < FOUND_DISTANCE)
        return found, total

    def measure(self, x: np.ndarray, radius: float) -> DMMOPMeasure:
        """Coverage of the current global optima (within ``radius``), for monitoring only."""
        near = np.linalg.norm(x[:, None, :] - self.o[None, :, :], axis=2) <= radius
        return DMMOPMeasure(coverage=float(near.any(axis=0).mean()), occupancy=float(near.any(axis=1).mean()),
                            error=float(self.of[0] - self._core(x).max()))

    def maybe_change(self, round_: int) -> bool:
        return False


def export_instances(problem: int, out: str | Path, dynamics_seed: int = 0, max_env: int = 60) -> None:
    """Write this port's environments in the format of scripts/matlab/dump_dmmop_instances.m (for tests)."""
    w = DMMOPWorld(problem, dynamics_seed, max_env)
    n, d = len(w.x), w.dim
    X, H, W = np.zeros((max_env, n, d)), np.full((max_env, n), np.nan), np.full((max_env, n), np.nan)
    M = np.zeros((max_env, w.gn_max, d, d)) if w.fun > 4 else np.zeros(0)
    GN, SEL, OF = np.zeros(max_env), np.zeros((max_env, n)), np.full((max_env, n), np.nan)
    for e in range(max_env):
        if e:
            w._advance()
        X[e] = w.x
        if w.fun <= 4:
            H[e], W[e] = w.h, w.w
        else:
            M[e] = np.stack(w.M)
        GN[e] = w.gn
        SEL[e, : len(w.selected)] = w.selected + 1
        OF[e, : len(w.of)] = w.of
    Path(out).mkdir(parents=True, exist_ok=True)
    sio.savemat(Path(out) / f"P{problem:02d}.mat", {"X": X, "H": H, "W": W, "M": M, "GN": GN, "SEL": SEL, "OF": OF})
