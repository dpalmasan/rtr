"""Python port (vectorised) of the CEC 2013 niching benchmark (Li, Engelbrecht and Epitropakis,
"Benchmark functions for CEC'2013 special session and competition on niching methods for multimodal
function optimization", 2013), from the organisers' python3 code (github.com/mikeagn/CEC2013) with
its data files (data/cec2013/, BSD licence).

Protocol: 20 maximisation problems, a fixed evaluation budget each (``MAXFES``); at the end the
algorithm reports its solutions and the peak ratio counts, at accuracies 1e-1 ... 1e-5, the global
optima found by the official procedure (``how_many_goptima``: sort by fitness, keep "seeds" more
than rho apart, count seeds within the accuracy of the optimum value), divided by the number of
global optima.

``CEC2013World`` works in normalised coordinates [0, 1]^D (the team code assumes one interval for
every coordinate); it maps to the problem's box before evaluating and counting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

DATA = Path(__file__).resolve().parents[3] / "data" / "cec2013"
FOPT = [200.0, 1.0, 1.0, 200.0, 1.031628453489877, 186.7309088310239, 1.0, 2709.093505572820, 1.0, -2.0] + [0.0] * 10
RHO = [0.01, 0.01, 0.01, 0.01, 0.5, 0.5, 0.2, 0.5, 0.2, 0.01] + [0.01] * 10
NOPT = [2, 5, 1, 4, 2, 18, 36, 81, 216, 12, 6, 8, 6, 6, 8, 6, 8, 6, 8, 8]
MAXFES = [50000] * 5 + [200000, 200000, 400000, 400000, 200000, 200000, 200000, 200000] + [400000] * 7
DIMS = [1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 5, 5, 10, 10, 20]
ACCURACY = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)


def _bounds(fn: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    if fn in (1, 2, 3):
        return np.zeros(d), np.full(d, 30.0 if fn == 1 else 1.0)
    if fn == 4:
        return np.full(d, -6.0), np.full(d, 6.0)
    if fn == 5:
        return np.array([-1.9, -1.1]), np.array([1.9, 1.1])
    if fn in (6, 8):
        return np.full(d, -10.0), np.full(d, 10.0)
    if fn in (7, 9):
        return np.full(d, 0.25), np.full(d, 10.0)
    if fn == 10:
        return np.zeros(d), np.ones(d)
    return np.full(d, -5.0), np.full(d, 5.0)


# --- basic functions (rows of x are points) -------------------------------------------------------
def _five_uneven_peak_trap(x: np.ndarray) -> np.ndarray:
    v = x[:, 0]
    conds = [v < 2.5, v < 5.0, v < 7.5, v < 12.5, v < 17.5, v < 22.5, v < 27.5, v <= 30.0]
    vals = [80 * (2.5 - v), 64 * (v - 2.5), 64 * (7.5 - v), 28 * (v - 7.5), 28 * (17.5 - v), 32 * (v - 17.5),
            32 * (27.5 - v), 80 * (v - 27.5)]
    return np.select(conds, vals, default=np.nan)


def _equal_maxima(x: np.ndarray) -> np.ndarray:
    return np.sin(5.0 * np.pi * x[:, 0]) ** 6


def _uneven_decreasing_maxima(x: np.ndarray) -> np.ndarray:
    v = x[:, 0]
    return np.exp(-2.0 * np.log(2) * ((v - 0.08) / 0.854) ** 2) * np.sin(5 * np.pi * (v**0.75 - 0.05)) ** 6


def _himmelblau(x: np.ndarray) -> np.ndarray:
    return 200 - (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 - (x[:, 0] + x[:, 1] ** 2 - 7) ** 2


def _six_hump_camel_back(x: np.ndarray) -> np.ndarray:
    a, b = x[:, 0], x[:, 1]
    return -((4.0 - 2.1 * a**2 + a**4 / 3.0) * a**2 + a * b + (4.0 * b**2 - 4.0) * b**2)


def _shubert(x: np.ndarray) -> np.ndarray:
    j = np.arange(1, 6)
    soma = np.sum(j * np.cos((j + 1) * x[:, :, None] + j), axis=2)
    return -np.prod(soma, axis=1)


def _vincent(x: np.ndarray) -> np.ndarray:
    return np.mean(np.sin(10 * np.log(x)), axis=1)


def _modified_rastrigin_all(x: np.ndarray) -> np.ndarray:
    k = np.array([3, 4])
    return -np.sum(10 + 9 * np.cos(2 * np.pi * k * x), axis=1)


def _sphere(z: np.ndarray) -> np.ndarray:
    return np.sum(z**2, axis=1)


def _rastrigin(z: np.ndarray) -> np.ndarray:
    return np.sum(z**2 - 10.0 * np.cos(2.0 * np.pi * z) + 10, axis=1)


def _griewank(z: np.ndarray) -> np.ndarray:
    i = np.sqrt(np.arange(z.shape[1]) + 1.0)
    return np.sum(z**2, axis=1) / 4000.0 - np.prod(np.cos(z / i), axis=1) + 1.0


_WK = np.arange(21)
_WC1, _WC2 = 0.5**_WK, 2.0 * np.pi * 3.0**_WK


def _weierstrass(z: np.ndarray) -> np.ndarray:
    d = z.shape[1]
    c = -d * np.sum(_WC1 * np.cos(_WC2 * 0.5))
    return np.sum(_WC1 * np.cos(_WC2 * (z[:, :, None] + 0.5)), axis=(1, 2)) + c


def _f8f2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    f2 = 100.0 * (a**2 - b) ** 2 + (1.0 - a) ** 2
    return 1.0 + f2**2 / 4000.0 - np.cos(f2)


def _ef8f2(z: np.ndarray) -> np.ndarray:
    y = z + 1
    return np.sum(_f8f2(y, np.roll(y, -1, axis=1)), axis=1)


_CF = {  # functions, sigma, lambda, rotation file
    1: ([_griewank, _griewank, _weierstrass, _weierstrass, _sphere, _sphere], [1] * 6, [1, 1, 8, 8, 1 / 5, 1 / 5], None),
    2: ([_rastrigin, _rastrigin, _weierstrass, _weierstrass, _griewank, _griewank, _sphere, _sphere], [1] * 8,
        [1, 1, 10, 10, 1 / 10, 1 / 10, 1 / 7, 1 / 7], None),
    3: ([_ef8f2, _ef8f2, _weierstrass, _weierstrass, _griewank, _griewank], [1, 1, 2, 2, 2, 2],
        [1 / 4, 1 / 10, 2, 1, 2, 5], "CF3"),
    4: ([_rastrigin, _rastrigin, _ef8f2, _ef8f2, _weierstrass, _weierstrass, _griewank, _griewank],
        [1, 1, 1, 1, 1, 2, 2, 2], [4, 1, 4, 1, 1 / 10, 1 / 5, 1 / 10, 1 / 40], "CF4"),
}
_CF_OF = {11: 1, 12: 2, 13: 3, 14: 3, 15: 4, 16: 3, 17: 4, 18: 3, 19: 4, 20: 4}


class _Composition:
    def __init__(self, cf: int, d: int) -> None:
        funcs, sigma, lam, rot = _CF[cf]
        k = len(funcs)
        self.funcs, self.sigma, self.lam, self.d = funcs, np.array(sigma, float), np.array(lam, float), d
        self.o = np.loadtxt(DATA / "optima.dat")[:k, :d]
        if rot is None:
            self.M = np.stack([np.eye(d)] * k)
        else:
            rows = np.loadtxt(DATA / f"{rot}_M_D{d}.dat")
            self.M = rows[: k * d, :d].reshape(k, d, d)
        x5 = np.full((1, d), 5.0)
        self.fmax = np.array([f((x5 / self.lam[i]) @ self.M[i])[0] for i, f in enumerate(funcs)])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        k, d = len(self.funcs), self.d
        w = np.exp(-np.sum((x[:, None, :] - self.o[None]) ** 2, axis=2) / (2.0 * d * self.sigma**2))
        top = w.max(axis=1, keepdims=True)
        w = np.where(w != top, w * (1.0 - top**10), w)
        total = w.sum(axis=1, keepdims=True)
        w = np.where(total == 0.0, 1.0 / k, w / np.where(total == 0.0, 1.0, total))
        fi = np.stack([f(((x - self.o[i]) / self.lam[i]) @ self.M[i]) for i, f in enumerate(self.funcs)], axis=1)
        return -np.sum(w * 2000.0 * fi / self.fmax, axis=1)


def problem_function(fn: int) -> Callable[[np.ndarray], np.ndarray]:
    """Vectorised fitness of problem ``fn`` (1..20) over rows of x (problem coordinates)."""
    basic = {1: _five_uneven_peak_trap, 2: _equal_maxima, 3: _uneven_decreasing_maxima, 4: _himmelblau,
             5: _six_hump_camel_back, 6: _shubert, 7: _vincent, 8: _shubert, 9: _vincent, 10: _modified_rastrigin_all}
    if fn in basic:
        return basic[fn]
    return _Composition(_CF_OF[fn], DIMS[fn - 1])


def how_many_goptima(pop: np.ndarray, fits: np.ndarray, fn: int, accuracy: float) -> int:
    """The official count: seeds (fitness order, more than rho apart) within accuracy of the optimum."""
    order = np.argsort(fits, kind="stable")[::-1]
    seeds: list[np.ndarray] = []
    count = 0
    for idx in order:
        p = pop[idx]
        if seeds and np.min(np.linalg.norm(np.array(seeds) - p, axis=1)) <= RHO[fn - 1]:
            continue
        seeds.append(p)
        if abs(fits[idx] - FOPT[fn - 1]) <= accuracy:
            count += 1
            if count == NOPT[fn - 1]:
                break
    return count


class CEC2013World:
    """One CEC 2013 niching problem as a static world in normalised coordinates [0, 1]^D."""

    def __init__(self, problem: int) -> None:
        if not 1 <= problem <= 20:
            raise ValueError("CEC 2013 problem must be 1..20")
        self.problem = problem
        self.dim = DIMS[problem - 1]
        self.lb, self.ub = _bounds(problem, self.dim)
        self.low, self.high = 0.0, 1.0
        self.budget = MAXFES[problem - 1]
        self.f = problem_function(problem)
        self.evaluations = 0
        self.changes = 0
        self.reporter: Callable[[], np.ndarray] | None = None
        self.final: np.ndarray | None = None

    def to_problem(self, u: np.ndarray) -> np.ndarray:
        return self.lb + np.clip(u, 0.0, 1.0) * (self.ub - self.lb)

    def values(self, u: np.ndarray) -> np.ndarray:
        return self.f(self.to_problem(np.atleast_2d(u)))

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        """Counted fitness; points beyond the budget get -inf and are not counted."""
        u = np.atleast_2d(u)
        out = np.full(len(u), -np.inf)
        take = max(0, min(len(u), self.budget - self.evaluations))
        if take:
            out[:take] = self.values(u[:take])
            self.evaluations += take
        return out

    @property
    def exhausted(self) -> bool:
        return self.evaluations >= self.budget

    def finish(self) -> None:
        if self.reporter is not None:
            self.final = np.atleast_2d(self.reporter())

    def peak_ratios(self, solutions: np.ndarray | None = None) -> np.ndarray:
        """Peak ratio at each accuracy (1e-1 ... 1e-5) of the reported (normalised) solutions."""
        u = self.final if solutions is None else np.atleast_2d(solutions)
        if u is None or not len(u):
            return np.zeros(len(ACCURACY))
        pop = self.to_problem(u)
        fits = self.f(pop)
        return np.array([how_many_goptima(pop, fits, self.problem, eps) for eps in ACCURACY]) / NOPT[self.problem - 1]

    def measure(self, u: np.ndarray, radius: float):
        return _NoMeasure

    def maybe_change(self, round_: int) -> bool:
        return False


class _NoMeasureType:
    coverage = occupancy = error = float("nan")


_NoMeasure = _NoMeasureType()
