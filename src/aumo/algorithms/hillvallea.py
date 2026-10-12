"""HillVallEA (Maree, Alderliesten, Thierens, Bosman; GECCO 2018/2019) driven through its original C++ code.

The authors' code (github.com/SCMaree/HillVallEA, GECCO 2019 commit, GPL-3.0) is built as a shared library
by third_party/hillvallea/build.sh and called through ctypes with the objective as a callback. Settings are
the authors' defaults (AMaLGaM-Univariate local optimizer), as in their CEC 2013 benchmark script.

Static problems (CEC 2013): one run with the full budget; the reported solutions are its elitist archive.

Dynamic problems (CEC 2022 DMMOP): HillVallEA has no change handling, so it is run once per environment
with that environment's budget, and its elitist archive is reported at the change. Evaluations it leaves
unused in an environment are spent re-evaluating its best elite, so every environment gets the same budget.
  restart: every environment starts from scratch (the unmodified algorithm);
  seeded:  the previous environment's elites are added to the first sample of the next run
           (third_party/hillvallea/seed.patch), the usual way to carry memory across changes.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

LIBRARY = Path(__file__).resolve().parents[3] / "third_party" / "hillvallea" / "libhve.so"
_OBJECTIVE = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int)
_DOUBLES = ctypes.POINTER(ctypes.c_double)
_lib = None
MAX_ELITES = 5000


def _library() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not LIBRARY.exists():
            raise FileNotFoundError(f"{LIBRARY} not found: run third_party/hillvallea/build.sh")
        _lib = ctypes.CDLL(str(LIBRARY))
        _lib.hve_run.restype = ctypes.c_int
        _lib.hve_run.argtypes = [ctypes.c_int, _DOUBLES, _DOUBLES, ctypes.c_int, ctypes.c_int, _OBJECTIVE,
                                 _DOUBLES, ctypes.c_int, _DOUBLES, _DOUBLES, ctypes.c_int]
    return _lib


def _run(world, budget: int, random_seed: int, seeds: np.ndarray) -> tuple[np.ndarray, int]:
    """One HillVallEA run of at most `budget` counted evaluations of `world`; returns (elites, evaluations used)."""
    d = world.dim
    used = [0]

    def objective(ptr, n):
        if used[0] >= budget:   # never cross into the next environment
            return 1e300
        used[0] += 1
        x = np.ctypeslib.as_array(ptr, (n,)).copy()
        return -float(world.evaluate(x[None, :])[0])   # HillVallEA minimizes

    lo = np.ascontiguousarray(np.broadcast_to(world.low, d), dtype=float)
    hi = np.ascontiguousarray(np.broadcast_to(world.high, d), dtype=float)
    seeds = np.ascontiguousarray(seeds, dtype=float).reshape(-1, d)
    out_x, out_f = np.zeros((MAX_ELITES, d)), np.zeros(MAX_ELITES)
    callback = _OBJECTIVE(objective)
    n = _library().hve_run(d, lo.ctypes.data_as(_DOUBLES), hi.ctypes.data_as(_DOUBLES), int(budget), int(random_seed),
                           callback, seeds.ctypes.data_as(_DOUBLES), len(seeds), out_x.ctypes.data_as(_DOUBLES),
                           out_f.ctypes.data_as(_DOUBLES), MAX_ELITES)
    return out_x[:n].copy(), used[0]


def run_hillvallea(world, seed: int, seeded: bool = True):
    d = world.dim
    state = {"elites": np.empty((0, d))}
    world.reporter = lambda: state["elites"].copy()
    if not hasattr(world, "freq"):   # static
        state["elites"], _ = _run(world, world.budget, 1000 * seed, np.empty((0, d)))
        world.finish()
        return world
    while not world.exhausted:
        cap = world.freq - world.evaluations % world.freq
        seeds = state["elites"] if seeded else np.empty((0, d))
        # One evaluation is kept back so the elites are in place when the last one triggers the change.
        elites, used = _run(world, cap - 1, 1000 * seed + world.changes, seeds)
        state["elites"] = elites if len(elites) else state["elites"]
        filler = state["elites"][:1] if len(state["elites"]) else np.full((1, d), world.low, dtype=float)
        world.evaluate(np.repeat(filler, cap - used, axis=0))
    world.finish()
    return world
