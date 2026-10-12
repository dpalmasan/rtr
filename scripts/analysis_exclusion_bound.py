"""Exclusion bound on the CEC 2022 suite (Proposition 1 of the analysis section).

With full perception and exclusion radius r, two agents at rest are at least r apart, so (with optima
held at distance < e and pairwise more than 2e apart) at most
P_{r-2e}(O) of the global optima O of an environment can be held at rest, where e = 0.05 is the
distance within which an optimum counts as found and P_s(O) is the size of the largest subset of O
whose points are pairwise more than s apart (computed exactly; |O| <= 8). (The first version used
P_r; the two differ only on composition problems, where the bound is not reached.)
This script computes, from the geometry of the instances alone (no algorithm is run),

    bound(r) = sum over environments of P_{r-2e}(O_e) / sum over environments of |O_e|

for r = the default exclusion radius and r = 5 units (0.5), the radius of the global5 variant of
scripts/ablation_followup.py, and compares bound(r_ex) with the peak ratio of the full-perception
ablation (artifacts/dmmop_ablation_2026_10_10). The bound at 5 units was computed and written to
artifacts/exclusion_bound_2026_10_11/bound.csv before the global5 results were read.

Usage: PYTHONPATH=src:scripts python scripts/analysis_exclusion_bound.py --out artifacts/exclusion_bound_2026_10_11
       (add --instances data/dmmop/instances to use the official instances)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os

import numpy as np

from aumo.problems.dmmop import DIMS, DMMOPWorld

UNIT = 0.1  # span / 100 on [-5, 5]^D
EPS = 0.05  # FOUND_DISTANCE of the suite


def default_exclusion(dim: int) -> float:
    return 10.0 / (2 * 10 ** (1 / dim))


def packing(points: np.ndarray, r: float) -> int:
    """Size of the largest subset of points that are pairwise more than r apart (exact)."""
    n = len(points)
    far = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2) > r
    for size in range(n, 0, -1):
        for subset in itertools.combinations(range(n), size):
            if all(far[a, b] for a, b in itertools.combinations(subset, 2)):
                return size
    return 0


def environments(problem: int, instances: str | None):
    world = DMMOPWorld(problem, instances=instances)
    for e in range(world.max_env):
        yield world.o.copy()
        if e + 1 < world.max_env:
            world._advance()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--instances", default=None)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for problem in range(1, 25):
        dim = DIMS[problem - 1]
        r_ex = default_exclusion(dim)
        total = held_ex = held_5 = 0
        gaps = []
        for o in environments(problem, args.instances):
            total += len(o)
            held_ex += packing(o, r_ex - 2 * EPS)
            held_5 += packing(o, 5 * UNIT - 2 * EPS)
            d = np.linalg.norm(o[:, None, :] - o[None, :, :], axis=2)
            gaps.append(d[np.triu_indices(len(o), 1)].min() if len(o) > 1 else np.inf)
        rows.append({"problem": problem, "dim": dim, "r_ex": round(r_ex, 3), "optima": total,
                     "bound_rex": round(held_ex / total, 4), "bound_5": round(held_5 / total, 4),
                     "min_gap_median": round(float(np.median(gaps)), 3)})
    name = "bound_official.csv" if args.instances else "bound.csv"
    path = os.path.join(args.out, name)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
