"""Second review round on CEC 2022 DMMOP (registered 2026-10-12, before any run beyond smoke tests on
P2 and P5 with seed 1; post hoc with respect to the confirmation, and reported as such).

Same protocol as scripts/ablation_followup.py: our instance draws (dynamics seed 0), all 24 problems,
seeds 1-10, compared with ours_cma (and cma_stay) from confirm_final on the same seeds; two-sided
Mann-Whitney on the per-run peak ratio at 1e-4, Holm over 24.

Variants:
  restart_far  no redundancy trigger (policy fixed:stay); a searcher restarts at the farthest of k = 8
               random points from all teammates when its CMA-ES converges or its best point has not
               improved for 10 + ceil(30 D / lambda) generations (stagnation). Tests the claim that RTR's
               benefit comes from restarting on redundancy rather than on convergence or stagnation.
  stale5/20/40 RTR with stale teammate information: the redundancy test and the destination see the
               teammates' positions and fitness as of the last snapshot, refreshed every k = 5, 20 or 40
               rounds and after every announced change (each agent knows its own state). Tests whether
               the rule tolerates the delays of a distributed implementation.

Decision rules (fixed before the runs):
  C  RTR is significantly better than restart_far on >= 12/24 problems and significantly worse on <= 2.
     If met, restarting on redundancy outperforms restarting on convergence or stagnation.
  S  for k = 5 and k = 20: the stale variant's mean is >= RTR's mean - 0.05, and it is significantly
     better than cma_stay on >= 16/24 problems and worse on <= 2 (the D1 threshold). If both hold, the rule
     degrades gracefully with stale information. k = 40 (one full cycle) is reported, not a criterion.

Usage: PYTHONPATH=src:scripts python scripts/ablation_review2.py --out artifacts/dmmop_review2_2026_10_12
       PYTHONPATH=src:scripts python scripts/ablation_review2.py --out artifacts/dmmop_review2_2026_10_12 --analyze
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ablation_dmmop import ABLATION_SEEDS, BASE

from aumo.algorithms.peak_hopping import run_peak_hopping
from aumo.problems.dmmop import DMMOPWorld
from aumo.progress import ProgressBar

VARIANTS = {
    "restart_far": replace(BASE, policy="fixed:stay", restart_far=True),
    "stale5": replace(BASE, stale_period=5),
    "stale20": replace(BASE, stale_period=20),
    "stale40": replace(BASE, stale_period=40),
}
CONFIRM = ["artifacts/confirm_final_2026_10_09/runs.csv"]
FIELDS = ["variant", "problem", "seed", "pr_1e-3", "pr_1e-4", "pr_1e-5", "hop_share", "far_restarts", "seconds"]


def run(task: tuple[str, int, int]) -> dict:
    variant, problem, seed = task
    started = time.time()
    world = DMMOPWorld(problem)
    result = run_peak_hopping(world, VARIANTS[variant], seed)
    found, total = world.peak_counts()
    return {"variant": variant, "problem": problem, "seed": seed, "pr_1e-3": found[0] / total, "pr_1e-4": found[1] / total,
            "pr_1e-5": found[2] / total, "hop_share": round(result.hop_share, 4),
            "far_restarts": result.summary_extra.get("far_restarts", 0), "seconds": round(time.time() - started, 1)}


def holm(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    order = np.argsort(p)
    reject = np.zeros(len(p), dtype=bool)
    for rank, i in enumerate(order):
        if p[i] > alpha / (len(p) - rank):
            break
        reject[i] = True
    return reject


def compare(a: dict, b: dict) -> tuple[list, list]:
    """Problems where a is significantly better / worse than b (rank direction), Holm over 24."""
    pv, sign = [], []
    for p in range(1, 25):
        x, y = np.asarray(a[p]), np.asarray(b[p])
        pv.append(1.0 if np.allclose(np.r_[x, y], x[0]) else mannwhitneyu(x, y, alternative="two-sided").pvalue)
        a12 = (np.sum(x[:, None] > y[None]) + 0.5 * np.sum(x[:, None] == y[None])) / (len(x) * len(y))
        sign.append(a12 - 0.5)
    rej = holm(np.array(pv))
    return [p for p in range(1, 25) if rej[p - 1] and sign[p - 1] > 0], [p for p in range(1, 25) if rej[p - 1] and sign[p - 1] < 0]


def analyze(path: str) -> None:
    runs = pd.read_csv(path)
    c = pd.read_csv(CONFIRM[0])
    c = c[(c.suite == "dmmop") & (c.seed.isin(ABLATION_SEEDS))].copy()
    c["problem"] = c.instance.astype(int)
    ref = {m: {p: g[g.problem == p]["pr_1e-4"].tolist() for p in range(1, 25)} for m, g in c.groupby("method") if m in ("ours_cma", "cma_stay")}
    rtr_mean = np.mean([np.mean(ref["ours_cma"][p]) for p in range(1, 25)])
    print(f"RTR mean {rtr_mean:.3f}; no relocation {np.mean([np.mean(ref['cma_stay'][p]) for p in range(1, 25)]):.3f}")
    var = {v: {p: g[g.problem == p]["pr_1e-4"].tolist() for p in range(1, 25)} for v, g in runs.groupby("variant")}
    for v, d in var.items():
        mean = np.mean([np.mean(d[p]) for p in range(1, 25)])
        better, worse = compare(ref["ours_cma"], d)
        vb, vw = compare(d, ref["cma_stay"])
        print(f"{v:12s} mean {mean:.3f}; RTR better on {len(better)} {better}, worse on {len(worse)} {worse}; "
              f"variant vs no relocation: better on {len(vb)}, worse on {len(vw)} {vw}")
        if v == "restart_far":
            print(f"  C -> {'MET' if len(better) >= 12 and len(worse) <= 2 else 'NOT MET'}")
        if v in ("stale5", "stale20"):
            ok = mean >= rtr_mean - 0.05 and len(vb) >= 16 and len(vw) <= 2
            print(f"  S part for {v} -> {'holds' if ok else 'fails'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    path = os.path.join(args.out, "runs.csv")
    if args.analyze:
        analyze(path)
        return
    os.makedirs(args.out, exist_ok=True)
    tasks = [(v, p, s) for v in VARIANTS for s in ABLATION_SEEDS for p in range(1, 25)]
    done = set()
    if os.path.exists(path):
        with open(path) as handle:
            done = {(r["variant"], int(r["problem"]), int(r["seed"])) for r in csv.DictReader(handle)}
    tasks = [t for t in tasks if t not in done]
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle, ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        bar = ProgressBar(len(tasks), prefix="review2 runs")
        for future in as_completed([pool.submit(run, t) for t in tasks]):
            writer.writerow(future.result())
            handle.flush()
            bar.update()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
