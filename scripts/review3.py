"""Third review round (registered 2026-10-12, before any run beyond smoke tests on P1 and P5 with seed 1).

R  The rule as recommended, on the organisers' instances (DMMOP_INSTANCES=data/dmmop/instances),
   seeds 1-30: RTR with uniform destinations and fixed log-uniform perception radii (no imitation);
   everything else as ours_cma. Compared with ours_cma (the evaluated configuration) and cma_stay from
   artifacts/dmmop_official_2026_10_11 on the same runs.
   Expectation: its mean peak ratio at 1e-4 is within 0.02 of ours_cma's, and it differs significantly
   from ours_cma (either direction) on at most 2 of 24 problems.
   Reported, not criteria: its comparison with cma_stay (the D1 counts) and its Friedman rank among
   {recommended, mqso, hmso_tuned, de}.
K  Fairer restart-on-convergence control, our draw (dynamics seed 0), seeds 1-10: as restart_far of
   scripts/ablation_review2.py, but the point a searcher leaves when it converges or stagnates is kept
   as a reported solution until the next change (a per-environment archive), and kept points count as
   occupied for destinations.
   Rule: RTR (ours_cma, confirm_final, same seeds) is significantly better on >= 12/24 problems and
   significantly worse on <= 2. If met, the trigger outperforms restarting on convergence even when the
   control does not forget the optima it leaves.
Statistics as in scripts/ablation_review2.py (two-sided Mann-Whitney on per-run peak ratio at 1e-4,
Holm over 24, direction from the rank shift).

Usage: PYTHONPATH=src:scripts python scripts/review3.py --out artifacts/dmmop_review3_2026_10_12
       PYTHONPATH=src:scripts python scripts/review3.py --out artifacts/dmmop_review3_2026_10_12 --analyze
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

from ablation_dmmop import ABLATION_SEEDS, BASE
from ablation_review2 import compare
from confirm_final import OURS_CMA

from aumo.algorithms.peak_hopping import run_peak_hopping
from aumo.problems.dmmop import DMMOPWorld
from aumo.progress import ProgressBar

OFFICIAL = "data/dmmop/instances"
JOBS = {
    "recommended": (replace(OURS_CMA, reevaluate="informed", destination="uniform", imitate=False), OFFICIAL, range(1, 31)),
    "restart_keep": (replace(BASE, policy="fixed:stay", restart_far=True, restart_keep=True), None, ABLATION_SEEDS),
}
FIELDS = ["variant", "problem", "seed", "pr_1e-3", "pr_1e-4", "pr_1e-5", "hop_share", "seconds"]


def run(task: tuple[str, int, int]) -> dict:
    variant, problem, seed = task
    cfg, instances, _ = JOBS[variant]
    started = time.time()
    world = DMMOPWorld(problem, instances=instances)
    result = run_peak_hopping(world, cfg, seed)
    found, total = world.peak_counts()
    return {"variant": variant, "problem": problem, "seed": seed, "pr_1e-3": found[0] / total, "pr_1e-4": found[1] / total,
            "pr_1e-5": found[2] / total, "hop_share": round(result.hop_share, 4), "seconds": round(time.time() - started, 1)}


def by_problem(frame: pd.DataFrame, column: str = "pr_1e-4") -> dict:
    return {p: frame[frame.problem == p].sort_values("seed")[column].tolist() for p in range(1, 25)}


def analyze(path: str) -> None:
    runs = pd.read_csv(path)
    off = pd.read_csv("artifacts/dmmop_official_2026_10_11/runs.csv")
    off["problem"] = off.instance.astype(int)
    rec = by_problem(runs[runs.variant == "recommended"])
    ref = {m: by_problem(off[off.method == m]) for m in ("ours_cma", "cma_stay", "mqso", "hmso_tuned", "de")}
    mean = lambda d: np.mean([np.mean(d[p]) for p in range(1, 25)])
    better, worse = compare(rec, ref["ours_cma"])
    ok = abs(mean(rec) - mean(ref["ours_cma"])) <= 0.02 and len(better) + len(worse) <= 2
    print(f"R recommended {mean(rec):.3f} vs evaluated {mean(ref['ours_cma']):.3f}; recommended better on {better}, worse on {worse} -> "
          f"{'MET' if ok else 'NOT MET'}")
    vb, vw = compare(rec, ref["cma_stay"])
    print(f"  vs no relocation {mean(ref['cma_stay']):.3f}: better on {len(vb)}, worse on {len(vw)} {vw}")
    table = pd.DataFrame({m: [np.mean(d[p]) for p in range(1, 25)] for m, d in
                          {"recommended": rec, "mqso": ref["mqso"], "hmso_tuned": ref["hmso_tuned"], "de": ref["de"]}.items()})
    print("  Friedman mean ranks:", table.rank(axis=1, ascending=False).mean().round(2).to_dict())
    c = pd.read_csv("artifacts/confirm_final_2026_10_09/runs.csv")
    c = c[(c.suite == "dmmop") & (c.method == "ours_cma") & c.seed.isin(ABLATION_SEEDS)].copy()
    c["problem"] = c.instance.astype(int)
    keep = by_problem(runs[runs.variant == "restart_keep"])
    better, worse = compare(by_problem(c), keep)
    print(f"K restart_keep {mean(keep):.3f} vs RTR {mean(by_problem(c)):.3f}; RTR better on {len(better)} {better}, worse on {len(worse)} {worse} -> "
          f"{'MET' if len(better) >= 12 and len(worse) <= 2 else 'NOT MET'}")


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
    tasks = [(v, p, s) for v, (_, _, seeds) in JOBS.items() for s in seeds for p in range(1, 25)]
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
        bar = ProgressBar(len(tasks), prefix="review3 runs")
        for future in as_completed([pool.submit(run, t) for t in tasks]):
            writer.writerow(future.result())
            handle.flush()
            bar.update()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
