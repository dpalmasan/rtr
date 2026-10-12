"""Follow-up ablation on DMMOP after review (registered 2026-10-11, before any run; post hoc with
respect to the confirmation, and reported as such).

Review points: (a) RTR may reduce to global exclusion at a smaller radius; (b) radius imitation is
inactive when f <= 0 (all composition functions); (c) the "fixed_vision" variant of
scripts/ablation_dmmop.py is fixed only where f <= 0: its radius grows from 5 towards 50 units with
successful relocations on the cone landscapes (career rule), so it is not a fixed radius there.

Variants (all 24 problems, seeds 1-10; compared with ours_cma from confirm_final on the same seeds;
two-sided Mann-Whitney on per-run peak ratio at 1e-4, Holm over 24):
  global5    full perception (global exclusion, as in mQSO) with the exclusion radius set to 5 units.
  fixed5     perception radius exactly 5 units for every agent (no imitation, no growth); exclusion default.
  imitate_hv radius imitation with the hold test judged by the hill-valley test alone (no f > 0 condition).
Decision rules (fixed before the runs):
  G: "the gain of RTR over global exclusion is explained by the radius scale" if global5 is significantly
     worse than RTR on <= 2 problems and its mean is within 0.02 of RTR's mean.
  F: "per-agent radii add nothing over one small fixed radius" by the same rule for fixed5.
  I: imitation is kept (with the sign-independent hold test) only if imitate_hv is significantly better
     than RTR on >= 1 problem and worse on none; otherwise, if F holds, imitation is dropped from the
     method and reported as a tested, optional module.

Usage: PYTHONPATH=src:scripts python scripts/ablation_followup.py --out artifacts/dmmop_followup_2026_10_11
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

from ablation_dmmop import ABLATION_SEEDS, BASE

from aumo.algorithms.peak_hopping import run_peak_hopping
from aumo.problems.dmmop import DMMOPWorld
from aumo.progress import ProgressBar

VARIANTS = {
    "global5": replace(BASE, vision=None, vision_social=False, exclusion=5.0),
    "fixed5": replace(BASE, vision=5.0, vision_max=5.0, vision_social=False),
    "imitate_hv": replace(BASE, hold_positive=False),
}


def run(task: tuple[str, int, int]) -> dict:
    variant, problem, seed = task
    started = time.time()
    world = DMMOPWorld(problem)
    result = run_peak_hopping(world, VARIANTS[variant], seed)
    found, total = world.peak_counts()
    extra = getattr(result, "summary_extra", None) or {}
    return {"variant": variant, "problem": problem, "seed": seed, "pr_1e-3": found[0] / total, "pr_1e-4": found[1] / total,
            "pr_1e-5": found[2] / total, "hop_share": round(result.hop_share, 4),
            "vision_mean": round(extra.get("vision_mean", float("nan")), 4), "seconds": round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tasks = [(v, p, s) for v in VARIANTS for s in ABLATION_SEEDS for p in range(1, 25)]
    path = os.path.join(args.out, "runs.csv")
    fields = ["variant", "problem", "seed", "pr_1e-3", "pr_1e-4", "pr_1e-5", "hop_share", "vision_mean", "seconds"]
    done = set()
    if os.path.exists(path):
        with open(path) as handle:
            done = {(r["variant"], int(r["problem"]), int(r["seed"])) for r in csv.DictReader(handle)}
    tasks = [t for t in tasks if t not in done]
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle, ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        bar = ProgressBar(len(tasks), prefix="follow-up runs")
        for future in as_completed([pool.submit(run, t) for t in tasks]):
            writer.writerow(future.result())
            handle.flush()
            bar.update()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
