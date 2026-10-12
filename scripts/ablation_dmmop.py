"""Component ablation and parameter sensitivity of the job-hopping team on DMMOP (registered
2026-10-10 before any run; answers reviewers' requests for ablations and sensitivity).

Ablation (all 24 problems, seeds 1-10; compared with ours_cma from confirm_final on the same seeds):
  random_hop   hop at random, with the same per-problem fraction of hop decisions as the rule
               (measured on dev seed 0: artifacts/dmmop_ablation_2026_10_10/hop_rate_dev_seed0.json).
               Isolates *when* to hop (redundancy) from *how often*.
  uniform_dest hop to a uniform random point instead of the farthest of 8 from seen colleagues.
  full_vision  every agent sees the whole team (no partial vision).
  fixed_vision every agent has vision 5 units (no social learning of the radius).
  A1 (registered): ours_cma has a significantly higher peak ratio at 1e-4 than random_hop on
     >= 12/24 problems and a lower one on <= 2 (Mann-Whitney, Holm over 24).
  The other three are reported with the same test, no threshold.
Sensitivity (8 problems P1, P5, P7, P10, P14, P16, P17, P21; seeds 1-5; one setting changed at a
time from the frozen method): agents 10 / 40; cycle 20 / 80; exclusion x0.5 / x2; sigma0 5 / 20 units.
  Reported as mean peak ratio relative to the frozen method; no threshold.

Usage: PYTHONPATH=src:scripts python scripts/ablation_dmmop.py --out artifacts/dmmop_ablation_2026_10_10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from confirm_final import OURS_CMA

from aumo.algorithms.peak_hopping import run_peak_hopping
from aumo.problems.dmmop import DIMS, DMMOPWorld
from aumo.progress import ProgressBar

BASE = replace(OURS_CMA, reevaluate="informed")
HOP_RATE = {int(k): v for k, v in json.loads(
    (Path(__file__).resolve().parents[1] / "artifacts/dmmop_ablation_2026_10_10/hop_rate_dev_seed0.json").read_text()).items()}
ABLATION_SEEDS, SENSITIVITY_SEEDS = range(1, 11), range(1, 6)
SENSITIVITY_PROBLEMS = (1, 5, 7, 10, 14, 16, 17, 21)


def default_exclusion(dim: int) -> float:
    """The default exclusion radius in units of span/100 (span / (2 * 10^(1/D)))."""
    return 50.0 / 10 ** (1 / dim)


def config(variant: str, problem: int):
    if variant == "random_hop":
        return replace(BASE, policy="random", hop_probability=HOP_RATE[problem])
    if variant == "uniform_dest":
        return replace(BASE, destination="uniform")
    if variant == "full_vision":
        return replace(BASE, vision=None, vision_social=False)
    if variant == "fixed_vision":
        return replace(BASE, vision_social=False)
    name, value = variant.split("=")
    if name == "agents":
        return replace(BASE, agents=int(value))
    if name == "cycle":
        return replace(BASE, cycle=int(value))
    if name == "exclusion":
        return replace(BASE, exclusion=float(value) * default_exclusion(DIMS[problem - 1]))
    if name == "sigma0":
        return replace(BASE, cma_sigma0=float(value))
    raise ValueError(variant)


def run(task: tuple[str, int, int]) -> dict:
    variant, problem, seed = task
    started = time.time()
    world = DMMOPWorld(problem)
    result = run_peak_hopping(world, config(variant, problem), seed)
    found, total = world.peak_counts()
    return {"variant": variant, "problem": problem, "seed": seed, "pr_1e-3": found[0] / total, "pr_1e-4": found[1] / total,
            "pr_1e-5": found[2] / total, "hop_share": round(result.hop_share, 4), "seconds": round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tasks = [(v, p, s) for v in ("random_hop", "uniform_dest", "full_vision", "fixed_vision") for s in ABLATION_SEEDS for p in range(1, 25)]
    for v in ("agents=10", "agents=40", "cycle=20", "cycle=80", "exclusion=0.5", "exclusion=2", "sigma0=5", "sigma0=20"):
        tasks += [(v, p, s) for s in SENSITIVITY_SEEDS for p in SENSITIVITY_PROBLEMS]
    path = os.path.join(args.out, "runs.csv")
    fields = ["variant", "problem", "seed", "pr_1e-3", "pr_1e-4", "pr_1e-5", "hop_share", "seconds"]
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
        bar = ProgressBar(len(tasks), prefix="ablation runs")
        for future in as_completed([pool.submit(run, t) for t in tasks]):
            writer.writerow(future.result())
            handle.flush()
            bar.update()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
