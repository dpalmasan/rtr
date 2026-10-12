"""Redundancy accounting on the CEC 2022 suite (Proposition 2 of the analysis section; post hoc,
registered 2026-10-11 before these runs).

Runs ours_cma (RTR) and cma_stay (no relocation) exactly as in confirm_final (our instance draws)
and reads the redundancy counters of run_peak_hopping: the share of agent-rounds spent redundant,
the share of rounds of active searchers spent redundant, the share of redundant agent-rounds spent
in a relocation stint, and the length of redundancy episodes (rounds from the first redundancy
verdict to relocation, to the end of redundancy, or to the end of the run).

Expectations (fixed before the runs):
  M0 the counters do not change the search: every run reproduces its pr_1e-4 in confirm_final.
  M1 (Proposition 2) under RTR no episode ends in relocation later than c + 1 = 41 rounds after it began;
     an episode can only exceed that if redundancy is interrupted (checked as episode_max <= 41 when
     no environment change intervenes; reported as is).
  M2 the mean episode is shorter under RTR than under cma_stay on every problem measured.
  M3 the share of agent-rounds spent redundant is lower under RTR than under cma_stay on every problem.

Usage: PYTHONPATH=src:scripts python scripts/analysis_redundancy.py --out artifacts/redundancy_2026_10_11
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

from confirm_final import TEAM_METHODS

from aumo.algorithms.peak_hopping import run_peak_hopping
from aumo.problems.dmmop import DMMOPWorld
from aumo.progress import ProgressBar

PROBLEMS = [1, 2, 5, 10, 17, 21]
SEEDS = [1, 2, 3]
METHODS = ["ours_cma", "cma_stay"]
FIELDS = ["red_share", "red_active_share", "red_recent_share", "episode_mean", "episode_p95", "episode_max"]


def run(task: tuple[str, int, int]) -> dict:
    method, problem, seed = task
    started = time.time()
    world = DMMOPWorld(problem)
    result = run_peak_hopping(world, replace(TEAM_METHODS[method], reevaluate="informed"), seed)
    found, total = world.peak_counts()
    row = {"method": method, "problem": problem, "seed": seed, "pr_1e-4": found[1] / total}
    row.update({k: round(result.summary_extra[k], 4) for k in FIELDS})
    row["seconds"] = round(time.time() - started, 1)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tasks = [(m, p, s) for m in METHODS for p in PROBLEMS for s in SEEDS]
    path = os.path.join(args.out, "runs.csv")
    with open(path, "w", newline="") as handle, ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(handle, fieldnames=["method", "problem", "seed", "pr_1e-4", *FIELDS, "seconds"])
        writer.writeheader()
        bar = ProgressBar(len(tasks), prefix="redundancy runs")
        for future in as_completed([pool.submit(run, t) for t in tasks]):
            writer.writerow(future.result())
            handle.flush()
            bar.update()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
