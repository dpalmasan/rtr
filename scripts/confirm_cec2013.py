"""Pre-registered confirmation of the job-hopping team on the static CEC 2013 niching benchmark.

Registered 2026-10-10, before any confirmation run. Method: the DMMOP team (scripts/confirm_final.py
OURS_CMA: 20 agents, CMA-ES workers, hop when a fitter colleague is seen within exclusion, social
vision) plus the team archive (peak_hopping.archive: a converged worker retires its point to the
archive and moves on; archived points count as colleagues who never hop). The archive is needed
because many problems have more global optima than agents (up to 216). Development (decided before
the archive was coded, then checked once): 3 seeds (100-102) on F1, F2, F4, F6, F7, F9, F10, F11,
F15, F20; nothing was tuned after it. There the team was better on F7, F9, F20, worse on F6, tied
elsewhere.

Protocol: official (20 problems, MAXFES per problem, peak ratio at accuracies 1e-1 ... 1e-5 by the
official count on the reported solutions = final team + archive); seeds 1-50 (the official 50 runs).
  N1 coordination: "ours" (team + archive) has significantly higher peak ratio at 1e-4 than "stay"
     (the same CMA workers and archive, never hop by choice: a restart CMA-ES with a shared archive)
     on >= 7/20 problems and significantly lower on <= 2 (two-sided Mann-Whitney on per-run peak
     ratio, Holm over 20).
  N2: the mean peak ratio at 1e-4 over the 20 problems is higher for ours than for stay.
  Reported, not criteria: "ours_noarchive" (the DMMOP team unchanged); other accuracies; published
  results (2013 entrants' tables; RS-CMSA-ESII and HillVallEA19 re-scored by our port from their
  submissions, artifacts/cec2013_published).

Also registered here for the official DMMOP instances (when data/dmmop/instances/ exists, dumped from
MATLAB by scripts/matlab/dump_dmmop_instances.m): the same criteria D1 and D2 as scripts/confirm_final.py,
on seeds 1-30, with D2's HmSO replaced by hmso_tuned (radii tuned per family on dev seed 0).

Usage: PYTHONPATH=src python scripts/confirm_cec2013.py --out artifacts/confirm_cec2013_2026_10_10
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

from confirm_final import OURS_CMA

from aumo.algorithms.peak_hopping import run_peak_hopping
from aumo.problems.cec2013 import ACCURACY, CEC2013World
from aumo.progress import ProgressBar

METHODS = {
    "ours": replace(OURS_CMA, archive=True, reevaluate="informed"),
    "stay": replace(OURS_CMA, archive=True, reevaluate="informed", policy="fixed:stay"),
    "ours_noarchive": replace(OURS_CMA, reevaluate="informed"),
}
SEEDS = range(1, 51)


def run(task: tuple[str, int, int]) -> dict:
    method, problem, seed = task
    started = time.time()
    world = CEC2013World(problem)
    result = run_peak_hopping(world, METHODS[method], seed)
    pr = world.peak_ratios()
    return {"method": method, "problem": problem, "seed": seed, **{f"pr_{e:g}": v for e, v in zip(ACCURACY, pr)},
            "archive": result.summary_extra.get("archive_size", ""), "evaluations": world.evaluations,
            "seconds": round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tasks = [(m, p, s) for m in METHODS for s in SEEDS for p in range(1, 21)]
    if args.smoke:
        tasks = [(m, p, 1) for m in METHODS for p in (1, 6, 20)]
    path = os.path.join(args.out, "runs.csv")
    fields = ["method", "problem", "seed"] + [f"pr_{e:g}" for e in ACCURACY] + ["archive", "evaluations", "seconds"]
    done = set()
    if os.path.exists(path):
        with open(path) as handle:
            done = {(r["method"], int(r["problem"]), int(r["seed"])) for r in csv.DictReader(handle)}
    tasks = [t for t in tasks if t not in done]
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle, ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        bar = ProgressBar(len(tasks), prefix="cec2013 runs")
        for future in as_completed([pool.submit(run, t) for t in tasks]):
            writer.writerow(future.result())
            handle.flush()
            bar.update()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
