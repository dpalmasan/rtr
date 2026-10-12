"""Pre-registered confirmation of the job-hopping team (coordination layer) on DMMOP and Moving Peaks.

Frozen method (development: artifacts/peaks_*_2026_10_09, artifacts/dmmop_dev_2026_10_09): 20 agents;
hop when you see a fitter colleague within the exclusion radius (rule); vision learned socially
(copy a colleague who holds hills more often, judged by a hill-valley test); cycle 40, probation 5.
The worker is fixed per benchmark family by development: CMA-ES workers on DMMOP (rugged composites),
hill-climbing workers on Moving Peaks (cone peaks). Both workers run on both suites and are reported.

DMMOP (our port of the CEC 2022 suite; 24 problems; official protocol: 60 environments of 5000*D
evaluations, informed changes; seeds 1-15 of the official 1-30, for compute). Peak ratio per run.
  D1 coordination: ours_cma beats cma_stay (same CMA workers, never hop) at accuracy 1e-4 on >= 16/24
     problems (two-sided Mann-Whitney on per-run peak ratio, Holm over 24) and is significantly worse on none.
  D2 head to head: ours_cma has the best mean Friedman rank at 1e-4 among {ours_cma, hmso, mqso, de},
     and beats HmSO (Holm over 24) on >= 12/24 problems.
  Reported, not criteria: ours_hill; accuracies 1e-3 and 1e-5; published CEC 2022 results (other instances).
Moving Peaks (scenario 2 and 9 standard variations; 100 changes; seeds 0-29; change detection by
re-evaluating the best solution).
  M1: ours_hill has significantly lower offline error than mQSO on >= 7/10 instances and is
     significantly worse on at most 1 (Mann-Whitney, Holm over 10).
  M2: the same against HmSO.
  M3: ours_hill has significantly higher coverage (visible peaks with a held solution within 1)
     than both mQSO and HmSO on >= 7/10 instances (Holm over 10 per baseline).
  Reported, not criteria: ours_cma, rule_full (no social vision), stay (never hop); published offline errors.

Usage: PYTHONPATH=src python scripts/confirm_final.py --out artifacts/confirm_final_2026_10_09
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

from aumo.algorithms.dmmop_baselines import run_de
from aumo.algorithms.hmso import HmSOConfig, run_hmso
from aumo.algorithms.mqso import run_mqso
from aumo.algorithms.peak_hopping import PeakHoppingConfig, run_peak_hopping
from aumo.problems.dmmop import DMMOPWorld
from aumo.problems.moving_peaks import scenario2
from aumo.progress import ProgressBar

TEAM = PeakHoppingConfig(policy="rule", cycle=40, probation=5, vision=5.0, vision_social=True)
OURS_CMA = replace(TEAM, worker="cma")
OURS_HILL = replace(TEAM, step_bounds=(1e-9, 20.0))
TEAM_METHODS = {
    "ours_cma": OURS_CMA,
    "cma_stay": replace(OURS_CMA, policy="fixed:stay"),
    "ours_hill": OURS_HILL,
    "rule_full": replace(OURS_HILL, vision=None, vision_social=False),
    "stay": replace(OURS_HILL, policy="fixed:stay"),
}
MPB = {
    "S2": {}, "peaks20": {"peaks": 20}, "peaks50": {"peaks": 50}, "peaks100": {"peaks": 100},
    "freq1000": {"change_frequency": 1000}, "freq2500": {"change_frequency": 2500}, "freq10000": {"change_frequency": 10000},
    "shift2": {"move_severity": 2.0}, "shift5": {"move_severity": 5.0}, "dim10": {"dim": 10},
}
DMMOP_SEEDS = range(1, 16)
# Added 2026-10-10 after the confirmation (not part of the pre-registration): HmSO with its radii tuned
# per DMMOP family on dev seed 0 (P2, P5), as our own worker is chosen per family. The default HmSO
# radii (Moving Peaks units) hibernate swarms 0.1 from a peak, far coarser than the 1e-4 accuracy.
HMSO_CONES = HmSOConfig(r=10.0, r_excl=10.0, r_conv=1e-4)
HMSO_COMPOSITES = HmSOConfig(r_conv=1e-5)
MPB_SEEDS = range(30)
# Priority order: primary DMMOP pair, then Moving Peaks, then the DMMOP baselines and the other worker.
PLAN = [("dmmop", m) for m in ("ours_cma", "cma_stay")] + \
       [("mpb", m) for m in ("ours_hill", "mqso", "hmso", "stay", "rule_full", "ours_cma")] + \
       [("dmmop", m) for m in ("hmso", "mqso", "de", "ours_hill")]


def run(task: tuple[str, str, str, int]) -> dict:
    suite, method, instance, seed = task
    started = time.time()
    if suite == "dmmop":
        # DMMOP_INSTANCES=data/dmmop/instances replays the official MATLAB environments (see dmmop.py).
        # DMMOP_DYNAMICS_SEED draws a different instance set (robustness to the port's random dynamics).
        world = DMMOPWorld(int(instance), dynamics_seed=int(os.environ.get("DMMOP_DYNAMICS_SEED", "0")),
                           instances=os.environ.get("DMMOP_INSTANCES") or None)
        if method in TEAM_METHODS:
            run_peak_hopping(world, replace(TEAM_METHODS[method], reevaluate="informed"), seed)
        elif method == "mqso":
            run_mqso(world, seed)
        elif method == "hmso":
            run_hmso(world, seed)
        elif method == "hmso_tuned":
            run_hmso(world, seed, config=HMSO_CONES if world.fun <= 4 else HMSO_COMPOSITES)
        else:
            run_de(world, seed)
        found, total = world.peak_counts()
        row = {"pr_1e-3": found[0] / total, "pr_1e-4": found[1] / total, "pr_1e-5": found[2] / total}
    else:
        overrides = MPB[instance]
        world = scenario2(**overrides)
        budget = 100 * world.change_frequency
        if method in TEAM_METHODS:
            result = run_peak_hopping(world, replace(TEAM_METHODS[method], reevaluate="detect"), seed, budget=budget)
            row = {"offline_error": result.summary["offline_error"], "coverage": result.summary["coverage"]}
        else:
            result = (run_mqso if method == "mqso" else run_hmso)(world, seed, budget)
            row = {"offline_error": result["offline_error"], "coverage": result["coverage"]}
    return {"suite": suite, "method": method, "instance": instance, "seed": seed, **row, "seconds": round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--smoke", action="store_true", help="tiny run of every method to check the pipeline")
    parser.add_argument("--dmmop-seeds", default=None, help="override DMMOP seeds as first-last (extension runs)")
    parser.add_argument("--only", default=None, help="comma-separated suite:method pairs to run (default: the full plan)")
    args = parser.parse_args()
    dmmop_seeds = DMMOP_SEEDS
    if args.dmmop_seeds:
        first, last = map(int, args.dmmop_seeds.split("-"))
        dmmop_seeds = range(first, last + 1)
    plan = [tuple(p.split(":")) for p in args.only.split(",")] if args.only else PLAN
    os.makedirs(args.out, exist_ok=True)
    tasks = []
    for suite, method in plan:
        if suite == "dmmop":
            tasks += [(suite, method, str(p), s) for s in dmmop_seeds for p in range(1, 25)]
        else:
            tasks += [(suite, method, inst, s) for s in MPB_SEEDS for inst in MPB]
    if args.smoke:
        seen, smoke = set(), []
        for t in tasks:
            if (t[0], t[1]) not in seen:
                seen.add((t[0], t[1]))
                smoke.append(t)
        tasks = smoke
    path = os.path.join(args.out, "runs.csv")
    fields = ["suite", "method", "instance", "seed", "pr_1e-3", "pr_1e-4", "pr_1e-5", "offline_error", "coverage", "seconds"]
    done = set()
    if os.path.exists(path):
        with open(path) as handle:
            done = {(r["suite"], r["method"], r["instance"], int(r["seed"])) for r in csv.DictReader(handle)}
    tasks = [t for t in tasks if t not in done]
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle, ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        if new_file:
            writer.writeheader()
        bar = ProgressBar(len(tasks), prefix="final runs")
        # Submitted in priority order; the pool runs them roughly in that order.
        for future in as_completed([pool.submit(run, t) for t in tasks]):
            writer.writerow(future.result())
            handle.flush()
            bar.update()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
