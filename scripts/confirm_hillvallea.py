"""Pre-registered head-to-head of RTR with HillVallEA (Maree et al., GECCO 2019 code) on CEC 2022 DMMOP.

Registered 2026-10-10, before any HillVallEA run beyond smoke tests (CEC 2013 F1, F6, F12 and DMMOP
P1, P5, P17 with 4 environments, used only to check the bridge). HillVallEA is the authors' C++ code at
commit 1a68ab7 with default settings (src/aumo/algorithms/hillvallea.py, third_party/hillvallea/).
Nothing in RTR is changed; its runs are the confirmed ones (artifacts/confirm_final_2026_10_09, seeds
1-15, and artifacts/confirm_dmmop_seeds16_30_2026_10_10, seeds 16-30).

  V  bridge validity: HillVallEA through our bridge on CEC 2013 (official protocol, seeds 1-50) has a
     mean peak ratio at 1e-4 over the 20 problems within 0.03 of the score of the authors' submitted
     solution files under our scorer (0.892, paper Table 5). If V fails, H1 and H2 are still reported,
     but as results for our build, not as a comparison with HillVallEA as published.
  H1 RTR (ours_cma) has the best Friedman mean rank at 1e-4 among {ours_cma, hve_restart, hve_seeded}
     over the 24 DMMOP problems (seeds 1-30, port instances, dynamics seed 0).
  H2 ours_cma has significantly higher peak ratio at 1e-4 than each HillVallEA variant on >= 12/24
     problems (two-sided Mann-Whitney, Holm over 24 per variant; direction from the rank shift).
  Reported, not criteria: problems where ours_cma is significantly worse; accuracies 1e-3 and 1e-5.

Variants (HillVallEA has no change handling): one run per environment with that environment's budget;
hve_restart starts each environment from scratch; hve_seeded adds the previous elites to the first sample.

Usage: PYTHONPATH=src:scripts python scripts/confirm_hillvallea.py --out artifacts/confirm_hillvallea_2026_10_10
       PYTHONPATH=src:scripts python scripts/confirm_hillvallea.py --out artifacts/confirm_hillvallea_2026_10_10 --analyze
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from aumo.algorithms.hillvallea import run_hillvallea
from aumo.problems.cec2013 import ACCURACY, CEC2013World
from aumo.problems.dmmop import DMMOPWorld
from aumo.progress import ProgressBar

CEC2013_SEEDS = range(1, 51)
DMMOP_SEEDS = range(1, 31)
HVE_SUBMISSION_1E4 = 0.892
RTR_RUNS = ["artifacts/confirm_final_2026_10_09/runs.csv", "artifacts/confirm_dmmop_seeds16_30_2026_10_10/runs.csv"]
FIELDS = ["suite", "method", "instance", "seed", "pr_1e-3", "pr_1e-4", "pr_1e-5", "evaluations", "seconds"]


def run(task: tuple[str, str, int, int]) -> dict:
    suite, method, instance, seed = task
    started = time.time()
    if suite == "cec2013":
        world = CEC2013World(instance)
        run_hillvallea(world, seed)
        pr = dict(zip(ACCURACY, world.peak_ratios()))
        row = {f"pr_{e}": pr[float(e)] for e in ("1e-3", "1e-4", "1e-5")}
    else:
        world = DMMOPWorld(instance)
        run_hillvallea(world, seed, seeded=method == "hve_seeded")
        found, total = world.peak_counts()
        row = {"pr_1e-3": found[0] / total, "pr_1e-4": found[1] / total, "pr_1e-5": found[2] / total}
    return {"suite": suite, "method": method, "instance": instance, "seed": seed, **row,
            "evaluations": world.evaluations, "seconds": round(time.time() - started, 1)}


def analyze(path: str) -> None:
    from analyze_confirm_final import decide

    cec, dm = defaultdict(list), defaultdict(lambda: defaultdict(list))
    with open(path) as handle:
        for r in csv.DictReader(handle):
            if r["suite"] == "cec2013":
                cec[int(r["instance"])].append(float(r["pr_1e-4"]))
            else:
                dm[int(r["instance"])][r["method"]].append(float(r["pr_1e-4"]))
    for rtr in RTR_RUNS:
        with open(rtr) as handle:
            for r in csv.DictReader(handle):
                if r["suite"] == "dmmop" and r["method"] == "ours_cma" and int(r["seed"]) in DMMOP_SEEDS:
                    dm[int(r["instance"])]["ours_cma"].append(float(r["pr_1e-4"]))
    if len(cec) == 20:
        mean = np.mean([np.mean(v) for v in cec.values()])
        print("CEC 2013 HillVallEA (our build) per problem: " + " ".join(f"F{p} {np.mean(cec[p]):.3f}" for p in sorted(cec)))
        print(f"V mean {mean:.3f} vs submission {HVE_SUBMISSION_1E4} -> {'PASS' if abs(mean - HVE_SUBMISSION_1E4) <= 0.03 else 'FAIL'}")
    else:
        print(f"V incomplete ({len(cec)}/20 problems)")
    methods = ["ours_cma", "hve_restart", "hve_seeded"]
    problems = sorted(dm)
    complete = [p for p in problems if all(len(dm[p][m]) == len(DMMOP_SEEDS) for m in methods)]
    print("\n| problem | " + " | ".join(methods) + " |\n|---|" + "---:|" * len(methods))
    for p in problems:
        print(f"| P{p} | " + " | ".join(f"{np.mean(dm[p][m]):.3f}" if dm[p][m] else "-" for m in methods) + " |")
    if len(complete) < 24:
        print(f"H1/H2 incomplete ({len(complete)}/24 problems)")
        return
    print("| **mean** | " + " | ".join(f"{np.mean([np.mean(dm[p][m]) for p in problems]):.3f}" for m in methods) + " |")
    ranks = []
    for p in problems:
        vals = np.array([-np.mean(dm[p][m]) for m in methods])
        ranks.append([(vals < v).sum() + ((vals == v).sum() + 1) / 2 for v in vals])
    mean_rank = dict(zip(methods, np.mean(ranks, axis=0)))
    print("\nH1 Friedman mean ranks: " + ", ".join(f"{m} {r:.2f}" for m, r in mean_rank.items())
          + f" -> {'PASS' if min(mean_rank, key=mean_rank.get) == 'ours_cma' else 'FAIL'}")
    ok = True
    for variant in methods[1:]:
        d = decide(dm, problems, "ours_cma", variant, True)
        won, lost = [p for p, v in d.items() if v == 1], [p for p, v in d.items() if v == -1]
        ok &= len(won) >= 12
        print(f"H2 vs {variant}: better on {len(won)}/24 {won}; worse on {lost or 'none'}")
    print(f"H2 -> {'PASS' if ok else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--only", choices=["cec2013", "dmmop"], default=None)
    args = parser.parse_args()
    path = os.path.join(args.out, "runs.csv")
    if args.analyze:
        analyze(path)
        return
    os.makedirs(args.out, exist_ok=True)
    tasks = []
    if args.only in (None, "cec2013"):
        tasks += [("cec2013", "hve", p, s) for s in CEC2013_SEEDS for p in range(1, 21)]
    if args.only in (None, "dmmop"):
        tasks += [("dmmop", m, p, s) for s in DMMOP_SEEDS for m in ("hve_restart", "hve_seeded") for p in range(1, 25)]
    done = set()
    if os.path.exists(path):
        with open(path) as handle:
            done = {(r["suite"], r["method"], int(r["instance"]), int(r["seed"])) for r in csv.DictReader(handle)}
    tasks = [t for t in tasks if t not in done]
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle, ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        bar = ProgressBar(len(tasks), prefix="hillvallea runs")
        for future in as_completed([pool.submit(run, t) for t in tasks]):
            writer.writerow(future.result())
            handle.flush()
            bar.update()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
