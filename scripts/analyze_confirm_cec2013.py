"""Score the pre-registered criteria of scripts/confirm_cec2013.py (works on partial results too).

Usage: PYTHONPATH=src:scripts python scripts/analyze_confirm_cec2013.py artifacts/confirm_cec2013_2026_10_10/runs.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_confirm_final import decide

PUBLISHED = Path(__file__).resolve().parents[1] / "artifacts" / "cec2013_published"
ACC = "1e-04"
ACCS = ["0.1", "0.01", "0.001", "0.0001", "1e-05"]


def published() -> dict[str, np.ndarray]:
    """Mean peak ratio at 1e-4 per problem: 2013 entrants (their tables) and re-scored later entries."""
    out = {}
    for path in sorted((PUBLISHED / "competition2013").glob("*_PR.dat")):
        out[path.stem[:-3] + " (2013)"] = np.loadtxt(path)[:, 3]
    rescored = PUBLISHED / "rescored_runs.csv"
    if rescored.exists():
        runs: dict = defaultdict(lambda: defaultdict(list))
        with open(rescored) as handle:
            for r in csv.DictReader(handle):
                runs[r["method"]][int(r["problem"])].append(float(r["pr_0.0001"]))
        for name, per in runs.items():
            out[name + " (re-scored)"] = np.array([np.mean(per[p]) for p in range(1, 21)])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs")
    args = parser.parse_args()
    data: dict = defaultdict(lambda: defaultdict(list))
    every: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(args.runs) as handle:
        for r in csv.DictReader(handle):
            data[int(r["problem"])][r["method"]].append(float(r["pr_0.0001"]))
            for a in ACCS:
                every[a][int(r["problem"])][r["method"]].append(float(r[f"pr_{a}"]))
    problems = sorted(data)
    methods = [m for m in ("ours", "stay", "ours_noarchive") if any(m in data[p] for p in problems)]
    pub = published()
    print("## CEC 2013 niching: mean peak ratio at 1e-4 (runs per cell in brackets when not 50)")
    print("| problem | " + " | ".join(methods) + " |\n|---|" + "---:|" * len(methods))
    for p in problems:
        cells = [f"{np.mean(data[p][m]):.3f}" + (f" [{len(data[p][m])}]" if len(data[p][m]) != 50 else "") if data[p].get(m) else "-"
                 for m in methods]
        print(f"| F{p} | " + " | ".join(cells) + " |")
    for a in ACCS:
        means = {m: np.mean([np.mean(every[a][p][m]) for p in problems if every[a][p].get(m)]) for m in methods}
        print(f"\nMean over problems at {a}: " + ", ".join(f"{m} {v:.3f}" for m, v in means.items()))

    print("\n### Pre-registered criteria")
    dec = decide(data, problems, "ours", "stay", True)
    if len(dec) == 20:
        won, lost = [p for p, v in dec.items() if v == 1], [p for p, v in dec.items() if v == -1]
        print(f"N1 ours beats stay on {len(won)}/20 {won}; worse on {lost or 'none'} -> {'PASS' if len(won) >= 7 and len(lost) <= 2 else 'FAIL'}")
        mo, ms = (np.mean([np.mean(data[p][m]) for p in problems]) for m in ("ours", "stay"))
        print(f"N2 mean peak ratio at 1e-4: ours {mo:.3f}, stay {ms:.3f} -> {'PASS' if mo > ms else 'FAIL'}")
    else:
        print(f"N1/N2 incomplete ({len(dec)}/20 problems with both methods)")

    if len(problems) == 20 and "ours" in methods and all(data[p].get("ours") for p in problems):
        ours = np.array([np.mean(data[p]["ours"]) for p in problems])
        print("\n## Published (context; 1e-4): mean over 20 problems, and problems where ours is higher / lower by > 0.02")
        print("| method | mean | ours higher | ours lower |\n|---|---:|---:|---:|")
        rows = sorted(pub.items(), key=lambda kv: -kv[1].mean())
        for name, v in rows:
            print(f"| {name} | {v.mean():.3f} | {(ours > v + 0.02).sum()} | {(ours < v - 0.02).sum()} |")
        print(f"| **ours** | {ours.mean():.3f} | | |")
        table = {**{k: v for k, v in rows}, "ours": ours}
        ranks = np.array([np.argsort(np.argsort([-table[k][i] for k in table])) + 1.0 for i in range(20)])
        for i in range(20):
            vals = np.array([table[k][i] for k in table])
            for v in np.unique(vals):
                tie = vals == v
                ranks[i, tie] = ranks[i, tie].mean()
        mean_rank = dict(zip(table, ranks.mean(axis=0)))
        order = sorted(mean_rank.items(), key=lambda kv: kv[1])
        print(f"\nFriedman mean rank among {len(table)} methods at 1e-4: " + ", ".join(f"{k} {r:.2f}" for k, r in order))


if __name__ == "__main__":
    main()
