"""Score the pre-registered criteria of scripts/confirm_final.py (works on partial results too).

Usage: PYTHONPATH=src:scripts python scripts/analyze_confirm_final.py artifacts/confirm_final_2026_10_09/runs.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

from stats import holm

ALPHA = 0.05
# Published CEC 2022 peak ratio at 1e-4 (MATLAB instances, 30 runs; context only): winner, runner-up, HmSO.
CEC2022_1E4 = {
    "AMLP-RS-CMSA-ESII": [0.696, 0.997, 0.954, 0.970, 0.547, 0.461, 0.634, 0.608, 0.608, 0.600, 0.564, 0.563, 0.570, 0.580, 0.365, 0.533,
                          0.505, 0.938, 0.873, 0.950, 0.420, 0.581, 0.651, 0.527],
    "AMPSO": [0.765, 1.000, 0.999, 1.000, 0.551, 0.264, 0.618, 0.441, 0.441, 0.287, 0.260, 0.256, 0.310, 0.288, 0.195, 0.213,
              0.543, 0.921, 0.717, 0.777, 0.335, 0.254, 0.543, 0.369],
    "HmSO (CEC 2022 entry)": [0.625, 0.993, 0.981, 0.994, 0.571, 0.255, 0.596, 0.259, 0.259, 0.103, 0.090, 0.073, 0.133, 0.115, 0.109,
                              0.127, 0.218, 0.239, 0.415, 0.421, 0.262, 0.182, 0.186, 0.100],
}
# Published Moving Peaks offline errors (alc1218.github.io/MPB; context only).
MPB_PUBLISHED = {
    "mQSO (2004)": {"S2": 1.75, "peaks20": 2.42, "peaks50": 2.50, "peaks100": 2.36, "shift2": 2.40, "shift5": 4.24, "dim10": 4.38,
                    "freq1000": 5.20, "freq2500": 2.85, "freq10000": 1.10},
    "HmSO (2010)": {"S2": 1.46, "peaks20": 1.67, "peaks50": 1.75, "peaks100": 1.78, "shift2": 2.54, "shift5": 6.04, "dim10": 4.63,
                    "freq1000": 4.79, "freq2500": 2.43, "freq10000": 0.90},
    "FTMPSO (2013)": {"S2": 0.67, "peaks20": 0.93, "peaks50": 1.32, "peaks100": 1.61, "shift2": 1.20, "shift5": 1.69,
                      "freq1000": 2.36, "freq2500": 1.21, "freq10000": 0.43},
}


def decide(samples: dict, keys: list, x: str, y: str, higher_better: bool) -> dict:
    """key -> +1 if x significantly better than y, -1 if worse, 0 otherwise (Mann-Whitney, Holm over keys)."""
    usable = [k for k in keys if len(samples[k].get(x, [])) > 1 and len(samples[k].get(y, [])) > 1]
    pvals = {}
    for k in usable:
        a, b = np.array(samples[k][x]), np.array(samples[k][y])
        pvals[k] = 1.0 if np.all(a == a[0]) and np.all(b == a[0]) else mannwhitneyu(a, b).pvalue
    adjusted = holm(pvals) if pvals else {}
    out = {}
    for k in usable:
        # Direction from the rank statistic the test is based on (A12 vs 0.5), not from the means: with
        # ties at a ceiling, a single outlier can put the mean on the other side of a significant shift.
        a, b = np.array(samples[k][x]), np.array(samples[k][y])
        shift = (a[:, None] > b[None, :]).mean() - (a[:, None] < b[None, :]).mean()
        better = shift > 0 if higher_better else shift < 0
        out[k] = 0 if adjusted[k] >= ALPHA else (1 if better else -1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs")
    parser.add_argument("--hmso", default="hmso", help="HmSO variant in D2 (hmso_tuned for the official instances)")
    args = parser.parse_args()
    dmmop: dict = defaultdict(lambda: defaultdict(list))
    dmmop_acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    mpb_err: dict = defaultdict(lambda: defaultdict(list))
    mpb_cov: dict = defaultdict(lambda: defaultdict(list))
    with open(args.runs) as handle:
        for r in csv.DictReader(handle):
            if r["suite"] == "dmmop":
                dmmop[int(r["instance"])][r["method"]].append(float(r["pr_1e-4"]))
                for acc in ("1e-3", "1e-4", "1e-5"):
                    dmmop_acc[acc][int(r["instance"])][r["method"]].append(float(r[f"pr_{acc}"]))
            else:
                mpb_err[r["instance"]][r["method"]].append(float(r["offline_error"]))
                mpb_cov[r["instance"]][r["method"]].append(float(r["coverage"]))

    if dmmop:
        problems = sorted(dmmop)
        methods = [m for m in ("ours_cma", "cma_stay", "ours_hill", "hmso", "hmso_tuned", "mqso", "de") if any(m in dmmop[p] for p in problems)]
        print(f"## DMMOP (our port), mean peak ratio at 1e-4 (runs per cell in brackets when not 15)")
        print("| problem | " + " | ".join(methods) + " | " + " | ".join(CEC2022_1E4) + " |")
        print("|---|" + "---:|" * (len(methods) + len(CEC2022_1E4)))
        for p in problems:
            cells = []
            for m in methods:
                v = dmmop[p].get(m, [])
                cells.append(f"{np.mean(v):.3f}" + (f" [{len(v)}]" if v and len(v) != 15 else "") if v else "-")
            print(f"| P{p} | " + " | ".join(cells) + " | " + " | ".join(f"{CEC2022_1E4[k][p - 1]:.3f}" for k in CEC2022_1E4) + " |")
        for acc in ("1e-3", "1e-5"):
            means = {m: np.mean([np.mean(dmmop_acc[acc][p][m]) for p in problems if dmmop_acc[acc][p].get(m)]) for m in methods}
            print(f"\nMean over problems at {acc}: " + ", ".join(f"{m} {v:.3f}" for m, v in means.items()))
        print("\n### Pre-registered DMMOP criteria")
        d1 = decide(dmmop, problems, "ours_cma", "cma_stay", True)
        if len(d1) == 24:
            won, lost = [p for p, v in d1.items() if v == 1], [p for p, v in d1.items() if v == -1]
            print(f"D1 ours_cma beats cma_stay on {len(won)}/24 {won}; worse on {lost or 'none'} -> "
                  f"{'PASS' if len(won) >= 16 and not lost else 'FAIL'}")
        else:
            print(f"D1 incomplete ({len(d1)}/24 problems with both methods)")
        rivals = ["ours_cma", args.hmso, "mqso", "de"]
        complete = [p for p in problems if all(len(dmmop[p].get(m, [])) > 0 for m in rivals)]
        if len(complete) == 24:
            ranks = np.array([np.argsort(np.argsort([-np.mean(dmmop[p][m]) for m in rivals])) + 1.0 for p in complete])
            # average ties
            for i, p in enumerate(complete):
                vals = np.array([np.mean(dmmop[p][m]) for m in rivals])
                for v in np.unique(vals):
                    tie = vals == v
                    ranks[i, tie] = ranks[i, tie].mean()
            mean_rank = dict(zip(rivals, ranks.mean(axis=0)))
            d2h = decide(dmmop, problems, "ours_cma", args.hmso, True)
            wins = [p for p, v in d2h.items() if v == 1]
            best = min(mean_rank, key=mean_rank.get)
            print("D2 Friedman mean ranks at 1e-4: " + ", ".join(f"{m} {r:.2f}" for m, r in sorted(mean_rank.items(), key=lambda kv: kv[1])))
            print(f"D2 ours_cma beats HmSO on {len(wins)}/24 {wins}; worse on {[p for p, v in d2h.items() if v == -1] or 'none'} -> "
                  f"{'PASS' if best == 'ours_cma' and len(wins) >= 12 else 'FAIL'}")
        else:
            print(f"D2 incomplete ({len(complete)}/24 problems with all four methods)")

    if mpb_err:
        order = ["S2", "peaks20", "peaks50", "peaks100", "freq1000", "freq2500", "freq10000", "shift2", "shift5", "dim10"]
        insts = [i for i in order if i in mpb_err]
        methods = [m for m in ("ours_hill", "ours_cma", "rule_full", "stay", "mqso", "hmso") if any(m in mpb_err[i] for i in insts)]
        for title, data, fmt in (("offline error (lower is better)", mpb_err, "{:.2f}"), ("coverage (higher is better)", mpb_cov, "{:.3f}")):
            print(f"\n## Moving Peaks: mean {title}")
            extra = list(MPB_PUBLISHED) if data is mpb_err else []
            print("| instance | " + " | ".join(methods + extra) + " |\n|---|" + "---:|" * (len(methods) + len(extra)))
            for i in insts:
                cells = [(fmt.format(np.mean(data[i][m])) + (f" [{len(data[i][m])}]" if len(data[i][m]) != 30 else "")) if data[i].get(m) else "-" for m in methods]
                cells += [f"{MPB_PUBLISHED[k].get(i, float('nan')):.2f}" for k in extra]
                print(f"| {i} | " + " | ".join(cells) + " |")
        print("\n### Pre-registered Moving Peaks criteria")
        for label, base in (("M1", "mqso"), ("M2", "hmso")):
            dec = decide(mpb_err, insts, "ours_hill", base, False)
            if len(dec) == 10:
                won, lost = [i for i, v in dec.items() if v == 1], [i for i, v in dec.items() if v == -1]
                print(f"{label} offline error vs {base}: better on {len(won)}/10 {won}; worse on {lost or 'none'} -> "
                      f"{'PASS' if len(won) >= 7 and len(lost) <= 1 else 'FAIL'}")
            else:
                print(f"{label} incomplete ({len(dec)}/10 instances)")
        covs = [decide(mpb_cov, insts, "ours_hill", base, True) for base in ("mqso", "hmso")]
        if all(len(c) == 10 for c in covs):
            both = [i for i in insts if covs[0][i] == 1 and covs[1][i] == 1]
            print(f"M3 coverage better than both mQSO and HmSO on {len(both)}/10 {both} -> {'PASS' if len(both) >= 7 else 'FAIL'}")
        else:
            print("M3 incomplete")


if __name__ == "__main__":
    main()
