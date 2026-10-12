"""Figure and per-problem table for the organisers' instances of CEC 2022 (RTR vs no relocation, 30 runs).

Writes figures/dmmop_official.pdf and artifacts/dmmop_official_2026_10_11/per_problem.tex.
Usage: PYTHONPATH=src:scripts python scripts/figure_official.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_confirm_final import decide
from paper_figures import NOHOP, OURS, a12, bars, boot_ci

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "dmmop_official_2026_10_11" / "runs.csv"


def main() -> None:
    data: dict = defaultdict(lambda: defaultdict(list))
    with open(RUNS) as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: int(r["seed"]))
    for r in rows:
        data[int(r["instance"])][r["method"]].append(float(r["pr_1e-4"]))
    probs = sorted(data)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    bars(ax, probs, data, ["ours_cma", "cma_stay"], [OURS, NOHOP], ["RTR", "no relocation"], "")
    ax.set_xticks(range(len(probs)), [f"P{p}" for p in probs], fontsize=7)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "dmmop_official.pdf")
    plt.close(fig)
    dec = decide(data, probs, "ours_cma", "cma_stay", True)
    lines = []
    for p in probs:
        a, b = data[p]["ours_cma"], data[p]["cma_stay"]
        lo, hi = boot_ci(a, b)
        fmt = lambda v: f"{v:+.3f}".replace("-", "$-$")
        verdict = {1: "better", -1: "worse", 0: "n.s."}[dec[p]]
        lines.append(f"    P{p:<6d}& {np.mean(a):.3f} & {np.mean(b):.3f} & {fmt(np.mean(a) - np.mean(b))} [{fmt(lo)}, {fmt(hi)}] & {a12(a, b):.2f} & {verdict} \\\\")
    out = ROOT / "artifacts" / "dmmop_official_2026_10_11" / "per_problem.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
