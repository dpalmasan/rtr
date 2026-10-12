"""Friedman tests with Holm post-hoc comparisons against RTR (Demsar 2006), on the per-problem mean
peak ratio at 1e-4. Reported for: the organisers' instances (D2 methods, plus no relocation), our
draws (five methods), the HillVallEA comparison (our draws), and the CEC 2013 ranking.

Usage: python scripts/analyze_friedman.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, rankdata

sys.path.insert(0, "scripts")


def friedman_holm(table: pd.DataFrame, control: str, name: str) -> None:
    """table: rows = problems, columns = methods (higher is better)."""
    ranks = np.array([rankdata(-row) for row in table.to_numpy()])
    mean_rank = dict(zip(table.columns, ranks.mean(axis=0)))
    n, k = table.shape
    stat, p = friedmanchisquare(*[table[c].to_numpy() for c in table.columns])
    se = np.sqrt(k * (k + 1) / (6 * n))
    others = [c for c in table.columns if c != control]
    pv = {c: 2 * norm.sf(abs(mean_rank[c] - mean_rank[control]) / se) for c in others}
    order = sorted(others, key=pv.get)
    holm, running = {}, 0.0
    for i, c in enumerate(order):
        running = max(running, min(1.0, pv[c] * (len(order) - i)))
        holm[c] = running
    print(f"\n{name}: N={n}, k={k}, Friedman chi2={stat:.1f}, p={p:.2g}")
    for c in sorted(table.columns, key=mean_rank.get):
        extra = "" if c == control else f"  Holm-adjusted p vs {control} = {holm[c]:.3g}"
        print(f"  {c:28s} mean rank {mean_rank[c]:.2f}{extra}")


def runs_table(path: str, methods: list[str], seeds=None) -> pd.DataFrame:
    r = pd.read_csv(path)
    r = r[(r.suite == "dmmop") & r.method.isin(methods)].copy()
    if seeds is not None:
        r = r[r.seed.isin(seeds)]
    r["p"] = r.instance.astype(int)
    return r.groupby(["p", "method"])["pr_1e-4"].mean().unstack()[methods]


def main() -> None:
    off = runs_table("artifacts/dmmop_official_2026_10_11/runs.csv", ["ours_cma", "cma_stay", "mqso", "hmso_tuned", "de"])
    friedman_holm(off.drop(columns="cma_stay"), "ours_cma", "Organisers' instances, D2 methods (30 runs)")
    friedman_holm(off, "ours_cma", "Organisers' instances, with no relocation (30 runs)")
    draws = pd.concat([pd.read_csv(f) for f in ["artifacts/confirm_final_2026_10_09/runs.csv",
                                                 "artifacts/confirm_dmmop_seeds16_30_2026_10_10/runs.csv",
                                                 "artifacts/dmmop_hmso_tuned_2026_10_10/runs.csv"]])
    draws = draws[draws.suite == "dmmop"].copy()
    draws["p"] = draws.instance.astype(int)
    t = draws.groupby(["p", "method"])["pr_1e-4"].mean().unstack()[["ours_cma", "mqso", "hmso", "hmso_tuned", "de"]]
    friedman_holm(t, "ours_cma", "Our draws, five methods (30 runs; default HmSO 15)")
    h = pd.read_csv("artifacts/confirm_hillvallea_2026_10_10/runs.csv")
    h = h[h.suite == "dmmop"].copy()
    h["p"] = h.instance.astype(int)
    hv = h.groupby(["p", "method"])["pr_1e-4"].mean().unstack()
    rt = draws[draws.method == "ours_cma"].groupby("p")["pr_1e-4"].mean()
    hv["ours_cma"] = rt
    friedman_holm(hv[["ours_cma", "hve_seeded", "hve_restart"]], "ours_cma", "HillVallEA comparison, our draws (30 runs)")


if __name__ == "__main__":
    main()
