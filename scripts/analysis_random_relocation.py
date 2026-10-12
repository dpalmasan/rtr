"""Displacement probability of random relocation (Proposition 3 of the analysis section; post hoc).

Under random relocation at rate q, an agent that holds an optimum at the start of an environment is
displaced before the environment ends with probability at least 1 - (1 - q)^m, m = floor(T / c)
(T rounds per environment, cycle c = 40). This estimate ignores rediscovery (which lowers the loss)
and other ways of losing an optimum (which raise it), so it is compared with the measured relative
loss 1 - PR_random / PR_RTR of the ablation (accuracy 1e-4, seeds 1-10) by rank correlation.

Usage: python scripts/analysis_random_relocation.py
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ABLATION = "artifacts/dmmop_ablation_2026_10_10"
CONFIRM = "artifacts/confirm_final_2026_10_09/runs.csv"
AGENTS, CYCLE = 20, 40


def rounds_per_environment(dim: int) -> float:
    lam = 4 + math.floor(3 * math.log(dim))
    return 5000 * dim / (AGENTS * lam)


def main() -> None:
    with open(f"{ABLATION}/hop_rate_dev_seed0.json") as handle:
        q = pd.Series({int(k): v for k, v in json.load(handle).items()}).sort_index()
    runs = pd.read_csv(f"{ABLATION}/runs.csv")
    random_pr = runs[runs.variant == "random_hop"].groupby("problem")["pr_1e-4"].mean()
    c = pd.read_csv(CONFIRM)
    c = c[(c.suite == "dmmop") & (c.method == "ours_cma") & (c.seed <= 10)].copy()
    c["problem"] = c.instance.astype(int)
    rtr_pr = c.groupby("problem")["pr_1e-4"].mean()
    dims = pd.Series({p: 5 if p <= 16 else 10 for p in q.index})
    m = dims.map(lambda d: math.floor(rounds_per_environment(d) / CYCLE))
    estimate = 1 - (1 - q) ** m
    measured = 1 - random_pr / rtr_pr
    table = pd.DataFrame({"q": q, "m": m, "displacement": estimate, "measured_loss": measured}).round(3)
    print(table.to_string())
    below = table[table.displacement < table.measured_loss]
    print(f"T: 5-D {rounds_per_environment(5):.0f}, 10-D {rounds_per_environment(10):.0f} rounds")
    print(f"estimate range {estimate.min():.3f}-{estimate.max():.3f}; Spearman rho {spearmanr(estimate, measured).statistic:.3f}")
    print(f"estimate above measured on {int((estimate > measured).sum())}/24; below on {below.index.tolist()} "
          f"by up to {float((below.measured_loss - below.displacement).max()):.3f}; "
          f"mean abs deviation {float(np.abs(estimate - measured).mean()):.3f}")


if __name__ == "__main__":
    main()
