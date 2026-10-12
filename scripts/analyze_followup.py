"""Score the follow-up ablation (rules G, F, I of scripts/ablation_followup.py).

Per problem: two-sided Mann-Whitney on the per-run peak ratio at 1e-4, variant vs ours_cma from
confirm_final on seeds 1-10, Holm over the 24 problems at alpha 0.05.

Usage: python scripts/analyze_followup.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

FOLLOWUP = "artifacts/dmmop_followup_2026_10_11/runs.csv"
CONFIRM = "artifacts/confirm_final_2026_10_09/runs.csv"
METRIC = "pr_1e-4"


def holm(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    order = np.argsort(p)
    reject = np.zeros(len(p), dtype=bool)
    for rank, i in enumerate(order):
        if p[i] > alpha / (len(p) - rank):
            break
        reject[i] = True
    return reject


def main() -> None:
    f = pd.read_csv(FOLLOWUP)
    c = pd.read_csv(CONFIRM)
    c = c[(c.suite == "dmmop") & (c.method == "ours_cma") & (c.seed.isin(f.seed.unique()))].copy()
    c["problem"] = c.instance.astype(int)
    ours = {p: g[METRIC].to_numpy() for p, g in c.groupby("problem")}
    verdicts = {}
    for variant, runs in f.groupby("variant"):
        rows = []
        for p in range(1, 25):
            v = runs[runs.problem == p][METRIC].to_numpy()
            o = ours[p]
            pv = 1.0 if np.allclose(np.r_[v, o], v[0]) else mannwhitneyu(v, o, alternative="two-sided").pvalue
            rows.append((p, o.mean(), v.mean(), pv))
        t = pd.DataFrame(rows, columns=["problem", "rtr", variant, "p"])
        t["sig"] = holm(t.p.to_numpy())
        better = t[t.sig & (t[variant] > t.rtr)].problem.tolist()
        worse = t[t.sig & (t[variant] < t.rtr)].problem.tolist()
        diff = t[variant].mean() - t.rtr.mean()
        extra = ""
        if "vision_mean" in runs:
            extra = f"  vision_mean (units) {runs.vision_mean.mean() / 0.1:.2f}"
        print(f"{variant:11s} mean {t[variant].mean():.3f} vs RTR {t.rtr.mean():.3f} (diff {diff:+.3f}); "
              f"variant better on {better}, worse on {worse}{extra}")
        verdicts[variant] = (len(better), len(worse), diff)
    for name, rule in (("G", "global5"), ("F", "fixed5")):
        b, w, d = verdicts[rule]
        print(f"rule {name}: worse on {w} (<= 2?) and |diff| {abs(d):.3f} (<= 0.02?) -> {'HOLDS' if w <= 2 and abs(d) <= 0.02 else 'FAILS'}")
    b, w, _ = verdicts["imitate_hv"]
    print(f"rule I: imitate_hv better on {b} (>= 1?), worse on {w} (0?) -> {'KEEP sign-independent imitation' if b >= 1 and w == 0 else 'do not keep'}")


if __name__ == "__main__":
    main()
