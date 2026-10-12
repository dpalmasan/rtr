"""Figures and tables of the RTR paper, from the committed run files (works on partial data).

Usage: PYTHONPATH=src:scripts python scripts/paper_figures.py
Writes figures/*.svg and artifacts/paper_tables.md.
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

ROOT = Path(__file__).resolve().parents[1]
ART, FIG = ROOT / "artifacts", ROOT / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
OURS, NOHOP, OTHER = "#1b6ca8", "#b0b0b0", "#e08a2c"


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def a12(x, y) -> float:
    """Vargha-Delaney A: probability that a run of x beats a run of y (ties count half)."""
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y)))


def boot_ci(x, y, n=4000, seed=0) -> tuple[float, float]:
    """95% bootstrap interval of mean(x) - mean(y)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    d = rng.choice(x, (n, len(x))).mean(axis=1) - rng.choice(y, (n, len(y))).mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


BY_SEED: dict = defaultdict(lambda: defaultdict(dict))  # problem -> method -> seed -> peak ratio at 1e-4


def dmmop_runs() -> dict:
    """problem -> method -> per-run peak ratio at 1e-4 (in seed order), all DMMOP files on the default instances."""
    for f in ("confirm_final_2026_10_09", "confirm_dmmop_seeds16_30_2026_10_10", "dmmop_hmso_tuned_2026_10_10"):
        for r in read(ART / f / "runs.csv"):
            if r.get("suite", "dmmop") == "dmmop":
                BY_SEED[int(r["instance"])][r["method"]][int(r["seed"])] = float(r["pr_1e-4"])
    data: dict = defaultdict(lambda: defaultdict(list))
    for p, methods in BY_SEED.items():
        for m, seeds in methods.items():
            data[p][m] = [seeds[s] for s in sorted(seeds)]
    return data


def seeds(p: int, m: str, chosen) -> list[float]:
    return [BY_SEED[p][m][s] for s in chosen if s in BY_SEED[p][m]]


def comparison_table(data: dict, keys: list, x: str, y: str, label) -> list[str]:
    dec = decide(data, keys, x, y, True)
    lines = [f"| problem | {x} | {y} | diff [95% CI] | A12 | Holm verdict |", "|---|---:|---:|---|---:|---|"]
    for k in keys:
        if not (data[k].get(x) and data[k].get(y)):
            continue
        a, b = data[k][x], data[k][y]
        lo, hi = boot_ci(a, b)
        verdict = {1: "better", -1: "worse", 0: "n.s."}.get(dec.get(k), "-")
        lines.append(f"| {label(k)} | {np.mean(a):.3f} | {np.mean(b):.3f} | {np.mean(a) - np.mean(b):+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                     f"{a12(a, b):.2f} | {verdict} |")
    return lines


def bars(ax, keys, data, methods, colors, labels, title):
    width = 0.8 / len(methods)
    for j, (m, c, lab) in enumerate(zip(methods, colors, labels)):
        means = [np.mean(data[k][m]) if data[k].get(m) else np.nan for k in keys]
        err = [1.96 * np.std(data[k][m], ddof=1) / np.sqrt(len(data[k][m])) if len(data[k].get(m, [])) > 1 else 0 for k in keys]
        ax.bar(np.arange(len(keys)) + (j - (len(methods) - 1) / 2) * width, means, width, yerr=err, color=c, label=lab,
               error_kw={"lw": 0.6, "capsize": 0})
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("peak ratio")
    ax.legend(frameon=False, ncol=len(methods), loc="upper left", fontsize=7.5)


def hop_rule_figure() -> None:
    """Illustration of the rule on a 1-D landscape."""
    x = np.linspace(0, 10, 600)
    f = np.maximum.reduce([8 - 2.2 * np.abs(x - 2.5), 6.5 - 2.0 * np.abs(x - 5.6), 5 - 1.5 * np.abs(x - 8.6), np.zeros_like(x)])
    fx = lambda p: float(np.interp(p, x, f))
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.plot(x, f, color="0.3", lw=1.2)
    agents = {"A": 2.45, "B": 1.6, "C": 5.5}
    for name, p in agents.items():
        ax.plot(p, fx(p), "o", ms=9, color=OURS if name != "B" else OTHER, zorder=3)
        ax.annotate(name, (p, fx(p)), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, weight="bold")
    ax.annotate("", xy=(8.5, fx(8.5) + 0.5), xytext=(1.6, fx(1.6) + 0.6),
                arrowprops={"arrowstyle": "->", "color": OTHER, "lw": 1.2, "connectionstyle": "arc3,rad=-0.25"})
    ax.text(6.3, 9.6, "B sees a fitter colleague (A) on its hill: redundant, so it hops\nto a place far from the colleagues it knows (here, the empty hill)",
            ha="center", fontsize=7.5, color=OTHER, va="top")
    ax.plot([5.5 - 1.2, 5.5 + 1.2], [fx(5.5) - 0.35] * 2, color=OURS, lw=2, alpha=0.5)
    ax.text(5.5, 1.0, "C sees only within its vision (bar): no fitter colleague\nin sight, so it stays and holds the second hill",
            ha="center", fontsize=7.5, color=OURS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 10)
    ax.set_ylabel("fitness")
    fig.tight_layout()
    fig.savefig(FIG / "hop_rule.svg")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    out: list[str] = []
    hop_rule_figure()

    # DMMOP: D1 with all available seeds; hmso_tuned
    dm = dmmop_runs()
    probs = sorted(dm)
    if probs:
        n_seeds = min(len(dm[p].get("ours_cma", [])) for p in probs)
        out += [f"## DMMOP: ours vs no-hop at 1e-4 ({n_seeds} runs per problem)", ""]
        out += comparison_table(dm, probs, "ours_cma", "cma_stay", lambda p: f"P{p}") + [""]
        if any(dm[p].get("hmso_tuned") for p in probs):
            out += ["## DMMOP: ours vs HmSO tuned at 1e-4", ""] + comparison_table(dm, probs, "ours_cma", "hmso_tuned", lambda p: f"P{p}") + [""]
        means = {m: np.mean([np.mean(dm[p][m]) for p in probs if dm[p].get(m)]) for m in ("ours_cma", "cma_stay", "mqso", "hmso", "hmso_tuned", "de")
                 if all(dm[p].get(m) for p in probs)}
        out += ["Mean peak ratio at 1e-4 over 24 problems: " + ", ".join(f"{m} {v:.3f}" for m, v in means.items()), ""]
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        bars(ax, probs, dm, ["ours_cma", "cma_stay"], [OURS, NOHOP], ["RTR", "no relocation"],
             "DMMOP: peak ratio at accuracy 1e-4 (mean, 95% CI)")
        ax.set_xticks(range(len(probs)), [f"P{p}" for p in probs], fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "dmmop_d1.svg")
        plt.close(fig)

    # DMMOP instance draws
    draws: dict = {0: defaultdict(lambda: defaultdict(list))}
    for p in range(1, 25):
        for m in ("ours_cma", "cma_stay"):
            draws[0][p][m] = seeds(p, m, range(1, 6))
    for d in range(1, 5):
        rows = read(ART / "dmmop_instance_draws_2026_10_10" / f"draw{d}" / "runs.csv")
        if rows:
            dd: dict = defaultdict(lambda: defaultdict(list))
            for r in rows:
                dd[int(r["instance"])][r["method"]].append(float(r["pr_1e-4"]))
            draws[d] = dd
    if len(draws) > 1:
        out += ["## DMMOP instance draws (seeds 1-5; draw 0 = the confirmation instances, seeds 1-5)", "",
                "| draw | ours | no-hop | problems complete |", "|---|---:|---:|---:|"]
        for d, dd in draws.items():
            complete = [p for p in range(1, 25) if len(dd[p].get("ours_cma", [])) >= 5 and len(dd[p].get("cma_stay", [])) >= 5]
            if complete:
                o = np.mean([np.mean(dd[p]["ours_cma"][:5]) for p in complete])
                s = np.mean([np.mean(dd[p]["cma_stay"][:5]) for p in complete])
                out.append(f"| {d} | {o:.3f} | {s:.3f} | {len(complete)} |")
        out.append("")

    # CEC 2013
    cec: dict = defaultdict(lambda: defaultdict(list))
    for r in read(ART / "confirm_cec2013_2026_10_10" / "runs.csv"):
        cec[int(r["problem"])][r["method"]].append(float(r["pr_0.0001"]))
    if cec:
        fns = sorted(cec)
        out += ["## CEC 2013: ours vs no-hop (both with the archive) at 1e-4", ""]
        out += comparison_table(cec, fns, "ours", "stay", lambda p: f"F{p}") + [""]
        pub: dict = defaultdict(lambda: defaultdict(list))
        for r in read(ART / "cec2013_published" / "rescored_runs.csv"):
            pub[int(r["problem"])][r["method"]].append(float(r["pr_0.0001"]))
        for p in fns:
            for m, v in pub[p].items():
                cec[p][m] = v
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        bars(ax, fns, cec, ["ours", "stay", "RS-CMSA-ESII"], [OURS, NOHOP, OTHER],
             ["RTR, with archive", "no relocation, with archive", "RS-CMSA-ESII (published runs)"],
             "CEC 2013 niching: peak ratio at accuracy 1e-4 (mean, 95% CI)")
        ax.set_xticks(range(len(fns)), [f"F{p}" for p in fns], fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "cec2013_n1.svg")
        plt.close(fig)

    # Ablation and sensitivity
    ab = read(ART / "dmmop_ablation_2026_10_10" / "runs.csv")
    if ab:
        abd: dict = defaultdict(lambda: defaultdict(list))
        for r in ab:
            abd[int(r["problem"])][r["variant"]].append(float(r["pr_1e-4"]))
        for p in range(1, 25):
            abd[p]["ours"] = seeds(p, "ours_cma", range(1, 11))   # seeds 1-10 of the confirmation
        out += ["## DMMOP component ablation (seeds 1-10, 1e-4)", "", "| variant | mean | better / worse than ours (Holm) |", "|---|---:|---|"]
        out.append(f"| ours | {np.mean([np.mean(abd[p]['ours']) for p in range(1, 25)]):.3f} | |")
        for v in ("random_hop", "uniform_dest", "full_vision", "fixed_vision"):
            if all(abd[p].get(v) for p in range(1, 25)):
                dec = decide(abd, list(range(1, 25)), "ours", v, True)
                won, lost = sum(x == 1 for x in dec.values()), sum(x == -1 for x in dec.values())
                out.append(f"| {v} | {np.mean([np.mean(abd[p][v]) for p in range(1, 25)]):.3f} | ours better on {won}, worse on {lost} |")
        out.append("")
        sens = [(v, k) for v, k in (("agents=10", "agents"), ("agents=40", "agents"), ("cycle=20", "cycle"), ("cycle=80", "cycle"),
                                     ("exclusion=0.5", "exclusion"), ("exclusion=2", "exclusion"), ("sigma0=5", "sigma0"), ("sigma0=20", "sigma0"))]
        sp = (1, 5, 7, 10, 14, 16, 17, 21)
        base = np.mean([np.mean(seeds(p, "ours_cma", range(1, 6))) for p in sp])
        rows = []
        for v, _ in sens:
            if all(len(abd[p].get(v, [])) >= 5 for p in sp):
                rows.append((v, np.mean([np.mean(abd[p][v]) for p in sp])))
        if rows:
            out += ["## Sensitivity (8 problems, seeds 1-5, 1e-4)", "", f"frozen method: {base:.3f}", "",
                    "| setting | mean | change |", "|---|---:|---:|"] + [f"| {v} | {m:.3f} | {m - base:+.3f} |" for v, m in rows] + [""]
            # Both accuracies: a drop only at 1e-4 means the searchers run out of budget to refine, not that they miss peaks.
            loose = {(int(r["instance"]), int(r["seed"])): float(r["pr_1e-3"]) for f in ("confirm_final_2026_10_09",)
                     for r in read(ART / f / "runs.csv") if r.get("suite") == "dmmop" and r["method"] == "ours_cma"}
            base3 = np.mean([np.mean([loose[(p, s)] for s in range(1, 6)]) for p in sp])
            ab3: dict = defaultdict(list)
            for r in ab:
                ab3[(r["variant"], int(r["problem"]))].append(float(r["pr_1e-3"]))
            label = {"agents": "team size $N$ = {}", "cycle": "cycle $c$ = {}", "exclusion": "$r_{{ex}}$ \u00d7 {}", "sigma0": "$\\sigma_0$ = {} units"}
            names = [label[v.split("=")[0]].format(v.split("=")[1]) for v, _ in rows]
            d3 = [np.mean([np.mean(ab3[(v, p)]) for p in sp]) - base3 for v, _ in rows]
            d4 = [m - base for _, m in rows]
            yy = np.arange(len(rows))
            fig, ax = plt.subplots(figsize=(5.2, 3.0))
            ax.barh(yy + 0.2, d3, 0.38, color=NOHOP, label="accuracy $10^{-3}$")
            ax.barh(yy - 0.2, d4, 0.38, color=OURS, label="accuracy $10^{-4}$")
            ax.set_yticks(yy, names)
            ax.axvline(0, color="0.3", lw=0.8)
            ax.set_xlabel("change in mean peak ratio vs. default settings")
            ax.legend(frameon=False, loc="upper left", fontsize=8)
            fig.tight_layout()
            fig.savefig(FIG / "sensitivity.svg")
            plt.close(fig)

    (ART / "paper_tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
