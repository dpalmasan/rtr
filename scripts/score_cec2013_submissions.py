"""Re-score published CEC 2013 niching-competition submissions with our port (context for the paper).

A submission folder holds problemPPPrunRRR.dat files in the GECCO 2017+ format: one solution per
line, "x_1 ... x_D = fitness @ evaluations time action", where action 0 empties the archive and adds
the solution, 1 adds it and -1 removes it (as in the official data-parser, load_file_keep_last_archive).
The final archive is re-evaluated and counted with the official procedure (how_many_goptima).

Usage: PYTHONPATH=src python scripts/score_cec2013_submissions.py NAME=DIR [NAME=DIR ...] --out FILE.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from aumo.problems.cec2013 import ACCURACY, DIMS, NOPT, how_many_goptima, problem_function


def final_archive(path: Path, dim: int) -> np.ndarray:
    archive: dict[str, np.ndarray] = {}
    with open(path) as handle:
        for line in handle:
            if "=" not in line:
                continue
            left, right = line.split("=", 1)
            coords = left.split()[:dim]
            action = int(float(right.split("@", 1)[1].split()[-1])) if "@" in right else 1
            key = " ".join(coords)
            if action == 0:
                archive.clear()
            if action in (0, 1):
                archive[key] = np.array([float(c) for c in coords])
            elif action == -1:
                archive.pop(key, None)
    return np.array(list(archive.values())) if archive else np.empty((0, dim))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submissions", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for spec in args.submissions:
        name, folder = spec.split("=", 1)
        for fn in range(1, 21):
            f = problem_function(fn)
            for path in sorted(Path(folder).glob(f"problem{fn:03d}run*.dat")):
                pop = final_archive(path, DIMS[fn - 1])
                fits = f(pop) if len(pop) else np.empty(0)
                counts = [how_many_goptima(pop, fits, fn, eps) if len(pop) else 0 for eps in ACCURACY]
                rows.append({"method": name, "problem": fn, "run": int(path.stem.split("run")[1]),
                             **{f"pr_{eps:g}": c / NOPT[fn - 1] for eps, c in zip(ACCURACY, counts)}})
            done = [r for r in rows if r["method"] == name and r["problem"] == fn]
            print(name, fn, len(done), np.round(np.mean([[r[f"pr_{e:g}"] for e in ACCURACY] for r in done], axis=0), 3) if done else "-")
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
