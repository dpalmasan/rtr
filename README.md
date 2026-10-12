# Redundancy-triggered relocation (RTR)

Code and raw results for the paper *Redundancy-Triggered Relocation: Decentralized Coordination of
Local Searchers for Static and Dynamic Multimodal Optimization* (D. Palma).

RTR coordinates a team of local searchers with one rule: a searcher that perceives a fitter teammate
within an exclusion radius is redundant, and it restarts elsewhere. Each searcher perceives only the
teammates within its own radius, so redundancy is judged at a small scale without a radius tuned to
the problem.

## Contents

| path | what |
|---|---|
| `src/aumo/algorithms/peak_hopping.py` | RTR (the package keeps its historical name, `aumo`) |
| `src/aumo/algorithms/{cma_workers,mqso,hmso,dmmop_baselines,hillvallea}.py` | CMA-ES searchers and the baselines |
| `src/aumo/problems/{dmmop,cec2013,moving_peaks}.py` | Python ports of the CEC 2022 dynamic multimodal suite, the CEC 2013 niching suite and Moving Peaks |
| `scripts/` | experiment runners (criteria in each docstring, fixed before the runs) and the scoring scripts |
| `artifacts/*/runs.csv` | raw per-run results of every experiment in the paper |
| `third_party/hillvallea/` | build script for HillVallEA (fetched from its authors, GPL-3.0; see NOTICE.md) |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
sh scripts/fetch_data.sh        # benchmark data from the organisers' repositories (not redistributed)
pytest -q tests
```

[REPRODUCE.md](REPRODUCE.md) lists every experiment with its command and output, and how to score
the criteria and regenerate the tables.

## Licence and citation

The code is MIT-licensed ([LICENSE](LICENSE)), except the files under `third_party/hillvallea/` (GPL-3.0).
The benchmark data keep the licences of their owners. If you use this code, please cite the paper
([CITATION.cff](CITATION.cff)).
