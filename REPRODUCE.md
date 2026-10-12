# Reproducing the experiments

Everything runs on open-source software (Python 3.10+, numpy, scipy, pandas, matplotlib). Every
confirmation and control was registered before its runs: the criteria are in the runner's docstring.
Runners are resumable and show a progress bar. Run all commands from the repository root with
`PYTHONPATH=src:scripts`, after `sh scripts/fetch_data.sh`.

## Benchmarks

| suite | code | validation |
|---|---|---|
| CEC 2022 dynamic multimodal suite (24 problems) | `src/aumo/problems/dmmop.py` | optima correct on all 24 problems; the organisers' instances are replayed exactly (below) |
| CEC 2013 niching suite (20 problems) | `src/aumo/problems/cec2013.py` | matches the organisers' python3 code to within 2e-11 on all 20 problems |
| Moving Peaks, scenario 2 and 9 variations | `src/aumo/problems/moving_peaks.py` | mQSO reaches offline error 1.52 on scenario 2 (published 1.75) |

**CEC 2022 instances.** The port draws its own instances from the organisers' generator (numpy
instead of MATLAB's random stream); the confirmation, the controls and the ablations use these draws
(dynamics seed 0). The organisers' own instances can be exported with MATLAB by
`scripts/matlab/dump_dmmop_instances.m` into `data/dmmop/instances/P01..P24.mat` and replayed with
`DMMOP_INSTANCES=data/dmmop/instances`; the loader checks every optimum value against the export.
They are not redistributed here.

## Experiments

| result in the paper | command | CPU-h |
|---|---|---:|
| Confirmation on our draw (D1, D2; seeds 1-15) and Moving Peaks (M1-M3) | `python scripts/confirm_final.py --out artifacts/confirm_final_2026_10_09` | ~90 |
| Our draw, seeds 16-30 | `python scripts/confirm_final.py --out artifacts/confirm_dmmop_seeds16_30_2026_10_10 --dmmop-seeds 16-30 --only dmmop:ours_cma,dmmop:cma_stay,dmmop:mqso,dmmop:de` | ~37 |
| Tuned HmSO | `python scripts/confirm_final.py --out artifacts/dmmop_hmso_tuned_2026_10_10 --dmmop-seeds 1-30 --only dmmop:hmso_tuned` | ~50 |
| Four further draws | `DMMOP_DYNAMICS_SEED=d python scripts/confirm_final.py --out artifacts/dmmop_instance_draws_2026_10_10/drawd --dmmop-seeds 1-5 --only dmmop:ours_cma,dmmop:cma_stay` (d = 1..4) | ~23 |
| Organisers' instances (D1, D2) | `DMMOP_INSTANCES=data/dmmop/instances python scripts/confirm_final.py --out artifacts/dmmop_official_2026_10_11 --dmmop-seeds 1-30 --only dmmop:ours_cma,dmmop:cma_stay,dmmop:hmso_tuned,dmmop:mqso,dmmop:de` | ~172 |
| CEC 2013 (N1, N2) | `python scripts/confirm_cec2013.py --out artifacts/confirm_cec2013_2026_10_10` | ~13 |
| HillVallEA (V, H1, H2); needs `sh third_party/hillvallea/build.sh` | `python scripts/confirm_hillvallea.py --out artifacts/confirm_hillvallea_2026_10_10` | ~711 |
| Component ablation (A1) and sensitivity | `python scripts/ablation_dmmop.py --out artifacts/dmmop_ablation_2026_10_10` | ~28 |
| Follow-up ablation (G, F, I) | `python scripts/ablation_followup.py --out artifacts/dmmop_followup_2026_10_11` | ~25 |
| Restart-on-convergence control and stale information (C, S) | `python scripts/ablation_review2.py --out artifacts/dmmop_review2_2026_10_12` | ~14 |
| Recommended rule on the organisers' instances and control with a per-environment archive (R, K) | `python scripts/review3.py --out artifacts/dmmop_review3_2026_10_12` | ~14 |
| Duplicate lifetime (Remark 1) | `python scripts/analysis_redundancy.py --out artifacts/redundancy_2026_10_11` | ~1 |
| Exclusion bound (Proposition 1), from geometry only | `python scripts/analysis_exclusion_bound.py --out artifacts/exclusion_bound_2026_10_11` | <1 |
| Published CEC 2013 entries, re-scored | `python scripts/score_cec2013_submissions.py NAME=DIR ... --out artifacts/cec2013_published/rescored_runs.csv` | <1 |

The submission folders for the re-scoring come from https://github.com/mikeagn/CEC2013
(`NichingCompetition2020GECCO-WCCIFinalData/RS-CMSA-ESII.zip`,
`NichingCompetition2019GECCOFinalData/HillVallEA19.zip`); the 2013 entrants' peak-ratio tables are
in `artifacts/cec2013_published/competition2013/`. The published CEC 2022-2024 results are in the
organisers' repository, https://github.com/MiLabHITSZ/dmmops.

## Scoring

```bash
python scripts/analyze_confirm_final.py artifacts/confirm_final_2026_10_09/runs.csv
python scripts/analyze_confirm_final.py artifacts/dmmop_official_2026_10_11/runs.csv --hmso hmso_tuned
python scripts/analyze_confirm_cec2013.py artifacts/confirm_cec2013_2026_10_10/runs.csv
python scripts/confirm_hillvallea.py --out artifacts/confirm_hillvallea_2026_10_10 --analyze
python scripts/analyze_followup.py
python scripts/ablation_review2.py --out artifacts/dmmop_review2_2026_10_12 --analyze
python scripts/review3.py --out artifacts/dmmop_review3_2026_10_12 --analyze
python scripts/analyze_friedman.py
python scripts/analysis_random_relocation.py
python scripts/figure_official.py
python -W ignore scripts/paper_figures.py
```

## Randomness

Every run is seeded: method generators derive from the run seed through `numpy.random.default_rng`,
the CEC 2022 dynamics from `dynamics_seed` (0 by default), and Moving Peaks from the run seed. The
same seed gives the same numbers on the same numpy version.
