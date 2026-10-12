## CEC 2013 niching: mean peak ratio at 1e-4 (runs per cell in brackets when not 50)
| problem | ours | stay | ours_noarchive |
|---|---:|---:|---:|
| F1 | 1.000 | 1.000 | 1.000 |
| F2 | 1.000 | 1.000 | 1.000 |
| F3 | 1.000 | 1.000 | 1.000 |
| F4 | 1.000 | 1.000 | 1.000 |
| F5 | 1.000 | 1.000 | 1.000 |
| F6 | 0.764 | 0.907 | 0.487 |
| F7 | 0.759 | 0.432 | 0.468 |
| F8 | 0.194 | 0.252 | 0.182 |
| F9 | 0.346 | 0.192 | 0.087 |
| F10 | 1.000 | 1.000 | 1.000 |
| F11 | 0.980 | 1.000 | 0.983 |
| F12 | 0.787 | 0.740 | 0.800 |
| F13 | 0.917 | 0.790 | 0.823 |
| F14 | 0.813 | 0.717 | 0.797 |
| F15 | 0.728 | 0.740 | 0.723 |
| F16 | 0.697 | 0.670 | 0.687 |
| F17 | 0.748 | 0.710 | 0.738 |
| F18 | 0.667 | 0.647 | 0.667 |
| F19 | 0.632 | 0.530 | 0.625 |
| F20 | 0.502 | 0.395 | 0.500 |

Mean over problems at 0.1: ours 0.787, stay 0.739, ours_noarchive 0.733

Mean over problems at 0.01: ours 0.777, stay 0.736, ours_noarchive 0.730

Mean over problems at 0.001: ours 0.777, stay 0.736, ours_noarchive 0.729

Mean over problems at 0.0001: ours 0.777, stay 0.736, ours_noarchive 0.728

Mean over problems at 1e-05: ours 0.777, stay 0.736, ours_noarchive 0.728

### Pre-registered criteria
N1 ours beats stay on 8/20 [7, 9, 12, 13, 14, 17, 19, 20]; worse on [6, 8] -> PASS
N2 mean peak ratio at 1e-4: ours 0.777, stay 0.736 -> PASS

## Published (context; 1e-4): mean over 20 problems, and problems where ours is higher / lower by > 0.02
| method | mean | ours higher | ours lower |
|---|---:|---:|---:|
| RS-CMSA-ESII (re-scored) | 0.906 | 0 | 11 |
| HillVallEA19 (re-scored) | 0.892 | 2 | 9 |
| nea2 (2013) | 0.801 | 3 | 7 |
| cma1 (2013) | 0.719 | 11 | 1 |
| dade1 (2013) | 0.712 | 10 | 4 |
| dade2 (2013) | 0.657 | 10 | 2 |
| Molina (2013) | 0.655 | 12 | 1 |
| denrand2 (2013) | 0.613 | 13 | 1 |
| nea1 (2013) | 0.594 | 14 | 1 |
| dels_ajitter (2013) | 0.586 | 13 | 1 |
| PNANSGAII (2013) | 0.585 | 13 | 1 |
| denrand1 (2013) | 0.581 | 14 | 0 |
| delg (2013) | 0.581 | 12 | 2 |
| decg (2013) | 0.558 | 12 | 2 |
| CDE (2013) | 0.499 | 14 | 1 |
| ipop1 (2013) | 0.359 | 18 | 0 |
| ANSGAII (2013) | 0.290 | 18 | 0 |
| **ours** | 0.777 | | |

Friedman mean rank among 18 methods at 1e-4: RS-CMSA-ESII (re-scored) 3.60, HillVallEA19 (re-scored) 3.88, ours 5.78, nea2 (2013) 5.80, dade1 (2013) 7.78, cma1 (2013) 7.90, Molina (2013) 8.50, dade2 (2013) 9.10, denrand2 (2013) 9.88, denrand1 (2013) 10.68, PNANSGAII (2013) 10.85, delg (2013) 10.97, dels_ajitter (2013) 11.07, nea1 (2013) 11.10, CDE (2013) 11.53, decg (2013) 11.62, ipop1 (2013) 14.88, ANSGAII (2013) 16.10
