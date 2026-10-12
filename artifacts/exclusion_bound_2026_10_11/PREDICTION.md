# Prediction for the global5 variant (written 2026-10-11, before its results were read)

bound_5 = 1.0 on all 24 problems (our draws and the official instances): no pair of global optima is
closer than 5 units (0.5), so exclusion at 5 units with full perception cannot lose optima through
false redundancy. Proposition 1 therefore predicts NO loss for global5 on P2, P3, P18, P19 (where
full perception at the default radius sits exactly on its bound: 0.50, 0.75, 0.50, 0.75, while RTR
reaches 1.00, 0.998, 1.00, 0.98). Any loss of global5 relative to RTR must come from another
mechanism (duplicates farther apart than 5 units go undetected until their searchers converge).
Consequence for the paper: the P2/P18 result defends RTR against exclusion at the default radius,
not against "mQSO exclusion with a smaller radius"; that question is decided by rule G alone.
