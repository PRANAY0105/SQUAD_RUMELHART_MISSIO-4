# DPO vs SimPO length-bias benchmark

| Method | Pairs | corr(len diff, margin) | Acc (chosen longer) | Acc (chosen shorter) | Exploit score | Verdict |
|---|---|---|---|---|---|---|
| Reference model (before training) | 200 | 0.668 | 0.750 | 0.120 | 0.630 | LENGTH BIAS DETECTED |
| DPO | 200 | -0.983 | 1.000 | 1.000 | 0.000 | NO STRONG LENGTH BIAS |
| SimPO | 200 | 0.787 | 1.000 | 1.000 | 0.000 | NO STRONG LENGTH BIAS |

- Reference model (before training): overall reward accuracy 0.435
- DPO: overall reward accuracy 1.000
- SimPO: overall reward accuracy 1.000

Training pairs had the chosen (benign) answer LONGER. The test set mixes both cases. 'Exploit score' = accuracy when chosen is longer minus accuracy when chosen is shorter. Toy model, single seed: do not read small differences as meaningful.
