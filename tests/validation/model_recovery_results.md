# Model Recovery Validation
C0 = 7.7980e-03 mol/L, N = 15 seeds x 7 archetypes, +/ 3 % noise, seed = 20260812

## Per-cutoff recovery rates

| cutoff | overall % | PSO/PFO rec % | mech rec % | false-PSO on mech data % |
|:---:|:---:|:---:|:---:|:---:|
| 1.00 | 52.4 | 85.0 | 8.9 | 15.6 |
| 0.95 | 30.5 | 53.3 | 0.0 | 24.4 |
| 0.90 | 19.0 | 31.7 | 2.2 | 22.2 |
| 0.85 | 8.6 | 15.0 | 0.0 | 22.2 |
| 0.80 | 1.9 | 3.3 | 0.0 | 13.3 |

## Per-catalyst best model (most common across seeds)

| catalyst | true model | 1.00 | 0.95 | 0.90 | 0.85 | 0.80 |
|---|---|---|---|---|---|---|
| PSO-A | Pseudo-second-order | Pseudo-second-order (100 %) | Pseudo-second-order (100 %) | Pseudo-second-order (87 %) | None (53 %) | None (87 %) |
| PSO-B | Pseudo-second-order | Pseudo-second-order (100 %) | Pseudo-second-order (87 %) | None (60 %) | None (87 %) | None (100 %) |
| PFO-A | Pseudo-first | Pseudo-first (87 %) | None (73 %) | None (93 %) | None (100 %) | None (100 %) |
| PFO-B | Pseudo-first | Pseudo-first (53 %) | None (93 %) | None (100 %) | None (100 %) | None (100 %) |
| PL-A | Power-Law | Pseudo-second-order (47 %) | Pseudo-second-order (73 %) | Pseudo-second-order (67 %) | Pseudo-second-order (67 %) | None (60 %) |
| LH-A | L-H | Pseudo-first (93 %) | None (93 %) | None (100 %) | None (100 %) | None (100 %) |
| AV-A | Avrami | Pseudo-first (100 %) | None (93 %) | None (100 %) | None (100 %) | None (100 %) |
