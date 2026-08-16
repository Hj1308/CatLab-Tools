# Model Recovery Validation
C0 = 7.7980e-03 mol/L, N = 200 seeds x 8 archetypes, seed = 20260812
Paired cutoff sweep: all 5 cutoffs applied to the same noisy dataset per (archetype, replicate). Completed tasks: 1600. Wall clock: 1076s.

## Per-cutoff sweep (saturation cutoff vs recovery)

| cutoff | overall % | PSO/PFO rec % | mech rec % | false-PSO on mech data % | mean n_pts | clamped? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1.00 | 74.1 | 92.2 | 55.9 | 0.0 | 7.0 | |
| 0.95 | 65.8 | 95.5 | 36.1 | 2.1 | 6.0 | yes |
| 0.90 | 65.8 | 95.5 | 36.1 | 2.1 | 6.0 | yes |
| 0.85 | 65.8 | 95.5 | 36.1 | 2.1 | 6.0 | yes |
| 0.80 | 65.8 | 95.5 | 36.1 | 2.1 | 6.0 | yes |

**Buckets:** PSO/PFO = Pseudo-first + Pseudo-second-order (simplified, flexible). mech = Power-Law + L-H + Avrami (mechanistic). Denominator = replicates (N_SEEDS x N_archetypes_in_bucket).

**Cutoff sweep granularity:** with only 7 raw time points and MIN_FIT_POINTS=6, at most 1 point can ever be excluded before clamping — so cutoffs 0.95 through 0.80 are clamped to the same retained set and are not independent evidence of a cutoff effect on this T array. Only cutoff 1.00 (no exclusion) differs meaningfully from the rest here.

## Per-archetype recovery (top-3 selections) | cutoff 1.00

| PSO-A | Pseudo-second-order | 90.5 % | Pseudo-second-order (181/200) | Avrami (12/200) | Power-Law (5/200) |
| PSO-B | Pseudo-second-order | 91.5 % | Pseudo-second-order (183/200) | Power-Law (11/200) | Avrami (6/200) |
| PFO-A | Pseudo-first | 94.5 % | Pseudo-first (189/200) | Elovich (6/200) | Avrami (3/200) |
| PFO-B | Pseudo-first | 92.5 % | Pseudo-first (185/200) | Avrami (10/200) | Elovich (2/200) |
| PL-A | Power-Law | 78.0 % | Power-Law (156/200) | Avrami (42/200) | Double-Exponential (2/200) |
| PL-B | Power-Law | 56.5 % | Power-Law (113/200) | Avrami (51/200) | Pseudo-first (33/200) |
| LH-A | L-H | 0.5 % | Power-Law (72/200) | Avrami (68/200) | Elovich (30/200) |
| AV-A | Avrami | 88.5 % | Avrami (177/200) | L-H (20/200) | Zero-order (3/200) |

## Per-archetype mean fit time

| archetype | mean fit ms |
|:---:|:---:|
| PSO-A | 7547 |
| PSO-B | 7293 |
| PFO-A | 2221 |
| PFO-B | 1147 |
| PL-A | 9466 |
| PL-B | 6989 |
| LH-A | 1349 |
| AV-A | 1373 |
