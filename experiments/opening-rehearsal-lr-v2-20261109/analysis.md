## per run
| cond | seed | openings | retention | lost | new | train CE | val CE | battle W-L-D (verified) | abnormal | truncated | engine timeout | suspected | eval error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 10000555 | 57/66 | fail | 8 | 0 | 0.0627 | 0.0689 | 12-20-0 (32) | 0 | 0 | 0 | 0 | 0 |
| A | 10000777 | 45/66 | fail | 20 | 0 | 0.0653 | 0.0734 | 12-20-0 (32) | 0 | 0 | 0 | 0 | 0 |
| A | 10000999 | 60/66 | fail | 5 | 0 | 0.0688 | 0.0912 | 14-18-0 (32) | 0 | 0 | 0 | 0 | 0 |
| B | 10000555 | 60/66 | fail | 5 | 0 | 0.0579 | 0.0665 | 16-16-0 (32) | 0 | 0 | 0 | 0 | 0 |
| B | 10000777 | 62/66 | fail | 3 | 0 | 0.0566 | 0.0719 | 17-15-0 (32) | 0 | 0 | 0 | 0 | 0 |
| B | 10000999 | 63/66 | fail | 2 | 0 | 0.0589 | 0.0867 | 16-16-0 (32) | 0 | 0 | 0 | 0 | 0 |
| C | 10000555 | 64/66 | fail | 1 | 0 | 0.0602 | 0.0755 | 16-16-0 (32) | 0 | 0 | 0 | 0 | 0 |
| C | 10000777 | 64/66 | fail | 1 | 0 | 0.0593 | 0.0751 | 17-15-0 (32) | 0 | 0 | 0 | 0 | 0 |
| C | 10000999 | 63/66 | fail | 2 | 0 | 0.0610 | 0.0880 | 16-16-0 (32) | 0 | 0 | 0 | 0 | 0 |
| D | 10000555 | 64/66 | fail | 1 | 0 | 0.0579 | 0.0746 | 17-15-0 (32) | 0 | 0 | 0 | 0 | 0 |
| D | 10000777 | 64/66 | fail | 1 | 0 | 0.0557 | 0.0771 | 14-18-0 (32) | 0 | 0 | 0 | 0 | 0 |
| D | 10000999 | 64/66 | fail | 1 | 0 | 0.0547 | 0.0809 | 13-19-0 (32) | 0 | 0 | 0 | 0 | 0 |

## lost cases per run
- A-10000555: lost ['cue-mix-jj-track-g08-p0', 'cue-reco-jj-zaren-g09-p0', 'mixed-rukina-cue-g08-p0', 'mixed-rukina-cue-g09-p0', 'mixed-rukina-cue-g10-p0', 'tune-juke-veiler-g08-p0', 'tune-juke-veiler-g09-p0', 'tune-stone-sweeper-g09-p0'] new []
- A-10000777: lost ['cue-mix-jj-track-g08-p0', 'cue-rotary-jj-zaren-g10-p0', 'elf-par-fire-cost-g09-p0', 'elf-par-fire-cost-g10-p0', 'elf-rukina-regina-g08-p0', 'elf-rukina-regina-g10-p0', 'mixed-rukina-cue-g08-p0', 'mixed-rukina-cue-g09-p0', 'mixed-rukina-cue-g10-p0', 'rukina-ash_parallelism-g09-p0', 'rukina-ash_spirit-g09-p0', 'rukina-ash_spirit-g10-p0', 'rukina-pass-g08-p0', 'rukina-pass-g09-p0', 'rukina-veiler_strelitzia-g08-p0', 'rukina-veiler_strelitzia-g09-p0', 'rukina-veiler_strelitzia-g10-p0', 'tune-juke-veiler-g08-p0', 'tune-juke-veiler-g09-p0', 'tune-stone-sweeper-g09-p0'] new []
- A-10000999: lost ['cue-reco-jj-zaren-g09-p0', 'cue-reco-jj-zaren-g10-p0', 'mixed-cue-spirit-g09-p0', 'tune-reco-g08-p0', 'tune-reco-g09-p0'] new []
- B-10000555: lost ['cue-mix-jj-track-g08-p0', 'mixed-rukina-cue-g08-p0', 'tune-juke-veiler-g08-p0', 'tune-juke-veiler-g09-p0', 'tune-stone-sweeper-g09-p0'] new []
- B-10000777: lost ['mixed-rukina-cue-g08-p0', 'mixed-rukina-cue-g09-p0', 'mixed-rukina-cue-g10-p0'] new []
- B-10000999: lost ['cue-reco-jj-zaren-g09-p0', 'cue-rotary-jj-zaren-g10-p0'] new []
- C-10000555: lost ['mixed-rukina-cue-g10-p0'] new []
- C-10000777: lost ['mixed-rukina-cue-g10-p0'] new []
- C-10000999: lost ['cue-reco-jj-zaren-g09-p0', 'mixed-cue-spirit-g09-p0'] new []
- D-10000555: lost ['mixed-rukina-cue-g10-p0'] new []
- D-10000777: lost ['mixed-rukina-cue-g10-p0'] new []
- D-10000999: lost ['cue-reco-jj-zaren-g09-p0'] new []

## paired comparisons (same seed): cases failing in one condition but not the other
- A vs C seed 10000555: lost 8 vs 1; only A: ['cue-mix-jj-track-g08-p0', 'cue-reco-jj-zaren-g09-p0', 'mixed-rukina-cue-g08-p0', 'mixed-rukina-cue-g09-p0', 'tune-juke-veiler-g08-p0', 'tune-juke-veiler-g09-p0', 'tune-stone-sweeper-g09-p0']; only C: []
- A vs C seed 10000777: lost 20 vs 1; only A: ['cue-mix-jj-track-g08-p0', 'cue-rotary-jj-zaren-g10-p0', 'elf-par-fire-cost-g09-p0', 'elf-par-fire-cost-g10-p0', 'elf-rukina-regina-g08-p0', 'elf-rukina-regina-g10-p0', 'mixed-rukina-cue-g08-p0', 'mixed-rukina-cue-g09-p0', 'rukina-ash_parallelism-g09-p0', 'rukina-ash_spirit-g09-p0', 'rukina-ash_spirit-g10-p0', 'rukina-pass-g08-p0', 'rukina-pass-g09-p0', 'rukina-veiler_strelitzia-g08-p0', 'rukina-veiler_strelitzia-g09-p0', 'rukina-veiler_strelitzia-g10-p0', 'tune-juke-veiler-g08-p0', 'tune-juke-veiler-g09-p0', 'tune-stone-sweeper-g09-p0']; only C: []
- A vs C seed 10000999: lost 5 vs 2; only A: ['cue-reco-jj-zaren-g10-p0', 'tune-reco-g08-p0', 'tune-reco-g09-p0']; only C: []
  total lost (counted per seed): A=33 C=4
- A vs B seed 10000555: lost 8 vs 5; only A: ['cue-reco-jj-zaren-g09-p0', 'mixed-rukina-cue-g09-p0', 'mixed-rukina-cue-g10-p0']; only B: []
- A vs B seed 10000777: lost 20 vs 3; only A: ['cue-mix-jj-track-g08-p0', 'cue-rotary-jj-zaren-g10-p0', 'elf-par-fire-cost-g09-p0', 'elf-par-fire-cost-g10-p0', 'elf-rukina-regina-g08-p0', 'elf-rukina-regina-g10-p0', 'rukina-ash_parallelism-g09-p0', 'rukina-ash_spirit-g09-p0', 'rukina-ash_spirit-g10-p0', 'rukina-pass-g08-p0', 'rukina-pass-g09-p0', 'rukina-veiler_strelitzia-g08-p0', 'rukina-veiler_strelitzia-g09-p0', 'rukina-veiler_strelitzia-g10-p0', 'tune-juke-veiler-g08-p0', 'tune-juke-veiler-g09-p0', 'tune-stone-sweeper-g09-p0']; only B: []
- A vs B seed 10000999: lost 5 vs 2; only A: ['cue-reco-jj-zaren-g10-p0', 'mixed-cue-spirit-g09-p0', 'tune-reco-g08-p0', 'tune-reco-g09-p0']; only B: ['cue-rotary-jj-zaren-g10-p0']
  total lost (counted per seed): A=33 B=10
- C vs D seed 10000555: lost 1 vs 1; only C: []; only D: []
- C vs D seed 10000777: lost 1 vs 1; only C: []; only D: []
- C vs D seed 10000999: lost 2 vs 1; only C: ['mixed-cue-spirit-g09-p0']; only D: []
  total lost (counted per seed): C=4 D=3
- B vs D seed 10000555: lost 5 vs 1; only B: ['cue-mix-jj-track-g08-p0', 'mixed-rukina-cue-g08-p0', 'tune-juke-veiler-g08-p0', 'tune-juke-veiler-g09-p0', 'tune-stone-sweeper-g09-p0']; only D: ['mixed-rukina-cue-g10-p0']
- B vs D seed 10000777: lost 3 vs 1; only B: ['mixed-rukina-cue-g08-p0', 'mixed-rukina-cue-g09-p0']; only D: []
- B vs D seed 10000999: lost 2 vs 1; only B: ['cue-rotary-jj-zaren-g10-p0']; only D: []
  total lost (counted per seed): B=10 D=3

## update size and gradients
| cond | seed | lr | base rel L2 | base max abs | head rel L2 | head changed | PPO grad norm | rehearsal grad norm | cosine(PPO,rehearsal) | rehearsal CE | rehearsal decisions | train s | RSS MiB | GPU peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 10000555 | 3.0e-05 | 1.08e-03 | 2.30e-04 | 4.82e-05 | 2080 | — | — | — | — | 0 | 42.2 | 1908.7 | 2382.0 |
| A | 10000777 | 3.0e-05 | 1.08e-03 | 2.20e-04 | 4.58e-05 | 2086 | — | — | — | — | 0 | 40.2 | 1923.0 | 2100.0 |
| A | 10000999 | 3.0e-05 | 9.67e-04 | 2.27e-04 | 6.26e-05 | 2065 | — | — | — | — | 0 | 40.2 | 1908.7 | 2019.0 |
| B | 10000555 | 3.0e-05 | 1.01e-03 | 2.28e-04 | 4.30e-05 | 2152 | 13.9 | 2.62 | +0.018 | 0.055 | 708 | 146.5 | 2741.7 | 2258.0 |
| B | 10000777 | 3.0e-05 | 9.49e-04 | 2.23e-04 | 4.06e-05 | 2148 | 21.4 | 6.22 | +0.136 | 0.071 | 729 | 50.2 | 2345.3 | 1986.0 |
| B | 10000999 | 3.0e-05 | 1.12e-03 | 2.23e-04 | 7.03e-05 | 2157 | 39.1 | 2.90 | -0.015 | 0.061 | 699 | 48.2 | 2350.4 | 1995.0 |
| C | 10000555 | 1.5e-05 | 5.57e-04 | 1.12e-04 | 2.83e-05 | 2080 | — | — | — | — | 0 | 42.4 | 1906.2 | 2007.0 |
| C | 10000777 | 1.5e-05 | 5.33e-04 | 1.10e-04 | 2.34e-05 | 2058 | — | — | — | — | 0 | 40.2 | 1912.1 | 1975.0 |
| C | 10000999 | 1.5e-05 | 5.08e-04 | 1.14e-04 | 3.30e-05 | 2059 | — | — | — | — | 0 | 40.2 | 1918.8 | 1984.0 |
| D | 10000555 | 1.5e-05 | 5.15e-04 | 1.11e-04 | 2.29e-05 | 2140 | 13.1 | 2.33 | +0.001 | 0.055 | 708 | 52.2 | 2398.2 | 2013.0 |
| D | 10000777 | 1.5e-05 | 4.65e-04 | 1.12e-04 | 2.28e-05 | 2126 | 19.1 | 3.03 | +0.131 | 0.063 | 729 | 50.2 | 2348.4 | 1983.0 |
| D | 10000999 | 1.5e-05 | 5.78e-04 | 1.13e-04 | 3.57e-05 | 2124 | 36.3 | 2.74 | -0.017 | 0.061 | 699 | 48.2 | 2330.1 | 1985.0 |

## per minibatch cosine (rehearsal arms)
- B-10000555: +0.083, +0.059, +0.047, -0.039, +0.026, +0.010, -0.142, +0.102
- B-10000777: +0.054, +0.012, -0.130, +0.020, +0.040, +0.537, +0.638, -0.085
- B-10000999: -0.026, -0.128, -0.011, -0.004, -0.087, -0.003, +0.073, +0.069
- D-10000555: +0.083, +0.042, +0.060, -0.071, -0.053, +0.014, -0.127, +0.059
- D-10000777: +0.054, +0.096, -0.032, +0.026, +0.079, +0.308, +0.493, +0.025
- D-10000999: -0.026, -0.123, -0.021, -0.016, -0.125, +0.001, +0.069, +0.107

## per-case table (cases changed anywhere)
| case | base | A555 | A777 | A999 | B555 | B777 | B999 | C555 | C777 | C999 | D555 | D777 | D999 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rukina-pass-g08-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| rukina-pass-g09-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| rukina-ash_spirit-g09-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| rukina-ash_spirit-g10-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| rukina-ash_parallelism-g09-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| rukina-veiler_strelitzia-g08-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| rukina-veiler_strelitzia-g09-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| rukina-veiler_strelitzia-g10-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| elf-rukina-regina-g08-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| elf-rukina-regina-g10-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| elf-par-fire-cost-g09-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| elf-par-fire-cost-g10-p0 | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| tune-reco-g08-p0 | ○ | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| tune-reco-g09-p0 | ○ | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| tune-stone-sweeper-g09-p0 | ○ | × | × | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| tune-juke-veiler-g08-p0 | ○ | × | × | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| tune-juke-veiler-g09-p0 | ○ | × | × | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| mixed-rukina-cue-g08-p0 | ○ | × | × | ○ | × | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| mixed-rukina-cue-g09-p0 | ○ | × | × | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| mixed-rukina-cue-g10-p0 | ○ | × | × | ○ | ○ | × | ○ | × | × | ○ | × | × | ○ |
| mixed-cue-spirit-g09-p0 | ○ | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | × | ○ | ○ | ○ |
| cue-reco-jj-zaren-g08-p0 | × | × | × | × | × | × | × | × | × | × | × | × | × |
| cue-reco-jj-zaren-g09-p0 | ○ | × | ○ | × | ○ | ○ | × | ○ | ○ | × | ○ | ○ | × |
| cue-reco-jj-zaren-g10-p0 | ○ | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| cue-mix-jj-track-g08-p0 | ○ | × | × | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| cue-rotary-jj-zaren-g10-p0 | ○ | ○ | × | ○ | ○ | ○ | × | ○ | ○ | ○ | ○ | ○ | ○ |
