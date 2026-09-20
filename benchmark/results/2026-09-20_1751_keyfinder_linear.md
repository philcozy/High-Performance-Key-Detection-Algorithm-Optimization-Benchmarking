# keyfinder_linear — 2026-09-20_1751

- pipeline: `keyfinder_linear`
- tracks: 1763
- per-track data: `2026-09-20_1751_keyfinder_linear.csv`

## Accuracy

| dataset | tracks | MIREX | correct | fifth | relative | parallel | wrong |
|---|---|---|---|---|---|---|---|
| giantsteps-key | 604 | 0.6914 | 60.3% | 11.8% | 6.5% | 5.3% | 16.2% |
| giantsteps-mtg-key | 1159 | 0.6406 | 53.9% | 10.7% | 5.7% | 15.4% | 14.3% |
| **all** | 1763 | 0.6580 | 56.1% | 11.1% | 6.0% | 11.9% | 15.0% |

Category columns are shares of that dataset. MIREX weights: correct 1.0, fifth 0.5, relative 0.3, parallel 0.2, wrong 0.0.

## Mode bias

| dataset | annotated minor / major | predicted minor / major | minor recall | major recall |
|---|---|---|---|---|
| giantsteps-key | 84.6% / 15.4% | 82.9% / 17.1% | 88.3% | 46.2% |
| giantsteps-mtg-key | 67.5% / 32.5% | 66.0% / 34.0% | 76.2% | 55.2% |
| **all** | 73.3% / 26.7% | 71.8% / 28.2% | 81.0% | 53.4% |

## Performance

- wall time: **18.8 s** with 4 workers (93.7 tracks/s)
- per track: **39.1 ms** median, 49.6 ms p95
- peak memory: 243 MB per worker

| stage | median ms | share of time |
|---|---|---|
| preprocess | 23.92 | 61.1% |
| spectrum | 13.83 | 35.5% |
| cqt | 1.02 | 2.8% |
| fold | 0.01 | 0.0% |
| classify | 0.19 | 0.5% |
