# keyfinder_linear — 2026-09-20_1530

- pipeline: `keyfinder_linear`
- tracks: 1763
- per-track data: `2026-09-20_1530_keyfinder_linear.csv`

## Accuracy

| dataset | tracks | MIREX |
|---|---|---|
| giantsteps-key | 604 | 0.6914 |
| giantsteps-mtg-key | 1159 | 0.6406 |
| **all** | 1763 | **0.6580** |

| category | tracks | share | score each |
|---|---|---|---|
| correct | 989 | 56.1% | 1.0 |
| fifth | 195 | 11.1% | 0.5 |
| relative | 105 | 6.0% | 0.3 |
| parallel | 210 | 11.9% | 0.2 |
| wrong | 264 | 15.0% | 0.0 |

## Mode bias

- annotations: **73.3%** minor
- predictions: **71.8%** minor

|  | predicted major | predicted minor | recall |
|---|---|---|---|
| true major | 251 | 219 | 53.4% |
| true minor | 246 | 1047 | 81.0% |

## Tonic error

Semitones from the annotated tonic to the predicted one.

| semitones | meaning | tracks | share |
|---|---|---|---|
| 0 | same tonic | 1199 | 68.0% |
| 1 |  | 16 | 0.9% |
| 2 |  | 41 | 2.3% |
| 3 | relative major | 66 | 3.7% |
| 4 |  | 37 | 2.1% |
| 5 | fourth | 94 | 5.3% |
| 6 |  | 9 | 0.5% |
| 7 | fifth | 170 | 9.6% |
| 8 |  | 8 | 0.5% |
| 9 | relative minor | 72 | 4.1% |
| 10 |  | 38 | 2.2% |
| 11 |  | 13 | 0.7% |

## Weakest and strongest annotated keys

| annotated key | mean MIREX | tracks |
|---|---|---|
| F# major | 0.418 | 39 |
| Ab major | 0.469 | 45 |
| G major | 0.487 | 40 |
| A major | 0.491 | 35 |
| D minor | 0.493 | 112 |
| ... |  |  |
| B minor | 0.728 | 89 |
| G minor | 0.737 | 127 |
| Eb minor | 0.748 | 79 |
| F minor | 0.759 | 176 |
| Ab minor | 0.830 | 63 |

## Performance

- wall time: **17.6 s** with 4 workers (100.1 tracks/s)
- per track: **37.5 ms** median, 41.4 ms p95
- peak memory: 257 MB per worker

| stage | median ms | share of time |
|---|---|---|
| preprocess | 23.03 | 61.7% |
| spectrum | 13.10 | 35.0% |
| cqt | 0.98 | 2.7% |
| fold | 0.01 | 0.0% |
| classify | 0.19 | 0.5% |
