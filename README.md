# Key Detection — Prototype, Optimization & Benchmarking
Pure Python key detection based on [libkeyfinder](https://github.com/mixxxdj/libkeyfinder), benchmarked on [GiantSteps](https://github.com/GiantSteps/giantsteps-key-dataset) using MIREX metrics.
---

## Motivation

libkeyfinder is the de-facto open-source key detector — it ships inside Mixxx and is widely used downstream. But its core algorithm comes from a 2011 master's thesis and the repository has seen no substantive algorithmic work in roughly the last three years. Modern MIR research has moved on: better key profiles, tuning estimation, harmonic-percussive separation, and energy-normalized chroma are all well-established techniques that are not represented in the most-used open-source tool.

This project:

1. **Rebuilds the libkeyfinder pipeline from scratch in Python**, so every stage is inspectable and every parameter has a documented reason.
2. **Establishes a transparent benchmark** against the [GiantSteps-Key-Dataset](https://github.com/GiantSteps/giantsteps-key-dataset) with MIREX-weighted scoring.
3. **Tests improvements** (switching profiles, adjust parameters etc...) one at a time against a fixed baseline.

The goal is not to beat deep-learning systems — those need ML infrastructure and labelled training data, and that is a different project. The goal is to push a hand-engineered DSP pipeline as far as it goes while keeping every design choice measurable.

---

## Pipeline

```
Raw Audio Input
 ├─ Downmixing & Resampling ── (4410 Hz/ Mono)
 ├─ Real STFT ── (via Numpy)
 ├─ Approximate constant-Q projection ── (72 bins)
 ├─ Log compression
 ├─ Octave fold ── (12-bin chroma vector)
 ├─ Cosine correlation against 24 rotated templates
 └─ MIREX Score Evaluation ── (Compared against GiantSteps Ground Truth)
```

---

## Benchmark

### Datasets
- [GiantSteps Key Dataset](https://github.com/GiantSteps/giantsteps-key-dataset) — 604 electronic-music tracks with single-key annotations.
- [GiantSteps MTG Key Dataset](https://github.com/GiantSteps/giantsteps-mtg-key-dataset) — 1159 tracks, keeping only annotations with confidence 2.

EDM is harder than average for chroma-based detectors because of distortion, heavy bass, and percussion.

### Scoring
MIREX weighted scheme:

| Result               | Score |
| -------------------- | ----- |
| Correct              | 1.0   |
| Perfect fifth        | 0.5   |
| Relative major/minor | 0.3   |
| Parallel major/minor | 0.2   |
| Wrong                | 0.0   |

### Current results

| Configuration    | MIREX score | Date       |
| ---------------- | ----------- | ---------- |
| current pipeline | 0.7151      | 2026/9/19  |
| libkeyfinder     | 0.58 – 0.62 | ---------- |
| non-deep-learning| 0.75        | ---------- |
| deep-learning    | 0.80        | ---------- |

### Modern Improvements
| Addons           | MIREX score |            |
| ---------------- | ----------- | ---------- |
| Baseline         | 0.6914      | -          |
| log compression  | 0.7151      | +3.4%      |
---

## Repository Structure
```text
├── prototype/            
├── benchmark/         # GiantSteps dataset annotations & results
└── README.md
```
---

## Installation
```bash
git clone https://github.com/philcozy/High-Performance-Key-Detection-Algorithm-Optimization-Benchmarking.git
cd High-Performance-Key-Detection-Algorithm-Optimization-Benchmarking
pip install -r requirements.txt
```

## Run Benchmark
```bash
cd benchmark
python run_benchmark.py                              # all datasets, 4 worker processes
python run_benchmark.py --datasets giantsteps-key    # one dataset
python run_benchmark.py --workers 1 --label my-idea  # single process, named run
```
Each run prints MIREX accuracy and per-stage timing, writes a per-track CSV to `benchmark/results/`, and appends one summary line to `benchmark/results/runs.csv`.

---

## References
- Catchmar, I. (2011). An Autonomous System for Key Detection in Polyphonic Music. Master's thesis, University of Southampton.

- Knees, P., et al. (2015). Two datasets for key detection in electronic dance music. (GiantSteps Dataset)

- MIREX Key Detection Evaluation Task Specifications.
