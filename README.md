# Key Detection — Prototype, Optimization & Benchmarking
"Pure Python key detection based on [libkeyfinder](https://github.com/mixxxdj/libkeyfinder), benchmarked on [GiantSteps](https://github.com/GiantSteps/giantsteps-key-dataset) using MIREX metrics."
---

## Motivation

libkeyfinder is the de-facto open-source key detector — it ships inside Mixxx and is widely used downstream. But its core algorithm comes from a 2011 master's thesis and the repository has seen no substantive algorithmic work in roughly the last three years. Modern MIR research has moved on: better key profiles, tuning estimation, harmonic-percussive separation, and energy-normalized chroma are all well-established techniques that are not represented in the most-used open-source tool.

This project:

1. **Rebuilds the libkeyfinder pipeline from scratch in Python**, so every stage is inspectable and every parameter has a documented reason.
2. **Establishes a transparent benchmark** against the [GiantSteps-Key-Dataset](https://github.com/GiantSteps/giantsteps-key-dataset) with MIREX-weighted scoring. Every change is attributable to a number.
3. **Tests modern improvements** (better profiles, tuning compensation, spectral compression, kernel tuning) one at a time against a fixed baseline.

The goal is not to beat deep-learning systems — those need ML infrastructure and labelled training data, and that is a different project. The goal is to push a hand-engineered DSP pipeline as far as it goes while keeping every design choice measurable.

---

## Pipeline

```
Raw Audio Input
 ├─ Downmixing & Resampling (4410hz/Mono)
 ├─ Real STFT (via Numpy)
 ├─ Approximate constant-Q projection (72 bins)
 ├─ Octave fold (12-bin chroma vector)
 ├─ Cosine correlation against 24 rotated templates
 └─ MIREX Score Evaluation ── (Compared against GIANTSTEP Ground Truth)
```

---

## Benchmark

**Dataset.** [GiantSteps Key Dataset](https://github.com/GiantSteps/giantsteps-key-dataset) — 604 electronic-music tracks with single-key annotations. EDM is harder than average for chroma-based detectors because of distortion, heavy bass, and percussion.

**Scoring.** MIREX weighted scheme:

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
| current pipeline | 0.70        | 2026/08/20 |
| libkeyfinder     | 0.58 – 0.62 | ---------- |
| non-deep-learning| 0.75        | ---------- |
| deep-learning    | 0.80        | ---------- |

---

## Repository Structure
```text
├── prototpye/            
├── benchmark/         # GiantSteps dataset annotations & resuls
└── README.md
```

## Installation
```text
git clone https://github.com/philcozy/High-Performance-Key-Detection-Algorithm-Optimization-Benchmarking.git
cd High-Performance-Key-Detection-Algorithm-Optimization-Benchmarking
pip install -r requirements.txt
```

## Run Benchmark
```text
python main.py --dataset giantstep --eval mirex
```
## References
- Catchmar, I. (2011). An Autonomous System for Key Detection in Polyphonic Music.

- Knees, P., et al. (2015). Two datasets for key detection in electronic dance music. (GIANTSTEP Dataset)

- MIREX Key Detection Evaluation Task Specifications.
