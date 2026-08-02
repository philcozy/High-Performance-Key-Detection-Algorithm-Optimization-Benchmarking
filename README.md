# Key Detection — Algorithm Optimization & Benchmarking

A Python reimplementation of musical key detection structured after [libkeyfinder](https://github.com/mixxxdj/libkeyfinder), with a benchmark harness for measuring modern improvements. Built as the foundation for a future real-time C++/JUCE plugin.

---

## Motivation

libkeyfinder is the de-facto open-source key detector — it ships inside Mixxx and is widely used downstream. But its core algorithm comes from a 2011 master's thesis and the repository has seen no substantive algorithmic work in roughly the last three years. Modern MIR research has moved on: better key profiles, tuning estimation, harmonic-percussive separation, and energy-normalized chroma are all well-established techniques that are not represented in the most-used open-source tool.

This project:

1. **Rebuilds the libkeyfinder pipeline from scratch in Python**, so every stage is inspectable and every parameter has a documented reason.
2. **Establishes a transparent benchmark** against the [GiantSteps-Key-Dataset](https://github.com/GiantSteps/giantsteps-key-dataset) with MIREX-weighted scoring. Every change is attributable to a number.
3. **Tests modern improvements** (better profiles, tuning compensation, spectral compression, kernel tuning) one at a time against a fixed baseline.
4. **Ports the validated pipeline to C++/JUCE** as a plugin, once the Python version is well-characterized.

The goal is not to beat deep-learning systems — those need ML infrastructure and labelled training data, and that is a different project. The goal is to push a hand-engineered DSP pipeline as far as it goes while keeping every design choice measurable.

---

## Pipeline

```
WAV file
 ├─ Mono reduction (channel mean)
 ├─ Downsample 44.1 kHz → 11.025 kHz via scipy.signal.resample_poly
 │      (built-in anti-aliasing filter)
 ├─ Framing: 16384-sample frames, 75 % overlap (hop = 4096)
 ├─ Hamming window per frame
 ├─ Real FFT → magnitude spectrum  (8193 bins, 0.673 Hz/bin)
 ├─ Per-frame L1 normalization → accumulated magnitude spectrum
 ├─ 72-band approximate constant-Q projection
 │      (Hann-shaped kernel per band, Q ≈ 20.2, centres A1 … G#6)
 ├─ Octave fold → 12-bin chroma vector
 └─ Pearson Correlation against 24 rotated Bellman-Budge templates → predicted key
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
| Sha'ash baseline | **0.64**    | 2026-08-02 |

Reference points for context:

- libkeyfinder on GiantSteps: ~0.58 – 0.62 (published)
- Best non-deep-learning systems: ~0.75
- Deep-learning state of the art: ~0.80

### Reproducing the benchmark

```bash
cd benchmark
python run_benchmark.py
```

Each run writes a dated CSV to `benchmark/results/`, so accuracy progression is in version control. CSVs are kept; the audio is gitignored.

---

## Roadmap

Each item is implemented in isolation, benchmarked, and committed with its own dated CSV so the contribution of every change is attributable.

- [ ] **Low-energy frame gating** — skip silent / fade frames in accumulation.
- [ ] **Tuning compensation** — estimate global pitch offset (many EDM tracks deviate from A=440), shift chroma bins. Expected gain +2–4 %.
- [ ] **Spectral compression** — `log(1 + |X|)` or `sqrt` to reduce bass dominance.
- [ ] **Octave weighting** — replace equal-weight octave sum with mid-octave-emphasized weights (libkeyfinder approach).
- [O] **Pearson correlation** instead of raw dot product for scale invariance.
- [ ] **Profile comparison** — Krumhansl-Schmuckler, Temperley, Albrecht-Shanahan benchmarked side-by-side with bellman's
- [ ] **Q-factor sweep** — kernel-width experiment, A/B against current Q.
- [ ] **Harmonic-percussive separation** — median-filter-based HPSS for percussion-heavy tracks.
- [ ] **C++ port** — `juce::dsp::FFT`-based reimplementation, per-stage equivalence verified by dumping arrays from C++ and diffing in Python.
- [ ] **JUCE plugin** — offline drag-file analyzer first; real-time chroma meter with lock-free audio→worker FIFO as stretch goal.

**Target trajectory:** 0.52 → 0.60 (match libkeyfinder) → 0.65+ (beat libkeyfinder with documented improvements) → C++ port → JUCE plugin.

---

## Repository structure

```
.
├── prototype/      # detect_key() + CQT kernel + key profiles
├── benchmark/             # GiantSteps evaluation, MIREX scoring, results/
└── README.md
```

## Tech stack

- **Python 3** — `numpy`, `scipy` (`fft`, `signal.resample_poly`, `signal.windows`)
- **Dataset** — GiantSteps Key Dataset, MIREX scoring
- **Planned** — Modern C++, JUCE 7, `juce::dsp::FFT`

---

## References

**Key profiles**

- Nápoles López, N., Arthur, C., & Fujinaga, I. (2019). Key-Finding Based on a Hidden Markov Model and Key Profiles.

**Dataset**

- Knees, P., Faraldo, Á., Herrera, P., Vogl, R., Böck, S., Hörschläger, F. & Le Goff, M. (2015). *Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections.* ISMIR 2015.