# Benchmark datasets

What the benchmark is measured on, and which way the data leans. Every number here was counted from the files in `benchmark/datasets/` (see *How these numbers were produced* at the end).

Read this before reading any score: both datasets are **electronic dance music and both are heavily minor**, so a detector that leans minor scores better than it deserves.

| | giantsteps-key | giantsteps-mtg-key | combined |
|---|---|---|---|
| tracks | 604 | 1159 | 1763 |
| minor | **511 (84.6%)** | **782 (67.5%)** | **1293 (73.3%)** |
| major | 93 (15.4%) | 377 (32.5%) | 470 (26.7%) |
| keys present | 24 / 24 | 24 / 24 | 24 / 24 |
| audio | 120 s excerpts, 44.1 kHz, mono, 16-bit WAV | same | same |
| shorter than 119 s | 4 | 4 | 8 |

Sources: [GiantSteps Key Dataset](https://github.com/GiantSteps/giantsteps-key-dataset) and [GiantSteps MTG Key Dataset](https://github.com/GiantSteps/giantsteps-mtg-key-dataset). The audio is Beatport preview mp3, converted to WAV with the sox command from `convert_audio.sh` (`rate 44100 gain -0.1 remix -`).

**The two sets share no tracks.** Comparing their Beatport ids gives 0 overlap, so one can be used for development and the other kept for testing. In the literature, giantsteps-mtg-key is the training set and giantsteps-key the test set.

## Mode imbalance is the thing to remember

The MIREX score never fully punishes a mode mistake: a major track answered with its parallel minor still earns 0.2, and its relative minor earns 0.3. Combined with 73% minor data, guessing minor is rewarded.

Two reference points for how far that alone gets you:

| strategy | giantsteps-key | giantsteps-mtg-key |
|---|---|---|
| answer the same key for every track (best one is C minor) | 0.2089 | 0.1801 |
| answer a random key out of 24 | 0.1042 | 0.1042 |
| **get every tonic right but always answer minor** | **0.8768** | **0.7398** |

The last row is the important one. A detector that never finds a major key can still reach 0.88 on giantsteps-key. So a high MIREX score does not prove a detector understands mode, and any result should be reported together with a mode-bias figure (predicted minor share, and major recall). Our run summaries in `results/*.md` include both.

The random-key baseline is 0.1042 on both sets, and always will be: it is the average of the MIREX weights over the 24 candidates, (1.0 + 0.5 + 0.5 + 0.3 + 0.2) / 24, and does not depend on the data.

## Key distribution

Tracks per tonic (pitch class, enharmonics merged):

| | C | Db | D | Eb | E | F | F# | G | Ab | A | Bb | B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| giantsteps-key | 68 | 26 | 50 | 45 | 58 | 82 | 36 | 70 | 34 | 65 | 33 | 37 |
| giantsteps-mtg-key | 135 | 114 | 116 | 70 | 107 | 132 | 80 | 97 | 74 | 81 | 74 | 79 |

Tonics are reasonably spread; the imbalance is in the mode, not the root note. The most common keys are the club standards:

- **giantsteps-key:** F minor 73, G minor 62, A minor 54, C minor 53, E minor 48, D minor 40. Rarest: Db major 1, F# major 5, Bb major 5, Ab major 5.
- **giantsteps-mtg-key:** C minor 112, F minor 103, E minor 79, Db minor 73, D minor 72, G minor 65. Rarest: B major 22, C major 23, A major 24, Eb major 27.

Note how thin the major keys are in giantsteps-key: with only 1 track in Db major and 5 in F# major, per-key major numbers on that set are anecdotes, not statistics. Use the combined set, or giantsteps-mtg-key, when looking at how a detector handles major keys.

## Annotation format

One `.key` file per audio file, same base name.

| | giantsteps-key | giantsteps-mtg-key |
|---|---|---|
| example | `C minor` | `d minor<TAB>2<TAB>` |
| tonic case | upper (`C`, `Eb`) | lower (`c`, `d#`) |
| accidentals | flats, plus `Gb` | sharps (`a#`, `c#`, `d#`, `f#`, `g#`) |
| extra columns | none | confidence, sometimes a comment |

Both spellings mean the same pitch class, and `mirex.parse_key()` accepts either; `Gb` and `f#` both map to pitch class 6. The mode word is always present, so the case of the tonic is never load-bearing here.

**The confidence column and what was removed.** giantsteps-mtg-key rates every annotation 0, 1 or 2. This copy keeps only confidence 2. Audio for lower confidences was never downloaded, and their `.key` files were deleted so that every file in the folder is usable. 23 further tracks were dropped although they are confidence 2: their annotations name more than one key (for example `c# minor / e major`, with comments such as "from 0:00 to 0:56 key is a minor"). A single-key benchmark cannot score those fairly, so they are treated like low confidence. What remains is 1159 tracks, all confidence 2 and all single-key.

## What this means for interpreting results

1. **Report mode bias next to the score.** On this data the score alone cannot tell a balanced detector from one that answers minor almost always.
2. **Read the two sets separately.** They differ by 17 percentage points in minor share (84.6% vs 67.5%), so a detector that leans minor will look better on giantsteps-key than on giantsteps-mtg-key. A large gap between the two columns is usually a sign of mode bias, not of dataset difficulty.
3. **Both sets are EDM only.** Sustained bass, heavy percussion and distortion are the norm, and quiet breakdowns or noise risers carry no key at all. Scores here say nothing about pop, rock or classical music, where published systems behave quite differently.
4. **One track is worth about 0.0006** of the combined score (0.0017 on giantsteps-key alone). Differences under roughly 0.01 are noise unless a per-track test says otherwise.

## How these numbers were produced

Counted from the files with `tracks.scan_dataset()` and `mirex.parse_key()`, the same code path the benchmark uses. Durations come from WAV file sizes (16-bit mono 44.1 kHz). The baselines were computed by scoring a fixed answer against every annotation with `mirex.mirex_score()`.
