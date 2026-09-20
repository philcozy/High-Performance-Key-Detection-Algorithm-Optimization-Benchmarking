"""Summarize TrackResults: terminal report, saved Markdown summary, per-track CSV, and a one-line-per-run history CSV."""

import csv
from collections import Counter
from dataclasses import dataclass

import numpy as np

from mirex import CATEGORY_SCORES, MAJOR, MINOR, parse_key
from tracks import DATASET_NAMES

MIREX_CATEGORIES = list(CATEGORY_SCORES)   # best to worst, as defined in mirex.py
BAR_WIDTH = 30                # width of the longest ASCII bar
LINE_WIDTH = 56               # width of the ─── separators
TAIL_PERCENTILE = 95          # "slow track" time reported next to the median

NOTE_NAMES = ('C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B')
# Tonic errors worth naming in the summary: semitones above the true tonic -> what it means
NAMED_TONIC_ERRORS = {0: 'same tonic', 3: 'relative major', 5: 'fourth', 7: 'fifth', 9: 'relative minor'}


# ---------- summary numbers ----------

@dataclass
class Accuracy:
    n: int
    score: float              # mean MIREX score
    categories: Counter       # category -> number of tracks


@dataclass
class Performance:
    n: int                    # tracks that ran successfully
    wall_s: float             # time the whole run took
    tracks_per_s: float
    median_ms: float          # median detect_key() time per track
    tail_ms: float            # TAIL_PERCENTILE detect_key() time per track
    stage_median_ms: dict     # {stage: median ms}, in pipeline order
    stage_share: dict         # {stage: fraction of all detect_key() time}
    peak_mb: float            # largest memory use of any worker


def summarize_accuracy(results):
    """Mean MIREX score and category counts over results."""
    n = len(results)
    score = sum(r.score for r in results) / n if n else 0.0
    return Accuracy(n, score, Counter(r.category for r in results))


@dataclass
class KeyErrors:
    """Where the mistakes are, beyond the single MIREX number."""
    n: int
    true_minor_share: float   # fraction of tracks whose annotation is minor
    pred_minor_share: float   # fraction of tracks predicted minor
    mode_recall: dict         # mode -> fraction of those tracks predicted with that mode
    mode_confusion: Counter   # (true mode, predicted mode) -> tracks
    tonic_errors: Counter     # semitones from true tonic to predicted tonic -> tracks
    key_scores: dict          # 'Eb minor' -> mean score of tracks annotated with that key
    key_counts: dict          # 'Eb minor' -> number of such tracks


def summarize_keys(results):
    """
    Break the results down by mode and by tonic.

    The MIREX score hides which way a system leans: on a dataset that is mostly
    minor, always answering minor scores well. These numbers make that visible.
    """
    scored = [r for r in results if r.status == 'ok']
    keys = [(parse_key(r.true), parse_key(r.pred), r.score) for r in scored]
    n = len(keys)

    mode_confusion = Counter((true[1], pred[1]) for true, pred, _ in keys)
    tonic_errors = Counter((pred[0] - true[0]) % 12 for true, pred, _ in keys)

    true_counts = Counter(true[1] for true, _, _ in keys)
    mode_recall = {
        mode: mode_confusion[(mode, mode)] / true_counts[mode]
        for mode in (MAJOR, MINOR) if true_counts[mode]
    }

    key_scores, key_counts = {}, {}
    for (tonic, mode), _, score in keys:
        name = f'{NOTE_NAMES[tonic]} {mode}'
        key_scores[name] = key_scores.get(name, 0.0) + score
        key_counts[name] = key_counts.get(name, 0) + 1
    key_scores = {name: total / key_counts[name] for name, total in key_scores.items()}

    return KeyErrors(
        n=n,
        true_minor_share=true_counts[MINOR] / n if n else 0.0,
        pred_minor_share=sum(c for (_, p), c in mode_confusion.items() if p == MINOR) / n if n else 0.0,
        mode_recall=mode_recall,
        mode_confusion=mode_confusion,
        tonic_errors=tonic_errors,
        key_scores=key_scores,
        key_counts=key_counts,
    )


def summarize_performance(results, wall_s):
    """Timing statistics over the tracks that ran successfully."""
    timed = [r for r in results if r.status == 'ok']
    if not timed:
        return None

    totals = np.array([r.total_ms for r in timed])
    stage_median_ms = {}
    stage_share = {}
    for stage in timed[0].stage_ms:
        stage_times = np.array([r.stage_ms[stage] for r in timed])
        stage_median_ms[stage] = float(np.median(stage_times))
        stage_share[stage] = float(stage_times.sum() / totals.sum())

    return Performance(
        n=len(timed),
        wall_s=wall_s,
        tracks_per_s=len(results) / wall_s,
        median_ms=float(np.median(totals)),
        tail_ms=float(np.percentile(totals, TAIL_PERCENTILE)),
        stage_median_ms=stage_median_ms,
        stage_share=stage_share,
        peak_mb=max(r.peak_mb for r in timed),
    )


# ---------- terminal report ----------

def bar(fraction):
    """ASCII bar for a 0-1 fraction."""
    return '█' * int(BAR_WIDTH * fraction)


def print_accuracy(by_dataset, overall):
    print('  ACCURACY')
    print(f'  {"dataset":22s} {"tracks":>6s}   MIREX')
    for name, acc in by_dataset.items():
        print(f'  {name:22s} {acc.n:6d}   {acc.score:.4f}')
    print(f'  {"all":22s} {overall.n:6d}   {overall.score:.4f}')
    print()
    for cat in MIREX_CATEGORIES:
        count = overall.categories[cat]
        fraction = count / overall.n if overall.n else 0.0
        print(f'  {cat:9s}  {count:4d}  ({100 * fraction:5.1f}%)  {bar(fraction)}')


def print_mode_bias(keys):
    """One line: which way the pipeline leans, next to what the data actually is."""
    print(f'  mode bias: predicted {100 * keys.pred_minor_share:5.1f}% minor, '
          f'annotations {100 * keys.true_minor_share:5.1f}% minor   '
          f'(major recall {100 * keys.mode_recall.get(MAJOR, 0.0):.1f}%)')


def print_performance(perf, pipeline, workers):
    print(f'  PERFORMANCE   ({pipeline}, {workers} workers)')
    if perf is None:
        print('  no track ran successfully')
        return
    print(f'  wall time:     {perf.wall_s:7.1f} s   ({perf.tracks_per_s:.1f} tracks/s)')
    print(f'  per track:     {perf.median_ms:7.1f} ms median, '
          f'{perf.tail_ms:.1f} ms p{TAIL_PERCENTILE}')
    print(f'  peak memory:   {perf.peak_mb:7.0f} MB per worker')
    print()
    print(f'  {"stage":12s} {"median ms":>9s}   share of time')
    for stage, share in perf.stage_share.items():
        print(f'  {stage:12s} {perf.stage_median_ms[stage]:9.2f}   '
              f'{100 * share:5.1f}%  {bar(share)}')


def print_failures(results):
    """Count tracks whose pipeline run failed; print nothing if there are none."""
    failures = Counter(r.status for r in results if r.status != 'ok')
    if failures:
        print('  FAILED')
        for status, count in failures.items():
            print(f'  {status:12s} {count:5d}   (tracebacks in benchmark_debug.log)')
        print('─' * LINE_WIDTH)


def print_report(by_dataset, overall, perf, keys, pipeline, workers, results, out_csv, out_md):
    """Print the full human-readable summary of a run."""
    print()
    print('─' * LINE_WIDTH)
    print_accuracy(by_dataset, overall)
    print()
    print_mode_bias(keys)
    print('─' * LINE_WIDTH)
    print_performance(perf, pipeline, workers)
    print('─' * LINE_WIDTH)
    print_failures(results)
    print(f'  Summary:   {out_md}')
    print(f'  Per-track: {out_csv}')


# ---------- saved summary ----------
# The per-track CSV is data for scripts; nobody reads 1763 rows. This file is the
# one a human opens: everything worth knowing about a run, on one page.

def markdown_table(header, rows):
    """Render one Markdown table from a header tuple and a list of row tuples."""
    lines = [f'| {" | ".join(header)} |', f'|{"---|" * len(header)}']
    lines += [f'| {" | ".join(str(cell) for cell in row)} |' for row in rows]
    return '\n'.join(lines)


def accuracy_section(by_dataset, overall):
    rows = [(name, acc.n, f'{acc.score:.4f}') for name, acc in by_dataset.items()]
    rows.append(('**all**', overall.n, f'**{overall.score:.4f}**'))
    table = markdown_table(('dataset', 'tracks', 'MIREX'), rows)

    category_rows = [
        (cat, overall.categories[cat], f'{100 * overall.categories[cat] / overall.n:.1f}%',
         CATEGORY_SCORES[cat])
        for cat in MIREX_CATEGORIES
    ]
    categories = markdown_table(('category', 'tracks', 'share', 'score each'), category_rows)
    return f'## Accuracy\n\n{table}\n\n{categories}'


def mode_section(keys):
    """Mode bias: which way the pipeline leans, and how well it finds each mode."""
    confusion_rows = [
        (f'true {mode}',
         keys.mode_confusion[(mode, MAJOR)],
         keys.mode_confusion[(mode, MINOR)],
         f'{100 * keys.mode_recall.get(mode, 0.0):.1f}%')
        for mode in (MAJOR, MINOR)
    ]
    return (
        '## Mode bias\n\n'
        f'- annotations: **{100 * keys.true_minor_share:.1f}%** minor\n'
        f'- predictions: **{100 * keys.pred_minor_share:.1f}%** minor\n\n'
        + markdown_table(('', 'predicted major', 'predicted minor', 'recall'), confusion_rows)
    )


def tonic_section(keys):
    """How far off the tonic is, in semitones, regardless of mode."""
    rows = []
    for semitones in range(12):
        count = keys.tonic_errors[semitones]
        rows.append((semitones, NAMED_TONIC_ERRORS.get(semitones, ''), count,
                     f'{100 * count / keys.n:.1f}%'))
    return f'## Tonic error\n\nSemitones from the annotated tonic to the predicted one.\n\n' \
           + markdown_table(('semitones', 'meaning', 'tracks', 'share'), rows)


def weakest_keys_section(keys, count=5):
    """The annotated keys the pipeline handles worst and best."""
    ranked = sorted(keys.key_scores.items(), key=lambda item: item[1])
    rows = [(name, f'{score:.3f}', keys.key_counts[name]) for name, score in ranked[:count]]
    rows.append(('...', '', ''))
    rows += [(name, f'{score:.3f}', keys.key_counts[name]) for name, score in ranked[-count:]]
    return f'## Weakest and strongest annotated keys\n\n' \
           + markdown_table(('annotated key', 'mean MIREX', 'tracks'), rows)


def performance_section(perf, workers):
    if perf is None:
        return '## Performance\n\nNo track ran successfully.'
    rows = [(stage, f'{perf.stage_median_ms[stage]:.2f}', f'{100 * share:.1f}%')
            for stage, share in perf.stage_share.items()]
    return (
        '## Performance\n\n'
        f'- wall time: **{perf.wall_s:.1f} s** with {workers} workers ({perf.tracks_per_s:.1f} tracks/s)\n'
        f'- per track: **{perf.median_ms:.1f} ms** median, {perf.tail_ms:.1f} ms p{TAIL_PERCENTILE}\n'
        f'- peak memory: {perf.peak_mb:.0f} MB per worker\n\n'
        + markdown_table(('stage', 'median ms', 'share of time'), rows)
    )


def write_run_summary(out_md, timestamp, label, pipeline, workers,
                      by_dataset, overall, perf, keys, results, out_csv):
    """Write the one-page Markdown overview of a run."""
    failures = Counter(r.status for r in results if r.status != 'ok')
    sections = [
        f'# {label} — {timestamp}',
        f'- pipeline: `{pipeline}`\n'
        f'- tracks: {overall.n}\n'
        f'- per-track data: `{out_csv.name}`'
        + (f'\n- failed: {dict(failures)} (tracebacks in benchmark_debug.log)' if failures else ''),
        accuracy_section(by_dataset, overall),
        mode_section(keys),
        tonic_section(keys),
        weakest_keys_section(keys),
        performance_section(perf, workers),
    ]
    out_md.write_text('\n\n'.join(sections) + '\n')


# ---------- CSV files ----------

def write_track_csv(results, out_csv):
    """One row per track: prediction, score and timing."""
    stages = list(results[0].stage_ms) if results else []
    stage_columns = [f'{stage}_ms' for stage in stages]
    fieldnames = ['dataset', 'track', 'true', 'pred', 'score', 'category',
                  'status', 'total_ms', *stage_columns]

    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                'dataset': r.dataset, 'track': r.track, 'true': r.true,
                'pred': r.pred, 'score': r.score, 'category': r.category,
                'status': r.status, 'total_ms': round(r.total_ms, 2),
            }
            for stage in stages:
                row[f'{stage}_ms'] = round(r.stage_ms.get(stage, 0.0), 3)
            writer.writerow(row)


def append_run_history(history_csv, timestamp, label, pipeline, workers, by_dataset, overall, perf):
    """
    Add one summary line for this run, so runs can be compared side by side.

    Pipelines have different stages, so all stage times share one column,
    e.g. 'preprocess 126.0 | spectrum 11.3 | cqt 0.9'.
    """
    row = {'timestamp': timestamp, 'label': label, 'pipeline': pipeline, 'workers': workers,
           'tracks': overall.n, 'mirex_all': round(overall.score, 4)}
    for name in DATASET_NAMES:
        acc = by_dataset.get(name)
        row[f'mirex_{name}'] = round(acc.score, 4) if acc else ''
    row['wall_s'] = round(perf.wall_s, 1) if perf else ''
    row['median_ms'] = round(perf.median_ms, 2) if perf else ''
    row['stage_median_ms'] = ' | '.join(
        f'{stage} {ms:.2f}' for stage, ms in perf.stage_median_ms.items()
    ) if perf else ''

    is_new_file = not history_csv.exists()
    with history_csv.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)
