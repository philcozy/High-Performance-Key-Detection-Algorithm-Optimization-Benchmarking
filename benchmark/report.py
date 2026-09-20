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
class ModeBias:
    """Which mode the pipeline answers, next to which mode the data actually has."""
    n: int
    annotated: Counter        # mode -> tracks annotated with it
    predicted: Counter        # mode -> tracks predicted with it
    recall: dict              # mode -> fraction of those tracks predicted with that mode

    def share(self, counts, mode):
        """counts[mode] as a fraction of all tracks."""
        return counts[mode] / self.n if self.n else 0.0


def summarize_modes(results):
    """
    Count major and minor, annotated versus predicted.

    The MIREX score hides which way a system leans: on a dataset that is mostly
    minor, always answering minor already scores well. These counts make it visible.
    """
    modes = [(parse_key(r.true)[1], parse_key(r.pred)[1]) for r in results if r.status == 'ok']
    annotated = Counter(true for true, _ in modes)
    predicted = Counter(pred for _, pred in modes)
    hits = Counter(true for true, pred in modes if true == pred)

    return ModeBias(
        n=len(modes),
        annotated=annotated,
        predicted=predicted,
        recall={mode: hits[mode] / annotated[mode] for mode in (MINOR, MAJOR) if annotated[mode]},
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


def print_mode_bias(modes):
    """One line: which way the pipeline leans, next to what the data actually is."""
    print(f'  mode bias: predicted {100 * modes.share(modes.predicted, MINOR):5.1f}% minor, '
          f'annotations {100 * modes.share(modes.annotated, MINOR):5.1f}% minor   '
          f'(recall: minor {100 * modes.recall.get(MINOR, 0.0):.1f}%, '
          f'major {100 * modes.recall.get(MAJOR, 0.0):.1f}%)')


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


def print_report(by_dataset, overall, perf, modes, pipeline, workers, results, out_csv, out_md):
    """Print the full human-readable summary of a run."""
    print()
    print('─' * LINE_WIDTH)
    print_accuracy(by_dataset, overall)
    print()
    print_mode_bias(modes)
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
    """MIREX score and error categories, each dataset on its own row."""
    def row(name, acc):
        shares = (f'{100 * acc.categories[cat] / acc.n:.1f}%' for cat in MIREX_CATEGORIES)
        return (name, acc.n, f'{acc.score:.4f}', *shares)

    rows = [row(name, acc) for name, acc in by_dataset.items()]
    if len(by_dataset) > 1:
        rows.append(row('**all**', overall))

    header = ('dataset', 'tracks', 'MIREX', *MIREX_CATEGORIES)
    weights = ', '.join(f'{cat} {score}' for cat, score in CATEGORY_SCORES.items())
    return (f'## Accuracy\n\n{markdown_table(header, rows)}\n\n'
            f'Category columns are shares of that dataset. MIREX weights: {weights}.')


def mode_section(modes_by_dataset, modes_overall):
    """
    Mode bias: what the data has, what the pipeline answered, and recall for each mode.

    A pipeline can score well by answering minor almost always, because these
    datasets are mostly minor and a mode mistake still earns 0.2 or 0.3.
    See DATASETS.md for the distribution of the datasets themselves.
    """
    def row(name, modes):
        return (
            name,
            f'{100 * modes.share(modes.annotated, MINOR):.1f}% / {100 * modes.share(modes.annotated, MAJOR):.1f}%',
            f'{100 * modes.share(modes.predicted, MINOR):.1f}% / {100 * modes.share(modes.predicted, MAJOR):.1f}%',
            f'{100 * modes.recall.get(MINOR, 0.0):.1f}%',
            f'{100 * modes.recall.get(MAJOR, 0.0):.1f}%',
        )

    rows = [row(name, modes) for name, modes in modes_by_dataset.items()]
    if len(modes_by_dataset) > 1:
        rows.append(row('**all**', modes_overall))

    header = ('dataset', 'annotated minor / major', 'predicted minor / major',
              'minor recall', 'major recall')
    return f'## Mode bias\n\n{markdown_table(header, rows)}'


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


def write_run_summary(out_md, timestamp, label, pipeline, workers, by_dataset, overall,
                      perf, modes_by_dataset, modes_overall, results, out_csv):
    """Write the one-page Markdown overview of a run: accuracy, mode bias, performance."""
    failures = Counter(r.status for r in results if r.status != 'ok')
    sections = [
        f'# {label} — {timestamp}',
        f'- pipeline: `{pipeline}`\n'
        f'- tracks: {overall.n}\n'
        f'- per-track data: `{out_csv.name}`'
        + (f'\n- failed: {dict(failures)} (tracebacks in benchmark_debug.log)' if failures else ''),
        accuracy_section(by_dataset, overall),
        mode_section(modes_by_dataset, modes_overall),
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
