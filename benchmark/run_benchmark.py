"""
Benchmark the prototype key-finder on the GiantSteps datasets using MIREX scoring,
and record how long every pipeline stage takes.

Usage (from the benchmark/ folder):
    python run_benchmark.py                              # all datasets, 4 workers
    python run_benchmark.py --datasets giantsteps-key    # one dataset
    python run_benchmark.py --workers 1                  # no worker processes (easy to debug)
    python run_benchmark.py --pipeline libkeyfinder_port # another pipeline file in prototype/
    python run_benchmark.py --label cosine-v2            # name the run in results/runs.csv
"""

import os

# Each worker process handles one track at a time. Stop numpy/scipy from starting
# extra threads inside every worker, or the workers would compete for the same cores
# and the timings would get noisy. Must be set before numpy is imported.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from evaluate import available_pipelines, init_worker, evaluate_track
from report import (
    summarize_accuracy, summarize_modes, summarize_performance,
    print_report, write_run_summary, write_track_csv, append_run_history,
)
from tracks import DATASET_NAMES, scan_dataset

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / 'results'
RUN_HISTORY_CSV = RESULTS_DIR / 'runs.csv'
LOG_FILE = BASE_DIR / 'benchmark_debug.log'

DEFAULT_WORKERS = 4           # = performance cores on this Mac; efficiency cores are slower and would blur the timings
DEFAULT_PIPELINE = 'keyfinder'
PROGRESS_INTERVAL = 200       # print a running score every N tracks


# ---------- setup ----------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', choices=DATASET_NAMES, default=list(DATASET_NAMES),
                        help='datasets to evaluate (default: all)')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'worker processes (default: {DEFAULT_WORKERS}); 1 runs everything in this process')
    parser.add_argument('--pipeline', choices=available_pipelines(), default=DEFAULT_PIPELINE,
                        help=f'pipeline file in prototype/ to benchmark (default: {DEFAULT_PIPELINE})')
    parser.add_argument('--label', default=None,
                        help='name for this run in the file names and runs.csv (default: the pipeline name)')
    return parser.parse_args()


def configure_logging():
    """Configure warning-level logging to LOG_FILE, overwriting any previous run."""
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.WARNING,
        format='%(asctime)s [%(levelname)s] %(message)s',
        filemode='w',
        encoding='utf-8',
    )


# ---------- step 1: collect tracks ----------

def collect_tracks(dataset_names):
    """Return the tracks of every requested dataset."""
    tracks = []
    for name in dataset_names:
        dataset_tracks = scan_dataset(name)
        print(f'  {name:22s} {len(dataset_tracks):5d} tracks')
        tracks += dataset_tracks
    return tracks


# ---------- step 2: evaluate tracks ----------

def handle_result(result, done, total, results):
    """Store one finished result, report failures, and print progress now and then."""
    results.append(result)

    if result.status != 'ok':
        print(f'  [{result.status}] {result.dataset}/{result.track}')
        logging.error('%s: %s/%s\n%s', result.status, result.dataset, result.track, result.error)

    if done % PROGRESS_INTERVAL == 0 or done == total:
        mean_score = sum(r.score for r in results) / len(results)
        print(f'  [{done:4d}/{total}] score so far: {mean_score:.4f}')


def evaluate_in_this_process(tracks, pipeline):
    """Evaluate tracks one after another. Slow, but breakpoints and print() just work."""
    init_worker(pipeline)
    results = []
    for done, track in enumerate(tracks, 1):
        handle_result(evaluate_track(track), done, len(tracks), results)
    return results


def evaluate_in_parallel(tracks, pipeline, workers):
    """
    Evaluate tracks in `workers` separate processes at the same time.

    Each process gets its own copy of Python and runs init_worker() once.
    Tracks finish in any order, so the results are sorted afterwards.
    """
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(pipeline,)) as pool:
        futures = [pool.submit(evaluate_track, track) for track in tracks]
        for done, future in enumerate(as_completed(futures), 1):
            handle_result(future.result(), done, len(tracks), results)
    return results


def evaluate_all(tracks, pipeline, workers):
    """Evaluate every track; return (results in dataset/track order, wall-clock seconds)."""
    start = time.perf_counter()
    if workers == 1:
        results = evaluate_in_this_process(tracks, pipeline)
    else:
        results = evaluate_in_parallel(tracks, pipeline, workers)
    wall_s = time.perf_counter() - start

    results.sort(key=lambda r: (r.dataset, r.track))
    return results, wall_s


# ---------- orchestration ----------

def main():
    args = parse_args()
    label = args.label or args.pipeline
    configure_logging()
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d_%H%M')

    # 1. collect tracks
    print('\ncollecting tracks...')
    tracks = collect_tracks(args.datasets)
    if not tracks:
        print('No usable tracks found.')
        logging.error('no usable tracks in %s', args.datasets)
        return

    # 2. evaluate
    print(f'\nevaluating {len(tracks)} tracks with {args.pipeline}, {args.workers} workers...')
    logging.warning('=== run %s (%s, %s): %d tracks ===', timestamp, label, args.pipeline, len(tracks))
    results, wall_s = evaluate_all(tracks, args.pipeline, args.workers)

    # 3. summarize
    by_dataset = {name: summarize_accuracy([r for r in results if r.dataset == name])
                  for name in args.datasets}
    modes_by_dataset = {name: summarize_modes([r for r in results if r.dataset == name])
                        for name in args.datasets}
    overall = summarize_accuracy(results)
    modes_overall = summarize_modes(results)
    perf = summarize_performance(results, wall_s)

    # 4. save and report
    out_csv = RESULTS_DIR / f'{timestamp}_{label}.csv'
    out_md = RESULTS_DIR / f'{timestamp}_{label}.md'
    write_track_csv(results, out_csv)
    write_run_summary(out_md, timestamp, label, args.pipeline, args.workers, by_dataset,
                      overall, perf, modes_by_dataset, modes_overall, results, out_csv)
    append_run_history(RUN_HISTORY_CSV, timestamp, label, args.pipeline, args.workers, by_dataset, overall, perf)
    print_report(by_dataset, overall, perf, modes_overall, args.pipeline, args.workers,
                 results, out_csv, out_md)

    logging.warning('done: evaluated=%d score=%.4f wall=%.1fs', overall.n, overall.score, wall_s)


if __name__ == '__main__':
    main()
