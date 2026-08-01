"""Benchmark the prototype key-finder against the GiantSteps annotations using MIREX scoring."""

import csv
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'prototype'))
from keyfinder import detect_key
from mirex import parse_key, mirex_score

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / 'data' / 'audio'
ANNOT_DIR = BASE_DIR / 'data' / 'annotations' / 'key'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
LOG_FILE = BASE_DIR / 'benchmark_debug.log'

MIREX_CATEGORIES = ['correct', 'fifth', 'relative', 'parallel', 'wrong']
PROGRESS_INTERVAL = 50   # print a running score every N tracks
BAR_WIDTH = 30           # width of the ASCII bar in the final report


# ---------- setup ----------

def configure_logging():
    """Configure warning-level logging to LOG_FILE, overwriting any previous run."""
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.WARNING,
        format='%(asctime)s [%(levelname)s] %(message)s',
        filemode='w',
        encoding='utf-8',
    )


# ---------- per-track evaluation ----------

def normalize_tonic(tonic):
    """Normalize a tonic string to standard capitalization, e.g. 'eb' -> 'Eb'."""
    if not tonic:
        return 'Unknown'
    t = tonic.strip()
    return (t[0].upper() + t[1:].lower()) if len(t) > 1 else t.upper()


def find_annotation(wav):
    """Return the annotation file matching wav's stem, or None if there isn't one."""
    return next(ANNOT_DIR.glob(f'{wav.stem}.*'), None)


def load_true_key(annot_path):
    """Read and parse an annotation file, returning (true_key, true_key_str)."""
    true_str = annot_path.read_text().strip()
    return parse_key(true_str), true_str


def score_prediction(true_key, pred_tonic, pred_mode):
    """Normalize a raw (tonic, mode) prediction and score it against true_key."""
    tonic_norm = normalize_tonic(pred_tonic)
    mode_norm = pred_mode.strip().lower()
    pred_key = parse_key(f'{tonic_norm} {mode_norm}')
    score, category = mirex_score(true_key, pred_key)
    return f'{tonic_norm} {mode_norm}', score, category


def evaluate_track(wav):
    """
    Evaluate one WAV file against its annotation.

    Returns (status, row):
      status is one of 'no_annot', 'bad_annot', 'fail_detect', 'fail_parse', 'ok'.
      row is a result dict (or None if there was no usable annotation).
    """
    annot = find_annotation(wav)
    if annot is None:
        logging.warning('no annotation for %s', wav.name)
        return 'no_annot', None

    try:
        true_key, true_str = load_true_key(annot)
    except Exception:
        print(f'  [skip-bad-annot]  {annot.name}')
        logging.exception('bad annotation: %s', annot.name)
        return 'bad_annot', None

    pred_str, score, category = 'FAILED', 0.0, 'wrong'

    try:
        pred_tonic, pred_mode = detect_key(wav)
    except Exception:
        print(f'  [fail-detect]     {wav.name}')
        logging.exception('detect failed: %s', wav.name)
        status = 'fail_detect'
    else:
        try:
            pred_str, score, category = score_prediction(true_key, pred_tonic, pred_mode)
            status = 'ok'
        except Exception:
            print(f'  [fail-parse-pred] {wav.name}: {pred_tonic!r} {pred_mode!r}')
            logging.exception(
                'parse pred failed: %s (%r %r)', wav.name, pred_tonic, pred_mode
            )
            status = 'fail_parse'

    row = {
        'track': wav.name,
        'true': true_str,
        'pred': pred_str,
        'score': score,
        'category': category,
    }
    return status, row


# ---------- reporting ----------

def write_csv(rows, out_csv, final_score, categories):
    """Write per-track results plus a summary row to out_csv."""
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['track', 'true', 'pred', 'score', 'category'])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            'track': 'SUMMARY',
            'true': f'n={len(rows)}',
            'pred': '',
            'score': round(final_score, 4),
            'category': '/'.join(f'{cat}={categories[cat]}' for cat in MIREX_CATEGORIES),
        })


def print_report(num_wavs, counts, rows, final_score, categories, out_csv):
    """Print a human-readable summary of a benchmark run."""
    n = len(rows)
    print()
    print('─' * 50)
    print(f'  WAV files found:        {num_wavs}')
    print(f'  skipped (no annot):     {counts["no_annot"]}')
    print(f'  skipped (bad annot):    {counts["bad_annot"]}')
    print(f'  failed  (detect):       {counts["fail_detect"]}')
    print(f'  failed  (parse pred):   {counts["fail_parse"]}')
    print(f'  evaluated:              {n}')
    print(f'  MIREX weighted score:   {final_score:.4f}')
    print()
    for cat in MIREX_CATEGORIES:
        c = categories[cat]
        bar = '█' * int(BAR_WIDTH * c / n) if n else ''
        print(f'  {cat:9s}  {c:4d}  ({100*c/n:5.1f}%)  {bar}')
    print('─' * 50)
    print(f'  Results: {out_csv}')


# ---------- orchestration ----------

def run_benchmark(wavs, timestamp):
    """Evaluate every WAV file, write a results CSV, print a report, and return the mean score."""
    out_csv = RESULTS_DIR / f'{timestamp}_prototype.csv'

    rows = []
    categories = Counter()
    counts = Counter()
    total_score = 0.0

    print(f'\nevaluating {len(wavs)} files...')
    logging.warning('=== run %s: %d wav files ===', timestamp, len(wavs))

    for i, wav in enumerate(wavs, 1):
        status, row = evaluate_track(wav)
        counts[status] += 1

        if row is not None:
            rows.append(row)
            total_score += row['score']
            categories[row['category']] += 1

        if i % PROGRESS_INTERVAL == 0 and rows:
            n_so_far = len(rows)
            print(f'  [{i}/{len(wavs)}] score so far: {total_score/n_so_far:.4f}  '
                  f'correct: {categories["correct"]}/{n_so_far}')

    final_score = total_score / len(rows) if rows else 0.0

    write_csv(rows, out_csv, final_score, categories)
    print_report(len(wavs), counts, rows, final_score, categories, out_csv)

    logging.warning(
        'done: found=%d no_annot=%d bad_annot=%d fail_detect=%d '
        'fail_parse=%d evaluated=%d score=%.4f',
        len(wavs), counts['no_annot'], counts['bad_annot'],
        counts['fail_detect'], counts['fail_parse'], len(rows), final_score,
    )

    return final_score


def main():
    """Run the key-finder benchmark over all annotated WAV files in AUDIO_DIR."""
    configure_logging()

    wavs = sorted(AUDIO_DIR.glob('*.wav'))
    if not wavs:
        print(f'No WAVs found in {AUDIO_DIR}')
        logging.error('no WAVs found in %s', AUDIO_DIR)
        return

    timestamp = time.strftime('%Y-%m-%d_%H%M')
    run_benchmark(wavs, timestamp)


if __name__ == '__main__':
    main()