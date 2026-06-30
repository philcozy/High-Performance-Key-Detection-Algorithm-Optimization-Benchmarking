import sys
import csv
import time
import logging
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'prototype'))
from keyfinder import detect_key
from mirex import parse_key, mirex_score

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / 'data' / 'audio'
ANNOT_DIR = Path(__file__).parent / 'data' / 'annotations' / 'key'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=BASE_DIR / 'benchmark_debug.log',
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='w',
    encoding='utf-8'
)

def normalize_tonic(tonic: str) -> str:
    """'eb ' -> 'Eb', 'C#' -> 'C#'"""
    if not tonic:
        return "Unknown"
    t = tonic.strip()
    if len(t) > 1:
        return t[0].upper() + t[1:].lower()
    return t.upper()


def main():
    timestamp = time.strftime('%Y-%m-%d_%H%M')
    out_csv = RESULTS_DIR / f'{timestamp}_baseline.csv'

    wavs = sorted(AUDIO_DIR.glob('*.wav'))
    if not wavs:
        print(f'No WAVs found in {AUDIO_DIR}')
        logging.error('no WAVs found in %s', AUDIO_DIR)
        return

    logging.warning('=== benchmark run %s: %d wav files found ===', timestamp, len(wavs))

    rows = []
    categories = Counter()
    total_score = 0.0

    skipped_no_annot = 0
    skipped_bad_annot = 0
    failed_detect = 0
    failed_parse_pred = 0

    print(f'Evaluating {len(wavs)} files...')

    for i, wav in enumerate(wavs, 1):
        # ── annotation ──────────────────────────────────────────────────────
        annot = next(ANNOT_DIR.glob(f'{wav.stem}.*'), None)
        if annot is None:
            skipped_no_annot += 1
            print(f'[skip-no-annot]   {wav.name}')
            logging.warning('no annotation for %s', wav.name)
            continue

        try:
            true_str = annot.read_text().strip()
            true = parse_key(true_str)
        except Exception:
            skipped_bad_annot += 1
            print(f'[skip-bad-annot]  {annot.name}')
            logging.exception('bad annotation: %s', annot.name)
            continue

        # ── detection ───────────────────────────────────────────────────────
        pred_str = 'FAILED'
        score, category = 0.0, 'wrong'

        try:
            pred_tonic, pred_mode = detect_key(wav)
        except Exception:
            failed_detect += 1
            print(f'[fail-detect]     {wav.name}')
            logging.exception('detect_key failed: %s', wav.name)
        else:
            try:
                tonic_norm = normalize_tonic(pred_tonic)
                pred_mode_norm = pred_mode.strip().lower()
                pred = parse_key(f'{tonic_norm} {pred_mode_norm}')
                score, category = mirex_score(true, pred)
                pred_str = f'{tonic_norm} {pred_mode_norm}'
            except Exception:
                failed_parse_pred += 1
                print(f'[fail-parse-pred] {wav.name}: {pred_tonic!r} {pred_mode!r}')
                logging.exception('could not parse prediction for %s (%r %r)',
                                   wav.name, pred_tonic, pred_mode)

        total_score += score
        categories[category] += 1
        rows.append({
            'track':    wav.name,
            'true':     true_str,
            'pred':     pred_str,
            'score':    score,
            'category': category,
        })

        if i % 50 == 0:
            n_so_far = len(rows)
            print(f'  [{i}/{len(wavs)}] score so far: {total_score/n_so_far:.4f}  '
                  f'correct: {categories["correct"]}/{n_so_far}')

    # ── results ─────────────────────────────────────────────────────────────
    n = len(rows)
    final_score = total_score / n if n else 0.0

    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['track', 'true', 'pred', 'score', 'category'])
        w.writeheader()
        w.writerows(rows)
        w.writerow({
            'track':    'SUMMARY',
            'true':     f'n={n}',
            'pred':     '',
            'score':    round(final_score, 4),
            'category': '/'.join(
                f'{cat}={categories[cat]}'
                for cat in ['correct', 'fifth', 'relative', 'parallel', 'wrong']
            ),
        })

    print()
    print('─' * 50)
    print(f'WAV files found:         {len(wavs)}')
    print(f'  skipped (no annot):    {skipped_no_annot}')
    print(f'  skipped (bad annot):   {skipped_bad_annot}')
    print(f'  failed  (detect):      {failed_detect}')
    print(f'  failed  (parse pred):  {failed_parse_pred}')
    print(f'  evaluated:             {n}')
    print('─' * 50)
    print(f'MIREX weighted score:    {final_score:.4f}')
    print()
    for cat in ['correct', 'fifth', 'relative', 'parallel', 'wrong']:
        c = categories[cat]
        bar = '█' * int(30 * c / n) if n else ''
        print(f'  {cat:9s}  {c:4d}  ({100*c/n:5.1f}%)  {bar}')
    print()
    print(f'Results written to: {out_csv}')

    logging.warning(
        'run complete: found=%d no_annot=%d bad_annot=%d fail_detect=%d '
        'fail_parse=%d evaluated=%d score=%.4f',
        len(wavs), skipped_no_annot, skipped_bad_annot,
        failed_detect, failed_parse_pred, n, final_score,
    )


if __name__ == '__main__':
    main()