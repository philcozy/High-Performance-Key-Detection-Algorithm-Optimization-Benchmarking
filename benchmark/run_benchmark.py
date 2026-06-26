import sys
import csv
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'prototype'))
from keyfinder import detect_key
from mirex import parse_key, mirex_score

AUDIO_DIR = Path(__file__).parent / 'data' / 'audio'
ANNOT_DIR = Path(__file__).parent / 'data' / 'annotations' / 'key'
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

def main():
    timestamp = time.strftime('%Y-%m-%d_%H%M')
    out_csv = RESULTS_DIR / f'{timestamp}_baseline.csv'

    wavs = sorted(AUDIO_DIR.glob('*.wav'))
    if not wavs:
        print(f'No WAVs found in {AUDIO_DIR}')
        return

    rows = []
    categories = Counter()
    total_score = 0.0

    for i, wav in enumerate(wavs, 1):
        # GiantSteps: annotation has same stem, extension .key or .txt
        annot = next(ANNOT_DIR.glob(f'{wav.stem}.*'), None)
        if annot is None:
            print(f'[skip] no annotation for {wav.name}')
            continue

        true_str = annot.read_text().strip()
        try:
            true = parse_key(true_str)
        except Exception as e:
            print(f'[skip] cannot parse {annot.name}: {true_str!r} ({e})')
            continue

        try:
            pred_tonic, pred_mode = detect_key(wav)
            # normalize from your states tuple ('C', 'c', etc.) to canonical
            pred = (parse_key(f'{pred_tonic.upper()} {pred_mode}'))
        except Exception as e:
            print(f'[fail] {wav.name}: {e}')
            continue

        score, category = mirex_score(true, pred)
        total_score += score
        categories[category] += 1
        rows.append({
            'track': wav.name,
            'true': true_str,
            'pred': f'{pred_tonic} {pred_mode}',
            'score': score,
            'category': category,
        })

    n = len(rows)
    final_score = total_score / n if n else 0.0

    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['track', 'true', 'pred', 'score', 'category'])
        w.writeheader()
        w.writerows(rows)
        w.writerow({
            'track': 'SUMMARY',
            'true': f'n={n}',
            'pred': '',
            'score': round(final_score, 4),
            'category': '/'.join(f'{cat}={categories[cat]}' for cat in ['correct','fifth','relative','parallel','wrong']),
        })

    print()
    print(f'Total tracks evaluated: {n}')
    print(f'MIREX weighted score:   {final_score:.3f}')
    print()
    for cat in ['correct', 'fifth', 'relative', 'parallel', 'wrong']:
        c = categories[cat]
        print(f'  {cat:9s} {c:4d}  ({100*c/n:5.1f}%)')
    print()
    print(f'Wrote {out_csv}')

if __name__ == '__main__':
    main()