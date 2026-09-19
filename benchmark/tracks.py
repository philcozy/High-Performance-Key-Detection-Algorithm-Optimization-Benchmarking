"""
Find the benchmark tracks in each dataset and read their key annotations.
    The two datasets use different layouts:
      giantsteps-key:      'C minor'
      giantsteps-mtg-key:  'd minor<TAB>2<TAB>'   (key, confidence)

Every dataset has exactly one .key file per .wav file, so every audio file becomes a Track.
"""

from dataclasses import dataclass
from pathlib import Path
from mirex import parse_key

DATASETS_DIR = Path(__file__).parent / 'datasets'
DATASET_NAMES = ('giantsteps-key', 'giantsteps-mtg-key')
AUDIO_SUBDIR = 'audio'        # <dataset>/audio/<id>.wav
KEY_SUBDIR = 'key'            # <dataset>/key/<id>.key


@dataclass(frozen=True)
class Track:
    """One track that will be evaluated."""
    dataset: str              # which dataset
    audio_path: Path
    true_key_str: str         # original annotation as written
    true_key: tuple           # parsed (tonic_idx, mode)


# ---------- reading one annotation ----------

def read_annotation(key_path):
    """Return the key. e.g. 'd minor'."""
    raw = key_path.read_text()
    key = raw.split('\t')[0]
    return key.strip()


# ---------- scanning a dataset ----------

def scan_dataset(name):
    """Return one Track per audio file in the dataset, sorted by file name."""
    audio_dir = DATASETS_DIR / name / AUDIO_SUBDIR
    key_dir = DATASETS_DIR / name / KEY_SUBDIR

    tracks = []
    for audio_path in sorted(audio_dir.glob('*.wav')):
        true_key_str = read_annotation(key_dir / f'{audio_path.stem}.key')
        track = Track(name, audio_path, true_key_str, parse_key(true_key_str))
        tracks.append(track)
    return tracks
