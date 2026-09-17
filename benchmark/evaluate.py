"""
Evaluate one track: run the pipeline, time it, and score the prediction.

Everything here runs inside a worker process. Each worker calls init_worker()
once, then evaluate_track() once per track it is given.
"""

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'prototype'))
import keyfinder
from mirex import parse_key, mirex_score
from timing import StageTimer, peak_memory_mb

# Functions of keyfinder.py to time. Keep in the order detect_key() calls them.
PIPELINE_STAGES = ('preprocess', 'spectrum', 'cqt', 'fold', 'classify')


@dataclass
class TrackResult:
    """Everything recorded about one track."""
    dataset: str
    track: str                # audio file name
    true: str                 # annotated key, as written in the .key file
    pred: str = 'FAILED'      # predicted key, e.g. 'D minor'
    score: float = 0.0        # MIREX score
    category: str = 'wrong'   # MIREX category
    status: str = 'ok'        # 'ok', 'fail_detect' or 'fail_parse'
    error: str = ''           # traceback when status is not 'ok'
    total_ms: float = 0.0     # time spent inside detect_key()
    stage_ms: dict = field(default_factory=dict)   # {stage: ms}
    peak_mb: float = 0.0      # peak memory of the worker process so far


# ---------- worker setup ----------

_stage_timer = None           # one per worker process, created by init_worker()


def init_worker():
    """Run once in each worker process: install the stage stopwatches."""
    global _stage_timer
    _stage_timer = StageTimer(keyfinder, PIPELINE_STAGES)


# ---------- scoring ----------

def normalize_tonic(tonic):
    """Normalize a tonic string to standard capitalization, e.g. 'eb' -> 'Eb'."""
    if not tonic:
        return 'Unknown'
    t = tonic.strip()
    return (t[0].upper() + t[1:].lower()) if len(t) > 1 else t.upper()


def score_prediction(true_key, pred_tonic, pred_mode):
    """Normalize a raw (tonic, mode) prediction and score it against true_key."""
    tonic_norm = normalize_tonic(pred_tonic)
    mode_norm = pred_mode.strip().lower()
    pred_key = parse_key(f'{tonic_norm} {mode_norm}')
    score, category = mirex_score(true_key, pred_key)
    return f'{tonic_norm} {mode_norm}', score, category


# ---------- one track ----------

def evaluate_track(track):
    """Run detect_key() on one Track and return a TrackResult."""
    result = TrackResult(track.dataset, track.audio_path.name, track.true_key_str)

    # 1. run the pipeline, timing the whole call and each stage
    _stage_timer.reset()
    start = time.perf_counter()
    try:
        pred_tonic, pred_mode = keyfinder.detect_key(track.audio_path)
    except Exception:
        result.status = 'fail_detect'
        result.error = traceback.format_exc()
        return result
    result.total_ms = 1000 * (time.perf_counter() - start)
    result.stage_ms = _stage_timer.milliseconds()
    result.peak_mb = peak_memory_mb()

    # 2. score the prediction
    try:
        result.pred, result.score, result.category = score_prediction(
            track.true_key, pred_tonic, pred_mode
        )
    except Exception:
        result.status = 'fail_parse'
        result.pred = f'{pred_tonic} {pred_mode}'
        result.error = traceback.format_exc()

    return result
