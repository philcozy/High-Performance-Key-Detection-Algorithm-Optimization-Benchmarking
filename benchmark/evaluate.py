"""
Evaluate one track: run the pipeline, time it, and score the prediction.

Everything here runs inside a worker process. Each worker calls init_worker()
once, then evaluate_track() once per track it is given.
"""

import importlib
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from mirex import parse_key, mirex_score
from timing import StageTimer, peak_memory_mb

PIPELINE_DIR = Path(__file__).resolve().parent.parent / 'prototype'
sys.path.insert(0, str(PIPELINE_DIR))


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

_pipeline = None              # the pipeline module, loaded by init_worker()
_stage_timer = None           # one per worker process, created by init_worker()


def available_pipelines():
    """
    Import names of the pipeline files in prototype/ and its subfolders,
    A pipeline is any .py file that defines STAGES.
    """
    names = []
    for path in PIPELINE_DIR.rglob('*.py'):
        if 'STAGES = ' in path.read_text():
            relative = path.relative_to(PIPELINE_DIR).with_suffix('')
            names.append('.'.join(relative.parts))
    return sorted(names)


def init_worker(pipeline_name):
    """Run once in each worker process: load the pipeline and time the stages it lists in STAGES."""
    global _pipeline, _stage_timer
    _pipeline = importlib.import_module(pipeline_name)
    _stage_timer = StageTimer(_pipeline, _pipeline.STAGES)


# ---------- scoring ----------

def score_prediction(true_key, pred_tonic, pred_mode):
    """
    Score a pipeline's raw (tonic, mode) prediction against true_key.

    Returns (prediction as written by the pipeline, score, category).
    Capitalisation and spelling are handled by parse_key().
    """
    pred_str = f'{pred_tonic} {pred_mode}'
    score, category = mirex_score(true_key, parse_key(pred_str))
    return pred_str, score, category


# ---------- one track ----------

def evaluate_track(track):
    """Run detect_key() on one Track and return a TrackResult."""
    result = TrackResult(track.dataset, track.audio_path.name, track.true_key_str)

    # 1. run the pipeline, timing the whole call and each stage
    _stage_timer.reset()
    start = time.perf_counter()
    try:
        pred_tonic, pred_mode = _pipeline.detect_key(track.audio_path)
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
