"""
Measure how long each pipeline stage takes, without editing the pipeline.

How it works: detect_key() calls its stages (preprocess, spectrum, ...) by looking
their names up in the keyfinder module every time it runs. StageTimer replaces
those module attributes with wrappers that call the original function and add the
elapsed time to a stopwatch. detect_key() then calls the wrappers without knowing
it, and keyfinder.py stays untouched.
"""

import resource
import sys
import time
from functools import wraps

BYTES_PER_MB = 1024 * 1024


class StageTimer:
    """Stopwatch for each named stage function of a module."""

    def __init__(self, module, stage_names):
        self.stage_names = tuple(stage_names)
        self.seconds = {}
        for name in self.stage_names:
            self._wrap(module, name)

    def _wrap(self, module, name):
        """Replace module.<name> with a version that records its running time."""
        if not hasattr(module, name):
            raise AttributeError(
                f'{module.__name__} has no stage function {name!r}; '
                f'update PIPELINE_STAGES in run_benchmark.py'
            )
        original = getattr(module, name)

        @wraps(original)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                self.seconds[name] = self.seconds.get(name, 0.0) + elapsed

        setattr(module, name, timed)

    def reset(self):
        """Zero every stopwatch; call before each track."""
        self.seconds = {name: 0.0 for name in self.stage_names}

    def milliseconds(self):
        """Return {stage: ms} for the track that just ran."""
        return {name: 1000 * self.seconds.get(name, 0.0) for name in self.stage_names}


def peak_memory_mb():
    """Return the largest amount of memory this process has used so far, in MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS but in kilobytes on Linux
    peak_bytes = peak if sys.platform == 'darwin' else peak * 1024
    return peak_bytes / BYTES_PER_MB
