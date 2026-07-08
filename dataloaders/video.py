# Ported from Wavy-Hec/CVBench bench/methods/stitch.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Frame-index sampling shared across the harness modules.

``sample_frame_indices`` is relocated here from ``bench/methods/stitch.py``
(docs/PORTING.md §1: frame helpers used by more than one module move to
``dataloaders.video``); ``harnesses.stitched`` imports it from here and the
montage compositors are otherwise unchanged.
"""
from __future__ import annotations

from typing import List


def sample_frame_indices(n_total: int, nframes: int) -> List[int]:
    """``linspace(0, n_total-1, nframes)`` rounded to ints (mirrors the per-clip
    nframes sampling used elsewhere in the harness)."""
    if n_total <= 0:
        return [0] * nframes
    if nframes <= 1:
        return [0]
    step = (n_total - 1) / (nframes - 1)
    return [int(round(i * step)) for i in range(nframes)]
