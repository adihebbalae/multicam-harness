# Ported from Wavy-Hec/CVBench bench/methods/base.py @ f65d6e043014b6e9090c32dec4893ebc14fa4320
"""Core abstractions for the multi-camera benchmark.

A ``Method`` is an *architecture* (how the camera streams are fed to a model);
a ``Backend`` is the underlying VLM (``Backend`` and ``GenOut`` live in
``models.clients``). ``Method.answer(rec, video_root)`` returns a ``Result``
carrying the prediction plus the latency / token / calibration metrics
(M1-M4).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

from inprocess.dataloaders.qa_json import num_images, num_videos
from inprocess.models.clients import Backend


@dataclass
class Result:
    """One (question x method x backend) row written to the per-question JSONL."""
    id: object
    task_type: str
    source: Optional[str]
    orig_num_cameras: Optional[int]
    cap_answer_safe: Optional[bool]
    num_videos: int
    method: str
    backend: str
    prediction: str
    gold: str
    correct: bool
    abstained: bool
    pair_idx: Optional[str] = None   # paired-question twin id (All-Angles-Bench IC metric)
    # 4-pass protocol (Table 1 std): a "pass" = one sampled generation (temp>0)
    # at a fixed seed, with frames held fixed; std is taken over the passes.
    pass_idx: Optional[int] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None
    # M2 latency
    latency_s: Optional[float] = None                  # end-to-end (serial) wall-clock
    perception_latency_par_s: Optional[float] = None   # A1: max over per-stream calls (true-parallel estimate)
    perception_latency_serial_s: Optional[float] = None  # A1: sum over per-stream calls
    aggregate_latency_s: Optional[float] = None        # A1: aggregator call
    # M3 cost
    input_tokens: Optional[int] = None
    video_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    num_model_calls: int = 1
    # reasoning/audit (set by every method on success; None on legacy rows/errors)
    response_text: Optional[str] = None    # raw g.text (full <think>...<answer> trace)
    think: Optional[str] = None            # extract_think(g.text)
    frame_alloc: Optional[dict] = None     # frame-audit methods: per-clip durations +
                                           # allocated frame counts; per_stream: the
                                           # per-view perception_texts
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def require_video_record(rec, method_name):
    """Raise unless ``rec`` carries video clips, which the clip/frame arms need.

    Those arms reach for ``video_paths()``, which looks only at ``video_i``. A
    still-image record (``image_i``, e.g. All-Angles) therefore yields an empty
    path list and the arm would run with ZERO visual items — scoring a blind run
    while being labelled a vision one. Fail loudly instead.
    """
    if num_videos(rec) == 0 and num_images(rec) > 0:
        raise ValueError(
            f"{method_name} needs video records but question id={rec.get('id')} "
            f"({rec.get('task_type')}) carries {num_images(rec)} still image(s) and no video. "
            "The clip/frame-selection arms sample frames out of clips, so they cannot run on "
            "still-image benchmarks. Use cvbench_native, centralized or per_stream instead.")


def result_fields(rec):
    """Stratification keys copied verbatim from the question record."""
    return dict(
        id=rec.get("id"),
        task_type=rec.get("task_type"),
        source=rec.get("source"),
        orig_num_cameras=rec.get("orig_num_cameras"),
        cap_answer_safe=rec.get("cap_answer_safe"),
        num_videos=num_videos(rec) or num_images(rec),
        pair_idx=rec.get("pair_idx") or None,
    )


class Method:
    name = "method"

    def __init__(self, backend: Backend, nframes: int = 8, max_new_tokens: int = 8192,
                 temperature: float = 0.0):
        self.backend = backend
        self.nframes = nframes
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def answer(self, rec, video_root, seed=None) -> Result:
        raise NotImplementedError
