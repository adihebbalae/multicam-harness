# Ported from Wavy-Hec/CVBench bench/methods/stitch.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
# Ported from Wavy-Hec/CVBench bench/methods/centralized.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Spatial-stitching for the CENTRALIZED harness.

The spec's centralized method "temporally aligns the video streams and
spatially stitches the corresponding images across multiple views to provide a
unified input." This module turns the K (<=4) camera clips of one question into
``T`` grid-montage images: for each of T aligned timesteps, the synchronized
frame from every camera is tiled into one labeled grid image.

Pure decord + PIL, no model. Frames are sampled at the SAME normalized positions
within each clip (proportional alignment), which degrades gracefully when clips
differ slightly in length/fps (MEVA: same 30fps but sub-second start offsets;
EgoExo4D: frame-aligned). Output is a list of ``PIL.Image`` consumed unchanged by
the Qwen backend and via ``load_image`` by the InternVL backend.

CENTRALIZED harness (spec-faithful): one model ingests a SINGLE unified input
built by temporally aligning the camera streams and spatially STITCHING the
synchronized frames into grid-montage images (see ``build_montages``).

The text scaffold (question/options/<think>/<answer>) is taken verbatim from the
existing harness (``build_messages(..., no_video=True)``) so only the visual
presentation differs from the blind/per-stream paths. The montages for a question
are built once and cached, so the 4 sampling passes reuse identical pixels.
"""
from __future__ import annotations

import math
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from decord import VideoReader, cpu

from harnesses.base import Method, Result, result_fields
from dataloaders.qa_json import build_messages, video_paths
from dataloaders.video import sample_frame_indices
from evaluation.scoring import parse_choice, gt_choice


def decode_aligned_frames(video_paths: List[str], nframes: int) -> List[List[Optional[Image.Image]]]:
    """Per camera, decode ``nframes`` frames at proportional positions.

    Returns ``frames[k][t]`` (PIL.Image), or ``None`` for a frame whose clip
    failed to decode (compose_montage fills those cells black).
    """
    per_cam: List[List[Optional[Image.Image]]] = []
    for vp in video_paths:
        try:
            vr = VideoReader(vp, ctx=cpu(0), num_threads=1)
            n = len(vr)
            idx = sample_frame_indices(n, nframes)
            frames = [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in idx]
        except Exception:
            frames = [None] * nframes  # decode failure -> black cells
        per_cam.append(frames)
    return per_cam


def grid_layout(k: int) -> tuple[int, int]:
    """(rows, cols) for K camera cells. cols = ceil(sqrt(k)); K<=4 -> at most 2x2."""
    k = max(1, k)
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k / cols)
    return rows, cols


def _label_font():
    try:
        return ImageFont.load_default()
    except Exception:  # extremely defensive; load_default is bundled with PIL
        return None


def compose_montage(frames: List[Optional[Image.Image]], labels: List[str],
                    cell_w: int = 448, cell_h: int = 448,
                    pad_color=(0, 0, 0), label_band: int = 22) -> Image.Image:
    """Tile one timestep's per-camera frames into a single labeled grid image."""
    k = len(frames)
    rows, cols = grid_layout(k)
    font = _label_font()
    cell_total_h = cell_h + label_band
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_total_h), pad_color)
    draw = ImageDraw.Draw(canvas)
    for i in range(k):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, r * cell_total_h
        # label band
        draw.rectangle([x0, y0, x0 + cell_w, y0 + label_band], fill=(30, 30, 30))
        if font is not None:
            draw.text((x0 + 4, y0 + 4), labels[i], fill=(255, 255, 255), font=font)
        # frame (black if missing)
        frame = frames[i]
        if frame is None:
            cell = Image.new("RGB", (cell_w, cell_h), pad_color)
        else:
            cell = frame.resize((cell_w, cell_h))
        canvas.paste(cell, (x0, y0 + label_band))
    return canvas


def build_montages(video_paths: List[str], nframes: int = 8, T: Optional[int] = None,
                   cell_px: int = 448, label_prefix: str = "Camera") -> List[Image.Image]:
    """Decode the K clips and compose ``T`` grid montages (one per aligned timestep).

    ``T`` defaults to ``nframes`` (each sampled timestep gets a montage); pass
    ``T=1`` for the strict "single unified image" reading. ``label_prefix`` sets the
    per-cell caption ("Camera" for synced views, "Video" for independent clips).
    """
    T = nframes if (T is None or T <= 0) else T
    per_cam = decode_aligned_frames(video_paths, nframes)  # [K][nframes]
    k = len(video_paths)
    labels = [f"{label_prefix} {i + 1}" for i in range(k)]
    # pick T timestep indices among the nframes decoded positions
    t_idx = sample_frame_indices(nframes, T)
    montages = []
    for t in t_idx:
        frames_t = [per_cam[c][t] if t < len(per_cam[c]) else None for c in range(k)]
        montages.append(compose_montage(frames_t, labels, cell_w=cell_px, cell_h=cell_px))
    return montages


# "camera" — MEVA-style synchronized multi-view (default, unchanged).
MONTAGE_PREFIX_CAMERA = (
    "The following {T} image(s) are time-synchronized grid montages of {k} camera "
    "view(s), shown in chronological order. Each montage tiles the cameras into a "
    "grid; every cell is labeled 'Camera i' (top-left). Reason across the views and "
    "over time to answer.")
# "video" — CVBench-style INDEPENDENT clips (corrected preamble: matches the
# 'Video i' labels used in the question, and does not falsely call them synchronized).
MONTAGE_PREFIX_VIDEO = (
    "The following {T} image(s) are grid montages built from {k} independent video "
    "clips (different, unrelated scenes), shown in chronological order. Each montage "
    "tiles the {k} clips into a grid; every cell is labeled 'Video i' (top-left), "
    "corresponding to Video 1..Video {k} in the question. Reason about each Video "
    "separately as well as together, and over time, to answer.")
MONTAGE_PREFIXES = {"camera": MONTAGE_PREFIX_CAMERA, "video": MONTAGE_PREFIX_VIDEO}
MONTAGE_LABELS = {"camera": "Camera", "video": "Video"}
MONTAGE_PREFIX = MONTAGE_PREFIX_CAMERA  # backward-compat alias


class CentralizedMethod(Method):
    name = "centralized"

    def __init__(self, backend, nframes=8, max_new_tokens=8192, temperature=0.0,
                 montage_frames=0, cell_px=448, montage_kind="camera"):
        super().__init__(backend, nframes=nframes, max_new_tokens=max_new_tokens,
                         temperature=temperature)
        self.T = montage_frames if montage_frames and montage_frames > 0 else nframes
        self.cell_px = cell_px
        self.montage_kind = montage_kind  # "camera" (synced views) | "video" (independent clips)
        self._prefix = MONTAGE_PREFIXES[montage_kind]
        self._label = MONTAGE_LABELS[montage_kind]
        self._cache = {}  # rec id -> (montages, scaffold_text, yn, gold); last rec only

    def _prepare(self, rec, video_root):
        key = rec.get("id")
        if key in self._cache:
            return self._cache[key]
        base_msgs, yn = build_messages(rec, video_root, self.nframes, no_video=True)
        scaffold = base_msgs[0]["content"][0]["text"]
        paths = video_paths(rec, video_root)
        montages = build_montages(paths, nframes=self.nframes, T=self.T, cell_px=self.cell_px,
                                  label_prefix=self._label)
        gold = gt_choice(rec["answer"], yn)
        self._cache = {key: (montages, scaffold, yn, gold, len(paths))}  # keep only last rec
        return self._cache[key]

    def answer(self, rec, video_root, seed=None) -> Result:
        f = result_fields(rec)
        try:
            montages, scaffold, yn, gold, k = self._prepare(rec, video_root)
        except Exception as e:
            gold = gt_choice(rec["answer"], all(o.strip().strip(".").lower() in ("yes", "no")
                                                for o in rec["options"]))
            return Result(**f, method=self.name, backend=self.backend.name,
                          prediction="", gold=gold, correct=False, abstained=True,
                          pass_idx=None, seed=seed, temperature=self.temperature,
                          num_model_calls=1, error=f"stitch:{type(e).__name__}: {e}")
        content = [{"type": "text", "text": self._prefix.format(T=len(montages), k=k)}]
        content += [{"type": "image", "image": m} for m in montages]
        content += [{"type": "text", "text": scaffold}]
        messages = [{"role": "user", "content": content}]
        try:
            g = self.backend.generate(messages, max_new_tokens=self.max_new_tokens,
                                      seed=seed, temperature=self.temperature)
            pred = parse_choice(g.text, yn)
            return Result(
                **f, method=self.name, backend=self.backend.name,
                prediction=pred, gold=gold,
                correct=(pred.strip().upper() == gold.strip().upper()),
                abstained=(pred == ""),
                seed=seed, temperature=self.temperature,
                latency_s=g.latency_s,
                input_tokens=g.input_tokens, video_tokens=g.video_tokens,
                output_tokens=g.output_tokens, num_model_calls=1,
            )
        except Exception as e:  # keep the sweep alive; record the failure
            return Result(**f, method=self.name, backend=self.backend.name,
                          prediction="", gold=gold, correct=False, abstained=True,
                          seed=seed, temperature=self.temperature,
                          num_model_calls=1, error=f"{type(e).__name__}: {e}")
