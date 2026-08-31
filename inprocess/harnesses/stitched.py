# Ported from Wavy-Hec/CVBench bench/methods/stitch.py @ 017f416ffd7fa8536c2787cee7d3b67665d4e048
# Ported from Wavy-Hec/CVBench bench/methods/centralized.py @ 7f3e480aa4fc4ce615a4735a593594f8b6e71a93
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
import os
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from decord import VideoReader, cpu

from inprocess.harnesses.base import Method, Result, result_fields
from inprocess.dataloaders.qa_json import (build_messages, image_paths, letters_of,
                                 num_images, video_paths)
from inprocess.dataloaders.video import sample_frame_indices
from inprocess.evaluation.scoring import extract_think, gt_choice, parse_choice


def decode_aligned_frames(video_paths: List[str], nframes: int) -> List[List[Optional[Image.Image]]]:
    """Per camera, decode ``nframes`` frames at proportional positions.

    Returns ``frames[k][t]`` (PIL.Image), or ``None`` for a frame whose clip
    failed to decode (compose_montage fills those cells black).

    A clip that decodes PARTIALLY still black-cells only its failed timesteps —
    decord is flaky per index, and the surviving frames of that clip carry the
    view. But a clip that cannot be read at ALL raises, because a montage built
    from mostly-black cells is still answerable: swallowing it makes the
    centralized harness the only one that answers (the other two raise on the
    same file), off a black canvas, with error=None and a scored prediction.
    """
    per_cam: List[List[Optional[Image.Image]]] = []
    unreadable: List[str] = []
    for vp in video_paths:
        try:
            vr = VideoReader(vp, ctx=cpu(0), num_threads=1)
            n = len(vr)
            idx = sample_frame_indices(n, nframes)
        except Exception as e:
            if not os.path.exists(vp):
                unreadable.append(f"{vp} (missing)")
            else:
                unreadable.append(f"{vp} ({type(e).__name__})")
            per_cam.append([None] * nframes)  # cannot open -> all black cells
            continue
        # per-frame, not one comprehension: an all-or-nothing read let a single
        # bad index blacken the whole camera
        frames: List[Optional[Image.Image]] = []
        for i in idx:
            try:
                frames.append(Image.fromarray(vr[i].asnumpy()).convert("RGB"))
            except Exception:
                frames.append(None)
        if all(f is None for f in frames):
            unreadable.append(f"{vp} (no decodable frames)")
        per_cam.append(frames)
    if unreadable:
        raise FileNotFoundError(
            f"{len(unreadable)} of {len(video_paths)} clips could not be read — "
            f"check --video-root. First: {unreadable[0]}")
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


def build_image_montage(image_paths: List[str], cell_px: int = 448,
                        label_prefix: str = "View") -> List[Image.Image]:
    """Still-image variant: tile the K view images of one question into a single
    labeled grid montage (no temporal axis — T=1 by construction).

    Unlike the video path, a view that fails to open RAISES. Black-celling a
    missing frame is right for video, where decord flakiness is expected and the
    other frames of that clip still carry the view. On stills the view has
    exactly one frame, so a swallowed failure hands the model an all-black
    montage and the row is still written with error=null and a scored
    prediction — while the other two arms raise on the same missing file. A
    wrong --video-root would then silently produce a plausible
    centralized-only number."""
    frames: List[Optional[Image.Image]] = []
    failed: List[str] = []
    for ip in image_paths:
        try:
            frames.append(Image.open(ip).convert("RGB"))
        except Exception as e:
            failed.append(f"{ip} ({type(e).__name__})")
            frames.append(None)
    if failed:
        raise FileNotFoundError(
            f"{len(failed)} of {len(image_paths)} view images failed to open — "
            f"check --video-root. First: {failed[0]}")
    labels = [f"{label_prefix} {i + 1}" for i in range(len(image_paths))]
    return [compose_montage(frames, labels, cell_w=cell_px, cell_h=cell_px)]


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
# "video" — INDEPENDENT clips (corrected preamble: matches the
# 'Video i' labels used in the question, and does not falsely call them synchronized).
MONTAGE_PREFIX_VIDEO = (
    "The following {T} image(s) are grid montages built from {k} independent video "
    "clips (different, unrelated scenes), shown in chronological order. Each montage "
    "tiles the {k} clips into a grid; every cell is labeled 'Video i' (top-left), "
    "corresponding to Video 1..Video {k} in the question. Reason about each Video "
    "separately as well as together, and over time, to answer.")
# "view" — still-image multi-view (All-Angles-Bench style): one montage of the K
# simultaneous view images; labels match the question text's "View 1..View k".
MONTAGE_PREFIX_VIEW = (
    "The following image is a grid montage of {k} camera views of the same scene, "
    "captured at the same moment. Every cell is labeled 'View i' (top-left), "
    "corresponding to View 1..View {k} in the question. Reason across the views "
    "to answer.")
MONTAGE_PREFIXES = {"camera": MONTAGE_PREFIX_CAMERA, "video": MONTAGE_PREFIX_VIDEO,
                    "view": MONTAGE_PREFIX_VIEW}
MONTAGE_LABELS = {"camera": "Camera", "video": "Video", "view": "View"}
MONTAGE_PREFIX = MONTAGE_PREFIX_CAMERA  # backward-compat alias


class CentralizedMethod(Method):
    name = "centralized"

    def __init__(self, backend, nframes=8, max_new_tokens=8192, temperature=0.0,
                 montage_frames=0, cell_px=448, montage_kind="camera",
                 total_frames=0, reasoning=True):
        super().__init__(backend, nframes=nframes, max_new_tokens=max_new_tokens,
                         temperature=temperature, reasoning=reasoning)
        self.T = montage_frames if montage_frames and montage_frames > 0 else nframes
        # total_frames > 0: hold the TOTAL source-frame count (T montages x K
        # cells) fixed per question by setting T = round(total/K) per record
        # (the mentor's fixed-budget protocol)
        self.total_frames = total_frames
        self.cell_px = cell_px
        self.montage_kind = montage_kind  # "camera" (synced views) | "video" (independent clips)
        self._prefix = MONTAGE_PREFIXES[montage_kind]
        self._label = MONTAGE_LABELS[montage_kind]
        self._cache = {}  # rec id -> (montages, scaffold_text, yn, gold); last rec only

    def _prepare(self, rec, video_root):
        key = rec.get("id")
        if key in self._cache:
            return self._cache[key]
        base_msgs, yn = build_messages(rec, video_root, self.nframes, no_video=True,
                                       reasoning=self.reasoning)
        scaffold = base_msgs[0]["content"][0]["text"]
        if num_images(rec) > 0:
            # still-image record: one montage of the view images; labels/preamble
            # are forced to "View" to match the question text regardless of
            # --montage-kind (using "Camera i" here is a known labeling artifact)
            paths = image_paths(rec, video_root)
            montages = build_image_montage(paths, cell_px=self.cell_px, label_prefix="View")
            prefix = MONTAGE_PREFIX_VIEW
            # cell_px and the canvas decide the whole visual budget on stills, and
            # neither is otherwise recoverable from a row — only echoed to the
            # sbatch log. Without them a leg run at a different cell size is
            # indistinguishable from one that was not.
            alloc = {"kind": "image_montage", "K": len(paths),
                     # the EFFECTIVE prompt/label variant (stills force "view");
                     # without it, rows of different montage_kind legs differ
                     # only by file path and a glob aggregation blends them
                     "montage_kind": "view",
                     "cell_px": self.cell_px,
                     "canvas_wh": list(montages[0].size),
                     # InternVL re-tiles the canvas by aspect ratio, so max_tiles
                     # — not cell_px — is what sets the montage token budget on
                     # that backend. Record it or the leg is unauditable.
                     "max_tiles": getattr(self.backend, "max_tiles", None)}
        else:
            paths = video_paths(rec, video_root)
            t = self.T
            if self.total_frames:
                # One montage is one timestep across ALL K cameras, so this
                # harness can only deliver t*K frames for integer t — it cannot
                # match an arbitrary budget unless K divides it. round() went
                # OVER the budget for some K and under for others, so an
                # equal-budget comparison was biased in a direction that flipped
                # with camera count. Floor instead: never exceed, so the montage
                # harness is never the flattered one, and record what was
                # actually delivered so comparisons can use that rather than
                # the nominal.
                # A budget below K cannot be honoured at all (one frame per
                # camera already exceeds it); clamping to t=1 silently delivered
                # K frames, the exact overshoot the floor exists to prevent.
                if self.total_frames < len(paths):
                    raise ValueError(
                        f"total_frames={self.total_frames} < {len(paths)} views: "
                        "the montage arm cannot hold this budget")
                t = self.total_frames // len(paths)
            montages = build_montages(paths, nframes=max(self.nframes, t), T=t,
                                      cell_px=self.cell_px, label_prefix=self._label)
            prefix = self._prefix
            alloc = {"kind": "montage", "T": t, "K": len(paths),
                     # prompt/label variant — see the still-image branch note
                     "montage_kind": self.montage_kind,
                     "frames_total": t * len(paths),
                     "total_frames": self.total_frames or None,
                     # same reason as the still-image branch above: cell_px and
                     # max_tiles set the whole visual budget and are otherwise
                     # recoverable only from the sbatch log, which leaves every
                     # video montage leg unauditable for tiling from its rows
                     "cell_px": self.cell_px,
                     "canvas_wh": list(montages[0].size) if montages else None,
                     "max_tiles": getattr(self.backend, "max_tiles", None)}
        gold = gt_choice(rec["answer"], yn, letters=letters_of(rec))
        self._cache = {key: (montages, scaffold, yn, gold, len(paths), prefix, alloc)}  # last rec only
        return self._cache[key]

    def answer(self, rec, video_root, seed=None) -> Result:
        f = result_fields(rec)
        letters = letters_of(rec)
        try:
            montages, scaffold, yn, gold, k, prefix, alloc = self._prepare(rec, video_root)
        except Exception as e:
            gold = gt_choice(rec["answer"], all(o.strip().strip(".").lower() in ("yes", "no")
                                                for o in rec["options"]), letters=letters)
            return Result(**f, method=self.name, backend=self.backend.name,
                          prediction="", gold=gold, correct=False, abstained=True,
                          pass_idx=None, seed=seed, temperature=self.temperature,
                          num_model_calls=1, error=f"stitch:{type(e).__name__}: {e}")
        content = [{"type": "text", "text": prefix.format(T=len(montages), k=k)}]
        content += [{"type": "image", "image": m} for m in montages]
        content += [{"type": "text", "text": scaffold}]
        messages = [{"role": "user", "content": content}]
        try:
            g = self.backend.generate(messages, max_new_tokens=self.max_new_tokens,
                                      seed=seed, temperature=self.temperature)
            pred = parse_choice(g.text, yn, letters=letters, options=rec.get('options'))
            return Result(
                **f, method=self.name, backend=self.backend.name,
                prediction=pred, gold=gold,
                correct=(pred.strip().upper() == gold.strip().upper()),
                abstained=(pred == ""),
                seed=seed, temperature=self.temperature,
                latency_s=g.latency_s,
                input_tokens=g.input_tokens, video_tokens=g.video_tokens,
                output_tokens=g.output_tokens, num_model_calls=1,
                response_text=g.text, think=extract_think(g.text),
                frame_alloc=alloc,
            )
        except Exception as e:  # keep the sweep alive; record the failure
            return Result(**f, method=self.name, backend=self.backend.name,
                          prediction="", gold=gold, correct=False, abstained=True,
                          seed=seed, temperature=self.temperature,
                          num_model_calls=1, error=f"{type(e).__name__}: {e}")
