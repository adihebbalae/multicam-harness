#!/usr/bin/env python3
"""Camera-wall MP4s for the four-camera MEVA demo: one clip per demo question.

For every question in ``analysis/figs_demo/manifest.json`` this renders a 2x2
wall of the question's four cameras, played on a shared clip clock across the
window that contains the release's annotated event moments, and writes it into
``analysis/figs_demo/walls/`` (gitignored):

  q<id>_wall.mp4           the wall itself: the question on top, the four
                           camera streams in slots 1-4, each with its scorer's
                           segment strip, its kept segments, the annotated
                           event windows and a playhead; a camera is dimmed
                           whenever the clock is outside the segments the
                           selector actually sent to the model, and the clip
                           ends on an answer card with every arm's letter.
  q<id>_wall_snapshot.png  the composed frame at the first event's start, for
                           the slides and for whoever verifies this.
  walls_all.mp4            every wall in the output directory concatenated in
                           subset order, rebuilt whenever at least two are
                           there, so a one-question re-render cannot truncate
                           it (walls_manifest.json lists the ids it carries).
  walls_manifest.json      per question: window, speed, duration, path, mode.

Nothing here is recomputed: every number drawn (segment scores, kept segments,
selected frame counts, event windows, predictions, latency) is read from
``analysis/figs_demo/manifest.json``, which ``analysis/render_demo.py`` wrote
from the demo result rows. This script only decodes video and draws.

Read-only with respect to the harness: it never loads a model, never submits
anything, never touches ``bench/``. Run from the repo root under ``cvbench``
(PIL + numpy + matplotlib + decord + ffmpeg):

  python3 analysis/make_demo_walls.py
  python3 analysis/make_demo_walls.py --ids 749 --mode global
  python3 analysis/make_demo_walls.py --target-seconds 20 --max-speed 3

MEDIA. MEVA records spell their clips ``.avi`` and an ``.avi`` must never reach
a decoder (decord returns the wrong frame on random access into those
containers - every seek lands on keyframe 0). The manifest already stores the
remuxed ``.mp4`` sibling for each camera; this script refuses anything else.
The release also mixes frame sizes - 13 of the 20 clips behind this manifest
are 1920x1080 and 7 are 1920x1072 - so the four tiles of a wall take one
shared fit (wall_geometry) rather than a fit per camera.

THE CLOCK is clip time on each camera's own clip, identical across the four
tiles. It is not wall-clock time: the MEVA release starts the clips of one slot
within a few seconds of each other; the footer on every frame carries that
question's own measured spread, read off the clip filenames.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

try:  # matplotlib >= 3.9 deprecates cm.get_cmap
    VIRIDIS = matplotlib.colormaps["viridis"]
except Exception:  # pragma: no cover - older matplotlib
    from matplotlib import cm as _cm
    VIRIDIS = _cm.get_cmap("viridis")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MANIFEST = "analysis/figs_demo/manifest.json"
DEFAULT_OUT = "analysis/figs_demo/walls"

# Subset order (analysis/demo_meva4cam_subset.json): the order walls_all.mp4
# concatenates in, and the order the narration follows.
SUBSET_ORDER = [860, 749, 251, 1, 130]

# ---------------------------------------------------------------------------
# canvas geometry
# ---------------------------------------------------------------------------
W, H = 1920, 1080
TOP_H = 120            # title + question, two lines at 30 px
BOT_H = 56             # the standing selection/clock caption, two lines
TILE_W = W // 2
TILE_H = (H - TOP_H - BOT_H) // 2      # 452
GRID_TOP = TOP_H
GRID_BOT = TOP_H + 2 * TILE_H

PAD_BEFORE = 3.0       # seconds of lead-in before the first annotated moment
PAD_AFTER = 3.0        # and of lead-out after the last one
DECODE_CHUNK = 60      # output frames per decord get_batch, to keep memory flat
MAX_CROP_FRAC = 0.12   # of source height; see tile_geometry()
DIM = 0.35             # brightness of a camera the selector did not send
MIN_TAIL_FRAC = 0.08   # of a caption line: a shorter last line is an orphan

# Okabe-Ito accents, same as analysis/render_demo.py so the video and the
# figures read as one artifact.
C_EVENT = (213, 94, 0)        # #D55E00 evidence badge, event bars, event border
C_GLOBAL = (0, 114, 178)      # #0072B2 the other leg's kept segments
C_OK = (0, 158, 115)
C_BAD = (204, 51, 17)
C_BG = (12, 14, 16)
C_BAND = (22, 25, 29)
C_INK = (242, 242, 240)
C_MUTE = (168, 172, 178)
C_GREY_BADGE = (88, 92, 98)
C_KEPT = (255, 255, 255)

FS_TITLE = 28
FS_Q = 30              # the question never goes below 30 px
FS_CLOCK = 26
FS_CAM = 23            # tile labels never go below 22 px
FS_META = 22
FS_BADGE = 22
FS_STATUS = 23
FS_CAP = 22
FS_FOOT = 21
FS_CARD_H = 44
FS_CARD = 28
FS_CARD_SM = 24

def scorer_words(leg):
    """Name the scorer and its query from the manifest's selection_leg, never
    from a flag: SigLIP legs score against the question text, the *_opt ViCLIP
    legs against the four answer options (max over options)."""
    scorer = str((leg or {}).get("scorer", "")).lower()
    method = str((leg or {}).get("method", "")).lower()
    if "siglip" in scorer:
        name = "SigLIP so400m"
    elif "viclip" in scorer:
        name = "ViCLIP"
    else:
        name = scorer or "the scorer"
    if method.endswith("_opt"):
        query = "against the four answer options (max over options)"
    elif method.endswith("_stmt"):
        query = "against the question's statements"
    else:
        query = "against the question"
    return name, query


def clip_start_spread(q):
    """Seconds between the earliest and the latest clip start on this question.

    MEVA spells the start in the clip filename's second field
    (``2018-03-11.14-10-00.14-15-00...`` -> ``14-10-00``), so the spread is read
    off the four camera paths rather than asserted as a constant. None when a
    path does not carry a parsable start."""
    secs = []
    for c in (q or {}).get("cameras", []):
        parts = os.path.basename(str(c.get("path", ""))).split(".")
        if len(parts) < 2:
            return None
        m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{2})", parts[1])
        if not m:
            return None
        h, mi, sec = (int(x) for x in m.groups())
        secs.append(h * 3600 + mi * 60 + sec)
    if len(secs) < 2:
        return None
    return max(secs) - min(secs)


def footer_text(mode, leg, spread=None):
    name, query = scorer_words(leg)
    head = f"Selection: {name} scores 8 segments per camera {query}; "
    if mode == "per_clip":
        # 4 of 8 segments x 8 frames = 128 frames, thinned down to the budget.
        body = ("top 4 segments per camera kept (128 frames), thinned to the "
                "96-frame budget, 24 per camera. ")
    else:
        # --seg-select global spends one budget across the four cameras (12 of
        # the 32 segments, one per camera as a floor), so the per-clip wording
        # would be wrong.
        body = ("the best 12 segments across all four cameras kept, one per "
                "camera as a floor (96 frames), 8 per segment. ")
    tail = ("Cameras share a clip clock; clip starts on this question differ "
            + (f"by up to {spread:.0f} s." if spread is not None
               else "by an unknown amount (clip start not parsable)."))
    return head + body + tail


MARK_OK = "✓"
MARK_BAD = "✗"
MID = "·"

def scorer_arms(leg):
    """Manifest arm keys of the scorer that drove this manifest's selection, read
    from selection_leg.scorer (never from a flag): the closing card must list and
    highlight the ViCLIP arms on a ViCLIP wall, the SigLIP arms on a SigLIP wall."""
    sc = str((leg or {}).get("scorer", "")).lower()
    if "viclip" in sc:
        return {"per_clip": "viclip_perclip", "global": "viclip_global"}
    return {"per_clip": "seg_perclip", "global": "seg_global"}


def arm_order(leg):
    m = scorer_arms(leg)
    return ["blind", "native", m["per_clip"], m["global"]]

NOTES: list[str] = []


def note(msg: str) -> None:
    """Something the viewer should not have to guess at: printed and recorded."""
    print(f"[note] {msg}")
    NOTES.append(msg)


def rel(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO, path)


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------
_FONT_CACHE: dict = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _FONT_CACHE:
        prop = fm.FontProperties(family="DejaVu Sans",
                                 weight="bold" if bold else "normal")
        _FONT_CACHE[key] = ImageFont.truetype(fm.findfont(prop), size)
    return _FONT_CACHE[key]


_MEASURE = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def tw(text: str, fnt) -> float:
    return _MEASURE.textlength(text, font=fnt)


def elide(text: str, fnt, max_w: float) -> str:
    if tw(text, fnt) <= max_w:
        return text
    out = text
    while out and tw(out + "…", fnt) > max_w:
        out = out[:-1]
    return out.rstrip() + "…"


def unorphan(lines: list, fnt, max_w: float) -> list:
    """Pull one word down when the wrap leaves a stub on the last line.

    Greedy wrapping happily breaks '17.7-19.1 s' into a full line and a lone
    's'. The caption's whole job is to carry those two numbers with their
    unit, so a last line under MIN_TAIL_FRAC of the width (or two characters)
    takes the previous line's last word with it.
    """
    if len(lines) < 2:
        return lines
    last = lines[-1]
    if len(last) > 2 and tw(last, fnt) >= MIN_TAIL_FRAC * max_w:
        return lines
    head = lines[-2].split()
    if len(head) < 2:
        return lines
    moved = f"{head[-1]} {last}"
    if tw(moved, fnt) > max_w:
        return lines
    return lines[:-2] + [" ".join(head[:-1]), moved]


def fit_caption(prefix: str, desc: str, suffix: str, fnt, line_w: float,
                max_lines: int) -> list:
    """Wrap '<label> . <description> . <start>-<end> s' into max_lines.

    The description is what gets shortened, never the label or the times: a
    caption that loses its timestamps is no use to anyone reading the strip.
    The chosen wrap then goes through unorphan(), so the unit never ends up
    alone on the last line.
    """
    def fits(txt):
        lines = wrap(txt, fnt, line_w, max_lines)
        return " ".join(lines) == " ".join(txt.split()), lines

    ok, lines = fits(prefix + desc + suffix)
    if ok:
        return unorphan(lines, fnt, line_w)
    lo, hi, best = 0, len(desc), None
    while lo <= hi:                       # longest description that still fits
        mid = (lo + hi) // 2
        ok, lines = fits(prefix + desc[:mid].rstrip() + "\u2026" + suffix)
        if ok:
            best, lo = lines, mid + 1
        else:
            hi = mid - 1
    if best is None:
        best = wrap(prefix + suffix, fnt, line_w, max_lines)
    return unorphan(best, fnt, line_w)


def wrap(text: str, fnt, max_w: float, max_lines: int) -> list:
    """Greedy word wrap; the last line is elided when the text does not fit."""
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if cur and tw(trial, fnt) > max_w:
            lines.append(cur)
            cur = word
            if len(lines) == max_lines:
                break
        else:
            cur = trial
    if len(lines) < max_lines:
        if cur:
            lines.append(cur)
        return lines
    rest = " ".join(words[sum(len(l.split()) for l in lines):])
    if rest:
        lines[-1] = elide(f"{lines[-1]} {rest}", fnt, max_w)
    return lines


# ---------------------------------------------------------------------------
# media
# ---------------------------------------------------------------------------
def open_reader(path: str):
    from decord import VideoReader, cpu  # late import: heavy, and only here
    if path.lower().endswith(".avi"):
        raise RuntimeError(
            f"refusing to decode an .avi: {path} - decord returns the wrong "
            f"frame on random access into the MEVA release containers; the "
            f"manifest is supposed to carry the remuxed .mp4 sibling")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing demo media: {path}")
    return VideoReader(path, ctx=cpu(0), num_threads=2)


def tile_geometry(sw: int, sh: int, tw_: int, th: int, max_crop=MAX_CROP_FRAC):
    """How one source frame fits a tile without distortion.

    The demo media is 1920x1080 or 1920x1072 - the MEVA release mixes the two,
    and of the 20 camera clips this manifest points at, 13 are 1080 and 7 are
    1072 - against a 960x452 tile (aspect 2.12), so filling the tile edge to
    edge would need to drop 16.3% of a 1080-tall source and 15.7% of a
    1072-tall one. We CROP AT MOST 12% of the source height (centred) and
    LETTERBOX whatever is still missing: 23 px of pillar on each side of a
    1080-tall source, 20 px on a 1072-tall one. Nothing is stretched, and at
    most an eighth of the frame height is ever hidden from the viewer.

    Those two pillar widths differ, so this per-source fit is NOT what the
    walls use; wall_geometry() below gives the four tiles of one question a
    single size and offset. This function stays the per-source primitive it
    derives from.
    """
    scale_w = tw_ / sw
    need_h = th / scale_w                       # source rows a full-width fit uses
    if need_h >= sh:                            # taller tile than source: letterbox
        s = min(tw_ / sw, th / sh)
        rw, rh = max(1, round(sw * s)), max(1, round(sh * s))
        return {"crop": (0, 0, sw, sh), "size": (rw, rh),
                "offset": ((tw_ - rw) // 2, (th - rh) // 2), "mode": "letterbox"}
    crop_frac = 1.0 - need_h / sh
    if crop_frac <= max_crop + 1e-9:            # a small crop fills the tile
        top = int(round((sh - need_h) / 2))
        keep = int(round(need_h))
        return {"crop": (0, top, sw, top + keep), "size": (tw_, th),
                "offset": (0, 0), "mode": f"crop {crop_frac * 100:.1f}% of height"}
    keep = int(round(sh * (1.0 - max_crop)))    # crop the cap, letterbox the rest
    top = int(round((sh - keep) / 2))
    s = min(tw_ / sw, th / keep)
    rw, rh = max(1, round(sw * s)), max(1, round(keep * s))
    return {"crop": (0, top, sw, top + keep), "size": (rw, rh),
            "offset": ((tw_ - rw) // 2, (th - rh) // 2),
            "mode": f"crop {max_crop * 100:.0f}% of height + letterbox"}


def wall_geometry(dims, tw_: int, th: int, max_crop=MAX_CROP_FRAC) -> list:
    """One size and offset for every tile of a wall, from mixed source sizes.

    Deriving tile_geometry() per camera gives neighbouring tiles pillars of
    different widths whenever a question mixes 1080-tall and 1072-tall clips
    (4 of the 5 demo questions do), and a 3 px step in the inner black gutter
    reads as a crooked wall. So the tiles share the most constrained camera's
    size and offset, and each source is cropped - centred, never stretched -
    to exactly that aspect, still inside the max_crop cap. Returns one dict
    per source in ``dims``; only ``crop`` and ``mode`` differ between them.
    """
    dims = [(int(w), int(h)) for w, h in dims]
    geoms = [tile_geometry(sw, sh, tw_, th, max_crop) for sw, sh in dims]
    rw = min(g["size"][0] for g in geoms)
    rh = min(g["size"][1] for g in geoms)
    for sw, sh in dims:                    # no source may lose more than the cap
        rw = min(rw, int(sw * rh / (sh * (1.0 - max_crop))))
    rw = max(2, rw - rw % 2)               # even: equal pillars, clean chroma
    rh = max(2, rh - rh % 2)
    pill_x, pill_y = (tw_ - rw) // 2, (th - rh) // 2
    out = []
    for sw, sh in dims:
        want_h = sw * rh / rw
        if want_h <= sh:                   # crop height (the usual case)
            keep_w, keep_h = sw, min(sh, int(round(want_h)))
        else:                              # source too narrow: crop width
            keep_w, keep_h = min(sw, int(round(sh * rw / rh))), sh
        left, top = (sw - keep_w) // 2, (sh - keep_h) // 2
        bits = []
        if keep_h < sh:
            bits.append(f"crop {(1 - keep_h / sh) * 100:.1f}% of height")
        if keep_w < sw:
            bits.append(f"crop {(1 - keep_w / sw) * 100:.1f}% of width")
        if pill_x:
            bits.append(f"{pill_x} px pillar")
        if pill_y:
            bits.append(f"{pill_y} px letterbox")
        out.append({"crop": (left, top, left + keep_w, top + keep_h),
                    "size": (rw, rh), "offset": (pill_x, pill_y),
                    "mode": ", ".join(bits) or "exact fit"})
    return out


class CamStream:
    """One camera of one question: its reader, its geometry, its overlays."""

    def __init__(self, cam: dict, mode: str, norm, qid=None):
        self.cam = cam
        self.qid = qid
        self.tag = f"q{qid}" if qid is not None else "q?"
        self.slot = int(cam["slot"])
        self.name = cam["camera"]
        self.path = rel(cam["path"])
        self.fps = float(cam["fps"]) or 30.0
        self.duration = float(cam["duration_s"])
        self.bounds = [(float(a), float(b)) for a, b in cam["segment_bounds_s"]]
        scores = cam.get("segment_scores") or {}
        self.scores = [scores.get(str(i)) for i in range(len(self.bounds))]
        self.evidence = bool(cam.get("evidence"))
        self.windows = cam.get("event_windows") or []
        self.norm = norm
        kept_pc = set(cam.get("segments_kept") or [])
        kept_gl = set(cam.get("segments_kept_global") or [])
        self.kept = kept_pc if mode == "per_clip" else kept_gl
        self.kept_other = kept_gl if mode == "per_clip" else kept_pc
        self.vr = open_reader(self.path)
        self.n_frames = int(cam.get("n_frames_mp4") or len(self.vr))
        if len(self.vr) != self.n_frames:
            note(f"{self.tag} {self.name}: manifest says {self.n_frames} frames, "
                 f"the file decodes {len(self.vr)}; using the smaller one")
            self.n_frames = min(self.n_frames, len(self.vr))
        img_h, img_w = self.vr[0].shape[:2]
        self.src = (int(img_w), int(img_h))
        # A per-source fit, so a stream is usable on its own; render_question
        # overwrites it with the wall's shared geometry.
        self.geom = tile_geometry(img_w, img_h, TILE_W, TILE_H)

    # -- decoding ---------------------------------------------------------
    def index_at(self, t: float) -> int:
        return max(0, min(self.n_frames - 1, int(round(t * self.fps))))

    def tiles_at(self, times) -> list:
        """Fitted tile images for a chunk of clip times (one get_batch)."""
        idx = [self.index_at(t) for t in times]
        try:
            batch = self.vr.get_batch(idx).asnumpy()
        except Exception as exc:  # pragma: no cover - decoder dependent
            note(f"{self.tag} {self.name}: get_batch failed "
                 f"({exc.__class__.__name__}), falling back to single seeks")
            batch = np.stack([self.vr[i].asnumpy() for i in idx])
        g = self.geom
        out = []
        for arr in batch:
            im = Image.fromarray(arr).crop(g["crop"]).resize(g["size"],
                                                             Image.BILINEAR)
            out.append(im)
        return out

    # -- state at a clip time ---------------------------------------------
    def sent_at(self, t: float) -> bool:
        for i in sorted(self.kept):
            if i < len(self.bounds):
                lo, hi = self.bounds[i]
                if lo <= t <= hi:
                    return True
        return False

    def window_at(self, t: float):
        """The annotated window the clock is inside, if any.

        No slop either side: the caption prints this window's own start and
        end, so the orange border has to be on for exactly the frames those
        two printed numbers bracket - the clock is accurate to a frame, so
        widening the test only made the wall highlight a moment whose printed
        start was still in the future.
        """
        for wdw in self.windows:
            if float(wdw["start_sec"]) <= t <= float(wdw["end_sec"]):
                return wdw
        return None


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
DIM_LUT = [int(v * DIM) for v in range(256)] * 3


def dashed_rect(draw, box, color, width=2, dash=9, gap=7):
    x0, y0, x1, y1 = box

    def run(a, b, horizontal, fixed):
        pos = a
        while pos < b:
            end = min(pos + dash, b)
            if horizontal:
                draw.line([(pos, fixed), (end, fixed)], fill=color, width=width)
            else:
                draw.line([(fixed, pos), (fixed, end)], fill=color, width=width)
            pos = end + gap

    run(x0, x1, True, y0)
    run(x0, x1, True, y1)
    run(y0, y1, False, x0)
    run(y0, y1, False, x1)


def badge(draw, x, y, text, fill, ink=(255, 255, 255), fs=FS_BADGE):
    f = font(fs, bold=True)
    pad = 9
    w = tw(text, f) + 2 * pad
    h = fs + 12
    draw.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=fill)
    draw.text((x + pad, y + 5), text, font=f, fill=ink)
    return x + w


def pill(draw, x, y, text, fs, ink, bg=(0, 0, 0, 170)):
    f = font(fs, bold=True)
    pad = 10
    w = tw(text, f) + 2 * pad
    h = fs + 12
    draw.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=bg)
    draw.text((x + pad, y + 5), text, font=f, fill=ink)
    return w, h


def score_color(value, norm):
    if value is None:
        return (70, 70, 70)
    lo, hi = norm
    frac = 0.5 if hi <= lo else (float(value) - lo) / (hi - lo)
    r, g, b, _ = VIRIDIS(max(0.0, min(1.0, frac)))
    return (int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------------
# one tile
# ---------------------------------------------------------------------------
def draw_tile(canvas, draw, cs: CamStream, tile_img, ox, oy, t, mode):
    """Paste one camera's frame at clip time t and draw everything over it."""
    sent = cs.sent_at(t)
    img = tile_img.point(DIM_LUT) if not sent else tile_img
    draw.rectangle([ox, oy, ox + TILE_W - 1, oy + TILE_H - 1], fill=(0, 0, 0))
    canvas.paste(img, (ox + cs.geom["offset"][0], oy + cs.geom["offset"][1]))

    # -- header strip: who this is, how the selector rated it --------------
    hh = 40
    draw.rectangle([ox, oy, ox + TILE_W, oy + hh], fill=(0, 0, 0, 155))
    fc = font(FS_CAM, bold=True)
    draw.text((ox + 12, oy + 8), f"Camera {cs.slot} {MID} {cs.name}",
              font=fc, fill=C_INK)
    bx = ox + 12 + tw(f"Camera {cs.slot} {MID} {cs.name}", fc) + 12
    if cs.evidence:
        badge(draw, bx, oy + 5, "EVIDENCE", C_EVENT)
    else:
        badge(draw, bx, oy + 5, "padding", C_GREY_BADGE, ink=(235, 235, 235))
    n_pc = int(cs.cam.get("n_frames_selected") or 0)
    n_gl = int(cs.cam.get("n_frames_selected_global") or 0)
    if mode == "per_clip":
        frames = f"frames sent {n_pc} (global {n_gl})"
    else:
        frames = f"frames sent {n_gl} (per-clip {n_pc})"
    mx = cs.cam.get("max_score")
    meta = (f"relevance max {float(mx):.3f} {MID} {frames}" if mx is not None
            else frames)
    fm_ = font(FS_META)
    draw.text((ox + TILE_W - 12 - tw(meta, fm_), oy + 9), meta,
              font=fm_, fill=C_INK)

    # -- bottom strip: the eight scored segments, kept marks, events -------
    sh = 70
    sy = oy + TILE_H - sh
    draw.rectangle([ox, sy, ox + TILE_W, oy + TILE_H], fill=(0, 0, 0, 165))
    x0, x1 = ox + 12, ox + TILE_W - 12
    span = max(1e-6, cs.duration)

    def xt(tt):
        return x0 + (max(0.0, min(span, tt)) / span) * (x1 - x0)

    lbl_y = sy + 3
    bar_y0, bar_y1 = sy + 22, sy + 30
    strip_y0, strip_y1 = sy + 33, sy + 61
    for i, (lo, hi) in enumerate(cs.bounds):
        a, b = xt(lo), xt(hi)
        draw.rectangle([a, strip_y0, b, strip_y1],
                       fill=score_color(cs.scores[i], cs.norm))
    for i in range(len(cs.bounds)):
        a, b = xt(cs.bounds[i][0]), xt(cs.bounds[i][1])
        if i in cs.kept_other:
            dashed_rect(draw, (a + 3, strip_y0 + 3, b - 3, strip_y1 - 3),
                        C_GLOBAL, width=2)
        if i in cs.kept:
            draw.rectangle([a, strip_y0, b, strip_y1], outline=C_KEPT, width=3)
    flbl = font(17, bold=True)
    for wdw in cs.windows:
        a, b = xt(float(wdw["start_sec"])), xt(float(wdw["end_sec"]))
        draw.rectangle([a, bar_y0, max(a + 3, b), bar_y1], fill=C_EVENT)
        lab = f"event {wdw.get('label', '')}".strip()
        draw.text((min(a, x1 - tw(lab, flbl)), lbl_y), lab, font=flbl,
                  fill=C_EVENT, stroke_width=2, stroke_fill=(0, 0, 0))
    px = xt(t)
    draw.line([(px, bar_y0 - 2), (px, strip_y1 + 2)], fill=(255, 255, 255),
              width=2)
    draw.polygon([(px - 5, strip_y1 + 2), (px + 5, strip_y1 + 2),
                  (px, strip_y1 + 9)], fill=(255, 255, 255))

    # -- was this camera's clock inside a kept segment? --------------------
    status = "sent to the model" if sent else "not sent to the model"
    ink = C_OK if sent else (235, 235, 235)
    pill(draw, ox + 12, sy - 42, status, FS_STATUS, ink,
         bg=(0, 0, 0, 175) if sent else (40, 40, 40, 205))

    # -- an annotated moment is on screen right now ------------------------
    wdw = cs.window_at(t)
    if wdw is not None:
        draw.rectangle([ox + 3, oy + 3, ox + TILE_W - 4, oy + TILE_H - 4],
                       outline=C_EVENT, width=6)
        fcap = font(FS_CAP, bold=True)
        lines = fit_caption(
            f"{wdw.get('label', '')} {MID} ",
            str(wdw.get("description", "")),
            f" {MID} {float(wdw['start_sec']):.1f}-{float(wdw['end_sec']):.1f} s",
            fcap, TILE_W - 48, 3)
        ch = len(lines) * (FS_CAP + 6) + 10
        cy = sy - 42 - ch - 6
        draw.rectangle([ox + 12, cy, ox + TILE_W - 12, cy + ch],
                       fill=(120, 52, 0, 205))
        for i, line in enumerate(lines):
            draw.text((ox + 22, cy + 5 + i * (FS_CAP + 6)), line,
                      font=fcap, fill=(255, 245, 235))


# ---------------------------------------------------------------------------
# one composed frame
# ---------------------------------------------------------------------------
def compose(q, streams, tiles, t, plan):
    canvas = Image.new("RGB", (W, H), C_BG)
    draw = ImageDraw.Draw(canvas, "RGBA")

    for cs, tile in zip(streams, tiles):
        col = (cs.slot - 1) % 2
        row = (cs.slot - 1) // 2
        draw_tile(canvas, draw, cs, tile, col * TILE_W,
                  GRID_TOP + row * TILE_H, t, plan["mode"])
    draw.line([(TILE_W, GRID_TOP), (TILE_W, GRID_BOT)], fill=C_BG, width=2)
    draw.line([(0, GRID_TOP + TILE_H), (W, GRID_TOP + TILE_H)], fill=C_BG,
              width=2)

    # -- top band ---------------------------------------------------------
    draw.rectangle([0, 0, W, TOP_H], fill=C_BAND)
    ft = font(FS_TITLE, bold=True)
    title = (f"Q{q['id']} {MID} {q.get('question_type', '?')} {MID} "
             f"4 camera streams")
    draw.text((24, 8), title, font=ft, fill=C_INK)
    fclk = font(FS_CLOCK, bold=True)
    clock = f"clip time {t:.1f} s {MID} x{plan['speed']:.1f}"
    # left-aligned at the position the widest clock would take, so the label
    # does not jitter as the clip time gains a digit
    widest = f"clip time {plan['t1']:.1f} s {MID} x{plan['speed']:.1f}"
    draw.text((W - 24 - tw(widest, fclk), 10), clock, font=fclk, fill=C_MUTE)
    fq = font(FS_Q)
    for i, line in enumerate(wrap(q.get("question", ""), fq, W - 48, 2)):
        draw.text((24, 44 + i * 36), line, font=fq, fill=C_INK)
    draw.line([(0, TOP_H - 1), (W, TOP_H - 1)], fill=(60, 64, 70), width=2)

    # -- bottom band ------------------------------------------------------
    draw.rectangle([0, GRID_BOT, W, H], fill=C_BAND)
    ff = font(FS_FOOT)
    footer = plan["footer"]
    for i, line in enumerate(wrap(footer, ff, W - 48, 2)):
        draw.text((24, GRID_BOT + 4 + i * 25), line, font=ff, fill=C_MUTE)
    return canvas


def answer_card(last_frame, q, plan):
    """The closing hold: gold, every arm's letter, the selection arm's cost."""
    canvas = last_frame.point(DIM_LUT)
    draw = ImageDraw.Draw(canvas, "RGBA")
    arms = {a["arm"]: a for a in q.get("arms", [])}
    rows = [arms[k] for k in arm_order(q.get("selection_leg")) if k in arms]
    driver = scorer_arms(q.get("selection_leg"))[plan["mode"]]

    # The cost line names both clocks - the answer call and, separately, the
    # segment scoring - so it is built before the card is sized and the card
    # widens to hold it rather than letting it run off the panel.
    drv = arms.get(driver)
    tail = None
    if drv is not None:
        proto = ("direct answer" if drv.get("protocol") == "direct"
                 else str(drv.get("protocol", "?")))
        temp = drv.get("temperature")
        # latency_s is the InternVL answer call only; the segment scoring
        # (decode + scorer) is its own stamp and is named separately.
        sel = drv.get("selection_latency_s")
        tail = (f"{drv.get('label', driver)}: {float(drv['latency_s']):.2f} s "
                f"answer call"
                + (f" + {float(sel):.0f} s segment scoring"
                   if isinstance(sel, (int, float)) else "")
                + f" per question {MID} {proto}"
                + (f", T={temp}" if temp is not None else ""))

    f_tail = font(FS_CARD_SM)
    cw = 1180
    if tail is not None:
        # +96: 40 px of left inset plus a little slack, so the line never
        # comes back elided by a fraction of a pixel.
        cw = int(max(cw, min(W - 80, tw(tail, f_tail) + 96)))
    ch = 180 + 54 * len(rows) + 70
    cx, cy = (W - cw) // 2, (H - ch) // 2
    draw.rounded_rectangle([cx, cy, cx + cw, cy + ch], radius=18,
                           fill=(16, 18, 22, 248), outline=(90, 96, 104),
                           width=3)
    draw.text((cx + 40, cy + 26),
              f"Q{q['id']} {MID} {q.get('question_type', '?')}",
              font=font(FS_CARD, bold=True), fill=C_MUTE)
    draw.text((cx + 40, cy + 64), f"Gold: {q.get('gold', '?')}",
              font=font(FS_CARD_H, bold=True), fill=C_INK)

    y = cy + 140
    for arm in rows:
        is_driver = arm["arm"] == driver
        ink = C_INK if is_driver else C_MUTE
        if is_driver:
            draw.rectangle([cx + 26, y - 6, cx + cw - 26, y + 44],
                           fill=(255, 255, 255, 18))
        draw.text((cx + 40, y), arm.get("label", arm["arm"]),
                  font=font(FS_CARD, bold=is_driver), fill=ink)
        pred = arm.get("prediction") or "-"
        ok = bool(arm.get("correct"))
        draw.text((cx + cw - 170, y - 4), pred,
                  font=font(FS_CARD_H, bold=True), fill=C_OK if ok else C_BAD)
        draw.text((cx + cw - 96, y - 2), MARK_OK if ok else MARK_BAD,
                  font=font(FS_CARD_H, bold=True), fill=C_OK if ok else C_BAD)
        y += 54

    if tail is not None:
        draw.text((cx + 40, y + 8), elide(tail, f_tail, cw - 80),
                  font=f_tail, fill=C_MUTE)
    return canvas


# ---------------------------------------------------------------------------
# planning and rendering
# ---------------------------------------------------------------------------
def plan_window(q, args):
    """The clip-time window to play, and how fast to play it."""
    cams = q["cameras"]
    dur = min(float(c["duration_s"]) for c in cams)
    wins = q.get("event_windows") or []
    if not wins:
        raise ValueError(f"q{q['id']}: no event windows to centre the wall on")
    t0 = max(0.0, min(float(w["start_sec"]) for w in wins) - PAD_BEFORE)
    t1 = min(dur, max(float(w["end_sec"]) for w in wins) + PAD_AFTER)
    speed = max(1.0, min(args.max_speed, (t1 - t0) / args.target_seconds))
    if (t1 - t0) < args.target_seconds:
        # Widen symmetrically so the wall always plays for target_seconds at
        # 1x, then shift (never shrink) whatever falls off either clip end.
        mid = 0.5 * (t0 + t1)
        half = min(0.5 * args.target_seconds, 0.5 * dur)
        t0, t1 = mid - half, mid + half
        if t0 < 0:
            t0, t1 = 0.0, min(dur, t1 - t0)
        if t1 > dur:
            t0, t1 = max(0.0, t0 - (t1 - dur)), dur
        speed = max(1.0, min(args.max_speed, (t1 - t0) / args.target_seconds))
    n_out = max(1, int(round((t1 - t0) / speed * args.fps)))
    n_card = max(0, int(round(args.answer_card_seconds * args.fps)))
    return {"t0": round(t0, 3), "t1": round(t1, 3), "speed": round(speed, 4),
            "mode": args.mode, "n_out": n_out, "n_card": n_card,
            "fps": args.fps, "clip_duration_s": dur}


def shared_norm(q):
    """One viridis scale across the four cameras of this question."""
    vals = [v for c in q["cameras"]
            for v in (c.get("segment_scores") or {}).values() if v is not None]
    if not vals:
        return (0.0, 1.0)
    return (float(min(vals)), float(max(vals)))


def ffmpeg_writer(path, fps, crf):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{fps:g}", "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
           "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_question(q, args, out_dir):
    plan = plan_window(q, args)
    plan["footer"] = footer_text(plan["mode"], q.get("selection_leg"),
                                 clip_start_spread(q))
    norm = shared_norm(q)
    streams = [CamStream(c, args.mode, norm, qid=q["id"])
               for c in sorted(q["cameras"], key=lambda c: c["slot"])[:4]]
    # One geometry for the whole wall: these four clips are not all the same
    # height, and a per-camera fit would step the inner gutter by 3 px.
    for cs, geom in zip(streams,
                        wall_geometry([s.src for s in streams], TILE_W, TILE_H)):
        cs.geom = geom
    tile_size, tile_off = streams[0].geom["size"], streams[0].geom["offset"]
    sources = sorted({f"{w}x{h}" for w, h in (s.src for s in streams)})
    modes = sorted({s.geom["mode"] for s in streams})
    print(f"  window {plan['t0']:.2f}-{plan['t1']:.2f} s  speed x{plan['speed']:.2f}  "
          f"frames {plan['n_out']}+{plan['n_card']}")
    print(f"  sources {', '.join(sources)} -> tile {tile_size[0]}x{tile_size[1]} "
          f"at +{tile_off[0]},{tile_off[1]}  fit: {'; '.join(modes)}")

    times = [plan["t0"] + i * plan["speed"] / args.fps
             for i in range(plan["n_out"])]
    path = os.path.join(out_dir, f"q{q['id']}_wall.mp4")
    proc = ffmpeg_writer(path, args.fps, args.crf)
    last = None
    try:
        for start in range(0, len(times), DECODE_CHUNK):
            chunk = times[start:start + DECODE_CHUNK]
            per_cam = [s.tiles_at(chunk) for s in streams]
            for j, t in enumerate(chunk):
                frame = compose(q, streams, [pc[j] for pc in per_cam], t, plan)
                proc.stdin.write(frame.tobytes())
                last = frame
            del per_cam
        card = answer_card(last, q, plan)
        blob = card.tobytes()
        for _ in range(plan["n_card"]):
            proc.stdin.write(blob)
        proc.stdin.close()
    except BrokenPipeError:  # pragma: no cover - ffmpeg died
        proc.stdin.close()
        raise RuntimeError(f"ffmpeg failed while writing {path}")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode} for {path}")

    # The snapshot is composed at the first annotated moment's start, not at a
    # rounded output-frame time, so the verifier sees exactly that instant.
    snap_t = float(q["event_windows"][0]["start_sec"])
    tiles = [s.tiles_at([snap_t])[0] for s in streams]
    snap = compose(q, streams, tiles, snap_t, plan)
    snap_path = os.path.join(out_dir, f"q{q['id']}_wall_snapshot.png")
    snap.save(snap_path)

    size = os.path.getsize(path)
    entry = {
        "id": q["id"],
        "t0": plan["t0"],
        "t1": plan["t1"],
        "speed": plan["speed"],
        "duration": round((plan["n_out"] + plan["n_card"]) / args.fps, 3),
        "path": os.path.relpath(path, REPO),
        "mode": plan["mode"],
        "footer": plan["footer"],
        "played_span_s": round(plan["t1"] - plan["t0"], 3),
        "answer_card_s": round(plan["n_card"] / args.fps, 3),
        "fps": args.fps,
        "crf": args.crf,
        "n_frames": plan["n_out"] + plan["n_card"],
        "snapshot": os.path.relpath(snap_path, REPO),
        "snapshot_time_s": round(snap_t, 3),
        "size_bytes": size,
        "question_type": q.get("question_type"),
        "gold": q.get("gold"),
        "score_scale": {"vmin": norm[0], "vmax": norm[1], "cmap": "viridis"},
        "tile": {"size": list(tile_size), "offset": list(tile_off),
                 "canvas_tile": [TILE_W, TILE_H]},
        "cameras": [{"slot": s.slot, "camera": s.name, "evidence": s.evidence,
                     "segments_kept_shown": sorted(s.kept),
                     "segments_kept_other": sorted(s.kept_other),
                     "source": f"{s.src[0]}x{s.src[1]}",
                     "fit": s.geom["mode"]} for s in streams],
    }
    print(f"  wrote {entry['path']} ({size / 1e6:.1f} MB, "
          f"{entry['duration']:.2f} s) and the snapshot")
    return entry


def concat(out_dir, rendered):
    """walls_all.mp4: every wall in out_dir, back to back, in subset order.

    Built from the q<id>_wall.mp4 files ON DISK, not from the ids this run
    happened to render, so `--ids 749` refreshes one clip and still leaves a
    complete reel instead of a one-question stub or a stale five-question one.
    Ids outside SUBSET_ORDER follow it in numeric order. The ids that went in
    are returned, and recorded in walls_manifest.json.
    """
    found = []
    for name in sorted(os.listdir(out_dir)):
        m = re.fullmatch(r"q(\d+)_wall\.mp4", name)
        if m:
            found.append((int(m.group(1)), os.path.join(out_dir, name)))
    order = {qid: i for i, qid in enumerate(SUBSET_ORDER)}
    found.sort(key=lambda f: (order.get(f[0], 10_000), f[0]))
    ids = [qid for qid, _ in found]
    out = os.path.join(out_dir, "walls_all.mp4")
    if len(found) < 2:
        if os.path.exists(out):
            os.remove(out)          # a stale reel is worse than no reel
            note(f"removed walls_all.mp4: only {ids} on disk, nothing to join")
        return None, ids
    reused = [qid for qid in ids if qid not in set(rendered)]
    if reused:
        note(f"walls_all.mp4 also carries walls this run did not render: "
             f"{reused} (the q<id>_wall.mp4 already in "
             f"{os.path.relpath(out_dir, REPO)}, possibly from another --mode)")
    with tempfile.TemporaryDirectory() as scratch:
        list_path = os.path.join(scratch, "walls_concat.txt")
        with open(list_path, "w") as fh:
            for _, path in found:
                fh.write("file '%s'\n" % path.replace("'", r"'\''"))
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out]
        if subprocess.call(cmd) != 0:
            raise RuntimeError("ffmpeg concat failed")
    print(f"  wrote {os.path.relpath(out, REPO)} "
          f"({os.path.getsize(out) / 1e6:.1f} MB, {ids})")
    return out, ids


def main():
    ap = argparse.ArgumentParser(
        description="Render one 2x2 camera-wall MP4 per demo question.")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="verified figure manifest (default: %(default)s)")
    ap.add_argument("--ids", nargs="+", type=int, default=None,
                    help="question ids to render (default: all in the manifest)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output directory (default: %(default)s)")
    ap.add_argument("--mode", choices=["per_clip", "global"], default="per_clip",
                    help="which leg's kept segments drive the dimming; the "
                         "other leg's frame count still appears in the label")
    ap.add_argument("--target-seconds", type=float, default=14.0,
                    help="target played length of the window (default: %(default)s)")
    ap.add_argument("--max-speed", type=float, default=4.0)
    ap.add_argument("--answer-card-seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--crf", type=int, default=20)
    args = ap.parse_args()

    man_path = rel(args.manifest)
    with open(man_path) as fh:
        man = json.load(fh)
    questions = man.get("questions") or []
    if args.ids:
        want = list(args.ids)
        by_id = {q["id"]: q for q in questions}
        missing = [i for i in want if i not in by_id]
        if missing:
            sys.exit(f"ids not in {args.manifest}: {missing}")
        questions = [by_id[i] for i in want]
    if not questions:
        sys.exit(f"no questions in {args.manifest}")

    out_dir = rel(args.out)
    os.makedirs(out_dir, exist_ok=True)
    print(f"manifest {os.path.relpath(man_path, REPO)} -> "
          f"{os.path.relpath(out_dir, REPO)}  mode={args.mode}")

    entries = []
    for q in questions:
        print(f"q{q['id']} ({q.get('question_type')}, gold {q.get('gold')})")
        entries.append(render_question(q, args, out_dir))

    all_path, all_ids = concat(out_dir, [e["id"] for e in entries])

    wm = {
        "generated_by": "analysis/make_demo_walls.py",
        "manifest": os.path.relpath(man_path, REPO),
        "mode": args.mode,
        "fps": args.fps,
        "crf": args.crf,
        "target_seconds": args.target_seconds,
        "max_speed": args.max_speed,
        "answer_card_seconds": args.answer_card_seconds,
        "pad_before_s": PAD_BEFORE,
        "pad_after_s": PAD_AFTER,
        "canvas": f"{W}x{H}",
        "footer": (entries[0].get("footer") if entries else None),
        "all_path": os.path.relpath(all_path, REPO) if all_path else None,
        "all_ids": all_ids,
        "rendered_ids": [e["id"] for e in entries],
        "subset_order": SUBSET_ORDER,
        "notes": NOTES,
        "walls": entries,
    }
    wm_path = os.path.join(out_dir, "walls_manifest.json")
    with open(wm_path, "w") as fh:
        json.dump(wm, fh, indent=2)
        fh.write("\n")
    print(f"wrote {os.path.relpath(wm_path, REPO)}")


if __name__ == "__main__":
    main()
