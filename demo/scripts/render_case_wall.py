#!/usr/bin/env python3
"""Animated 2x2 camera wall for a case question, in the style of the deck's question-860 wall.

Reads <out>/case_q<ID>.json (written by render_case_contrast.py into the same --out) and the
release videos; the four cameras are synchronised in wall-clock time (clip start times differ
between cameras) over a window around the annotated events. Per question and case, three files:

  wall_case1_q<ID>.gif / .mp4   each tile carries that camera's answer when it is the only input
  wall_case2_q<ID>.gif / .mp4   the same wall with the multi-camera answer in a banner
  wall_case<N>_q<ID>_mid.png    the middle frame, the still the demo shows before the clip plays

The GIF is for the slides; the MP4 (H.264, same route as make_demo_walls.py) is what the demo
page plays. A tile is outlined in the selection orange while its annotated event is happening.
CPU only.

Usage:
  ~/anaconda3/envs/cvbench/bin/python demo/scripts/render_case_wall.py \
      --release-root ~/MultiCam/crossview-release-annotations/crossview-release --qid 954 \
      --out demo/figs_case            # the demo page reads figs_case/ next to demo_ui.html
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import matplotlib  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H = 960, 540                     # same canvas as the question-860 wall
TW, TH = W // 2, H // 2
N_FRAMES, FRAME_MS = 210, 100       # a 21 s loop: about 42 s of wall-clock at 2x
GIF_W = 720                         # encoded width; drawn at 960 so the labels match the existing wall
PAD_S = 5.0                         # wall-clock seconds shown before the first / after the last event
TEAL, WHITE, ORANGE, GREEN, RED = (27, 175, 160), (255, 255, 255), (235, 104, 52), (27, 175, 122), (226, 72, 60)
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")


def font(size, bold=True):
    return ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def find_video(root, name):
    hits = glob.glob(os.path.join(root, "videos", "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def strip(draw, xy, wh, alpha_img, fill=(0, 0, 0, 150)):
    ImageDraw.Draw(alpha_img).rectangle([xy, (xy[0] + wh[0], xy[1] + wh[1])], fill=fill)


def letters(answers):
    return " ".join(a.strip()[0] for a in answers)


def encode(frames, base, ffmpeg):
    """Write base.gif and base.mp4 from one PNG sequence.

    GIF: one shared palette and frame differencing (ffmpeg) - about 7x smaller than PIL's
    per-frame palettes. MP4: H.264 yuv420p with the moov atom in front, the same route
    make_demo_walls.py takes, so the demo page plays it through the same <video> path as
    the question walls (pause, scrub, no 6 MB GIF decode).
    """
    with tempfile.TemporaryDirectory() as tmp:
        for k, f in enumerate(frames):
            f.save(os.path.join(tmp, "f%04d.png" % k))
        seq = os.path.join(tmp, "f%04d.png")
        rate = str(1000 // FRAME_MS)
        vf = ("scale=%d:-2:flags=lanczos,split[a][b];" % GIF_W +
              "[a]palettegen=max_colors=128:stats_mode=diff[p];"
              "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", rate, "-i", seq,
                        "-vf", vf, "-loop", "0", base + ".gif"], check=True)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", rate, "-i", seq,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                        base + ".mp4"], check=True)


def render(case, root, out, mode, ffmpeg):
    cams = case["cameras"][:4]
    events = [e for c in cams for e in c["events"]]
    t0 = min(e["wall_start_s"] for e in events) - PAD_S
    t1 = max(e["wall_end_s"] for e in events) + PAD_S
    step = (t1 - t0) / (N_FRAMES - 1)
    readers = [VideoReader(find_video(root, c["file"]), ctx=cpu(0), width=TW, height=TH) for c in cams]
    names = case["short"]["events"]
    arm = case["arms"][case["case2_arm"]] if mode == "case2" else None
    f_lab, f_ans, f_clock = font(19), font(20), font(15, bold=False)
    frames = []
    for k in range(N_FRAMES):
        t = t0 + k * step
        canvas = Image.new("RGB", (W, H))
        over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        active = []
        for i, (c, vr) in enumerate(zip(cams, readers)):
            x, y = (i % 2) * TW, (i // 2) * TH
            idx = min(max(int(round((t - c["clip_start_s"]) * c["fps"])), 0), len(vr) - 1)
            canvas.paste(Image.fromarray(vr[idx].asnumpy()), (x, y))
            strip(None, (x, y), (TW, 30), over)
            if mode == "case1":
                strip(None, (x, y + TH - 32), (TW, 32), over)
            active.append(any(e["wall_start_s"] <= t <= e["wall_end_s"] for e in c["events"]))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), over).convert("RGB")
        d = ImageDraw.Draw(canvas)
        for i, c in enumerate(cams):
            x, y = (i % 2) * TW, (i // 2) * TH
            has = bool(c["events"])
            lab = "%s · %s" % (c["camera"], names[c["events"][0]["label"]] + " event" if has else "no event")
            d.text((x + 10, y + 4), lab, font=f_lab, fill=TEAL if has else WHITE)
            if mode == "case1":
                s = c["single"]
                ok = s["n_right"] == len(s["answers"])
                d.text((x + 10, y + TH - 29), "alone:  %s   %s" % (letters(s["answers"]), "✓" if ok else "✗"),
                       font=f_ans, fill=GREEN if ok else RED)
            if active[i]:
                d.rectangle([x + 2, y + 2, x + TW - 3, y + TH - 3], outline=ORANGE, width=5)
                d.text((x + TW - 150, y + 4), "HAPPENING", font=f_lab, fill=ORANGE)
        if mode == "case2":
            ok = arm["n_right"] == len(arm["answers"])
            txt = "all four cameras together:  %s   %s" % (letters(arm["answers"]), "✓" if ok else "✗")
            tw = d.textlength(txt, font=f_ans)
            d.rectangle([(W - tw) / 2 - 16, H - 40, (W + tw) / 2 + 16, H - 2], fill=(0, 0, 0))
            d.text(((W - tw) / 2, H - 34), txt, font=f_ans, fill=GREEN if ok else RED)
        hh, mm, ss = int(t // 3600), int(t % 3600 // 60), int(t % 60)
        clock = "%02d:%02d:%02d  ·  %.0fx" % (hh, mm, ss, step / (FRAME_MS / 1000))
        d.text((W - d.textlength(clock, font=f_clock) - 8, TH - 24 if mode == "case2" else TH - 54), clock,
               font=f_clock, fill=WHITE)
        frames.append(canvas)
    base = os.path.join(out, "wall_%s_q%d" % (mode, case["id"]))
    encode(frames, base, ffmpeg)
    frames[N_FRAMES // 2].save(base + "_mid.png")
    for ext in (".gif", ".mp4"):
        print("%s: %.1f MB, %.0f s of wall-clock in %d frames" % (
            base + ext, os.path.getsize(base + ext) / 1e6, t1 - t0, N_FRAMES))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--qid", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg", default=os.path.join(os.path.dirname(sys.executable), "ffmpeg"),
                    help="ffmpeg binary (default: the one in this Python environment)")
    a = ap.parse_args()
    for q in a.qid:
        case = json.load(open(os.path.join(a.out, "case_q%d.json" % q)))
        for mode in ("case1", "case2"):
            render(case, os.path.expanduser(a.release_root), a.out, mode, a.ffmpeg)


if __name__ == "__main__":
    main()
