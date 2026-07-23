# Ported from Wavy-Hec/CVBench bench/frame_sweep_by_category_fig.py @ db18c06633ffd3183d008bf3fd8ac8138d273cd5
"""Editorial line chart: accuracy vs. frame budget, by task category.

Sweep legs are 2x2-stitch presentations at 8/16/32/64/128 frames; the pooled
JSONL carries one row per (question, method, pass).

The numbers are recomputed from the raw pooled run so the figure is reproducible
end-to-end: for each frame budget and task category we take the per-pass accuracy
(correct / total within each pass), then report the mean and the population std
across passes -- the same statistic the summary path prints (statistics.pstdev).

Each series also gets a dashed random-guessing (chance) line in its own colour,
computed by evaluation.chance from the subset's real option lists: chance = mean
over that category's questions of 1 / (number of answer options). It is not a
flat 25% -- CVBench mixes four-option questions with yes/no ones.

Two deviations from the fork source, both deliberate:

1. The fork keeps frozen fallback arrays to plot when the pooled JSONL is
   missing. Results never live in this repo, so this port requires --jsonl.
2. The fork draws a second "effective fps" tick row under the x axis, derived
   from a hardcoded 300 s clip length. That constant does not describe this
   subset -- sampled clip durations run from a few seconds to ~17 minutes with a
   median near 80 s, and because the stitched harness samples against the
   shorter clip of each pair, the length that actually governs a question has a
   median nearer 50 s. One scalar cannot stand in for that spread, and 300 s
   understated the real rate several-fold, so the derived axis is dropped rather
   than published as fact. The frame budget itself is exact and still plotted.

Usage (from repo root):
  python -m plotting.frame_sweep_by_category --jsonl results/stitch_sweep_combined.jsonl
"""
import argparse
import json
import os
from collections import defaultdict
from statistics import mean as _mean, pstdev as _pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

from evaluation.chance import chance_table, default_qa_json, pooled_detail  # noqa: E402

# method tag (in the pooled jsonl) -> frame budget
METHOD_FRAMES = {
    "stitch08_f8": 8, "stitch16_f16": 16, "stitch32_f32": 32,
    "stitch64_f64": 64, "stitch128_f128": 128,
}
FRAMES = [8, 16, 32, 64, 128]

# category display name -> task_type string in the data (None = pooled overall)
CATS = {
    "Overall":            None,
    "Temporal Reasoning": "Multi-video Temporal Reasoning",
    "Event Retrieval":    "Cross-video Event Retrieval",
    "Spatial Navigating": "Joint-video Spatial Navigating",
}


def load_from_raw(paths):
    """Return (mean%, std%) dicts keyed by category, each a list over FRAMES."""
    # method -> cat -> pass_idx -> [correct, total]
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                m = r.get("method")
                if m not in METHOD_FRAMES:
                    continue
                tt, pi, correct = r.get("task_type"), r.get("pass_idx"), bool(r.get("correct"))
                for cat, filt in CATS.items():
                    if filt is None or tt == filt:
                        cell = agg[m][cat][pi]
                        cell[1] += 1
                        cell[0] += 1 if correct else 0

    means, stds = {}, {}
    inv = {v: k for k, v in METHOD_FRAMES.items()}
    for cat in CATS:
        m_row, s_row = [], []
        for f in FRAMES:
            method = inv[f]
            per_pass = []
            for pi in sorted(agg[method][cat]):
                cor, tot = agg[method][cat][pi]
                if tot:
                    per_pass.append(cor / tot)
            if not per_pass:
                raise SystemExit(
                    f"ERROR: no rows for method {method!r} -- is this the pooled "
                    "sweep JSONL (all five frame legs concatenated)?")
            m_row.append(100 * _mean(per_pass))
            s_row.append(100 * (_pstdev(per_pass) if len(per_pass) > 1 else 0.0))
        means[cat], stds[cat] = m_row, s_row
    return means, stds


def get_chance(qa_json):
    """Chance (random-guessing) accuracy % per plotted category, + option mix."""
    per_task, overall, detail = chance_table(qa_json)
    chance, mix = {}, {}
    for cat, filt in CATS.items():
        chance[cat] = overall if filt is None else per_task[filt]
        mix[cat] = pooled_detail(detail) if filt is None else detail.get(filt)
    return chance, mix


# --------------------------------------------------------------------------- #
# Reusable editorial style -- an rcParams block, drop into any figure script.
# --------------------------------------------------------------------------- #
INK = "#1a1a1a"      # primary text / near-black
INK_SOFT = "#4d4d4d"  # secondary text
MUTED = "#8a8a8a"    # tertiary (fps row, footnote)
GRID = "#ededed"
SPINE = "#3d3d3d"

EDITORIAL_RC = {
    # type: a clean sans; resolves to Liberation Sans (Helvetica-metric) on most
    # Linux boxes, honouring the Inter/Source Sans/Helvetica preference where installed.
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Source Sans Pro", "Source Sans 3", "Helvetica",
                        "Helvetica Neue", "Arial", "Liberation Sans",
                        "Nimbus Sans", "DejaVu Sans"],
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.labelsize": 11,
    "axes.titlesize": 15,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": SPINE,
    "ytick.color": SPINE,
    "xtick.labelcolor": INK_SOFT,
    "ytick.labelcolor": INK_SOFT,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 0,      # y ticks read off the gridlines; no stubs
    "xtick.major.width": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",     # keep SVG text as editable text
    "pdf.fonttype": 42,         # embed TrueType in PDF (text stays text)
    "figure.dpi": 150,
}

# entity -> hue (validated: all six data-viz checks pass, light surface)
COLOR = {
    "Overall":            "#1F1B18",  # near-black reference
    "Temporal Reasoning": "#A63D2C",  # muted rust  (faller, warm)
    "Event Retrieval":    "#BC8A1E",  # muted ochre (faller, warm)
    "Spatial Navigating": "#068A72",  # muted teal-green (riser, cool)
}
LW = {"Overall": 2.4, "Temporal Reasoning": 1.7,
      "Event Retrieval": 1.7, "Spatial Navigating": 1.7}
ZORD = {"Temporal Reasoning": 3, "Event Retrieval": 3,
        "Spatial Navigating": 3, "Overall": 5}
ORDER = ["Overall", "Temporal Reasoning", "Event Retrieval", "Spatial Navigating"]


def decollide(targets, min_gap, lo, hi):
    """Nudge label y-positions apart by >= min_gap, keeping within [lo, hi].

    targets: dict name -> desired y. Returns dict name -> adjusted y.
    """
    names = sorted(targets, key=lambda k: targets[k])
    ys = [targets[n] for n in names]
    # forward pass: push up
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    # if we overran the top, slide the whole stack down
    overflow = ys[-1] - hi
    if overflow > 0:
        ys = [y - overflow for y in ys]
    # backward pass in case sliding down collided at the bottom
    for i in range(len(ys) - 2, -1, -1):
        if ys[i + 1] - ys[i] < min_gap:
            ys[i] = ys[i + 1] - min_gap
    ys = [max(lo, y) for y in ys]
    return dict(zip(names, ys))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jsonl", nargs="+", required=True,
                    help="pooled sweep JSONL(s): all five frame legs concatenated")
    ap.add_argument("--qa-json", default=default_qa_json(),
                    help="QA subset json whose option lists define the chance level")
    ap.add_argument("--out-dir", default="figures/frame_sweep")
    ap.add_argument("--stem", default="frame_sweep_by_category")
    ap.add_argument("--backend-label", default="InternVL3-8B",
                    help="model name shown in the subtitle")
    args = ap.parse_args()

    means, stds = load_from_raw(args.jsonl)
    chance, mix = get_chance(args.qa_json)

    # ---- print the exact arrays being plotted -----------------------------
    print(f"\nData source   : {', '.join(args.jsonl)} (recomputed)")
    print(f"Chance source : {args.qa_json}")
    print(f"Frame budgets : {FRAMES}\n")
    print(f"{'category':20s} | " + " | ".join(f"f{f:<3d}" for f in FRAMES) + " | chance")
    print("-" * 76)
    for cat in ORDER:
        cells = " | ".join(f"{m:4.1f}" for m in means[cat])
        print(f"{cat:20s} | {cells} | {chance[cat]:5.1f}   (accuracy %)")
        scells = " | ".join(f"{s:4.1f}" for s in stds[cat])
        print(f"{'  +/- pass std':20s} | {scells} |")
        gcells = " | ".join(f"{m - chance[cat]:+4.1f}" for m in means[cat])
        print(f"{'  above chance':20s} | {gcells} |")
        d = mix.get(cat)
        if d:
            print(f"{'  option mix':20s} | n={d['n']}, "
                  f"{{n_options: count}} = {d['hist']}")
    print()

    # ---- figure ------------------------------------------------------------
    plt.rcParams.update(EDITORIAL_RC)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=0.115, right=0.815, top=0.84, bottom=0.215)

    ax.set_xscale("log", base=2)

    # faint +/-1 std bands (drawn first, under the lines)
    for cat in ORDER:
        m, s = means[cat], stds[cat]
        ax.fill_between(FRAMES, [a - b for a, b in zip(m, s)],
                        [a + b for a, b in zip(m, s)],
                        color=COLOR[cat], alpha=0.09, linewidth=0, zorder=1)

    # ---- chance (random-guessing) reference, one dashed line per series ----
    # Drawn under the data. Each category's own chance level, because the option
    # counts are not uniform -- see evaluation/chance.py.
    for cat in ORDER:
        ax.plot([8, 128], [chance[cat]] * 2, color=COLOR[cat], lw=1.0,
                linestyle=(0, (4, 3)), alpha=0.55, zorder=2, solid_capstyle="butt")

    # lines + markers
    for cat in ORDER:
        ax.plot(FRAMES, means[cat], color=COLOR[cat], lw=LW[cat],
                solid_capstyle="round", solid_joinstyle="round",
                marker="o", markersize=5.5, markerfacecolor=COLOR[cat],
                markeredgecolor="white", markeredgewidth=0.9, zorder=ZORD[cat])

    # ---- axes limits & ticks ----------------------------------------------
    ax.set_xlim(7.1, 205)          # right room for the direct labels
    ax.set_ylim(20, 72)            # low enough to seat the ~25-29% chance lines
    ax.set_yticks([20, 30, 40, 50, 60, 70])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))

    ax.xaxis.set_major_locator(FixedLocator(FRAMES))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))

    # Tufte-ish: trim spines to the data range so they don't run into margins
    ax.spines["left"].set_bounds(20, 70)
    ax.spines["bottom"].set_bounds(8, 128)

    ax.set_ylabel("accuracy (%)")

    # row label at the far left, aligned to the tick row
    ax.text(-0.028, -0.038, "frames", transform=ax.transAxes, ha="right",
            va="top", fontsize=9, color=INK_SOFT)

    # ---- direct labels at the right end, de-collided ----------------------
    ends = {cat: means[cat][-1] for cat in ORDER}
    label_y = decollide(dict(ends), min_gap=2.8, lo=30.5, hi=71.0)
    x_text = 128 * 2 ** 0.07       # just right of the f128 marker
    for cat in ORDER:
        ye, yl = ends[cat], label_y[cat]
        if abs(yl - ye) > 0.35:    # thin leader when the label was nudged
            ax.plot([128, x_text], [ye, yl], color=COLOR[cat], lw=0.6,
                    alpha=0.55, zorder=2, clip_on=False)
        ax.text(x_text * 1.04, yl, cat, color=COLOR[cat], fontsize=10,
                va="center", ha="left", clip_on=False)

    # ---- direct labels for the chance lines, same treatment ----------------
    c_label_y = decollide(dict(chance), min_gap=1.9, lo=21.0, hi=30.2)
    for cat in ORDER:
        ye, yl = chance[cat], c_label_y[cat]
        if abs(yl - ye) > 0.25:
            ax.plot([128, x_text], [ye, yl], color=COLOR[cat], lw=0.6,
                    alpha=0.45, zorder=2, clip_on=False)
        ax.text(x_text * 1.04, yl, f"{chance[cat]:.1f}%", color=COLOR[cat],
                fontsize=9, alpha=0.9, va="center", ha="left", clip_on=False)
    ax.text(x_text * 1.04, max(c_label_y.values()) + 2.7,
            "chance (random guessing)", color=MUTED, fontsize=8.5,
            style="italic", va="center", ha="left", clip_on=False)

    # ---- title (neutral, descriptive) & finding caption (below axes) ------
    n_q = mix["Overall"]["n"] if mix.get("Overall") else "?"
    ax.set_title("Accuracy vs. frame budget, by task category\n"
                 f"(2×2-stitch, {args.backend_label})", loc="left", linespacing=1.35)

    fig.text(0.115, 0.078,
             "Overall flat then falling; temporal-ordering tasks degrade with "
             "more frames, spatial navigation improves — all well above chance.",
             ha="left", va="center", fontsize=9.5, color=INK_SOFT)
    fig.text(0.115, 0.040,
             f"{n_q} questions × {len(FRAMES)} frame budgets; bands = ±1 std across "
             f"passes.",
             ha="left", va="center", fontsize=8, color=MUTED)
    fig.text(0.115, 0.012,
             "Dashed = chance, the score of a uniform random guesser: mean over that "
             "category's questions of 1∕(number of answer options).",
             ha="left", va="center", fontsize=8, color=MUTED)

    # ---- export -----------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, args.stem)
    for ext, dpi in (("png", 300), ("pdf", None), ("svg", None)):
        fig.savefig(f"{stem}.{ext}", **({"dpi": dpi} if dpi else {}))
    plt.close(fig)
    print("Wrote:")
    for ext in ("png", "pdf", "svg"):
        print(f"  {stem}.{ext}")


if __name__ == "__main__":
    main()
