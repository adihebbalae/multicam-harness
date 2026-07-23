# Ported from Wavy-Hec/CVBench bench/frame_budget_smallmultiples_fig.py @ db18c06633ffd3183d008bf3fd8ac8138d273cd5
"""Small multiples: accuracy vs. frame budget, one panel per task type.

Sweep legs are 2x2-stitch presentations at 8/16/32/64/128 frames; the pooled
JSONL carries one row per (question, method, pass).

Everything is recomputed from the raw pooled run: for each frame budget and task
type we take the per-pass accuracy (correct / total within each pass), then the
mean and the population std across passes -- the same statistic the summary path
prints. Panels are ordered Overall first, then by accuracy gain from the
smallest to the largest frame budget (descending).

The dashed line in each panel is that task type's chance level from
evaluation.chance: the accuracy of a guesser picking uniformly among the options
it is shown, = mean over that panel's questions of 1 / n_options. It is not a
flat 25% -- Cross-video Scene Recognition is mostly yes/no, so its chance is
45.5%, and the pooled Overall panel sits at 28.5%.

Usage (from repo root):
  python -m plotting.frame_budget_smallmultiples --jsonl results/stitch_sweep_combined.jsonl
"""
import argparse
import json
import os
import textwrap
import warnings
from collections import defaultdict
from statistics import mean as _mean, pstdev as _pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

from evaluation.chance import chance_table, default_qa_json  # noqa: E402

METHOD_FRAMES = {
    "stitch08_f8": 8, "stitch16_f16": 16, "stitch32_f32": 32,
    "stitch64_f64": 64, "stitch128_f128": 128,
}
FRAMES = [8, 16, 32, 64, 128]
OVERALL = "Overall"          # pooled pseudo-category key

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
INK = "#26251f"
MUTED = "#6e6a5e"
GRID = "#e8e6df"
SPINE = "#ced4da"
BLUE = "#1f5fbf"

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Source Sans Pro", "Source Sans 3", "Helvetica",
                        "Helvetica Neue", "Arial", "Liberation Sans",
                        "Nimbus Sans", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": MUTED,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": SPINE,
    "ytick.color": SPINE,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 3,
    "ytick.major.size": 0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.dpi": 150,
}

# Low enough to seat the ~25% chance lines with a clear strip underneath for the
# gain annotation, high enough for the tallest +/-1 std band.
YLIM = (16, 92)
YTICKS = [40, 60, 80]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_from_raw(paths):
    """-> {task_type or 'Overall': {'mean': [...], 'std': [...], 'n': int}}."""
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
                correct = 1 if r.get("correct") else 0
                for key in (r.get("task_type"), OVERALL):
                    cell = agg[m][key][r.get("pass_idx")]
                    cell[1] += 1
                    cell[0] += correct

    inv = {v: k for k, v in METHOD_FRAMES.items()}
    if not agg[inv[8]]:
        raise SystemExit(
            "ERROR: no rows for the smallest frame leg -- is this the pooled "
            "sweep JSONL (all five frame legs concatenated)?")
    out = {}
    for key in agg[inv[8]]:
        means, stds = [], []
        for f in FRAMES:
            per_pass = [c / t for c, t in agg[inv[f]][key].values() if t]
            means.append(100 * _mean(per_pass))
            stds.append(100 * (_pstdev(per_pass) if len(per_pass) > 1 else 0.0))
        n_passes = len(agg[inv[8]][key]) or 1
        n_q = sum(t for _, t in agg[inv[8]][key].values()) // n_passes
        out[key] = {"mean": means, "std": stds, "n": n_q}
    return out


def get_chance(series, qa_json):
    per_task, overall, _ = chance_table(qa_json)
    chance = {}
    for key in series:
        if key == OVERALL:
            chance[key] = overall
        elif key in per_task:
            chance[key] = per_task[key]
        else:
            warnings.warn(f"no chance level for {key!r}; falling back to 25%")
            chance[key] = 25.0
    return chance


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def panel(ax, ys, ss, chance, color, is_overall, title, n_q, bottom_row, left_col):
    lo = [a - b for a, b in zip(ys, ss)]
    hi = [a + b for a, b in zip(ys, ss)]

    ax.set_xscale("log", base=2)
    ax.set_xlim(7.2, 142)
    ax.set_ylim(*YLIM)

    # chance reference, drawn under everything
    ax.plot([8, 128], [chance] * 2, color=color, lw=1.0, linestyle=(0, (4, 3)),
            alpha=0.5, zorder=1, solid_capstyle="butt")
    ax.text(8.3, chance + 1.2, f"chance {chance:.1f}%", color=color, alpha=0.85,
            fontsize=7.5, ha="left", va="bottom", zorder=4)

    ax.fill_between(FRAMES, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=2)
    ax.plot(FRAMES, ys, color=color, lw=2.0, solid_capstyle="round",
            solid_joinstyle="round", zorder=3)

    # dot + value at the best budget
    best = max(range(len(ys)), key=lambda i: ys[i])
    ax.plot([FRAMES[best]], [ys[best]], marker="o", markersize=6, color=color,
            markeredgecolor="white", markeredgewidth=0.9, zorder=5)
    ha = "left" if best == 0 else ("right" if best == len(ys) - 1 else "center")
    dx = {"left": 1.06, "right": 0.94, "center": 1.0}[ha]
    ax.text(FRAMES[best] * dx, ys[best] + 3.0, f"{ys[best]:.0f}", color=color,
            fontsize=10, fontweight="bold", ha=ha, va="bottom", zorder=5)

    # gain annotation, bottom-right; white bbox so the chance line reads as
    # interrupted rather than struck through
    gain = ys[-1] - ys[0]
    ax.text(0.975, 0.03, f"{gain:+.1f} pts @{FRAMES[-1]} vs {FRAMES[0]}",
            transform=ax.transAxes, color=MUTED, fontsize=8.5, ha="right",
            va="bottom", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    ax.xaxis.set_major_locator(FixedLocator(FRAMES))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    ax.set_yticks(YTICKS)
    if not bottom_row:
        ax.set_xticklabels([])
    if not left_col:
        ax.set_yticklabels([])

    label = f"{title} ({n_q} Q)" if is_overall else title
    ax.set_title("\n".join(textwrap.wrap(label, 26)), fontsize=10.5,
                 color=INK, fontweight="bold" if is_overall else "normal",
                 pad=8, linespacing=1.25)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jsonl", nargs="+", required=True,
                    help="pooled sweep JSONL(s): all five frame legs concatenated")
    ap.add_argument("--qa-json", default=default_qa_json(),
                    help="QA subset json whose option lists define the chance level")
    ap.add_argument("--out-dir", default="figures/frame_sweep")
    ap.add_argument("--stem", default="frame_budget_smallmultiples")
    ap.add_argument("--backend-label", default="InternVL3-8B",
                    help="model name shown in the subtitle")
    args = ap.parse_args()

    series = load_from_raw(args.jsonl)
    chance = get_chance(series, args.qa_json)

    # Overall first, then by gain across the frame range, descending
    tasks = [k for k in series if k != OVERALL]
    tasks.sort(key=lambda k: -(series[k]["mean"][-1] - series[k]["mean"][0]))
    order = [OVERALL] + tasks

    # ---- print the exact arrays being plotted ------------------------------
    print(f"\nData source   : {', '.join(args.jsonl)} (recomputed)")
    print(f"Chance source : {args.qa_json}")
    print(f"Frame budgets : {FRAMES}\n")
    print(f"{'panel':38s} {'n':>5s}  " + "".join(f"f{f:<6d}" for f in FRAMES)
          + f"{'gain':>7s} {'chance':>7s} {'best-chance':>12s}")
    print("-" * 108)
    for k in order:
        d = series[k]
        cells = "".join(f"{m:5.1f} " for m in d["mean"])
        gain = d["mean"][-1] - d["mean"][0]
        print(f"{k:38s} {d['n']:5d}  {cells}{gain:+7.1f} {chance[k]:7.1f} "
              f"{max(d['mean']) - chance[k]:+12.1f}")
    print()

    # ---- figure ------------------------------------------------------------
    plt.rcParams.update(RC)
    nrow = 4
    ncol = max(1, -(-len(order) // nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.25 * ncol, 7.7), squeeze=False)
    fig.subplots_adjust(left=0.055, right=0.988, top=0.825, bottom=0.10,
                        hspace=0.62, wspace=0.10)

    for i, key in enumerate(order):
        r, c = divmod(i, ncol)
        d = series[key]
        panel(axes[r][c], d["mean"], d["std"], chance[key],
              INK if key == OVERALL else BLUE, key == OVERALL,
              key, d["n"], bottom_row=(r == nrow - 1), left_col=(c == 0))
    for j in range(len(order), nrow * ncol):    # blank any unused cells
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle("More frames don't help: overall flat then falling — "
                 "temporal-ordering categories degrade most",
                 x=0.055, y=0.978, ha="left", fontsize=15, fontweight="bold",
                 color=INK)
    fig.text(0.055, 0.945,
             f"2×2-stitch presentation · {args.backend_label} · "
             f"{series[OVERALL]['n']} questions · line = mean, band = ±1 std, "
             f"dot = best budget · panels sorted by gain "
             f"{FRAMES[0]}→{FRAMES[-1]}",
             ha="left", va="center", fontsize=10, color=MUTED)
    fig.text(0.055, 0.915,
             "Dashed = chance, the score of a uniform random guesser: mean over that "
             "panel's questions of 1∕(number of answer options). Not a flat 25% — "
             "Cross-video Scene Recognition is mostly yes/no.",
             ha="left", va="center", fontsize=9, color=MUTED)

    fig.text(0.012, 0.46, "accuracy (%)", rotation=90, ha="center", va="center",
             fontsize=10, color=MUTED)
    fig.text(0.52, 0.026,
             "frame budget (stitched 2×2 timesteps per question, log scale)",
             ha="center", va="center", fontsize=10, color=MUTED)

    # ---- export -------------------------------------------------------------
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
