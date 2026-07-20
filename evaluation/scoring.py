# Ported from Wavy-Hec/CVBench Video-R1/src/eval_thinking.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
# Ported from Wavy-Hec/CVBench bench/metrics.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Answer scoring + benchmark metric aggregation.

``extract_think`` / ``extract_answer`` / ``parse_choice`` / ``gt_choice`` are
vendored byte-faithfully from ``Video-R1/src/eval_thinking.py`` so this repo
parses and scores answers identically to the original harness.

The rest aggregates per-question Result rows into the benchmark metrics (M1-M4).

Pure-stdlib (no numpy) so it runs anywhere, including the login node for the
CPU scoring-validation gate.
"""
import re
from collections import defaultdict
from statistics import mean as _mean, pstdev as _pstdev


def extract_think(text):
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Thinking checkpoints (e.g. Qwen3-VL-*-Thinking) open <think> inside the
    # generation prompt, so the decoded output holds only "trace...</think>".
    if "</think>" in text:
        return text.split("</think>", 1)[0].strip()
    return ""


def extract_answer(text):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


# Explicit "the answer is X" style conclusions, used ONLY when the model never
# emitted an <answer> tag (e.g. a truncated trace). We take the LAST such match
# (closest to the model's conclusion). Scanning the whole trace for any bare
# A/B/C/D is wrong: it grabs prose like "a vehicle" / "options A to D", which
# fabricated a lucky 'A' on truncated runs and made 0/20 structural.
_CONCLUDE_MC = re.compile(
    r"(?i)(?:final\s+answer|best\s+answer|correct\s+answer|the\s+answer|answer)\s*"
    r"(?:is|:|=|would\s+be)?\s*\(?([ABCD])\b"
)
_CONCLUDE_YN = re.compile(
    r"(?i)(?:final\s+answer|the\s+answer|answer)\s*(?:is|:|=)?\s*\(?(yes|no)\b"
)


def parse_choice(text, is_yesno):
    """Final answer = <answer>..</answer> if present. If the tag is missing
    (e.g. a truncated trace), fall back ONLY to an explicit "the answer is X"
    conclusion (last match); otherwise abstain (return "") rather than grabbing
    an incidental letter from the reasoning prose."""
    ans = extract_answer(text)
    if is_yesno:
        if ans:
            m = re.search(r"(?i)\b(yes|no)\b", ans)
            return m.group(1).capitalize() if m else ans.strip()
        ms = list(_CONCLUDE_YN.finditer(text))
        return ms[-1].group(1).capitalize() if ms else ""
    if ans:
        m = re.search(r"(?i)\b([ABCD])\b", ans)
        return m.group(1).upper() if m else ans.strip()
    ms = list(_CONCLUDE_MC.finditer(text))
    return ms[-1].group(1).upper() if ms else ""


def gt_choice(answer, is_yesno):
    a = answer.strip()
    if is_yesno:
        return a.capitalize()
    m = re.search(r"(?i)([ABCD])", a)
    return m.group(1).upper() if m else a.upper()


def _pct(xs, q):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


def _acc(rows):
    n = len(rows)
    c = sum(1 for r in rows if r.get("correct"))
    return {"correct": c, "total": n, "acc": (c / n if n else None)}


def _by(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r.get(key)].append(r)
    return {str(k): _acc(v) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}


def summarize(rows):
    """rows: list of Result dicts (a single method x backend, or pooled)."""
    lat = [r.get("latency_s") for r in rows if r.get("latency_s") is not None]
    intok = [r.get("input_tokens") for r in rows if r.get("input_tokens") is not None]
    outtok = [r.get("output_tokens") for r in rows if r.get("output_tokens") is not None]
    calls = [r.get("num_model_calls") for r in rows if r.get("num_model_calls") is not None]
    n = len(rows)
    return {
        "n": n,
        "overall": _acc(rows),                       # M1
        "by_task_type": _by(rows, "task_type"),
        "by_orig_num_cameras": _by(rows, "orig_num_cameras"),
        "by_source": _by(rows, "source"),
        "by_cap_answer_safe": _by(rows, "cap_answer_safe"),
        "latency_s": {                               # M2
            "p50": _pct(lat, 0.50), "p95": _pct(lat, 0.95),
            "mean": (sum(lat) / len(lat) if lat else None), "n": len(lat),
        },
        "tokens": {                                  # M3
            "input_mean": (sum(intok) / len(intok) if intok else None),
            "output_mean": (sum(outtok) / len(outtok) if outtok else None),
            "calls_mean": (sum(calls) / len(calls) if calls else None),
        },
        "abstain_rate": (sum(1 for r in rows if r.get("abstained")) / n if n else None),  # M4
        "errors": sum(1 for r in rows if r.get("error")),
    }


def summarize_by_method_backend(rows):
    """Group rows by (method, backend) and summarize each -> the headline table."""
    g = defaultdict(list)
    for r in rows:
        g[(r.get("method"), r.get("backend"))].append(r)
    return {f"{m}/{b}": summarize(v) for (m, b), v in sorted(g.items())}


# --- 4-pass mean +/- std (Table 1 + plot error bars) -------------------------
# A "pass" = one sampled generation at a fixed seed. Accuracy is computed WITHIN
# each pass, then we report mean +/- std over the passes (std = decoding variance).

def _pass_accs(rows, filt=None):
    """Per-pass accuracies over rows (optionally filtered), as a list."""
    g = defaultdict(lambda: [0, 0])  # pass_idx -> [correct, total]
    for r in rows:
        if filt is not None and not filt(r):
            continue
        pi = r.get("pass_idx")
        g[pi][1] += 1
        if r.get("correct"):
            g[pi][0] += 1
    accs = []
    for _, (c, n) in sorted(g.items(), key=lambda kv: str(kv[0])):
        if n:
            accs.append(c / n)
    return accs


def _mstd(accs):
    if not accs:
        return {"mean": None, "std": None, "n_passes": 0, "per_pass": []}
    return {"mean": _mean(accs), "std": (_pstdev(accs) if len(accs) > 1 else 0.0),
            "n_passes": len(accs), "per_pass": accs}


def summarize_passes(rows):
    """Like ``summarize`` but adds mean+/-std-over-passes for overall, by task_type,
    by orig_num_cameras, and the task_type x cameras cross-tab (Plot 4)."""
    tts = sorted({r.get("task_type") for r in rows}, key=str)
    cams = sorted({r.get("orig_num_cameras") for r in rows},
                  key=lambda x: (x is None, x))
    base = summarize(rows)
    base["overall_passes"] = _mstd(_pass_accs(rows))
    base["by_task_type_passes"] = {
        str(tt): _mstd(_pass_accs(rows, lambda r, tt=tt: r.get("task_type") == tt))
        for tt in tts}
    base["by_orig_num_cameras_passes"] = {
        str(c): _mstd(_pass_accs(rows, lambda r, c=c: r.get("orig_num_cameras") == c))
        for c in cams}
    base["by_task_camera_passes"] = {
        str(tt): {str(c): _mstd(_pass_accs(
            rows, lambda r, tt=tt, c=c: r.get("task_type") == tt
            and r.get("orig_num_cameras") == c)) for c in cams}
        for tt in tts}
    return base


def summarize_by_method_backend_passes(rows):
    """Group by (method, backend) and summarize_passes each -> Table 1 + plot data."""
    g = defaultdict(list)
    for r in rows:
        g[(r.get("method"), r.get("backend"))].append(r)
    return {f"{m}/{b}": summarize_passes(v) for (m, b), v in sorted(g.items())}


def format_summary(rows):
    out = []
    for key, s in summarize_by_method_backend(rows).items():
        ov = s["overall"]
        lat = s["latency_s"]
        out.append(f"\n=== {key} ===")
        acc = f'{ov["acc"]*100:.1f}%' if ov["acc"] is not None else "n/a"
        out.append(f'  overall: {ov["correct"]}/{ov["total"]} = {acc}   '
                   f'abstain={s["abstain_rate"]*100:.0f}%  errors={s["errors"]}'
                   if ov["total"] else "  (no rows)")
        if lat["p50"] is not None:
            out.append(f'  latency_s: p50={lat["p50"]:.1f} p95={lat["p95"]:.1f} mean={lat["mean"]:.1f}')
        if s["tokens"]["input_mean"] is not None:
            out.append(f'  tokens: in~{s["tokens"]["input_mean"]:.0f} out~{s["tokens"]["output_mean"]:.0f} '
                       f'calls~{s["tokens"]["calls_mean"]:.1f}')
        out.append("  by task_type: " + ", ".join(
            f'{k} {v["correct"]}/{v["total"]}' for k, v in s["by_task_type"].items()))
    return "\n".join(out)
