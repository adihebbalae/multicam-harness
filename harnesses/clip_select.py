# Ported from Wavy-Hec/CVBench bench/methods/clip_select.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
# Deliberate delta vs source: gen_clip_summaries references updated from the
# fork's `bench/gen_clip_summaries.py` module path to this repo's
# `scripts/gen_clip_summaries.py` (docstring, comment, and SystemExit hint).
"""CLIP-SELECTION harness (D3, the PRIMARY decision): choose WHICH of the K
independent clips a question actually needs, then spend the whole 64-frame
budget on the selected clips only. Two selector families, three method arms:

  summary_select_route   Per-clip QUESTION-AGNOSTIC text summaries (precomputed
                         once per unique clip by ``scripts/gen_clip_summaries.py``
                         and cached in a JSONL) are handed to the SAME backend
                         model in ONE text-only call, which returns either the
                         minimal set of clips needed or ALL. The ALL branch
                         emits a prompt BYTE-IDENTICAL to ``temporal_weighted``
                         (PREFIX/MARKER imported verbatim), so any delta vs the
                         baseline is attributable to the questions it pruned.

  summary_select_top1    Same summaries + selector, but forced to pick exactly
                         ONE clip. Only ~4% of CVBench questions name a single
                         video, so this arm is a DIAGNOSTIC of what hard
                         pruning costs, not a candidate winner.

  clip_select_top{m}     No summaries/LLM: score each clip by CLIP text-image
                         similarity between the question and a few uniform
                         thumbnails (max over thumbnails), keep the top-m.
                         The cheap no-LLM rival to the summary selector.

Selection is DETERMINISTIC (greedy text decode / CLIP scores) and runs once per
question: run_bench iterates all passes of a record consecutively, so the
keep-last-record ``_prepare`` cache reuses the pass-1 selection and the 4-pass
std isolates the answer stage.

Numbering invariant: questions and options reference clips by ORIGINAL index
("Video 2"), so a selected subset is always presented under its ORIGINAL
numbers (``MARKER_SEL``) — never renumbered "1..m".

Accounting convention: the headline Result latency/token fields (and
``num_model_calls=1``) cover the ANSWER call only, so cost columns stay
comparable with temporal_weighted. The selector call's cost + raw output — and
any live cache-miss summary cost (``live_summary_cost``) — live in
``Result.frame_alloc``; the amortized summary-cache cost lives in the cache
JSONL itself. Note the prepared ``frame_alloc`` dict is shared by ALL passes
of a question, so selector cost appears identically on each of its 4 rows —
dedup by question id before summing.
"""
import glob
import json
import os
import re

import numpy as np
from decord import VideoReader, cpu

from harnesses.base import Method, Result, result_fields
# ALL-branch prompt must be byte-identical to temporal_weighted:
# PREFIX/MARKER/SPLIT_DESC are imported verbatim, and the budget split over
# the selected clips reuses allocate_frames.
from harnesses.uniform import allocate_frames, PREFIX, MARKER, SPLIT_DESC
from dataloaders.qa_json import build_messages, video_paths
from evaluation.scoring import parse_choice, gt_choice, extract_think


def clip_scores(clip_bundle, text, pil_frames, batch=32):
    """CLIP/SigLIP cosine similarity between ``text`` and each PIL frame
    (higher = closer). Raw normalized cosine is monotonic in the model's own
    logit, so it ranks identically for both families. (Relocated from the
    retired frame-level adaptive ablation, whose result — motion −6.4 pts,
    CLIP-frames tie vs uniform — is archived in
    analysis/adaptive_frames_experiment.md §B.)"""
    import torch
    model, proc, device = clip_bundle
    # SigLIP was trained with fixed 64-token max_length padding; CLIP with
    # dynamic padding. Mismatched padding shifts SigLIP text embeddings.
    pad = "max_length" if getattr(model.config, "model_type", "") == "siglip" else True
    tok = proc(text=[text], return_tensors="pt", padding=pad, truncation=True).to(device)
    with torch.no_grad():
        t_emb = model.get_text_features(**tok)
        t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
        sims = []
        for i in range(0, len(pil_frames), batch):
            chunk = proc(images=pil_frames[i:i + batch], return_tensors="pt").to(device)
            i_emb = model.get_image_features(**chunk)
            i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
            sims.append((i_emb @ t_emb.T).squeeze(1).float().cpu().numpy())
    return np.concatenate(sims)

# --------------------------------------------------------------------------- #
# prompts                                                                      #
# --------------------------------------------------------------------------- #
# Question-agnostic per-clip summary (generated once per unique clip by
# scripts/gen_clip_summaries.py, which imports these two constants so the cache
# and any live fallback are guaranteed to use the same prompt).
SUMMARY_PROMPT_VER = "v1"
SUMMARY_NFRAMES = 16
SUMMARY_MAX_NEW_TOKENS = 384
SUMMARY_PROMPT = (
    "Describe this video clip so that someone who has NOT seen it could answer "
    "questions about it. Cover: (1) the setting / type of scene; (2) the people "
    "— how many, their appearance and clothing colors; (3) the main objects and "
    "any visible text, logos or scoreboards; (4) the actions and events in "
    "temporal order (beginning, middle, end); (5) the overall activity or topic. "
    "Be concrete and specific. Write 5-8 sentences. Do not speculate beyond what "
    "is visible.")

# Answer-stage prompt when only a SUBSET of the clips is shown. Keeps the
# clips' ORIGINAL numbers (questions/options refer to "Video k" by original
# index — renumbering would silently corrupt them).
PREFIX_SEL = (
    "The question below refers to {K} INDEPENDENT video clips (different, "
    "unrelated scenes), numbered Video 1 to Video {K}. Only {m} of them "
    "{is_are} shown here, keeping the ORIGINAL numbering: {shown}. The other "
    "clips were judged not needed for this question. A total budget of {budget} "
    "frames is split across the shown clips in proportion to each clip's "
    "duration; within each clip the frames are sampled UNIFORMLY across its "
    "full duration. A banner '=== Video k ===' precedes each shown clip's "
    "frames, and frames are labeled Frame1, Frame2, ... within that clip. If "
    "the question requires a Video that is not shown, answer as best you can "
    "from what is shown.")

MARKER_SEL = ("=== Video {orig} — {n_k} frames sampled uniformly across its "
              "full duration (~{dur_s:.0f}s) ===")

SELECTOR_ROUTE_PROMPT = (
    "You will decide which video clips are needed to answer a multiple-choice "
    "question. Below are short text summaries of {K} INDEPENDENT video clips "
    "(different, unrelated scenes), numbered Video 1 to Video {K}, followed by "
    "the question and its options.\n\n{summaries}\n\nQuestion:\n{question}\n\n"
    "Options:\n{options}\n\nWhich videos must be watched to answer this "
    "question? If the question or its options compare videos, ask 'which "
    "video...', or mention both/all/neither videos, then ALL videos are "
    "needed. If it asks only about one specific video or about content that "
    "clearly appears in only some of them, pick the minimal set.\n"
    "Do NOT answer the question itself. Reply with exactly one line, either:\n"
    "SELECTED: <comma-separated video numbers>\n"
    "or\n"
    "SELECTED: ALL")

SELECTOR_TOP1_PROMPT = (
    "You will pick the single most relevant video clip for a multiple-choice "
    "question. Below are short text summaries of {K} INDEPENDENT video clips "
    "(different, unrelated scenes), numbered Video 1 to Video {K}, followed by "
    "the question and its options.\n\n{summaries}\n\nQuestion:\n{question}\n\n"
    "Options:\n{options}\n\nWhich single video is the most relevant to this "
    "question? Do NOT answer the question itself. Reply with exactly one "
    "line:\nBEST: <one video number>")


# --------------------------------------------------------------------------- #
# selection-output parsing                                                     #
# --------------------------------------------------------------------------- #
def _strip_think(text):
    """Drop <think>...</think> spans (and an unclosed <think> tail from a
    token-capped reply) so chatter never masquerades as the decision."""
    t = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    return re.sub(r"<think>.*\Z", " ", t, flags=re.DOTALL)


def _numbers_in_line(line, K):
    """All in-range clip numbers on one reply line, expanding 'a-b' ranges
    (so 'SELECTED: 1, 2, and 4' -> {1,2,4} and 'SELECTED: 1-3' -> {1,2,3})."""
    nums = set()
    for a, b in re.findall(r"(\d+)\s*[-–]\s*(\d+)", line):
        a, b = int(a), int(b)
        nums.update(range(min(a, b), max(a, b) + 1))
    nums.update(int(n) for n in re.findall(r"\d+", line))
    return sorted(n for n in nums if 1 <= n <= K)


def parse_selection(text, K, mode):
    """Selector reply -> (sorted 1-based indices, fallback_reason|None).

    The structured line is parsed from think-stripped text first (last match
    wins), then from the raw text. Route mode has NO mentions fallback: any
    unparseable reply degrades to ALL, so on a parse failure the route arm can
    never present fewer clips than temporal_weighted — the baseline, not below
    it. Top1 falls back to the first 'Video k' mention outside <think>, then
    to clip 1.
    """
    everything = list(range(1, K + 1))
    raw = text or ""
    stripped = _strip_think(raw)
    key = "SELECTED" if mode == "route" else "BEST"
    for t in (stripped, raw):
        # line-bounded capture: a rationale on the NEXT line can't leak digits
        hits = re.findall(rf"{key}\s*:(.*)", t, flags=re.IGNORECASE)
        if not hits:
            continue
        last = hits[-1]
        if mode == "route" and re.search(r"\bALL\b", last, flags=re.IGNORECASE):
            return everything, None
        idx = _numbers_in_line(last, K)
        if idx:
            return ([idx[0]], None) if mode == "top1" else (idx, None)
    if mode == "top1":
        mentioned = [int(n) for n in re.findall(r"[Vv]ideo\s*(\d+)", stripped)
                     if 1 <= int(n) <= K]
        if mentioned:
            return [mentioned[0]], "fallback:video_mentions"
        return [1], "fallback:first"
    return everything, "fallback:all"


# --------------------------------------------------------------------------- #
# shared answer-stage presentation                                             #
# --------------------------------------------------------------------------- #
def present_selected(paths, durs, ncaps, sel_idx, budget, floor, scaffold,
                     weights=None):
    """Build the answer-stage content over the SELECTED clips.

    ``sel_idx``: sorted ORIGINAL 1-based indices of the kept clips. When all K
    are kept the prompt is byte-identical to temporal_weighted (duration split);
    a strict subset uses PREFIX_SEL/MARKER_SEL under the original numbers.
    ``weights``: optional relevance weights over the selected clips (v2 hook;
    None -> duration weighting, the v1 default, so the selection lever is
    measured in isolation). Returns (content, nframes_per_selected_clip).
    """
    K = len(paths)
    m = len(sel_idx)
    sel_durs = [durs[i - 1] for i in sel_idx]
    sel_caps = [ncaps[i - 1] for i in sel_idx]
    w = list(weights) if weights is not None else sel_durs
    nframes = allocate_frames(w, budget, floor=floor, caps=sel_caps)

    if m == K:
        split = SPLIT_DESC["duration"].format(K=K)
        content = [{"type": "text", "text": PREFIX.format(K=K, budget=budget,
                                                          split=split)}]
        for k, (vp, n_k, d) in enumerate(zip(paths, nframes, sel_durs), 1):
            content.append({"type": "text",
                            "text": MARKER.format(k=k, K=K, n_k=n_k,
                                                  dur_s=(d if d is not None else 0.0))})
            content.append({"type": "video", "video": vp, "nframes": n_k})
    else:
        shown = ", ".join(f"Video {i}" for i in sel_idx)
        content = [{"type": "text", "text": PREFIX_SEL.format(
            K=K, m=m, is_are=("is" if m == 1 else "are"), shown=shown,
            budget=budget)}]
        for i, n_k, d in zip(sel_idx, nframes, sel_durs):
            content.append({"type": "text",
                            "text": MARKER_SEL.format(orig=i, n_k=n_k,
                                                      dur_s=(d if d is not None else 0.0))})
            content.append({"type": "video", "video": paths[i - 1], "nframes": n_k})
    content.append({"type": "text", "text": scaffold})
    return content, nframes


def _clip_meta(vp):
    """(duration_s, n_decoded) via decord; (None, None) on decode failure."""
    try:
        vr = VideoReader(vp, ctx=cpu(0), num_threads=1)
        n = len(vr)
        fps = float(vr.get_avg_fps())
        return (n / fps if fps > 0 else None), n
    except Exception:
        return None, None


def _rel_keys(rec):
    """The record's raw video_1..4 values, in video_paths() order — the summary
    cache keys (they match analysis/cvbench_all_videos.txt lines exactly)."""
    return [rec.get(f"video_{i}") for i in range(1, 5) if rec.get(f"video_{i}")]


def summary_cache_files(path_or_glob):
    """The cache file set for a path: the base JSONL (if present) PLUS its
    generator ``_shard*.jsonl`` siblings — the same union the generator's
    resume and --check use, so 'check passed' implies 'eval will see it'.
    A glob is used as-is; an explicit _shardN path is used alone."""
    if any(ch in path_or_glob for ch in "*?["):
        return sorted(glob.glob(path_or_glob))
    paths = [path_or_glob] if os.path.exists(path_or_glob) else []
    if "_shard" not in os.path.basename(path_or_glob):
        base = path_or_glob[:-6] if path_or_glob.endswith(".jsonl") else path_or_glob
        paths += sorted(p for p in glob.glob(base + "_shard*.jsonl") if p not in paths)
    return paths


def load_summary_cache(path_or_glob):
    """{rel_video_path: summary} over summary_cache_files().

    Rows with an ``error``, an empty ``summary``, or a prompt_ver other than
    the current SUMMARY_PROMPT_VER are skipped (stale-version rows count as
    missing, so a prompt bump regenerates instead of silently mixing versions);
    last row per video wins.
    """
    paths = summary_cache_files(path_or_glob)
    cache = {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if (row.get("summary") and not row.get("error")
                        and row.get("prompt_ver", SUMMARY_PROMPT_VER) == SUMMARY_PROMPT_VER):
                    cache[row["video"]] = row["summary"]
    return cache, paths


# --------------------------------------------------------------------------- #
class SummarySelectMethod(Method):
    """summarize -> select -> answer (modes: "route" | "top1")."""
    name = "summary_select"

    def __init__(self, backend, summaries_path, mode="route", budget=64, floor=2,
                 sel_max_new_tokens=512, nframes=8, max_new_tokens=8192,
                 temperature=0.0):
        super().__init__(backend, nframes=nframes, max_new_tokens=max_new_tokens,
                         temperature=temperature)
        assert mode in ("route", "top1")
        self.mode = mode
        self.budget = budget
        self.floor = floor
        self.sel_max_new_tokens = sel_max_new_tokens
        self.name = f"summary_select_{mode}"
        self.summaries_path = summaries_path
        self.summaries, found = load_summary_cache(summaries_path)
        if not self.summaries:
            raise SystemExit(
                f"summary_select_{mode}: no clip summaries found at "
                f"'{summaries_path}' (files matched: {found}). Generate the cache "
                f"first:\n  python scripts/gen_clip_summaries.py "
                f"--subset <subset.json> --backend internvl3 "
                f"--out {summaries_path}")
        self._cache = {}   # rec id -> prepared tuple; last rec only

    def _summary_for(self, rel, abspath):
        """(summary_text, source, GenOut|None) — cache hit, or a live one-off.

        Live summaries are NOT persisted (concurrent Slurm shards must not
        write the cache); keep the cache complete before sharded runs. The
        live call's GenOut is returned so its cost lands in frame_alloc.
        """
        s = self.summaries.get(rel)
        if s:
            return s, "cache", None
        try:
            g = self.backend.generate(
                [{"role": "user", "content": [
                    {"type": "video", "video": abspath, "nframes": SUMMARY_NFRAMES},
                    {"type": "text", "text": SUMMARY_PROMPT}]}],
                max_new_tokens=SUMMARY_MAX_NEW_TOKENS, seed=None, temperature=0.0)
            return g.text.strip(), "live", g
        except Exception as e:
            return f"(no summary available: {type(e).__name__})", "error", None

    def _prepare(self, rec, video_root):
        key = rec.get("id")
        if key in self._cache:
            return self._cache[key]
        base_msgs, yn = build_messages(rec, video_root, self.nframes, no_video=True)
        scaffold = base_msgs[0]["content"][0]["text"]
        paths = video_paths(rec, video_root)
        rels = _rel_keys(rec)
        K = len(paths)
        durs, ncaps = [], []
        for vp in paths:
            d, n = _clip_meta(vp)
            durs.append(d)
            ncaps.append(n)

        sel_meta = {"mode": self.mode, "summaries_path": self.summaries_path}
        if K <= 1:                                  # nothing to select
            sel_idx, fallback = list(range(1, K + 1)), "single_clip"
        else:
            sums, sources = [], []
            live = {"calls": 0, "latency_s": 0.0, "input_tokens": 0, "output_tokens": 0}
            for rel, vp in zip(rels, paths):
                s, src, g_live = self._summary_for(rel, vp)
                sums.append(s)
                sources.append(src)
                if g_live is not None:
                    live["calls"] += 1
                    live["latency_s"] += g_live.latency_s
                    live["input_tokens"] += g_live.input_tokens
                    live["output_tokens"] += g_live.output_tokens
            if live["calls"]:
                live["latency_s"] = round(live["latency_s"], 3)
                sel_meta["live_summary_cost"] = live
            summaries_block = "\n\n".join(
                f"Video {i} summary:\n{s}" for i, s in enumerate(sums, 1))
            tmpl = SELECTOR_ROUTE_PROMPT if self.mode == "route" else SELECTOR_TOP1_PROMPT
            sel_prompt = tmpl.format(K=K, summaries=summaries_block,
                                     question=rec["question"],
                                     options="\n".join(rec["options"]))
            # Deterministic (greedy) selection, once per question.
            g_sel = self.backend.generate(
                [{"role": "user", "content": [{"type": "text", "text": sel_prompt}]}],
                max_new_tokens=self.sel_max_new_tokens, seed=None, temperature=0.0)
            sel_idx, fallback = parse_selection(g_sel.text, K, self.mode)
            sel_meta.update(
                selector_output=g_sel.text[-1200:],
                selector_latency_s=round(g_sel.latency_s, 3),
                selector_input_tokens=g_sel.input_tokens,
                selector_output_tokens=g_sel.output_tokens,
                summary_source=sources,
                summaries_trunc=[s[:200] for s in sums])

        content, nframes = present_selected(paths, durs, ncaps, sel_idx,
                                            self.budget, self.floor, scaffold)
        gold = gt_choice(rec["answer"], yn)
        alloc_meta = {
            "budget": self.budget,
            "floor": self.floor,
            "K": K,
            "selected": sel_idx,
            "selection_fallback": fallback,
            "durations_s": [round(d, 2) if d is not None else None for d in durs],
            "nframes": nframes,
            "n_decoded": ncaps,
            "sum_nframes": sum(nframes),
            **sel_meta,
        }
        self._cache = {key: (content, yn, gold, alloc_meta)}   # keep only last rec
        return self._cache[key]

    def answer(self, rec, video_root, seed=None) -> Result:
        f = result_fields(rec)
        try:
            content, yn, gold, alloc_meta = self._prepare(rec, video_root)
        except Exception as e:
            gold = gt_choice(rec["answer"], all(o.strip().strip(".").lower() in ("yes", "no")
                                                for o in rec["options"]))
            return Result(**f, method=self.name, backend=self.backend.name,
                          prediction="", gold=gold, correct=False, abstained=True,
                          seed=seed, temperature=self.temperature, num_model_calls=1,
                          error=f"prepare:{type(e).__name__}: {e}")
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
                seed=seed, temperature=self.temperature, latency_s=g.latency_s,
                input_tokens=g.input_tokens, video_tokens=g.video_tokens,
                output_tokens=g.output_tokens, num_model_calls=1,
                response_text=g.text, think=extract_think(g.text),
                frame_alloc=alloc_meta,
            )
        except Exception as e:  # keep the sweep alive; record the failure
            return Result(**f, method=self.name, backend=self.backend.name,
                          prediction="", gold=gold, correct=False, abstained=True,
                          seed=seed, temperature=self.temperature, num_model_calls=1,
                          error=f"{type(e).__name__}: {e}", frame_alloc=alloc_meta)


# --------------------------------------------------------------------------- #
class ClipScoreSelectMethod(Method):
    """CLIP-scored clip selection: question-vs-thumbnails similarity, top-m."""
    name = "clip_select"

    def __init__(self, backend, top_m=1, thumbs=8,
                 clip_model="openai/clip-vit-base-patch32", budget=64, floor=2,
                 nframes=8, max_new_tokens=8192, temperature=0.0,
                 stat="max", name=None):
        super().__init__(backend, nframes=nframes, max_new_tokens=max_new_tokens,
                         temperature=temperature)
        self.top_m = int(top_m)
        self.thumbs = thumbs
        self.clip_model_name = clip_model
        self.budget = budget
        self.floor = floor
        if stat not in ("max", "mean"):
            raise ValueError(f"stat must be 'max' or 'mean', got {stat!r}")
        self.stat = stat
        # The CLI method string (e.g. clip_select_siglip_top2) is passed in as
        # name so rows/resume keys distinguish scorer variants; the bare
        # default keeps old clip_select_top1 rows compatible.
        self.name = name or f"clip_select_top{top_m}"
        self._cache = {}
        self._clip = None

    def _ensure_clip(self):
        if self._clip is None:
            # AutoModel resolves CLIPModel for openai/clip-* and SiglipModel
            # for google/siglip-*; both expose get_{text,image}_features.
            from transformers import AutoModel, AutoProcessor
            dev = getattr(self.backend, "device", "cuda:0")
            model = AutoModel.from_pretrained(self.clip_model_name).to(dev).eval()
            proc = AutoProcessor.from_pretrained(self.clip_model_name)
            self._clip = (model, proc, dev)
        return self._clip

    def _score_clip(self, vp, question):
        """(max_sim, mean_sim) of the question vs ``thumbs`` uniform thumbnails."""
        from PIL import Image
        vr = VideoReader(vp, ctx=cpu(0), num_threads=1)
        n_total = len(vr)
        if n_total <= 0:
            raise ValueError("empty clip")
        idx = sorted({min(n_total - 1, int((j + 0.5) * n_total / self.thumbs))
                      for j in range(self.thumbs)})
        frames = vr.get_batch(idx).asnumpy()
        pil = [Image.fromarray(fr).convert("RGB") for fr in frames]
        s = clip_scores(self._ensure_clip(), question, pil)
        return float(np.max(s)), float(np.mean(s))

    def _prepare(self, rec, video_root):
        key = rec.get("id")
        if key in self._cache:
            return self._cache[key]
        base_msgs, yn = build_messages(rec, video_root, self.nframes, no_video=True)
        scaffold = base_msgs[0]["content"][0]["text"]
        paths = video_paths(rec, video_root)
        K = len(paths)
        durs, ncaps = [], []
        for vp in paths:
            d, n = _clip_meta(vp)
            durs.append(d)
            ncaps.append(n)

        fallback = None
        smax, smean = [], []
        for vp in paths:
            try:
                mx, mn = self._score_clip(vp, rec.get("question", ""))
            except Exception:
                mx, mn = float("-inf"), float("-inf")
            smax.append(mx)
            smean.append(mn)
        m = min(self.top_m, K)
        ranking = smax if self.stat == "max" else smean
        if all(x == float("-inf") for x in ranking):  # scoring failed everywhere
            sel_idx, fallback = list(range(1, K + 1)), "fallback:score_error_all"
        else:
            order = sorted(range(K), key=lambda i: -ranking[i])
            sel_idx = sorted(i + 1 for i in order[:m])

        content, nframes = present_selected(paths, durs, ncaps, sel_idx,
                                            self.budget, self.floor, scaffold)
        gold = gt_choice(rec["answer"], yn)
        alloc_meta = {
            "budget": self.budget,
            "floor": self.floor,
            "K": K,
            "selected": sel_idx,
            "selection_fallback": fallback,
            "top_m": self.top_m,
            "thumbs": self.thumbs,
            "clip_model": self.clip_model_name,
            "sel_stat": self.stat,
            "clip_scores_max": [None if x == float("-inf") else round(x, 4) for x in smax],
            "clip_scores_mean": [None if x == float("-inf") else round(x, 4) for x in smean],
            "durations_s": [round(d, 2) if d is not None else None for d in durs],
            "nframes": nframes,
            "n_decoded": ncaps,
            "sum_nframes": sum(nframes),
        }
        self._cache = {key: (content, yn, gold, alloc_meta)}
        return self._cache[key]

    # identical answer skeleton to SummarySelectMethod / temporal_weighted
    answer = SummarySelectMethod.answer
