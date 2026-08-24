# Ported from Wavy-Hec/CVBench bench/methods/option_union.py @ 967fe27;
# the _per_option_tau small-n caveat additionally ported from
# @ 8b4fbd117b6084dafb743eb30b5298e9a1ba91e3.
"""Option-aware subsampling (Follow-up 1) and tool-based query search (Follow-up 2).

Three new arms, all reducing the visual input BEFORE the answer call:

  frame_select[_<scorer>]_optu   Follow-up 1, frame level. Every answer option is
                                 a separate retrieval query; a candidate frame is
                                 KEPT when it passes that option's similarity
                                 threshold, and the kept sets are UNIONED across
                                 options ("any option would retrieve it"). The
                                 union is capped at the frame budget; it is NOT
                                 backfilled when small — showing fewer, more
                                 relevant frames is the lever under test.

  clip_select[_<scorer>]_optu    Follow-up 1, clip level, across the multi-video
                                 streams. Scorer 'viclip' embeds each clip's 8
                                 frames JOINTLY (video-native, motion-aware);
                                 CLIP/SigLIP fall back to max-over-thumbnails.
                                 A clip is kept when any option's threshold
                                 passes; each option always contributes at least
                                 its best clip, so the union is never empty.

  query_search[_<scorer>]        Follow-up 2. The backend first writes short
                                 visual search phrases from Question + Options
                                 (one deterministic text-only call, cost recorded
                                 in frame_alloc), then CLIP/SigLIP retrieves the
                                 top-budget frames matching any phrase, and the
                                 answer call runs on those.

Budget parity: --budget 0 (the default when the flag is omitted) means "match
the sequential arm" — nframes x K total frames per question. Frame COUNT parity
is not token parity, though: the sequential arm feeds video frames, these arms
feed still images, and the two are tokenized differently. On Qwen backends the
selected frames are therefore sent UNRESIZED with per-item pixel caps mirroring
qwen_vl_utils' video caps (VIDEO_MIN/MAX_TOKEN_NUM), so each image is sized
exactly like a video frame; on InternVL an image is tiled by max_tiles while a
video frame is pinned to 1 tile, so these arms REFUSE to construct when
internvl_max_tiles != 1 (run the montage arm's max_tiles>1 legs separately).
Rows record which parity mode applied (frame_alloc.img_token_parity); compare
input_tokens/video_tokens across arms before reading any difference as an
architecture effect. Thresholds: --sel-tau > 0 is an absolute cosine cutoff
(scorer-specific — calibrate with analysis/calibrate_union_tau.py); the default
--sel-tau 0 uses the per-option --sel-tau-q quantile of that option's own
scores, which is scale-free across CLIP/SigLIP/ViCLIP.

This module is deliberately a NEW file: the queued similarity gate (job 86147)
imports clip_select.py and friends at job start, so everything here reuses the
frozen modules by import/subclass only. The duplicated _prepare scaffolding
(vs FrameSelectMethod) is the price of that freeze; fold the shared parts back
into clip_select.py only after the gate has run.
"""
import re

import numpy as np
from decord import VideoReader, cpu

from inprocess.harnesses.base import require_video_record
from inprocess.harnesses.clip_select import (
    FrameSelectMethod, ClipScoreSelectMethod, clip_scores, option_texts,
    present_selected, _clip_meta, FRAME_MARKER)
from inprocess.dataloaders.qa_json import build_messages, letters_of, video_paths
from inprocess.evaluation.scoring import gt_choice

OPTION_UNION_PREFIX = (
    "The question below refers to {K} INDEPENDENT video clips (different, "
    "unrelated scenes), numbered Video 1 to Video {K}. From across ALL of "
    "them, {n} frames were selected: a frame is kept when it is visually "
    "similar to at least ONE of the answer options (each option acts as a "
    "separate retrieval query and the kept sets are unioned), plus each "
    "clip's single best frame so no clip goes unrepresented. They are shown "
    "below grouped by their source Video and in temporal order within each "
    "Video; a banner '=== Video k ===' precedes each Video's frames. Reason "
    "over the shown frames to answer.")

QUERY_SEARCH_PREFIX = (
    "The question below refers to {K} INDEPENDENT video clips (different, "
    "unrelated scenes), numbered Video 1 to Video {K}. To choose what to "
    "show, {nq} short visual search phrases were first written from the "
    "question and its answer options, and the {n} frames most similar to any "
    "of those phrases were retrieved from across ALL clips (one shared "
    "budget chosen globally, plus each clip's single best frame). They are "
    "shown below grouped by their source Video and in temporal order within "
    "each Video; a banner '=== Video k ===' precedes each Video's frames. "
    "Reason over the shown frames to answer.")

QUERY_GEN_PROMPT = (
    "You will help retrieve relevant video frames for a multiple-choice "
    "question about {K} independent video clips. Read the question and its "
    "answer options, then write up to {n} short SEARCH PHRASES describing "
    "concrete visual content to look for (objects, people, clothing colors, "
    "actions, scene types). One phrase per line, at most 12 words each, no "
    "numbering. Do NOT answer the question and do NOT mention video "
    "numbers.\n\nQuestion:\n{question}\n\nOptions:\n{options}\n\n"
    "Search phrases:")

_QLINE_STRIP = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s*")

# qwen_vl_utils sizes a video frame into [VIDEO_MIN_TOKEN_NUM,
# VIDEO_MAX_TOKEN_NUM] visual tokens; one token covers (2*patch)^2 pixels.
_QWEN_VIDEO_MIN_TOK, _QWEN_VIDEO_MAX_TOK = 128, 768


def _check_token_parity(backend, name):
    """Refuse configurations whose image tokenization cannot match the
    sequential arm's video tokenization. InternVL tiles an image by max_tiles
    but pins a video frame to 1 tile, so max_tiles > 1 silently inflates these
    arms' visual budget (the montage arm legitimately uses max_tiles > 1 —
    run those legs separately). Returns the parity mode stamped on rows."""
    if type(backend).__name__ == "InternVL3Backend":
        tiles = getattr(backend, "max_tiles", 1)
        if tiles != 1:
            raise SystemExit(
                f"{name}: INTERNVL_MAX_TILES={tiles} tokenizes selected-frame "
                "images at up to {tiles}+1 tiles while sequential video frames "
                "stay at 1 tile — the same-budget comparison would be a token "
                "artifact. Launch these arms in their own leg with "
                "INTERNVL_MAX_TILES=1.")
        return "internvl_tiles1"
    if getattr(backend, "patch_size", None):
        return "qwen_video_caps"
    return "cell_px"


def _frame_content(im, backend, resize):
    """One selected frame as a content item whose visual-token cost matches a
    sequential-arm video frame on this backend. Qwen: send the frame UNRESIZED
    with per-item pixel caps mirroring qwen_vl_utils' video caps, so the
    processor sizes it exactly like a video frame (resizing to 448px first
    would re-create the montage-cell token deficit). Others: the 448px cell."""
    ps = getattr(backend, "patch_size", None)
    if ps:
        f = 2 * ps
        return {"type": "image", "image": im,
                "min_pixels": _QWEN_VIDEO_MIN_TOK * f * f,
                "max_pixels": _QWEN_VIDEO_MAX_TOK * f * f}
    return {"type": "image", "image": resize(im)}


def _per_option_tau(scores_2d, tau, tau_q):
    """One threshold per option (column). tau > 0: the absolute cutoff for all;
    else: each option's own tau_q quantile, so 'passing' means 'in this
    option's top (1-tau_q) fraction' regardless of the scorer's cosine scale.

    Small-n caveat: np.quantile interpolates, so with few rows the q=0.85
    default lands BETWEEN the top scores — over the CLIP-level arm's K clips,
    exactly 1 passes per option for K <= 7 and 2 for K = 8-13. At typical K
    the clip-level 'union' therefore degenerates to each option's best clip
    (the backstop's set), i.e. effectively top-1 per option, NOT a spread of
    passing clips. The frame-level arms pool K x candidates rows (~128+),
    where the quantile keeps the intended ~(1-q) fraction. The straight top-K
    selection discussed for the segment pipeline lives in segment_select
    (``--segments-keep`` / 0 = budget-derived auto K), not here — this arm
    stays the Follow-up-1 threshold-union record."""
    if tau > 0:
        return [float(tau)] * scores_2d.shape[1]
    return [float(np.quantile(scores_2d[:, j], tau_q))
            for j in range(scores_2d.shape[1])]


def _floor_then_fill(pool_videos, order, eligible, budget, floor):
    """Selection shared by the frame-level arms: reserve each clip's best
    ``floor`` frames first (starvation guard, same rationale as
    FrameSelectMethod), then fill the remaining budget best-first from
    ``eligible`` (None = every candidate). Returns kept indices, best-first."""
    keep, per_clip, picked = [], {}, set()
    if floor:
        for j in order:
            v = pool_videos[j]
            if per_clip.get(v, 0) < floor:
                keep.append(j)
                picked.add(j)
                per_clip[v] = per_clip.get(v, 0) + 1
    for j in order:
        if len(keep) >= budget:
            break
        if j in picked or (eligible is not None and j not in eligible):
            continue
        keep.append(j)
        picked.add(j)
    return keep[:budget]


class OptionUnionFrameSelect(FrameSelectMethod):
    """frame_select[_<scorer>]_optu — union-of-per-option-thresholds selection."""

    def __init__(self, backend, tau=0.0, tau_q=0.85, **kw):
        super().__init__(backend, **kw)
        self.tau = float(tau)
        self.tau_q = float(tau_q)
        self.img_token_parity = _check_token_parity(backend, self.name)

    def _prepare(self, rec, video_root):
        key = rec.get("id")
        if key in self._cache:
            return self._cache[key]
        require_video_record(rec, self.name)
        base_msgs, yn = build_messages(rec, video_root, self.nframes, no_video=True,
                                       reasoning=self.reasoning)
        scaffold = base_msgs[0]["content"][0]["text"]
        paths = video_paths(rec, video_root)
        K = len(paths)
        budget_eff = self.budget if self.budget > 0 else self.nframes * K
        if budget_eff < K:
            # the prompt promises every clip its best frame; below K frames
            # that is unkeepable — same policy as the montage arm's budget raise
            raise ValueError(
                f"budget {budget_eff} < {K} views: cannot keep one frame per "
                "clip as the prompt promises — raise --budget (or use 0 = "
                "matched nframes x K)")

        pool, per_video_cand = [], []
        for i, vp in enumerate(paths, 1):
            cands = self._candidates(vp)          # fail-loud decode, inherited
            per_video_cand.append(len(cands))
            for t, im in cands:
                pool.append((i, t, im))
        if not pool:
            raise FileNotFoundError("no candidate frames from any clip")

        opts = [o for o in option_texts(rec) if o]
        qmode = "options" if opts else "question"
        queries = opts if opts else [rec.get("question", "")]
        # scorer failures raise (inherited policy): no silent uniform fallback
        S = clip_scores(self._ensure_clip(), queries, [im for _, _, im in pool])
        taus = _per_option_tau(S, self.tau, self.tau_q)
        passing = S >= np.array(taus)[None, :]     # [n_frames, n_opts]
        in_union = passing.any(axis=1)
        # each option always retrieves its own best frame (mirrors the clip
        # arm), so an absolute tau above every score can never empty the union
        backstop_fired = False
        for j in range(S.shape[1]):
            jbest = int(np.argmax(S[:, j]))
            if not in_union[jbest]:
                in_union[jbest] = True
                backstop_fired = True
        frame_best = S.max(axis=1)
        order = sorted(range(len(pool)), key=lambda j: -float(frame_best[j]))
        floor = max(0, min(self.floor, budget_eff // K)) if K else 0
        eligible = {j for j in range(len(pool)) if in_union[j]}
        keep = _floor_then_fill([p[0] for p in pool], order, eligible,
                                budget_eff, floor)
        if not keep:
            raise ValueError(
                f"{self.name}: empty selection (sel_tau={self.tau}, floor="
                f"{floor}) — a zero-image answer call must never run as a "
                "vision row")
        floor_only = [j for j in keep if j not in eligible]
        n_union_kept = len(keep) - len(floor_only)

        sel = [(pool[j][0], pool[j][1], pool[j][2], float(frame_best[j]))
               for j in keep]
        by_video = {}
        for vidx, t, im, sc in sel:
            by_video.setdefault(vidx, []).append((t, im, sc))
        for v in by_video:
            by_video[v].sort(key=lambda x: (x[0] if x[0] is not None else 0.0))

        content = [{"type": "text", "text": OPTION_UNION_PREFIX.format(
            K=K, n=len(sel))}]
        for v in sorted(by_video):
            content.append({"type": "text",
                            "text": FRAME_MARKER.format(orig=v, cnt=len(by_video[v]))})
            for t, im, sc in by_video[v]:
                content.append(_frame_content(im, self.backend, self._resize))
        content.append({"type": "text", "text": scaffold})

        letters = letters_of(rec)
        gold = gt_choice(rec["answer"], yn, letters=letters)
        alloc_meta = {
            "mode": "option_union",
            "budget": budget_eff,
            "budget_matched": self.budget <= 0,
            "img_token_parity": self.img_token_parity,
            "K": K,
            "candidates_per_video": self.candidates_per_video,
            "n_candidates": len(pool),
            "n_selected": len(sel),
            "union_size": int(in_union.sum()),
            "union_backstop": backstop_fired,
            "union_truncated": n_union_kept < int(in_union.sum()),
            "n_floor_only": len(floor_only),
            "floor": floor,
            "tau_mode": "abs" if self.tau > 0 else f"q{self.tau_q}",
            "tau_per_option": [round(t, 4) for t in taus],
            "pass_count_per_option": [int(passing[:, j].sum())
                                      for j in range(passing.shape[1])],
            "selected_per_video": {v: len(by_video[v]) for v in sorted(by_video)},
            "candidates_decoded_per_video": per_video_cand,
            "clip_model": self.clip_model_name,
            "query_mode": qmode,
            "n_query_texts": len(queries),
            "selection_fallback": None,
        }
        self._cache = {key: (content, yn, gold, alloc_meta)}
        return self._cache[key]


def _decode_thumbs(vp, n, exact):
    """n uniform PIL thumbnails; fail-loud like the sibling arms. ``exact``
    keeps duplicates so short clips still yield exactly n (ViCLIP's tube needs
    a fixed length); otherwise duplicates collapse (the CLIP-thumbs contract)."""
    from PIL import Image
    try:
        vr = VideoReader(vp, ctx=cpu(0), num_threads=1)
        n_total = len(vr)
    except Exception as e:
        raise FileNotFoundError(f"unreadable clip {vp} ({type(e).__name__}) "
                                "— check --video-root") from e
    if n_total <= 0:
        raise FileNotFoundError(f"unreadable clip {vp} (0 frames)")
    idx = [min(n_total - 1, int((j + 0.5) * n_total / n)) for j in range(n)]
    if not exact:
        idx = sorted(set(idx))
    frames = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(fr).convert("RGB") for fr in frames]


class OptionUnionClipSelect(ClipScoreSelectMethod):
    """clip_select[_<scorer>]_optu — per-option thresholds over whole clips,
    union across options, ViCLIP (joint 8-frame embedding) or CLIP/SigLIP."""

    def __init__(self, backend, scorer="openai/clip-vit-base-patch32",
                 tau=0.0, tau_q=0.85, **kw):
        self.use_viclip = (scorer == "viclip")
        super().__init__(backend,
                         clip_model=("viclip" if self.use_viclip else scorer),
                         query="options", **kw)
        self.tau = float(tau)
        self.tau_q = float(tau_q)
        # presented as whole video items (present_selected), so token parity
        # with sequential holds on both backends without special casing

    def _option_matrix(self, paths, queries):
        """[K, n_opts] similarity — ViCLIP: one joint embedding per clip;
        CLIP/SigLIP: max over thumbnails, the same reduction the gate uses."""
        rows = []
        if self.use_viclip:
            from inprocess.harnesses.viclip_scorer import viclip_option_scores, VICLIP_NFRAMES
            dev = getattr(self.backend, "device", "cuda:0")
            text_cache = {}      # encode each option once per question, not per clip
            for vp in paths:
                pil = _decode_thumbs(vp, VICLIP_NFRAMES, exact=True)
                rows.append(viclip_option_scores(pil, queries, device=dev,
                                                 text_cache=text_cache))
        else:
            bundle = self._ensure_clip()
            for vp in paths:
                pil = _decode_thumbs(vp, self.thumbs, exact=False)
                s = clip_scores(bundle, queries, pil)   # [thumbs, n_opts]
                rows.append(s.max(axis=0))
        return np.stack(rows)

    def _prepare(self, rec, video_root):
        key = rec.get("id")
        if key in self._cache:
            return self._cache[key]
        require_video_record(rec, self.name)
        base_msgs, yn = build_messages(rec, video_root, self.nframes, no_video=True,
                                       reasoning=self.reasoning)
        scaffold = base_msgs[0]["content"][0]["text"]
        paths = video_paths(rec, video_root)
        K = len(paths)
        budget_eff = self.budget if self.budget > 0 else self.nframes * K
        durs, ncaps = [], []
        for vp in paths:
            d, n = _clip_meta(vp)
            durs.append(d)
            ncaps.append(n)

        opts = [o for o in option_texts(rec) if o]
        qmode = "options" if opts else "question"
        queries = opts if opts else [rec.get("question", "")]
        if K <= 1:
            sel_idx, M, taus, pass_counts = list(range(1, K + 1)), None, [], []
        else:
            M = self._option_matrix(paths, queries)        # [K, n_opts]
            taus = _per_option_tau(M, self.tau, self.tau_q)
            selected = set()
            pass_counts = []
            for j in range(M.shape[1]):
                passing = {i for i in range(K) if M[i, j] >= taus[j]}
                passing.add(int(np.argmax(M[:, j])))   # each option's best clip
                pass_counts.append(len(passing))
                selected |= passing
            sel_idx = sorted(i + 1 for i in selected)
        if budget_eff < len(sel_idx):
            raise ValueError(
                f"budget {budget_eff} < {len(sel_idx)} selected clips: cannot "
                "give every selected clip a frame — raise --budget (or use "
                "0 = matched nframes x K)")

        content, nframes = present_selected(paths, durs, ncaps, sel_idx,
                                            budget_eff, self.floor, scaffold)
        letters = letters_of(rec)
        gold = gt_choice(rec["answer"], yn, letters=letters)
        alloc_meta = {
            "mode": "option_union",
            "budget": budget_eff,
            "budget_matched": self.budget <= 0,
            "K": K,
            "selected": sel_idx,
            "selection_fallback": "single_clip" if K <= 1 else None,
            "thumbs": self.thumbs,
            "clip_model": self.clip_model_name,
            "tau_mode": "abs" if self.tau > 0 else f"q{self.tau_q}",
            "tau_per_option": [round(t, 4) for t in taus],
            "pass_count_per_option": pass_counts,
            "option_scores": (None if M is None else
                              [[round(float(x), 4) for x in row] for row in M]),
            "query_mode": qmode,
            "n_query_texts": len(queries),
            "durations_s": [round(d, 2) if d is not None else None for d in durs],
            "nframes": nframes,
            "n_decoded": ncaps,
            "sum_nframes": sum(nframes),
        }
        self._cache = {key: (content, yn, gold, alloc_meta)}
        return self._cache[key]


class QuerySearchMethod(FrameSelectMethod):
    """query_search[_<scorer>] — the backend writes visual search phrases from
    Question + Options; CLIP/SigLIP retrieves the top-budget frames matching
    any phrase; the answer call runs on the retrieved frames."""

    def __init__(self, backend, n_queries=4, query_max_new_tokens=256, **kw):
        super().__init__(backend, **kw)
        self.n_queries = int(n_queries)
        self.query_max_new_tokens = int(query_max_new_tokens)
        self.img_token_parity = _check_token_parity(backend, self.name)

    def _gen_queries(self, rec, K):
        """(queries, meta) — one deterministic text-only call per question
        (cached across the 4 passes via _prepare's record cache). Unparseable
        output falls back to the question text, and the row says so."""
        prompt = QUERY_GEN_PROMPT.format(K=K, n=self.n_queries,
                                         question=rec["question"],
                                         options="\n".join(rec["options"]))
        g = self.backend.generate(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_new_tokens=self.query_max_new_tokens, seed=None, temperature=0.0)
        text = re.sub(r"<think>.*?(?:</think>|\Z)", " ", g.text, flags=re.DOTALL)
        queries, seen = [], set()
        for line in text.splitlines():
            q = _QLINE_STRIP.sub("", line).strip().strip('"').strip()
            if not q or len(q) > 120 or q.lower().startswith("search phrase"):
                continue
            if q.lower() in seen:
                continue
            seen.add(q.lower())
            queries.append(q)
            if len(queries) >= self.n_queries:
                break
        meta = {
            "queries": queries,
            "query_fallback": None if queries else "question_text",
            "qgen_output": g.text[-800:],
            "qgen_latency_s": round(g.latency_s, 3),
            "qgen_input_tokens": g.input_tokens,
            "qgen_output_tokens": g.output_tokens,
        }
        return (queries if queries else [rec.get("question", "")]), meta

    def _prepare(self, rec, video_root):
        key = rec.get("id")
        if key in self._cache:
            return self._cache[key]
        require_video_record(rec, self.name)
        base_msgs, yn = build_messages(rec, video_root, self.nframes, no_video=True,
                                       reasoning=self.reasoning)
        scaffold = base_msgs[0]["content"][0]["text"]
        paths = video_paths(rec, video_root)
        K = len(paths)
        budget_eff = self.budget if self.budget > 0 else self.nframes * K
        if budget_eff < K:
            raise ValueError(
                f"budget {budget_eff} < {K} views: cannot keep one frame per "
                "clip as the prompt promises — raise --budget (or use 0 = "
                "matched nframes x K)")

        pool, per_video_cand = [], []
        for i, vp in enumerate(paths, 1):
            cands = self._candidates(vp)
            per_video_cand.append(len(cands))
            for t, im in cands:
                pool.append((i, t, im))
        if not pool:
            raise FileNotFoundError("no candidate frames from any clip")

        queries, qmeta = self._gen_queries(rec, K)
        S = clip_scores(self._ensure_clip(), queries, [im for _, _, im in pool])
        frame_best = S.max(axis=1) if S.ndim == 2 else S
        order = sorted(range(len(pool)), key=lambda j: -float(frame_best[j]))
        floor = max(0, min(self.floor, budget_eff // K)) if K else 0
        keep = _floor_then_fill([p[0] for p in pool], order, None,
                                budget_eff, floor)

        sel = [(pool[j][0], pool[j][1], pool[j][2], float(frame_best[j]))
               for j in keep]
        by_video = {}
        for vidx, t, im, sc in sel:
            by_video.setdefault(vidx, []).append((t, im, sc))
        for v in by_video:
            by_video[v].sort(key=lambda x: (x[0] if x[0] is not None else 0.0))

        content = [{"type": "text", "text": QUERY_SEARCH_PREFIX.format(
            K=K, nq=len(queries), n=len(sel))}]
        for v in sorted(by_video):
            content.append({"type": "text",
                            "text": FRAME_MARKER.format(orig=v, cnt=len(by_video[v]))})
            for t, im, sc in by_video[v]:
                content.append(_frame_content(im, self.backend, self._resize))
        content.append({"type": "text", "text": scaffold})

        letters = letters_of(rec)
        gold = gt_choice(rec["answer"], yn, letters=letters)
        alloc_meta = {
            "mode": "query_search",
            "budget": budget_eff,
            "budget_matched": self.budget <= 0,
            "img_token_parity": self.img_token_parity,
            "K": K,
            "candidates_per_video": self.candidates_per_video,
            "n_candidates": len(pool),
            "n_selected": len(sel),
            "floor": floor,
            "selected_per_video": {v: len(by_video[v]) for v in sorted(by_video)},
            "candidates_decoded_per_video": per_video_cand,
            "clip_model": self.clip_model_name,
            "query_mode": "generated",
            "n_query_texts": len(queries),
            "selection_fallback": None,
            **qmeta,
        }
        self._cache = {key: (content, yn, gold, alloc_meta)}
        return self._cache[key]
