# Ported from Wavy-Hec/CVBench bench/methods/segment_select.py @ 8b4fbd117b6084dafb743eb30b5298e9a1ba91e3;
# the global (cross-clip) selection mode (select_segments, --seg-select /
# --seg-floor, SEGMENT_SELECT_GLOBAL_PREFIX) additionally ported from
# @ a8e995dd0c069911c1c99acbc5854cc22337d34c; the Qwen-timing guard, the
# image-tower skip under ViCLIP and the selection-provenance stamps
# (image_tower_ran, options_are_clip_refs, segments_keep_effective,
# selection_noop, qwen_timing_unsafe, 6-dp segment scores) additionally ported
# from @ e9ea59088dd286ab3609a61cfaeb8591d1fe11d6.
# Deliberate delta vs source: the docstring's dedup-collapse caveat and its
# ordering paragraph state the failure mode and rationale but omit the fork's
# measured figures and internal review/audit references, which are
# unpublished results.
# Deliberate delta vs source: the Qwen parity note (docstring and __init__
# comment) keeps the 2x patch-merge mechanism but drops the fork's measured
# verification figure.
# Deliberate delta vs source: the global-mode docstring and comments name
# neither the fork's relevance-free control arm nor its coverage reduce (this
# port carries neither), and state the sampling-density confound without the
# fork's unpublished measured figures.
# Deliberate delta vs source: the image-tower skip, the Qwen-timing guard and
# the clip-reference stamp keep the fork's mechanism but drop its measured cost
# and prevalence figures and its dated internal audit references, which are
# unpublished. segments_keep_effective follows the fork's later global-aware
# form (this port carries global mode, which the fork added after e9ea590).
"""SEGMENT selection: split each clip into temporal segments, keep each clip's
most question-relevant segments, pool their frames, drop near-duplicates, and
answer from an evenly-thinned subset of the unique pool.

  segment_select[_<scorer>][_opt]
      (``--seg-select global``: the top-K is taken over ALL clips at once
      instead of within each clip — every (clip, segment) pair is ranked by
      the same reduced score and the best N pairs survive, N = budget //
      frames_per_segment (``--segments-keep``/``--seg-pool`` do not bind).
      ``--seg-floor`` segments per clip are reserved first (default 1, capped
      at N // K), so a camera goes unrepresented only when the segment budget
      cannot seat one apiece. A clip may then contribute several segments,
      one, or none; presentation still groups the survivors by clip in
      original slot order. At a FIXED frame budget global concentrates the
      frames into fewer, better-scored segments than per_clip, so a
      global-vs-per_clip delta confounds segment CHOICE with sampling
      density — run a relevance-free control beside it at the same
      seg_select and budget. Rows stamp ``segment_select_mode``,
      ``seg_floor``, ``segments_kept_total`` and ``clips_with_no_segment``.)
      1. Each of the K clips is split into ``segments_per_video`` contiguous
         equal-time segments (fewer when the clip is shorter than that).
      2. ``frames_per_segment`` frames are sampled uniformly WITHIN each
         segment; a segment's relevance is the best CLIP/SigLIP similarity of
         its frames to the question (``_opt``: to any answer option), or —
         scorer tag ``viclip`` — the ViCLIP similarity of the segment embedded
         JOINTLY as one 8-frame tube (motion-aware; scored against each option
         under ``_opt``). Selection is straight TOP-K: the per-option score
         matrix is reduced to one best score per segment, sorted descending,
         and the best ``segments_keep`` segments PER CLIP are kept.
         ``segments_keep 0`` = AUTO: K is derived from the pooled-frame target
         and the stream count instead — K = seg_pool // (frames_per_segment x
         n_streams), clamped to [1, min(16, segments_per_video)] — so one
         shared pool target (default 128 frames) splits into whole segments
         evenly across streams (4 streams -> 4 segments each; 16+ streams ->
         1; reaching 16 on a 1-stream record needs --segments-per-video >=
         16, since a clip cannot contribute more segments than it was split
         into). A
         within-clip lever, unlike clip_select (whole clips) and frame_select
         (a global frame pool with no temporal structure).
      3. The kept segments' frames are pooled per question and near-duplicates
         removed: frames are visited in DESCENDING relevance order and one is
         dropped when its image-embedding cosine to ANY already-kept frame
         reaches ``dedup_tau`` — so every near-duplicate cluster keeps its
         best-scoring frame, and presentation afterwards reverts to canonical
         (clip, playback) order. The embeddings are the scorer's own
         (clip_scores returns them), so dedup costs no second encoder pass.
         Dedup is question-wide by design — on synchronized-camera datasets a
         frame can be dropped as a duplicate of another CLIP's frame;
         ``dedup_dropped`` in frame_alloc records it.

         Threshold semantics: ``dedup_tau`` must be in (0, 1]; 1.0 DISABLES
         dedup (this is NOT the --sel-tau "0 = off" convention — a tau of 0
         would collapse every question to one frame, so it is rejected).
         Two caveats: (a) static-camera footage self-similarity sits ABOVE
         the 0.95 default — a long fixed camera's pooled frames can all
         exceed it pairwise and collapse to ONE frame, so on CrossView/MEVA
         the default measures a one-still-per-view baseline, not segment
         selection; run those legs with a calibrated tau or DEDUP_TAU=1, and
         read ``n_unique``/``dedup_dropped`` before trusting a leg. (b) the
         cosine scale is scorer-specific (SigLIP dedups harder than CLIP at
         the same tau), so cross-scorer comparisons at one tau confound
         ranking quality with effective visual budget — hold the scorer fixed
         or set tau per scorer; every row stamps dedup_tau + clip_model.
      4. The unique set is reduced to the frame budget: the budget is split
         across clips in proportion to each clip's unique-frame count
         (allocate_frames, per-clip floor so no clip is starved), and each
         clip's share is thinned EVENLY IN TIME over its unique frames —
         relevance chose the segments; within them, coverage stays uniform.

  Ordering is chronological everywhere and clips keep their ORIGINAL slot
  order/numbers: options reference clips by list position ("A. Video 1"), and
  on MVU-Eval's temporal-reasoning records recovering the source-time order IS
  the question — re-sorting clips would corrupt the answer key. Frames within
  a clip are presented in playback order.

  Budget parity: --budget 0/omitted = MATCHED (nframes x K). On Qwen each
  clip's kept frames are fed as ONE video item (a list of PIL frames), so the
  reader's 2-frame temporal merge applies exactly as in the sequential arm —
  feeding them as still images at video pixel caps (the _optu convention)
  costs a 2x visual-token surplus because images skip the merge.
  The reader pads an odd frame list to even by duplicating the last frame
  (``qwen_even_pad_per_video`` records it). On InternVL frames go as still
  images at max_tiles=1 (refused otherwise), where image/video-frame parity
  is exact at 1 tile = 256 tokens. Rows stamp which mode applied
  (``img_token_parity``). Selection is deterministic and cached across the 4
  passes, so the std isolates the answer stage. Under the ``viclip`` tag the
  tube scorer replaces only the segment-RELEVANCE signal; the dedup step
  still needs per-frame image embeddings (a joint tube embedding has none),
  so the CLIP/SigLIP image tower (--clip-model) runs alongside it when dedup
  is on (``dedup_tau`` < 1) and rows stamp both (``segment_scorer`` +
  ``clip_model``); with dedup off the tower is skipped (``image_tower_ran``
  False). Rows also stamp ``options_are_clip_refs`` (every query text is a
  bare "Video k" / permutation string, so the ranking carried no content),
  ``segments_keep_effective`` and ``selection_noop``.
"""
import os
import re

import numpy as np
from decord import VideoReader, cpu

from inprocess.harnesses.base import require_video_record
from inprocess.harnesses.clip_select import FrameSelectMethod, clip_scores, query_for
from inprocess.harnesses.option_union import (_check_token_parity,
                                              _floor_then_fill,
                                              _frame_content)
from inprocess.harnesses.uniform import allocate_frames
from inprocess.dataloaders.qa_json import build_messages, letters_of, video_paths
from inprocess.evaluation.scoring import gt_choice

SEGMENT_SELECT_PREFIX = (
    "The question below refers to {K} INDEPENDENT video clips (different, "
    "unrelated scenes), numbered Video 1 to Video {K} in their original "
    "order. Each clip was split into up to {S} equal time segments and only "
    "its most relevant segments (at most {top}) are represented: frames were "
    "sampled uniformly within those segments, near-duplicate frames were "
    "removed, and the {n} frames shown remain (thinned evenly in time "
    "wherever they exceeded the frame budget). Frames are shown grouped by "
    "their source Video (ORIGINAL numbering) and in temporal order within "
    "each Video; a banner '=== Video k ===' precedes each Video's frames. A "
    "clip whose frames were all removed as duplicates is omitted. Reason "
    "over the shown frames to answer.")

# --seg-select global: identical to SEGMENT_SELECT_PREFIX except for the
# sentence naming WHICH segments survived (one ranking across all clips, not
# each clip's own top-K) and the added omission clause — a global ranking can
# leave a clip with no represented segment, which the per-clip prompt promises
# never happens. Every other sentence is byte-identical so the prompt
# difference between the two modes is the selection rule and nothing else.
SEGMENT_SELECT_GLOBAL_PREFIX = (
    "The question below refers to {K} INDEPENDENT video clips (different, "
    "unrelated scenes), numbered Video 1 to Video {K} in their original "
    "order. Each clip was split into up to {S} equal time segments and only "
    "the most relevant segments across ALL clips (at most {top} in total) "
    "are represented: frames were "
    "sampled uniformly within those segments, near-duplicate frames were "
    "removed, and the {n} frames shown remain (thinned evenly in time "
    "wherever they exceeded the frame budget). Frames are shown grouped by "
    "their source Video (ORIGINAL numbering) and in temporal order within "
    "each Video; a banner '=== Video k ===' precedes each Video's frames. A "
    "clip may contribute several segments, one, or none; a clip with no "
    "represented segment is omitted. A "
    "clip whose frames were all removed as duplicates is omitted. Reason "
    "over the shown frames to answer.")

SEGMENT_MARKER = ("=== Video {orig} — {cnt} frame(s) drawn from its {nseg} "
                  "most relevant segment(s), in temporal order ===")

# Ceiling of the auto top-K rule (segments_keep 0): K = seg_pool //
# (frames_per_segment x n_streams), clamped to [1, SEG_KEEP_MAX] and never
# above segments_per_video (top-16-of-8 would select nothing while the row
# stamped a binding-looking 16). With the 128-frame pool default, 8-frame
# segments and segments_per_video >= 16, K spans exactly 1..16 as the stream
# count runs 16..1.
SEG_KEEP_MAX = 16


_CLIP_REF_RE = re.compile(
    r"^\s*(?:(?:video|view|clip|camera)\s*\d+|[ivx]+(?:\s*(?:->|→|,|then)\s*[ivx]+)+"
    r"|\d+(?:\s*(?:->|→)\s*\d+)+)\s*\.?\s*$", re.I)


def _options_are_clip_refs(texts):
    """True when EVERY query text is a bare clip/slot reference or a
    permutation of numerals ("Video 2", "II -> I -> III", "1, 3, 2") — a
    content-free query the scorer cannot ground."""
    return bool(texts) and all(_CLIP_REF_RE.match(t or "") for t in texts)


def _dedup_keep_best(indices, scores, embs, tau):
    """Greedy near-duplicate removal over ``indices`` (kept-frame pool).

    Frames are visited in DESCENDING relevance-score order, so the survivor
    of every near-duplicate cluster is its best-scoring frame — visiting in
    canonical order instead kept whichever frame happened to come first,
    discarding the relevance ranking exactly where dedup bites hardest
    (static cameras). Returns (kept, dropped), both in the original order of
    ``indices``. ``tau`` >= 1.0 disables dedup entirely."""
    if tau >= 1.0:
        return list(indices), []
    order = sorted(indices, key=lambda j: -float(scores[j]))
    kept_set = set()
    kept_mat = None
    for j in order:
        e = embs[j]
        if kept_mat is not None and float((kept_mat @ e).max()) >= tau:
            continue
        kept_set.add(j)
        kept_mat = e[None, :] if kept_mat is None else np.vstack([kept_mat,
                                                                  e[None, :]])
    return ([j for j in indices if j in kept_set],
            [j for j in indices if j not in kept_set])


def _spread(items, n):
    """n evenly-spaced elements of ``items`` (order preserved). For
    1 < n <= len(items) the picked positions are distinct because consecutive
    linspace points are >= 1 apart."""
    if n >= len(items):
        return list(items)
    if n <= 0:
        return []
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def select_segments(seg_scores, mode, keep_top, seg_floor, K):
    """Which segments each clip contributes: {video: sorted [segment_id]}.

    ``seg_scores`` is the max-reduced matrix {video: {segment_id: float}}.
    Every video present in ``seg_scores`` is present in the result, possibly
    with an EMPTY list (global mode only) — the caller indexes it per clip.

    ``mode="per_clip"`` (historic, the default): each clip keeps its own best
    ``keep_top`` segments, so every clip is represented and the frame budget
    spreads over all K of them. Ties keep the earlier-inserted segment
    (Python's stable sort over the score dict, which is built in ascending
    segment order) — reproduced here expression-for-expression.

    ``mode="global"``: every (clip, segment) pair competes in ONE ranking and
    the best ``keep_top`` pairs overall survive, so a clip may contribute
    several segments, one, or none. ``seg_floor`` segments per clip are
    reserved first — capped at ``keep_top // K``, i.e. dropped entirely when
    the segment budget cannot seat one per clip — so a global ranking cannot
    silently blind a camera unless the budget forces it. Ties break on
    (video, segment): deterministic across passes, shards and reruns, like
    every other choice this arm makes.
    """
    if mode == "per_clip":
        return {v: sorted(sorted(d, key=lambda s: -d[s])[: keep_top])
                for v, d in seg_scores.items()}
    if mode != "global":
        raise ValueError(
            f"select_segments: mode must be 'per_clip' or 'global', got "
            f"{mode!r}")
    if int(seg_floor) < 0:
        raise ValueError(
            f"select_segments: seg_floor must be >= 0, got {seg_floor}")
    pairs = [(v, s) for v, d in seg_scores.items() for s in d]
    pair_video = [v for v, _ in pairs]
    order = sorted(range(len(pairs)),
                   key=lambda j: (-seg_scores[pairs[j][0]][pairs[j][1]],
                                  pairs[j][0], pairs[j][1]))
    floor = max(0, min(int(seg_floor), keep_top // K)) if K else 0
    out = {v: [] for v in seg_scores}
    for j in _floor_then_fill(pair_video, order, None, keep_top, floor):
        v, s = pairs[j]
        out[v].append(s)
    return {v: sorted(segs) for v, segs in out.items()}


class SegmentSelectMethod(FrameSelectMethod):
    """segment_select[_<scorer>][_opt] — top segments per clip (or, under
    --seg-select global, across all clips), dedup, thin."""
    name = "segment_select"

    def __init__(self, backend, segments_per_video=8, segments_keep=4,
                 frames_per_segment=8, dedup_tau=0.95, seg_scorer=None,
                 seg_pool=128, seg_select="per_clip", seg_floor=1, **kw):
        super().__init__(backend, **kw)
        self.segments_per_video = int(segments_per_video)
        # 0 = AUTO: K per clip derived from seg_pool and the record's stream
        # count at _prepare time (clamped to [1, SEG_KEEP_MAX])
        self.segments_keep = int(segments_keep)
        self.frames_per_segment = int(frames_per_segment)
        self.dedup_tau = float(dedup_tau)
        # None -> the image tower scores segments too (historic); "viclip" ->
        # each segment is embedded jointly as one tube for segment relevance,
        # while the image tower still supplies dedup embeddings
        self.seg_scorer = seg_scorer
        self.seg_pool = int(seg_pool)
        # "per_clip" (historic) = top-K segments WITHIN each clip; "global" =
        # one ranking over every (clip, segment) pair, with seg_floor segments
        # reserved per clip first. See select_segments().
        self.seg_select = str(seg_select)
        self.seg_floor = int(seg_floor)
        if self.seg_select not in ("per_clip", "global"):
            raise ValueError(
                f"{self.name}: seg_select must be 'per_clip' or 'global', got "
                f"{self.seg_select!r}")
        if self.seg_floor < 0:
            raise ValueError(
                f"{self.name}: seg_floor must be >= 0, got {self.seg_floor}")
        if self.segments_per_video < 1 or self.segments_keep < 0 \
                or self.frames_per_segment < 1:
            raise ValueError(
                f"{self.name}: segments_per_video/frames_per_segment must be "
                ">= 1 and segments_keep >= 0 (0 = auto top-K from --seg-pool)")
        if self.segments_keep == 0 and self.seg_pool < 1:
            raise ValueError(
                f"{self.name}: segments_keep 0 (auto) needs --seg-pool >= 1, "
                f"got {self.seg_pool}")
        if not (0.0 < self.dedup_tau <= 1.0):
            # NOT the --sel-tau convention: tau 0 here is not "off" — every
            # CLIP/SigLIP image pair has positive cosine (cone effect), so
            # tau <= 0 silently collapses each question to ONE frame with
            # error=null. 1.0 disables dedup.
            raise ValueError(
                f"{self.name}: dedup_tau must be in (0, 1], got "
                f"{self.dedup_tau} (1.0 disables dedup; 0 is NOT 'off' — it "
                "would drop every frame after the first)")
        self.img_token_parity = _check_token_parity(backend, self.name)
        if getattr(backend, "patch_size", None):
            # true-parity path: kept frames go as per-clip VIDEO items so
            # Qwen's 2-frame temporal merge applies as in the sequential arm;
            # still images at video pixel caps cost a 2x visual-token surplus
            self.img_token_parity = "qwen_video_list"
            # BLOCKED until real timing is attached: qwen_vl_utils fabricates
            # fps=2.0 / indices=range(n) for a PIL-list video item, so frames
            # drawn from segments minutes apart are labelled seconds apart
            # (Qwen3-VL prints the wrong "<t s>" text; Qwen2.5-VL gets a
            # false M-RoPE time scale). SEGMENT_SELECT_QWEN_UNSAFE=1 overrides
            # and stamps qwen_timing_unsafe on every row it produces.
            if os.environ.get("SEGMENT_SELECT_QWEN_UNSAFE", "0") != "1":
                raise SystemExit(
                    f"{self.name}: the Qwen path feeds per-clip PIL-list "
                    "video items whose timestamps qwen_vl_utils fabricates; "
                    "attach real frame times before running a Qwen segment "
                    "leg (or set SEGMENT_SELECT_QWEN_UNSAFE=1 knowingly).")

    def _decode_segments(self, vp):
        """[(segment_id, time_s, PIL)] in playback order, plus decode meta.

        Same fail-loud recovery policy as ``_candidates``: decord flakiness
        costs only the failed indices, a fully unreadable clip raises."""
        from PIL import Image
        try:
            vr = VideoReader(vp, ctx=cpu(0), num_threads=1)
            n = len(vr)
            fps = float(vr.get_avg_fps())
        except Exception as e:
            kind = "missing" if not os.path.exists(vp) else type(e).__name__
            raise FileNotFoundError(
                f"unreadable clip {vp} ({kind}) — check --video-root") from e
        if n <= 0:
            raise FileNotFoundError(f"unreadable clip {vp} (0 frames)")
        S = max(1, min(self.segments_per_video, n))
        bounds = [round(s * n / S) for s in range(S + 1)]
        out = []
        for sid in range(S):
            a, b = bounds[sid], bounds[sid + 1]
            if b <= a:
                continue
            m = min(self.frames_per_segment, b - a)
            idx = sorted({a + min(b - a - 1, int((j + 0.5) * (b - a) / m))
                          for j in range(m)})
            for fi in idx:
                try:
                    fr = vr[fi].asnumpy()
                except Exception:
                    continue
                out.append((sid, (fi / fps) if fps > 0 else None,
                            Image.fromarray(fr).convert("RGB")))
        if not out:
            raise FileNotFoundError(f"unreadable clip {vp} (no decodable frames)")
        return out, {"n_total": n,
                     "fps": round(fps, 2) if fps > 0 else None,
                     "n_segments": S}

    def _viclip_segment_scores(self, pool, queries):
        """{video: {seg_id: np.ndarray [n_queries]}} — each segment's decoded
        frames as ONE ViCLIP tube (uniformly sampled, short segments padded by
        repetition, to the fixed tube length), scored jointly against every
        query. Per-record text cache: each option is encoded once, not once
        per segment. Fail-loud like the image tower — a scorer failure must
        never degrade to unguided selection."""
        from inprocess.harnesses.viclip_scorer import viclip_option_scores, VICLIP_NFRAMES
        dev = getattr(self.backend, "device", "cuda:0")
        frames_by_seg = {}                 # (video, seg_id) -> [PIL, ...]
        for v, sid, t, im in pool:         # pool is in playback order
            frames_by_seg.setdefault((v, sid), []).append(im)
        text_cache = {}
        out = {}
        for (v, sid), ims in frames_by_seg.items():
            n = len(ims)
            tube = [ims[min(n - 1, int((j + 0.5) * n / VICLIP_NFRAMES))]
                    for j in range(VICLIP_NFRAMES)]
            s = viclip_option_scores(tube, queries, device=dev,
                                     text_cache=text_cache)
            out.setdefault(v, {})[sid] = np.asarray(s, dtype=np.float64)
        return out

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
                "clip — raise --budget (or use 0 = matched nframes x K)")

        # 1+2. decode per-segment frames; pool is canonical order by
        # construction (clip 1..K, ascending frame index within each clip)
        pool, decode_meta = [], []            # (video 1-based, seg_id, t, im)
        for i, vp in enumerate(paths, 1):
            entries, meta = self._decode_segments(vp)
            decode_meta.append(meta)
            for sid, t, im in entries:
                pool.append((i, sid, t, im))
        if not pool:
            raise FileNotFoundError("no candidate frames from any clip")

        # score every pooled frame once with the image tower — it ranks the
        # segments unless ViCLIP does, and its per-frame embeddings feed the
        # dedup step. Under viclip with dedup OFF (tau >= 1) neither consumer
        # exists, so the pass is skipped: it cost a second encoder's FLOPs and
        # resident weights for embeddings _dedup_keep_best never read
        # (selection verified byte-identical either way). Scorer failures
        # raise (inherited policy — no silent uniform fallback).
        query, qmode = query_for(rec, self.query)
        queries = query if isinstance(query, list) else [query]
        need_img = self.seg_scorer != "viclip" or self.dedup_tau < 1.0
        if need_img:
            scores, embs = clip_scores(self._ensure_clip(), query,
                                       [p[3] for p in pool], return_image_embs=True)
            S2 = scores if scores.ndim == 2 else None    # [frames, options]
            if scores.ndim == 2:
                # [frames, options] -> a frame's score is its best option
                scores = scores.max(axis=1)
        else:
            scores, embs, S2 = None, None, None

        # per-segment per-option score matrix: ViCLIP embeds each segment
        # jointly as one tube; the image tower reduces max-over-frames per
        # option (same segment scores as the historic scalar path)
        if self.seg_scorer == "viclip":
            seg_opt = self._viclip_segment_scores(pool, queries)
        else:
            seg_opt = {}                      # video -> {seg_id: [per-option]}
            for j, (v, sid, t, im) in enumerate(pool):
                row = S2[j] if S2 is not None else np.array([float(scores[j])])
                d = seg_opt.setdefault(v, {})
                d[sid] = np.maximum(d[sid], row) if sid in d else row.copy()
        seg_scores = {v: {sid: float(r.max()) for sid, r in d.items()}
                      for v, d in seg_opt.items()}

        # straight TOP-K over the reduced matrix (descending; a tie keeps the
        # earlier segment), then restore chronological (segment-id) order
        # within the clip. PER CLIP by default; --seg-select global takes the
        # same top-K across all clips at once (select_segments).
        # segments_keep 0 = AUTO: fill the seg_pool frame target with whole
        # segments split evenly across the K streams, clamped to
        # [1, SEG_KEEP_MAX]. Auto and global are separate rules for keep_top.
        glob = self.seg_select == "global"
        if glob:
            # GLOBAL: keep_top is a budget of segments for the whole question,
            # not a per-clip K, so it comes from the FRAME budget (--seg-pool
            # plays no part) — as many whole segments as the budget seats.
            # Consequence to keep in view when reading a global leg: at the
            # same frame budget global concentrates the frames into fewer,
            # higher-scored segments than per_clip does (per_clip spends
            # keep_top x K segments' worth on the same budget), so a
            # global-vs-per_clip delta confounds "the RIGHT segments" with
            # "denser sampling inside fewer segments". A relevance-free
            # segment-selection control must run BESIDE a global leg, at the
            # identical seg_select/budget, to separate the two.
            keep_top = max(1, budget_eff // self.frames_per_segment)
        else:
            keep_top = self.segments_keep or max(1, min(
                SEG_KEEP_MAX, self.segments_per_video,
                self.seg_pool // (self.frames_per_segment * K)))
        kept_segs = select_segments(seg_scores, self.seg_select, keep_top,
                                    self.seg_floor, K)
        # the per-clip floor that actually bound (global only; per_clip has no
        # such lever — every clip keeps its own top-K by construction)
        seg_floor_eff = (max(0, min(self.seg_floor, keep_top // K))
                         if glob and K else None)
        kept = [j for j, (v, sid, t, im) in enumerate(pool)
                if sid in kept_segs[v]]

        # 3. near-duplicate removal; each cluster keeps its BEST-scoring frame
        unique, dropped = _dedup_keep_best(kept, scores, embs, self.dedup_tau)
        dropped_per_video = {}
        for j in dropped:
            v = pool[j][0]
            dropped_per_video[v] = dropped_per_video.get(v, 0) + 1
        n_unique = len(unique)

        # 4. thin the unique set to the budget, proportional per clip
        by_video_unique = {}
        for j in unique:
            by_video_unique.setdefault(pool[j][0], []).append(j)
        vids = sorted(by_video_unique)
        counts = [len(by_video_unique[v]) for v in vids]
        floor = max(0, min(self.floor, budget_eff // len(vids))) if vids else 0
        if n_unique <= budget_eff:
            alloc = counts
        else:
            alloc = allocate_frames(counts, budget=budget_eff, floor=floor,
                                    caps=counts)
        selected = {v: _spread(by_video_unique[v], a)
                    for v, a in zip(vids, alloc)}
        n_selected = sum(len(s) for s in selected.values())
        if n_selected == 0:
            raise ValueError(
                f"{self.name}: empty selection (unique={n_unique}, "
                f"budget={budget_eff}) — a zero-image answer call must never "
                "run as a vision row")

        qwen_video_list = self.img_token_parity == "qwen_video_list"
        qwen_pad = {}
        prefix = SEGMENT_SELECT_GLOBAL_PREFIX if glob else SEGMENT_SELECT_PREFIX
        content = [{"type": "text", "text": prefix.format(
            K=K, S=self.segments_per_video, top=keep_top,
            n=n_selected)}]
        for v in vids:
            if not selected[v]:
                continue
            content.append({"type": "text", "text": SEGMENT_MARKER.format(
                orig=v, cnt=len(selected[v]), nseg=len(kept_segs[v]))})
            frames = [pool[j][3] for j in selected[v]]
            if qwen_video_list:
                # one video item per clip: the reader sizes the frames with
                # its own video pixel caps AND applies the 2-frame temporal
                # merge; an odd list is padded by duplicating the last frame
                if len(frames) % 2 == 1:
                    qwen_pad[v] = 1
                content.append({"type": "video", "video": frames})
            else:
                for im in frames:
                    content.append(_frame_content(im, self.backend,
                                                  self._resize))
        content.append({"type": "text", "text": scaffold})

        letters = letters_of(rec)
        gold = gt_choice(rec["answer"], yn, letters=letters)
        alloc_meta = {
            "mode": "segment_select",
            "budget": budget_eff,
            "budget_matched": self.budget <= 0,
            "img_token_parity": self.img_token_parity,
            "K": K,
            "segments_per_video": self.segments_per_video,
            # "per_clip" = top-K WITHIN each clip; "global" = one ranking over
            # every (clip, segment) pair (select_segments)
            "segment_select_mode": self.seg_select,
            # the EFFECTIVE per-clip K (auto rows resolve it per record);
            # under global it is instead the question-wide segment budget N
            "segments_keep": keep_top,
            # auto-K is a per_clip rule: global derives N from the frame
            # budget, so --segments-keep/--seg-pool never bind there
            "segments_keep_auto": self.segments_keep == 0 and not glob,
            "seg_pool": (self.seg_pool
                         if self.segments_keep == 0 and not glob else None),
            # segments reserved per clip before the global fill, AFTER the
            # keep_top // K cap; None in per_clip (no such lever)
            "seg_floor": seg_floor_eff,
            "frames_per_segment": self.frames_per_segment,
            "segments_kept_per_video": kept_segs,
            # how much of the segment budget was actually spent, and how many
            # clips the ranking left unrepresented (always 0 under per_clip)
            "segments_kept_total": sum(len(s) for s in kept_segs.values()),
            "clips_with_no_segment": sum(1 for s in kept_segs.values() if not s),
            # keyed by segment id (a bare list shifts silently when decord
            # drops a whole segment's frames, mismapping scores post-hoc)
            # 6 dp: on static-camera sets the top-K boundary can fall below
            # 4-dp resolution, where the matrix no longer reproduces the kept
            # set
            "segment_scores": {v: {s: round(d[s], 6) for s in sorted(d)}
                               for v, d in seg_scores.items()},
            # the full per-option matrix behind the top-K (options mode only;
            # with one query it would just duplicate segment_scores)
            "segment_option_scores": (
                {v: {s: [round(float(x), 6) for x in d[s]] for s in sorted(d)}
                 for v, d in seg_opt.items()} if len(queries) > 1 else None),
            "segment_scorer": ("viclip" if self.seg_scorer == "viclip"
                               else self.clip_model_name),
            # True when every option is a bare clip reference ("Video 3",
            # "II -> I -> III"): the scorer then ranks segments by similarity
            # to that token, not to any content. None outside options mode
            "options_are_clip_refs": _options_are_clip_refs(queries) if qmode == "options" else None,
            # the per-clip K that actually bound (a clip cannot contribute
            # more than its segment count) — under global, each clip's ACTUAL
            # kept count, which the one ranking sets per clip — and whether
            # selection did anything at all (auto-K on 1-2 streams keeps
            # every segment)
            "segments_keep_effective": (
                {v: len(kept_segs[v]) for v in seg_scores} if glob
                else {v: min(keep_top, len(d)) for v, d in seg_scores.items()}),
            "selection_noop": all(len(kept_segs[v]) == len(d) for v, d in seg_scores.items()),
            "n_pool": len(pool),
            "n_kept_segment_frames": len(kept),
            "dedup_tau": self.dedup_tau,
            "n_unique": n_unique,
            "dedup_dropped": len(kept) - n_unique,
            "dedup_dropped_per_video": dropped_per_video,
            "n_selected": n_selected,
            "selected_per_video": {v: len(selected[v]) for v in vids},
            "selected_times_s": {v: [round(pool[j][2], 2)
                                     if pool[j][2] is not None else None
                                     for j in selected[v]] for v in vids},
            "floor": floor,
            "qwen_even_pad_per_video": qwen_pad if qwen_video_list else None,
            "per_video_decode": decode_meta,
            # the CONFIGURED dedup/image tower (unchanged meaning); whether it
            # actually ran this record is image_tower_ran
            "clip_model": self.clip_model_name,
            "image_tower_ran": need_img,
            # True only under the SEGMENT_SELECT_QWEN_UNSAFE override: the
            # PIL-list items carry fabricated timestamps
            "qwen_timing_unsafe": True if qwen_video_list else None,
            "query_mode": qmode,
            "n_query_texts": len(query) if isinstance(query, list) else 1,
            "selection_fallback": None,
        }
        self._cache = {key: (content, yn, gold, alloc_meta)}
        return self._cache[key]
