# Ported from Wavy-Hec/CVBench bench/methods/temporal.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
# Ported from Wavy-Hec/CVBench bench/methods/cvbench_native.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""TEMPORAL-WEIGHTED harness: the K (<=4) independent CVBench clips of one
question are presented to ONE model SEQUENTIALLY (clip 1, then clip 2, ...), with
a single TOTAL frame budget split across the clips IN PROPORTION TO EACH CLIP'S
DURATION (longer clips get more frames). Within each clip the frames are sampled
UNIFORMLY across its full duration (the backend's own per-clip ``nframes``
sampler). A banner precedes each clip so the model can tell where one clip ends
and the next begins.

This contrasts with ``centralized`` (spatial 2x2 grid stitching) and with a flat
per-clip ``nframes`` (``cvbench_native``): here the *total* frame count per
question is held fixed (default 64) regardless of how many clips it has, and the
per-clip split is duration-weighted. Passing ``weighting="even"`` keeps the same
sequencing/markers/total but splits the budget evenly — the budget-matched
control for "does duration weighting help?".

The text scaffold (question/options/<think>/<answer>) is taken verbatim from the
existing harness (``build_messages(..., no_video=True)``), so reasoning rides the
same <think> template as every other method. Per-clip durations + allocated frame
counts are recorded in ``Result.frame_alloc`` and the raw reasoning trace in
``Result.response_text``/``Result.think`` so failures stay interpretable.

CVBench-native presentation: the exact method described in CVBench --- the
multiple clips are fed to ONE model as separate, sequential video blocks (each
clip's frames presented as a temporal sequence with a ``"Video k:"`` marker),
not stitched. This is the baseline the stitching harness is compared against at
an equal frame budget.

It is the production ``build_messages(no_video=False)`` path, wrapped as a
Method with per-question timing + seed/temperature plumbing.
"""
import math

from decord import VideoReader, cpu

from harnesses.base import Method, Result, result_fields
from dataloaders.qa_json import build_messages, video_paths
from evaluation.scoring import parse_choice, gt_choice, extract_think

PREFIX = (
    "The following {K} video clips are INDEPENDENT (different, unrelated scenes) "
    "and are presented SEQUENTIALLY, one after another. They are NOT "
    "time-synchronized. A total budget of {budget} frames is {split}; within each "
    "clip the frames are sampled UNIFORMLY across its full duration. A banner "
    "'=== Video k of {K} ===' precedes each clip's frames, and frames are labeled "
    "Frame1, Frame2, ... within that clip. Keep track of which frames belong to "
    "which Video; reason about each Video separately and then jointly to answer.")

SPLIT_DESC = {
    "duration": "split across the {K} clips in proportion to each clip's duration "
                "(longer clips get more frames)",
    "even": "split EVENLY across the {K} clips",
}

MARKER = ("=== Video {k} of {K} — {n_k} frames sampled uniformly across its full "
          "duration (~{dur_s:.0f}s); clip {k} of the sequence ===")


def largest_remainder(weights, budget):
    """Hamilton/largest-remainder apportionment: split ``budget`` integers across
    ``len(weights)`` bins in proportion to ``weights``, summing EXACTLY to budget."""
    k = len(weights)
    if k == 0:
        return []
    total = sum(weights)
    if total <= 0:                                  # all-unknown -> even split
        out = [budget // k] * k
        for i in range(budget - sum(out)):
            out[i] += 1
        return out
    quotas = [w * budget / total for w in weights]
    floors = [int(math.floor(q)) for q in quotas]
    order = sorted(range(k), key=lambda i: quotas[i] - floors[i], reverse=True)
    for i in order[: budget - sum(floors)]:
        floors[i] += 1
    return floors


def allocate_frames(weights, budget=64, floor=2, caps=None):
    """Allocate a TOTAL ``budget`` of frames across the clips of one question.

    ``weights``: per-clip weight (clip duration in seconds, or equal weights for
        an even split); a non-positive/NaN weight is treated as 0 (and if every
        weight is non-positive, all clips are weighted equally).
    ``floor``:   per-clip minimum frames (clamped to ``budget // k`` so the floors
        can never exceed the budget).
    ``caps``:    per-clip max usable frames (= ``len(VideoReader)``); a clip is
        never asked for more frames than it has. Surplus from a saturated clip is
        redistributed to the others.

    Returns a list summing to ``min(budget, sum(caps))`` (under-fill only when
    every clip is cap-bound — recorded via ``sum_nframes`` for auditing).
    """
    k = len(weights)
    if k == 0:
        return []
    w = [x if (isinstance(x, (int, float)) and x == x and x > 0) else 0.0 for x in weights]
    if sum(w) <= 0:
        w = [1.0] * k                               # all unknown -> equal weights
    caps = [(budget if c in (None, 0) else int(c)) for c in (caps or [None] * k)]
    target = min(budget, sum(caps))
    floor = max(0, min(floor, target // k))         # k*floor never exceeds target

    # Proportional largest-remainder allocation, water-filling the budget into
    # clips with headroom and re-normalizing each time a clip saturates its cap
    # (its surplus floods the non-saturated clips). Pure proportional when no cap
    # binds, so 120/240/120 -> 16/32/16 exactly (floor applied only afterward).
    alloc = [0] * k
    remaining = target
    active = list(range(k))
    while remaining > 0 and active:
        add = largest_remainder([w[i] for i in active], remaining)
        moved = 0
        nxt = []
        for j, i in enumerate(active):
            give = min(add[j], caps[i] - alloc[i])
            alloc[i] += give
            moved += give
            if alloc[i] < caps[i]:
                nxt.append(i)
        remaining -= moved
        active = nxt
        if moved == 0:                              # residual rounding: 1 at a time
            for i in active:
                if remaining <= 0:
                    break
                alloc[i] += 1
                remaining -= 1
            break

    # Floor is a safety net: raise any clip below its floor (where the clip can
    # hold it), borrowing from the clips most above their own floor. Sum-preserving.
    for i in range(k):
        need = min(floor, caps[i]) - alloc[i]
        while need > 0:
            donors = [j for j in range(k) if j != i and alloc[j] > min(floor, caps[j])]
            if not donors:
                break
            j = max(donors, key=lambda j: alloc[j] - min(floor, caps[j]))
            alloc[j] -= 1
            alloc[i] += 1
            need -= 1
    return alloc


class TemporalWeightedMethod(Method):
    name = "temporal_weighted"

    def __init__(self, backend, budget=64, floor=2, weighting="duration",
                 nframes=8, max_new_tokens=8192, temperature=0.0):
        super().__init__(backend, nframes=nframes, max_new_tokens=max_new_tokens,
                         temperature=temperature)
        self.budget = budget
        self.floor = floor
        self.weighting = weighting          # "duration" (weighted) | "even" (control)
        # Distinct per-instance label so the weighted + even arms don't collide on
        # the JSONL resume key (id, method, backend, pass_idx) in a shared results file.
        self.name = "temporal_weighted" if weighting == "duration" else "temporal_even"
        self._cache = {}                    # rec id -> (content, yn, gold, alloc_meta); last rec only

    @staticmethod
    def _clip_meta(vp):
        """(duration_s, n_decoded) via decord; (None, None) on decode failure."""
        try:
            vr = VideoReader(vp, ctx=cpu(0), num_threads=1)
            n = len(vr)
            fps = float(vr.get_avg_fps())
            return (n / fps if fps > 0 else None), n
        except Exception:
            return None, None

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
            d, n = self._clip_meta(vp)
            durs.append(d)
            ncaps.append(n)
        weights = durs if self.weighting == "duration" else [1.0] * K
        nframes = allocate_frames(weights, self.budget, floor=self.floor, caps=ncaps)

        split = SPLIT_DESC.get(self.weighting, SPLIT_DESC["duration"]).format(K=K)
        content = [{"type": "text", "text": PREFIX.format(K=K, budget=self.budget, split=split)}]
        for k, (vp, n_k, d) in enumerate(zip(paths, nframes, durs), 1):
            content.append({"type": "text",
                            "text": MARKER.format(k=k, K=K, n_k=n_k,
                                                  dur_s=(d if d is not None else 0.0))})
            content.append({"type": "video", "video": vp, "nframes": n_k})
        content.append({"type": "text", "text": scaffold})

        gold = gt_choice(rec["answer"], yn)
        alloc_meta = {
            "weighting": self.weighting,
            "budget": self.budget,
            "floor": self.floor,
            "durations_s": [round(d, 2) if d is not None else None for d in durs],
            "nframes": nframes,
            "n_decoded": ncaps,
            "sum_nframes": sum(nframes),
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


class CVBenchNativeMethod(Method):
    name = "cvbench_native"

    def answer(self, rec, video_root, seed=None) -> Result:
        messages, yn = build_messages(rec, video_root, self.nframes, no_video=False)
        f = result_fields(rec)
        gold = gt_choice(rec["answer"], yn)
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
        except Exception as e:
            return Result(**f, method=self.name, backend=self.backend.name,
                          prediction="", gold=gold, correct=False, abstained=True,
                          seed=seed, temperature=self.temperature,
                          num_model_calls=1, error=f"{type(e).__name__}: {e}")
