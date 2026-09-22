# Ported from Wavy-Hec/MultiCam bench/tests/test_segment_seconds.py
# @ b5f666ca990458059cd072048625ccfe629b2737.
# Deliberate delta vs source: the fork keys resume identity on a
# (seg_select, seg_floor[, segment_seconds]) tuple stamped top-level by
# run_bench.seg_identity / stamp_seg_identity and read back by
# existing_identity; this runner stamps every knob on every row and compares
# them through IDENTITY_KNOBS / scan_output, so the identity tests exercise
# those instead. The fork's viclip_text_overflow monkeypatch and its
# ``image_tower_ran`` assertions on the count-partition leg are kept where the
# port has the stamp; the fork's measured-figure docstrings are not repeated.
"""CPU gate for the --segment-seconds time partition (segment_select).

The rule under test: every camera stream is cut into 8-second segments, a
segment scores s(c, j) = max over the query texts of its similarity, and the
12 best (camera, segment) pairs over ALL cameras are kept — 96 frames. These
tests pin the partition arithmetic, that the count partition is untouched by
the new flag, the run identity, and — on a synthetic clip, with a fake scorer
and no model — that the whole _prepare path delivers exactly those 12 x 8
frames at full resolution.

No backend, no scorer model, no GPU; the synthetic-clip tests need cv2 to
encode and decord to read. Runnable either way:

    python -m tests.test_segment_seconds
    pytest tests/test_segment_seconds.py
"""
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402
from PIL import Image                                             # noqa: E402

from inprocess.harnesses import segment_select as ss              # noqa: E402
from inprocess.harnesses.segment_select import (                  # noqa: E402
    SEGMENT_SELECT_GLOBAL_PREFIX, SEGMENT_SELECT_PREFIX,
    SegmentSelectMethod, _prefix_for, segment_bounds)
from inprocess.harnesses.viclip_scorer import VICLIP_SIDE         # noqa: E402
from inprocess.run import (                                       # noqa: E402
    FRAMES_PER_SEGMENT, IDENTITY_KNOB_DEFAULTS, IDENTITY_KNOBS,
    SEGMENTS_PER_VIDEO, build_parser, make_method, scan_output)


# ---------------------------------------------------------------- partition

def test_example_64s_is_8_segments_under_both_rules():
    for fps in (10, 25, 30, 60):
        n = 64 * fps
        t = segment_bounds(n, fps, 8, 8.0)
        assert t == [8 * fps * s for s in range(9)], (fps, t)
        assert t == segment_bounds(n, fps, 8, 0.0)


def test_meva_camera_is_38_segments_not_8():
    b = segment_bounds(9000, 30.0, 8, 8.0)            # 300 s at 30 fps
    assert len(b) - 1 == 38
    assert all(b[s + 1] - b[s] == 240 for s in range(37))
    assert b[-1] - b[-2] == 120                       # a 4 s tail stands alone
    assert segment_bounds(9000, 30.0, 8, 0.0) == [1125 * s for s in range(9)]


def test_short_tail_joins_the_last_segment():
    b = segment_bounds(37 * 240 + 119, 30.0, 8, 8.0)  # tail just under half
    assert len(b) - 1 == 37 and b[-1] - b[-2] == 240 + 119
    b = segment_bounds(37 * 240, 30.0, 8, 8.0)        # no tail at all
    assert len(b) - 1 == 37 and b[-1] == 37 * 240


def test_clip_shorter_than_a_window_is_one_segment():
    assert segment_bounds(100, 30.0, 8, 8.0) == [0, 100]
    assert segment_bounds(1, 30.0, 8, 8.0) == [0, 1]


def test_bounds_are_a_partition():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 40000))
        fps = float(rng.choice([7.5, 10, 23.976, 25, 29.97, 30, 59.94]))
        secs = float(rng.choice([1, 2.5, 8, 10]))
        b = segment_bounds(n, fps, 8, secs)
        assert b[0] == 0 and b[-1] == n
        assert all(b[i] < b[i + 1] for i in range(len(b) - 1)), (n, fps, b)
        w = max(1, round(fps * secs))
        assert all(b[i + 1] - b[i] == w for i in range(len(b) - 2))
        assert b[-1] - b[-2] < 1.5 * w or len(b) == 2


def test_time_partition_needs_a_frame_rate():
    for fps in (0, 0.0, None, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            segment_bounds(9000, fps, 8, 8.0)


def test_rows_can_rebuild_the_partition_from_their_stamps():
    """per_video_decode stamps n_total, n_segments and segment_frames (= w);
    a reader must rebuild a time partition from those, never the count rule."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        n = int(rng.integers(1, 40000))
        fps = float(rng.choice([10, 23.976, 25, 29.97, 30, 59.94]))
        b = segment_bounds(n, fps, 8, 8.0)
        w, S = ss.segment_window_frames(fps, 8.0), len(b) - 1
        assert [s * w for s in range(S)] + [n] == b


def test_count_partition_is_unchanged():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(1, 40000))
        spv = int(rng.integers(1, 20))
        S = max(1, min(spv, n))
        want = [round(s * n / S) for s in range(S + 1)]   # the pre-flag rule
        assert segment_bounds(n, 30.0, spv, 0.0) == want
        assert segment_bounds(n, 0.0, spv) == want        # fps never read


# ------------------------------------------------------------------- prompt

def test_count_partition_prompt_is_byte_identical():
    for glob, const in ((False, SEGMENT_SELECT_PREFIX),
                        (True, SEGMENT_SELECT_GLOBAL_PREFIX)):
        old = const.format(K=4, S=8, top=12, n=96)
        assert _prefix_for(glob, 0.0, 8).format(K=4, top=12, n=96) == old


def test_time_partition_prompt_names_the_window():
    p = _prefix_for(True, 8.0, 8).format(K=4, top=12, n=96)
    assert "consecutive segments of about 8 seconds" in p
    assert "equal time segments" not in p and "{" not in p
    old = SEGMENT_SELECT_GLOBAL_PREFIX.format(K=4, S=8, top=12, n=96)
    assert p.replace("consecutive segments of about 8 seconds",
                     "up to 8 equal time segments") == old


# ----------------------------------------------------------------- identity

def _args(**kw):
    """The runner's parsed defaults, overridden by ``kw``."""
    a = build_parser().parse_args(
        ["--subset", "x.json", "--video-root", ".", "--model", "m",
         "--strict-answer-prompt", "1"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_flag_defaults_to_the_count_partition():
    a = _args()
    assert a.segment_seconds == 0.0
    assert _args(segment_seconds=8).segment_seconds == 8


def _prior_values(seen, knob):
    """The values main() compares this run's knob against: everything the
    file holds for it, absence read through IDENTITY_KNOB_DEFAULTS."""
    return set(seen.get(knob, ()))


def test_identity_carries_the_window_and_separates_the_partitions():
    """The window is a run-defining knob: stamped on every row, read back by
    scan_output, so a file of 8-second rows refuses a count-partition
    append (and vice versa). Rows written before the flag existed carry no
    stamp and ARE count-partition rows (IDENTITY_KNOB_DEFAULTS): a count run still
    resumes such a file, a seconds run is refused by it."""
    assert "segment_seconds" in IDENTITY_KNOBS
    assert IDENTITY_KNOB_DEFAULTS["segment_seconds"] == 0.0
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "leg.jsonl")
        base = {"dataset": "toy", "id": 1, "method": "segment_select_viclip_auto",
                "backend": "m", "pass_idx": 1}
        with open(p, "w") as fh:
            fh.write(json.dumps(dict(base, segment_seconds=8.0)) + "\n")
        _, _, seen, _ = scan_output(p)
        prior = _prior_values(seen, "segment_seconds")
        assert prior == {8.0}
        # what main() does: any prior value other than this run's refuses
        assert not {v for v in prior if v != 8.0}                       # resumes itself
        assert {v for v in prior if v != 0.0}                           # refuses 0
        assert {v for v in prior if v != 4.0}                           # refuses 4
        with open(p, "w") as fh:
            fh.write(json.dumps(base) + "\n")                           # legacy row
        _, _, seen, _ = scan_output(p)
        prior = _prior_values(seen, "segment_seconds")
        assert prior == {0.0}                                           # absent = count
        assert not {v for v in prior if v != 0.0}                       # count run resumes
        assert {v for v in prior if v != 8.0}                           # seconds run refused
        # a knob with no default is still simply absent on a legacy row
        assert "dedup_tau" not in seen


def test_health_filter_reads_a_legacy_row_as_count_partition():
    """The end-of-run 'mine' filter applies the same default, so a legacy
    row counts toward a count leg and never toward a seconds leg."""
    legacy = {"method": "segment_select_viclip_auto"}
    for k in IDENTITY_KNOBS:
        legacy.setdefault(k, "x")
    legacy.pop("segment_seconds")
    def mine(knobs):
        return all(legacy.get(k, IDENTITY_KNOB_DEFAULTS.get(k, knobs[k])) == knobs[k]
                   for k in IDENTITY_KNOBS)
    knobs = {k: "x" for k in IDENTITY_KNOBS}
    assert mine(dict(knobs, segment_seconds=0.0))
    assert not mine(dict(knobs, segment_seconds=8.0))


def test_make_method_hands_the_window_to_the_arm():
    a = _args(segment_seconds=8, dedup_tau=1.0, seg_select="global", seg_floor=0)
    m = make_method("segment_select_viclip_auto", "segment_auto", "viclip", object(), a)
    assert m.segment_seconds == 8.0 and m.query == "auto" and m.seg_scorer == "viclip"
    assert m.segments_per_video == SEGMENTS_PER_VIDEO
    assert m.frames_per_segment == FRAMES_PER_SEGMENT
    m0 = make_method("segment_select_siglip_opt", "segment_opt",
                     "google/siglip-so400m-patch14-384", object(),
                     _args(dedup_tau=1.0))
    assert m0.segment_seconds == 0.0 and m0.query == "options"
    m1 = make_method("segment_select", "segment", "openai/clip-vit-base-patch32",
                     object(), _args(dedup_tau=1.0))
    assert m1.query == "question"


# --------------------------------------------------------------- thumbnails

def test_thumbnail_is_what_the_scorer_would_have_made():
    rng = np.random.default_rng(2)
    full = Image.fromarray(rng.integers(0, 256, (1072, 1920, 3), dtype=np.uint8))
    side = (VICLIP_SIDE, VICLIP_SIDE)
    from_full = np.asarray(full.convert("RGB").resize(side))       # _tube(full)
    thumb = full.convert("RGB").resize(side)                       # what we hold
    from_thumb = np.asarray(thumb.convert("RGB").resize(side))     # _tube(thumb)
    assert from_full.shape == (VICLIP_SIDE, VICLIP_SIDE, 3)
    assert np.array_equal(from_full, from_thumb)


def test_constructor():
    base = dict(name="segment_select_viclip_auto", dedup_tau=1.0)
    assert SegmentSelectMethod(object(), **base).segment_seconds == 0.0
    m = SegmentSelectMethod(object(), seg_scorer="viclip", segment_seconds=8, **base)
    assert m.segment_seconds == 8.0 and not m._needs_image_tower()
    with pytest.raises(ValueError):
        SegmentSelectMethod(object(), segment_seconds=-1, **base)
    assert SegmentSelectMethod(object(), name="segment_select_siglip_auto",
                               dedup_tau=1.0)._needs_image_tower()
    # dedup on needs the tower's per-frame embeddings even under viclip
    assert SegmentSelectMethod(object(), name="segment_select_viclip_auto",
                               dedup_tau=0.95, seg_scorer="viclip")._needs_image_tower()


# --------------------------------------------------- synthetic clip, no model

FPS, W, H = 10, 96, 64


def _write_clip(path, seconds, tint):
    cv2 = pytest.importorskip("cv2")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not vw.isOpened():
        pytest.skip("cv2 cannot encode mp4v here")
    for i in range(seconds * FPS):
        fr = np.full((H, W, 3), tint, dtype=np.uint8)
        fr[:, : 1 + (i * (W - 1)) // (seconds * FPS)] = 255   # a moving edge
        vw.write(fr)
    vw.release()


@pytest.fixture(scope="module")
def clips():
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for k, secs in enumerate((64, 64, 64, 30)):               # 8+8+8+4 segments
            p = os.path.join(d, f"cam{k + 1}.mp4")
            _write_clip(p, secs, 40 * (k + 1))
            paths.append(p)
        yield d, paths


def test_decode_time_partition_and_full_reload(clips):
    _, paths = clips
    m = SegmentSelectMethod(object(), name="segment_select_viclip_auto",
                            dedup_tau=1.0, seg_scorer="viclip", segment_seconds=8)
    thumbs, meta = m._decode_segments(paths[0], thumb=True)
    fulls, meta2 = m._decode_segments(paths[0], thumb=False)
    assert meta == meta2 and meta["n_segments"] == 8 and meta["segment_frames"] == 80
    assert [e[0] for e in thumbs] == [s for s in range(8) for _ in range(8)]
    assert [e[3] for e in thumbs] == [e[3] for e in fulls]        # same indices
    # 8 frames per 8 s window = 1 fps, centred: 5, 15, ... inside each window
    assert [e[3] for e in thumbs[:8]] == [5 + 10 * j for j in range(8)]
    assert all(e[2].size == (VICLIP_SIDE, VICLIP_SIDE) for e in thumbs)
    assert all(e[2].size == (W, H) for e in fulls)
    _, meta30 = m._decode_segments(paths[3], thumb=True)
    assert meta30["n_segments"] == 4                              # 8+8+8+6 s
    assert meta30["segment_frames"] == 80                         # w, not a length

    pool = [(1, sid, t, im) for sid, t, im, fi in thumbs]
    pool_fi = [fi for *_, fi in thumbs]
    kept = [j for j, e in enumerate(pool) if e[1] in (2, 5)]
    back = m._load_full_frames(paths, pool, pool_fi, kept)
    assert back == kept
    for j in kept:
        assert pool[j][3].size == (W, H)
        assert np.array_equal(np.asarray(pool[j][3]), np.asarray(fulls[j][2]))
    assert all(pool[j][3].size == (VICLIP_SIDE, VICLIP_SIDE)
               for j in range(len(pool)) if j not in kept)


def test_count_partition_decode_carries_no_window_stamp(clips):
    _, paths = clips
    m = SegmentSelectMethod(object(), name="segment_select", dedup_tau=1.0)
    entries, meta = m._decode_segments(paths[0])
    assert meta == {"n_total": 640, "fps": 10.0, "n_segments": 8}
    assert "segment_frames" not in meta
    assert all(e[2].size == (W, H) for e in entries)              # never thumbnails
    assert [e[3] for e in entries[:8]] == [5 + 10 * j for j in range(8)]


def _fake_scores(self, pool, queries):
    """Deterministic per-(camera, segment, query) scores; camera 2 is hottest."""
    out = {}
    for v, sid, t, im in pool:
        rng = np.random.default_rng(1000 * v + sid)
        out.setdefault(v, {})[sid] = rng.random(len(queries)) + (0.5 if v == 2 else 0)
    return out


def _record(rid, paths, question, options, task="toy"):
    rec = {"id": rid, "task_type": task, "question": question,
           "options": options, "answer": "A"}
    rec.update({f"video_{k + 1}": os.path.basename(p) for k, p in enumerate(paths)})
    return rec


def test_prepare_delivers_the_global_top12_at_full_resolution(clips, monkeypatch):
    d, paths = clips
    monkeypatch.setattr(SegmentSelectMethod, "_viclip_segment_scores", _fake_scores)
    rec = _record(7, paths,
                  "Order these: I. A man opens a door. II. A car stops. "
                  "III. A dog runs. Which sequence is correct?",
                  ["A. I -> II -> III", "B. III -> II -> I"],
                  task="CrossView-Toy-Event-Ordering")
    m = SegmentSelectMethod(object(), name="segment_select_viclip_auto",
                            dedup_tau=1.0, seg_scorer="viclip", query="auto",
                            seg_select="global", seg_floor=0, budget=96,
                            segment_seconds=8)
    m.pool_records = [rec]
    content, yn, gold, fa = m._prepare(rec, d)

    # the query set is the three statements, and the score is their max
    assert fa["query_mode"] == "statements" and fa["n_query_texts"] == 3
    assert fa["query_rule"] == "auto" and fa["query_dropped_options"] == {}
    assert fa["query_generic_test"] is True                       # a pool was given
    assert fa["query_texts"] == ["A man opens a door.", "A car stops.", "A dog runs."]
    raw = {}
    for v in range(1, 5):
        for sid in range(8 if v < 4 else 4):
            r = np.random.default_rng(1000 * v + sid).random(3) + (0.5 if v == 2 else 0)
            raw[(v, sid)] = float(r.max())
    assert {(int(v), int(s)): x for v, d_ in fa["segment_scores"].items()
            for s, x in d_.items()} == {k: round(x, 6) for k, x in raw.items()}

    # 28 candidate segments (8 + 8 + 8 + 4), the 12 best overall survive
    want = sorted(raw, key=lambda k: (-raw[k], k))[:12]
    got = [(int(v), s) for v, segs in fa["segments_kept_per_video"].items() for s in segs]
    assert sorted(got) == sorted(want)
    assert fa["segments_keep"] == 12 and fa["seg_floor"] == 0
    assert sum(1 for k in want if k[0] == 2) == 8          # no per-camera quota
    assert fa["selection_noop"] is False

    # 12 segments x 8 frames = 96 full-resolution frames reach the VLM
    images = [c["image"] for c in content if c.get("type") == "image"]
    assert len(images) == 96 == fa["n_selected"] == fa["n_kept_segment_frames"]
    assert all(im.size == (W, H) for im in images)
    assert fa["selection_on_thumbnails"] is True and fa["full_decode_dropped"] == 0
    assert fa["segment_seconds"] == 8.0 and fa["segments_per_video"] is None
    assert [pv["n_segments"] for pv in fa["per_video_decode"]] == [8, 8, 8, 4]
    assert all(pv["segment_frames"] == 80 for pv in fa["per_video_decode"])
    assert fa["image_tower_ran"] is False
    assert "consecutive segments of about 8 seconds" in content[0]["text"]
    # every kept frame's time lies inside one of its camera's kept 8 s windows
    for v, times in fa["selected_times_s"].items():
        segs = fa["segments_kept_per_video"][v]
        assert all(int(t // 8) in segs for t in times), (v, times, segs)


def test_prepare_with_the_image_tower_scores_full_frames(clips, monkeypatch):
    """The SigLIP arm under --segment-seconds: the image tower reads the
    frames, so nothing is thumbnailed and the same 12 x 8 frames come out."""
    d, paths = clips

    def fake_clip_scores(bundle, query, frames, batch=32, return_image_embs=False):
        assert all(im.size == (W, H) for im in frames)            # never thumbnails
        n_q = len(query) if isinstance(query, list) else None
        rng = np.random.default_rng(len(frames))
        sc = rng.random((len(frames), n_q)) if n_q else rng.random(len(frames))
        embs = rng.standard_normal((len(frames), 16))
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        return (sc, embs) if return_image_embs else sc

    monkeypatch.setattr(ss, "clip_scores", fake_clip_scores)
    monkeypatch.setattr(SegmentSelectMethod, "_ensure_clip", lambda self: None)
    rec = _record(9, paths, "What happens?",
                  ["A. A man opens a door", "B. A car stops at a light"])
    m = SegmentSelectMethod(object(), name="segment_select_siglip_auto",
                            dedup_tau=1.0, query="auto", seg_select="global",
                            seg_floor=0, budget=96, segment_seconds=8)
    content, yn, gold, fa = m._prepare(rec, d)
    assert fa["selection_on_thumbnails"] is False and fa["full_decode_dropped"] is None
    assert fa["image_tower_ran"] is True and fa["segment_seconds"] == 8.0
    assert fa["query_mode"] == "options" and fa["query_generic_test"] is False
    assert [pv["n_segments"] for pv in fa["per_video_decode"]] == [8, 8, 8, 4]
    assert fa["segments_kept_total"] == 12 and fa["n_selected"] == 96
    images = [c["image"] for c in content if c.get("type") == "image"]
    assert len(images) == 96


def test_record_with_fewer_segments_than_the_budget_keeps_them_all(clips, monkeypatch):
    """Cameras shorter than ~12 s give one segment each: top-12 of 2 keeps
    both, 16 frames are shown, and the row says selection did nothing."""
    d, paths = clips
    monkeypatch.setattr(SegmentSelectMethod, "_viclip_segment_scores", _fake_scores)
    short = []
    for k in range(2):
        p = os.path.join(d, f"short{k + 1}.mp4")
        _write_clip(p, 10, 60 * (k + 1))
        short.append(p)
    rec = _record(10, short, "What happens?",
                  ["A. A man opens a door", "B. A car stops at a light"])
    m = SegmentSelectMethod(object(), name="segment_select_viclip_auto",
                            dedup_tau=1.0, seg_scorer="viclip", query="auto",
                            seg_select="global", seg_floor=0, budget=96,
                            segment_seconds=8)
    content, yn, gold, fa = m._prepare(rec, d)
    assert [pv["n_segments"] for pv in fa["per_video_decode"]] == [1, 1]
    assert fa["segments_keep"] == 12 and fa["segments_kept_total"] == 2
    assert fa["selection_noop"] is True and fa["n_selected"] == 16


def test_per_clip_auto_k_caps_at_the_longest_clips_segment_count(clips, monkeypatch):
    """segments_keep 0 under --segment-seconds: the clamp reads the record's
    longest clip's segment count, not segments_per_video."""
    d, paths = clips
    monkeypatch.setattr(SegmentSelectMethod, "_viclip_segment_scores", _fake_scores)
    rec = _record(11, paths[3:], "What happens?",           # the 30 s clip alone
                  ["A. A man opens a door", "B. A car stops at a light"])
    m = SegmentSelectMethod(object(), name="segment_select_viclip_auto",
                            dedup_tau=1.0, seg_scorer="viclip", query="auto",
                            segments_keep=0, seg_pool=128, budget=96,
                            segment_seconds=8)
    content, yn, gold, fa = m._prepare(rec, d)
    # 128 // (8 x 1) = 16, clamped to the clip's 4 segments (not to S = 8)
    assert fa["segments_keep"] == 4 and fa["segments_keep_auto"] is True
    assert fa["segments_kept_per_video"] == {1: [0, 1, 2, 3]}
    assert fa["selection_noop"] is True


def test_prepare_count_partition_still_holds_full_frames(clips, monkeypatch):
    d, paths = clips
    monkeypatch.setattr(SegmentSelectMethod, "_viclip_segment_scores", _fake_scores)
    rec = _record(8, paths, "What happens?",
                  ["A. A man opens a door", "B. A car stops at a light"])
    m = SegmentSelectMethod(object(), name="segment_select_viclip_auto",
                            dedup_tau=1.0, seg_scorer="viclip", query="auto",
                            seg_select="global", seg_floor=0, budget=96)
    content, yn, gold, fa = m._prepare(rec, d)
    assert fa["segment_seconds"] is None and fa["segments_per_video"] == 8
    assert fa["selection_on_thumbnails"] is False and fa["full_decode_dropped"] is None
    assert fa["image_tower_ran"] is False                         # viclip, dedup off
    assert [pv["n_segments"] for pv in fa["per_video_decode"]] == [8, 8, 8, 8]
    assert "segment_frames" not in fa["per_video_decode"][0]
    assert fa["n_selected"] == 96 and "up to 8 equal time segments" in content[0]["text"]
    images = [c["image"] for c in content if c.get("type") == "image"]
    assert all(im.size == (W, H) for im in images)


def main():
    sys.exit(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    main()
