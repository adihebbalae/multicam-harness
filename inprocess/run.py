# New file (not a port): the package's entry point. docs/PORTING.md's
# byte-faithfulness rule covers ported bodies; nothing here is ported, and this
# file modifies nothing that already exists.
r"""Entry point for the ``inprocess`` arms: one subset x one model x N arms x P passes.

The package ships the arms but nothing that invokes them; this is that CLI. Run it
as a MODULE from the repo root — the package imports itself absolutely
(``from inprocess.harnesses...``), so ``python inprocess/run.py`` puts the wrong
directory on ``sys.path`` and dies before it reaches argparse:

    python -m inprocess.run \
        --subset data/subsets/mvueval_qa.json \
        --video-root <root holding the multi-view media> \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --strict-answer-prompt 1 \
        --no-reasoning --temperature 0.1 --passes 4 --seeds 1,2,3,4 --nframes 8

``--strict-answer-prompt 1`` above is not a default this file chose for you: the
flag is REQUIRED and has no default, and the 1 shown here is the value the legs
this file exists to reproduce were launched with. Read the next section before
copying it.

The command runs the three option-guided selection arms at a MATCHED frame budget
(``--budget 0``, the default -> nframes x K, the sequential arm's own budget).
Swap ``--model OpenGVLab/InternVL3-8B`` for the second backend; that one needs the
older-transformers environment, and ``--internvl-max-tiles`` must stay at 1 or the
selection arms refuse to construct, because an image is tiled while a video frame
is not and the "same budget" comparison would be a tokenization artifact.

One process cannot finish a subset of this size inside a normal wall clock. Shard
it — ``--chunk`` is the NUMBER OF SHARDS, not the shard size — and give every
shard its own output file, which the default ``--out`` naming already does:

    python -m inprocess.run --subset data/subsets/mvueval_qa.json \
        --video-root <root> --model Qwen/Qwen2.5-VL-7B-Instruct \
        --strict-answer-prompt 1 --no-reasoning --temperature 0.1 \
        --passes 4 --seeds 1,2,3,4 --nframes 8 \
        --chunk 8 --offset $SLURM_ARRAY_TASK_ID

THE KNOBS THAT MUST BE IDENTICAL AT BOTH SITES
----------------------------------------------
``--strict-answer-prompt {0,1}`` selects the answer-hygiene prompt version (v2
enumerates every legal letter and declares the options exhaustive). It rewrites
the prompt of essentially every record of a multi-option subset, so rows produced
with 1 ARE NOT poolable with rows produced with 0. It is REQUIRED with no default
precisely so neither operator can pick a side by omission: decide the value with
whoever else is running these legs BEFORE launching, pass the same number at both
sites, and check the ``strict_prompt`` field on the rows before pooling anything.

The package README documents ``STRICT_ANSWER_PROMPT`` in the environment as the
library-level switch, and it still is — this entry point simply takes the flag as
authoritative for its own rows and refuses to start when an exported value
disagrees with the flag, rather than letting a stale shell export change
generation invisibly.

Everything else that changes what the model is shown is either a flag that is
stamped on every row (``--nframes``, ``--budget``, ``--sel-tau``, ``--sel-tau-q``,
``--n-queries``, ``--max-new-tokens``, ``--internvl-max-tiles``, ``--temperature``,
``--seeds``, reasoning on/off) or a pinned module constant that is also stamped on
every row (the selection geometry: candidates per clip, thumbnails per clip, cell
px, and the per-clip frame floors). The constants are deliberately not flags —
they change what the model sees, so a leg that deviates cannot pool, and this
entry point exists to produce poolable rows. Appending to a file whose rows carry
a different value for any stamped knob is refused (see below).

RESUME AND ERROR POLICY
-----------------------
The file keeps exactly ONE row per ``(dataset, id, method, backend, pass_idx)``,
which is the reference runner's rule, and it is the rule because two operators'
JSONLs are meant to be concatenated: a consumer that pools them counts rows, so a
duplicated key inflates a denominator silently and permanently.

Concretely: a row is "done" whether or not it carries an ``error``, so a resume
never re-runs a key and never appends a second row for one. A failed key stays
failed until you ask for it back with ``--retry-errors``, which physically REMOVES
this run's error rows from the output file before the loop starts, so the retry
replaces them instead of shadowing them. The invariant is checked, not assumed:
duplicate keys in the output file are refused at startup (before any compute) and
again before the summary is written.

MEDIA LAYOUT
------------
``--video-root`` must be the directory the records' own paths are relative to, NOT
a flattened dump of every clip: a large minority of this subset's video references
carry a subdirectory component, and flattening the tree turns exactly those
records into unreadable-clip errors. A missing-file preflight runs before the
model loads, names what it could not find, and says which of the two failures it
looks like. ``--allow-missing-media`` downgrades it to a warning for a
nearly-complete tree; the affected records then become error rows, as they would
under the reference runner.

VICLIP
------
``clip_select_viclip_optu`` scores whole clips with a video-native embedding
loaded from a local download of the gated OpenGVLab/ViCLIP repo named by
``VICLIP_DIR``. That download is verified for real before the model loads — the
full-model checkpoint, the BPE vocab and the four class files — because the
alternative is discovering an incomplete download hours in, as a leg of error
rows. Note the repo also ships a larger vision-encoder-only ``.pth``; it is not a
substitute and the loader deliberately ignores it.

POOLING
-------
Question ids are small integers and collide across the subsets in this repo, so
the resume key and the append guard both include the dataset (the subset file's
basename). Rows also carry the subset's sha256, so two operators can prove they
ran the same questions rather than assume it. The default output name is
``results/<dataset>_<model>_strict<0|1>_reasoning<0|1>[_shard<i>].jsonl`` — the
protocol is in the filename so two prompt versions cannot land in one file even by
accident, and so the receiving side has something to glob for.
"""
import argparse
import hashlib
import json
import os
import re
import socket
from collections import Counter, defaultdict

from inprocess.dataloaders import qa_json
from inprocess.dataloaders.qa_json import media_remap, video_paths
from inprocess.evaluation.scoring import (format_summary,
                                          summarize_by_method_backend_passes)
from inprocess.harnesses.blind import BlindMethod
from inprocess.harnesses.decentralized import PerStreamMethod
from inprocess.harnesses.option_union import (OptionUnionClipSelect,
                                              OptionUnionFrameSelect,
                                              QuerySearchMethod)
from inprocess.harnesses.segment_select import SegmentSelectMethod
from inprocess.harnesses.stitched import CentralizedMethod
from inprocess.harnesses.uniform import (CVBenchNativeMethod,
                                         TemporalWeightedMethod)
from inprocess.models.clients import (INTERNVL_ALIASES, QWEN_ALIASES,
                                      make_backend)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The sequential baseline: every clip gets a flat --nframes, i.e. nframes x K
# frames per question — exactly what the selection arms match at --budget 0. It
# is the arm the selection arms are read against.
BASELINE = "cvbench_native"

# Selection-arm names are generated, not enumerated, so the scorer tag lands in
# the recorded method name and two scorers can never collide on a resume key.
OPTION_UNION_FRAME_RE = re.compile(
    r"^frame_select(?:_(?P<tag>(?!optu(?:_|$))[a-z0-9]+))?_optu$")
OPTION_UNION_CLIP_RE = re.compile(
    r"^clip_select(?:_(?P<tag>(?!optu(?:_|$))[a-z0-9]+))?_optu$")
QUERY_SEARCH_RE = re.compile(r"^query_search(?:_(?P<tag>[a-z0-9]+))?$")
# segment_select: top-K segments PER clip -> per-segment frames -> question-wide
# near-duplicate removal -> even thinning to the budget. Same tag/_opt grammar
# as frame_select, plus 'viclip' (each segment embedded jointly as one tube for
# segment relevance; an image tower still supplies the dedup embeddings).
SEGMENT_SELECT_RE = re.compile(
    r"^segment_select(?:_(?P<tag>(?!opt(?:_|$))[a-z0-9]+))?(?P<opt>_opt)?$")

# HF image-text scorers. 'viclip' is deliberately absent: it is not a
# transformers id but a local download named by VICLIP_DIR, and it embeds a
# whole clip jointly, so only the clip-level and segment-level arms can use it
# (they score whole clips / whole segment tubes, never single frames).
SCORER_ALIASES = {"siglip": "google/siglip-so400m-patch14-384",
                  "siglip2": "google/siglip2-so400m-patch14-384"}
DEFAULT_SCORER = "openai/clip-vit-base-patch32"
VICLIP_SCORER = "viclip"

# Selection geometry, pinned at the reference runner's values rather than
# exposed as flags: they change what the model is shown, so a leg that deviates
# cannot be pooled with one that does not. Every one of them is stamped on every
# row, so the other site can verify rather than trust.
FRAME_CANDIDATES = 32      # uniform candidate frames decoded per clip
SEL_THUMBS = 8             # thumbnails scored per clip by the clip-level arm
CELL_PX = 448              # frame size on backends without per-item pixel caps
FRAME_FLOOR = 1            # frame-level arms: per-clip minimum kept frames
CLIP_FLOOR = 2             # clip-level arm: per-clip minimum allocated frames
TEMPORAL_FLOOR = 2         # temporal_weighted: per-clip minimum frames
SEGMENTS_PER_VIDEO = 8     # segment_select: segments each clip is split into
FRAMES_PER_SEGMENT = 8     # segment_select: frames sampled per segment
# segment_select_viclip*: the dedup tower is pinned to the reference campaign's
# SigLIP id, so viclip-vs-siglip scorer comparisons share one dedup space and
# the only moving part is what scores the segments.
SEG_DEDUP_TOWER = SCORER_ALIASES["siglip"]

DEFAULT_METHODS = "frame_select_siglip_optu,clip_select_viclip_optu,query_search_siglip"

# The kinds whose arms SELECT frames and send them as images. The tile-parity
# guard and the still-image preflight apply to these, not to the classic arms:
# a montage leg legitimately tiles (its budget saturates at max_tiles — compare
# its token columns before reading a centralized-vs-sequential difference as
# architecture), and blind reads no media at all.
SELECTION_KINDS = {"frame_optu", "clip_optu", "query_search",
                   "segment", "segment_opt"}

# Run-defining values compared against the rows already in an output file before
# appending to it. A knob absent from a prior row is not compared (that row
# predates the stamp); a knob present and different refuses the append, because
# the alternative is a resume that runs nothing and writes a summary labelled
# with this run's knobs over another run's data.
IDENTITY_KNOBS = ("dataset", "backend", "subset_sha256", "strict_prompt",
                  "reasoning", "temperature", "seeds", "nframes", "budget",
                  "sel_tau", "sel_tau_q", "n_queries", "query_max_new_tokens",
                  "max_new_tokens", "internvl_max_tiles", "frame_candidates",
                  "sel_thumbs", "cell_px", "frame_floor", "clip_floor",
                  "segments_per_video", "segments_keep", "frames_per_segment",
                  "dedup_tau", "seg_pool", "total_frames", "montage_kind",
                  "stream_kind", "weighting", "perception_max_new_tokens")

# The four ViCLIP class files, in the order the loader executes them.
VICLIP_CLASS_FILES = ("simple_tokenizer.py", "viclip_text.py",
                      "viclip_vision.py", "viclip.py")
VICLIP_VOCAB = "bpe_simple_vocab_16e6.txt.gz"


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def parse_method(mname):
    """``(kind, scorer)`` for a supported arm name; ``SystemExit`` otherwise.

    Called for every requested arm BEFORE the model is loaded, so a typo in a
    name or a scorer tag costs a second rather than a model load and a GPU
    allocation.
    """
    if mname == BASELINE:
        return "baseline", None

    if mname in ("centralized", "per_stream", "temporal_weighted", "blind"):
        return mname, None

    def scorer_for(tag):
        if tag is None:
            return DEFAULT_SCORER
        if tag not in SCORER_ALIASES:
            raise SystemExit(f"unknown scorer tag '{tag}' in method '{mname}'. "
                             f"Known: {sorted(SCORER_ALIASES)}")
        return SCORER_ALIASES[tag]

    m = OPTION_UNION_FRAME_RE.match(mname)
    if m:
        if m.group("tag") == VICLIP_SCORER:
            raise SystemExit(
                "frame_select_viclip_optu: ViCLIP embeds a whole clip jointly "
                "and has no per-frame scores — use clip_select_viclip_optu, or "
                "a CLIP/SigLIP tag for the frame-level arm.")
        return "frame_optu", scorer_for(m.group("tag"))
    m = OPTION_UNION_CLIP_RE.match(mname)
    if m:
        tag = m.group("tag")
        return "clip_optu", (VICLIP_SCORER if tag == VICLIP_SCORER else scorer_for(tag))
    m = QUERY_SEARCH_RE.match(mname)
    if m:
        if m.group("tag") == VICLIP_SCORER:
            raise SystemExit("query_search_viclip: ViCLIP has no per-frame "
                             "scores; use a CLIP/SigLIP tag.")
        return "query_search", scorer_for(m.group("tag"))
    m = SEGMENT_SELECT_RE.match(mname)
    if m:
        tag = m.group("tag")
        kind = "segment_opt" if m.group("opt") else "segment"
        return kind, (VICLIP_SCORER if tag == VICLIP_SCORER else scorer_for(tag))
    raise SystemExit(
        f"unknown method '{mname}'. Known: {BASELINE}, centralized, per_stream, "
        "temporal_weighted, blind, "
        "frame_select[_<scorer>]_optu, clip_select[_<scorer>|_viclip]_optu, "
        "query_search[_<scorer>], segment_select[_<scorer>|_viclip][_opt], "
        f"with <scorer> in {sorted(SCORER_ALIASES)}.")


def make_method(mname, kind, scorer, backend, args):
    """Build one arm, then re-read what it actually stored.

    ``budget`` and ``floor`` are passed EXPLICITLY to every selection arm rather
    than left to defaulting. The option-union / query-search constructors do not
    name ``budget`` in their own signatures — it reaches the parent through
    ``**kw``, and the parent's default is a flat frame count per question, not
    the matched nframes x K. Omitting it therefore does not fail; it silently
    runs a different experiment that cannot be pooled. The checks below re-read
    the attributes the arm ended up with, so the same trap cannot reappear if a
    signature changes underneath this file.
    """
    common = dict(nframes=args.nframes, max_new_tokens=args.max_new_tokens,
                  temperature=args.temperature, reasoning=not args.no_reasoning)
    if kind == "baseline":
        method = CVBenchNativeMethod(backend, total_frames=args.total_frames,
                                     **common)              # --budget does not apply
        want = {"total_frames": args.total_frames}
    elif kind == "centralized":
        method = CentralizedMethod(backend, cell_px=CELL_PX,
                                   montage_kind=args.montage_kind,
                                   total_frames=args.total_frames, **common)
        want = {"cell_px": CELL_PX, "montage_kind": args.montage_kind,
                "total_frames": args.total_frames}
    elif kind == "per_stream":
        method = PerStreamMethod(
            backend, perception_max_new_tokens=args.perception_max_new_tokens,
            stream_kind=args.stream_kind, total_frames=args.total_frames,
            **common)
        want = {"perception_max_new_tokens": args.perception_max_new_tokens,
                "stream_kind": args.stream_kind,
                "total_frames": args.total_frames}
    elif kind == "temporal_weighted":
        method = TemporalWeightedMethod(backend, budget=args.budget,
                                        floor=TEMPORAL_FLOOR,
                                        weighting=args.weighting, **common)
        want = {"budget": args.budget, "floor": TEMPORAL_FLOOR,
                "weighting": args.weighting}
    elif kind == "blind":
        method = BlindMethod(backend, **common)
        want = {}
    elif kind == "frame_optu":
        method = OptionUnionFrameSelect(
            backend, tau=args.sel_tau, tau_q=args.sel_tau_q,
            budget=args.budget, floor=FRAME_FLOOR,
            candidates_per_video=FRAME_CANDIDATES, clip_model=scorer,
            cell_px=CELL_PX, name=mname, **common)
        want = {"budget": args.budget, "floor": FRAME_FLOOR,
                "candidates_per_video": FRAME_CANDIDATES, "cell_px": CELL_PX,
                "tau": args.sel_tau, "tau_q": args.sel_tau_q}
    elif kind == "clip_optu":
        method = OptionUnionClipSelect(
            backend, scorer=scorer, tau=args.sel_tau, tau_q=args.sel_tau_q,
            thumbs=SEL_THUMBS, budget=args.budget, floor=CLIP_FLOOR,
            name=mname, **common)
        want = {"budget": args.budget, "floor": CLIP_FLOOR, "thumbs": SEL_THUMBS,
                "tau": args.sel_tau, "tau_q": args.sel_tau_q}
    elif kind in ("segment", "segment_opt"):
        # under the viclip tag the tube scorer replaces only the segment-
        # RELEVANCE signal; the dedup step needs per-frame image embeddings (a
        # joint tube embedding has none), so an image tower runs alongside —
        # pinned to SEG_DEDUP_TOWER so scorer A/Bs share one dedup space.
        seg_scorer = VICLIP_SCORER if scorer == VICLIP_SCORER else None
        method = SegmentSelectMethod(
            backend, budget=args.budget, floor=FRAME_FLOOR,
            segments_per_video=SEGMENTS_PER_VIDEO,
            segments_keep=args.segments_keep,
            frames_per_segment=FRAMES_PER_SEGMENT,
            dedup_tau=args.dedup_tau, seg_scorer=seg_scorer,
            seg_pool=args.seg_pool,
            clip_model=SEG_DEDUP_TOWER if seg_scorer else scorer,
            cell_px=CELL_PX, name=mname,
            query="options" if kind == "segment_opt" else "question", **common)
        want = {"budget": args.budget, "floor": FRAME_FLOOR,
                "segments_per_video": SEGMENTS_PER_VIDEO,
                "segments_keep": args.segments_keep,
                "frames_per_segment": FRAMES_PER_SEGMENT,
                "dedup_tau": args.dedup_tau, "seg_pool": args.seg_pool,
                "cell_px": CELL_PX}
    else:
        method = QuerySearchMethod(
            backend, n_queries=args.n_queries,
            query_max_new_tokens=args.query_max_new_tokens,
            budget=args.budget, floor=FRAME_FLOOR,
            candidates_per_video=FRAME_CANDIDATES, clip_model=scorer,
            cell_px=CELL_PX, name=mname, **common)
        want = {"budget": args.budget, "floor": FRAME_FLOOR,
                "candidates_per_video": FRAME_CANDIDATES, "cell_px": CELL_PX,
                "n_queries": args.n_queries}
    bad = {k: getattr(method, k, None) for k, v in want.items()
           if getattr(method, k, None) != v}
    if bad:
        raise SystemExit(
            f"{mname}: the arm did not store what it was asked for — "
            + ", ".join(f"{k}: asked {want[k]!r}, stored {v!r}" for k, v in bad.items())
            + ".\n  A keyword that does not reach the frame budget means this leg "
              "runs a different experiment from the one it is labelled with and "
              "cannot be pooled. Fix the constructor before launching.")
    # temporal_weighted deliberately renames itself 'temporal_even' under the
    # even weighting so the two variants never collide on a resume key; every
    # other arm must record exactly the name it was launched under.
    renamed_ok = (mname == "temporal_weighted"
                  and method.name in ("temporal_weighted", "temporal_even"))
    if method.name != mname and not renamed_ok:
        raise SystemExit(
            f"{mname}: the arm records itself as '{method.name}'. Rows and resume "
            "keys use the recorded name, so the two must agree or a resume "
            "re-runs everything.")
    return method


# --------------------------------------------------------------------------- #
# output file: keys, identity, error policy
# --------------------------------------------------------------------------- #
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_rows(path):
    """Every parseable row of ``path``. A SIGKILL mid-write (a wall-clock kill)
    can leave one torn trailing line; skip it here so the resume, the guards and
    the summary all see the same rows."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def row_key(row):
    """``(dataset, id, method, backend, pass_idx)`` — the one row-per-key key.

    The DATASET is part of it because question ids are small integers that
    repeat across the subsets in this repo: without it, a relaunch pointed at
    another subset's file would treat foreign keys as done and report a full run
    having done nothing.
    """
    return (row.get("dataset"), row.get("id"), row.get("method"),
            row.get("backend"), row.get("pass_idx"))


def scan_output(path):
    """``(key counts, keys whose row carries an error, {knob: values seen},
    media_remap values seen — 'unstamped' for rows that predate the stamp)``."""
    counts = Counter()
    errored = set()
    seen = defaultdict(set)
    remaps = set()
    for row in iter_rows(path):
        key = row_key(row)
        counts[key] += 1
        if row.get("error"):
            errored.add(key)
        remaps.add(row.get("media_remap", "unstamped"))
        for knob in IDENTITY_KNOBS:
            if knob in row:
                value = row[knob]
                seen[knob].add(value if isinstance(value, (str, int, float, bool))
                               or value is None else json.dumps(value, sort_keys=True))
    return counts, errored, seen, remaps


def drop_error_rows(path, keys):
    """Rewrite ``path`` without the error rows whose key is in ``keys``.

    This is what makes ``--retry-errors`` safe: the retry REPLACES the failed row
    instead of appending a second row for the same key. Written to a sibling file
    and moved into place, so an interruption leaves the original intact. Blank
    and torn lines are dropped in passing, which is the same set the readers
    already skip.
    """
    if not keys or not os.path.exists(path):
        return 0
    tmp = path + ".rewrite"
    dropped = 0
    with open(path) as src, open(tmp, "w") as dst:
        for line in src:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error") and row_key(row) in keys:
                dropped += 1
                continue
            dst.write(line if line.endswith("\n") else line + "\n")
    os.replace(tmp, path)
    return dropped


def describe_duplicates(counts, limit=5):
    dups = [k for k, n in counts.items() if n > 1]
    if not dups:
        return None
    shown = "\n".join(f"    {k} x{counts[k]}" for k in dups[:limit])
    return (f"{len(dups)} duplicated (dataset, id, method, backend, pass) key(s):\n"
            f"{shown}\n" + ("    ...\n" if len(dups) > limit else ""))


def run_identity(subset):
    """``(dataset, run_id, node)`` stamped onto every row of this run."""
    dataset = os.path.splitext(os.path.basename(subset))[0]
    jid = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    node = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    run_id = (f"slurm-{jid}" + (f"_{task}" if task else "")) if jid else \
        f"{node}-{os.getpid()}"
    return dataset, run_id, node


# --------------------------------------------------------------------------- #
# prechecks
# --------------------------------------------------------------------------- #
def viclip_precheck():
    """Verify the ViCLIP download before anything loads a model.

    The scorer is not a transformers checkpoint: it is a local copy of a gated
    repo, and it needs the full-model checkpoint, the BPE vocab and four class
    files. An incomplete copy raises inside the arm's per-record prepare step,
    which is caught into an error row — so without this the failure surfaces as a
    whole leg of error rows, hours in, after the other arms have been paid for.
    The names are read off the scorer module itself rather than restated here, so
    the two cannot drift apart.
    """
    from inprocess.harnesses import viclip_scorer
    root = viclip_scorer.VICLIP_DIR
    size = viclip_scorer.VICLIP_SIZE
    ckpts = tuple(viclip_scorer._CKPTS)
    recipe = ("  huggingface-cli login\n"
              f"  huggingface-cli download OpenGVLab/ViCLIP --local-dir {root}")
    if size != "l":
        raise SystemExit(
            f"VICLIP_SIZE={size}: the scorer supports only 'l' and would die in a "
            "strict state-dict load at job start. Unset VICLIP_SIZE.")
    if not os.path.isdir(root):
        raise SystemExit(
            f"VICLIP_DIR={root} is not a directory. The clip-level arm's "
            "video-native scorer loads from a local download of the gated "
            f"OpenGVLab/ViCLIP repo:\n{recipe}")
    # the class files and the vocab may sit either flat in VICLIP_DIR or inside a
    # viclip/ package directory, which is exactly what the loader accepts
    pkg = os.path.join(root, "viclip")
    if not os.path.isdir(pkg):
        pkg = root
    missing = [os.path.join(pkg, f) for f in (VICLIP_VOCAB,) + VICLIP_CLASS_FILES
               if not os.path.exists(os.path.join(pkg, f))]
    have_ckpt = any(os.path.exists(os.path.join(root, c)) for c in ckpts)
    if not have_ckpt:
        strays = sorted(f for f in os.listdir(root)
                        if f.endswith(".pth") and f not in ckpts)
        detail = ""
        if strays:
            detail = (f"\n  {root} holds {strays}, which is not it: the larger "
                      ".pth on that repo is the vision-encoder-only file and "
                      "dies on a missing state-dict key, so it is deliberately "
                      "never used as a fallback.")
        missing.append(f"{os.path.join(root, ckpts[0])} (the full-model checkpoint)"
                       + detail)
    if missing:
        shown = "\n".join(f"    {m}" for m in missing)
        raise SystemExit(
            "the ViCLIP download under VICLIP_DIR is incomplete — missing:\n"
            f"{shown}\n  Re-download the repo in full (no --include filters); the "
            "clip-level arm cannot run without every one of these, and finding "
            f"out at the first record costs a whole leg:\n{recipe}")


def preflight_media(data, video_root, need_video, allow_missing):
    """Resolve every referenced clip before the model loads.

    A missing clip is not fatal to the arms — they raise and the row is written
    as an error — but a leg that is half error rows is worse than a refusal at
    second zero, and error rows are terminal under this file's resume policy. The
    hint branches, because "the root is wrong" and "the root was flattened" are
    different mistakes with the same symptom.
    """
    missing, seen, no_video = [], set(), 0
    unremuxed = []
    for rec in data:
        try:
            paths = video_paths(rec, video_root)
        except FileNotFoundError as e:
            # the resolver refuses a bare .avi (decord decodes the wrong frames
            # out of those containers); a missing sibling must fail here, at
            # second zero, not after the model load on a GPU node
            unremuxed.append(str(e).split(":")[0])
            continue
        if not paths:
            no_video += 1
            continue
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            if not os.path.exists(path):
                missing.append(path)
    if unremuxed:
        raise SystemExit(
            f"{len(unremuxed)} record(s) name an .avi without a verified .mp4 "
            "sibling. decord decodes the wrong frames out of a bare .avi, so "
            "this is deliberately not downgradable by --allow-missing-media: "
            "run scripts/data/remux_avi.py, then its --check. First: "
            f"{unremuxed[:3]}")
    if no_video and need_video:
        raise SystemExit(
            f"{no_video} record(s) carry no video_i slot. The selection arms "
            "sample frames out of clips and cannot run on still-image records; "
            "check that --subset is the multi-view VIDEO subset.")
    if not missing:
        return len(seen), 0
    root = os.path.abspath(video_root)
    in_subdir = sum(1 for p in missing
                    if os.path.dirname(os.path.relpath(os.path.abspath(p), root)))
    if len(missing) == len(seen):
        hint = ("  every reference is missing, so --video-root is pointing at the "
                "wrong place entirely — it must be the directory the records' own "
                "relative paths hang off.")
    elif in_subdir >= max(1, len(missing) // 2):
        hint = ("  most of the missing references carry a subdirectory component, "
                "which is the flattened-tree case: --video-root must preserve the "
                "dataset's own subdirectories, not be a dump of every clip into "
                "one directory.")
    else:
        hint = "  those clips are absent from an otherwise resolvable tree."
    shown = "\n".join(f"    {p}" for p in missing[:5])
    report = (f"{len(missing)} of {len(seen)} referenced clips are missing under "
              f"--video-root={video_root}:\n{shown}\n"
              + ("    ...\n" if len(missing) > 5 else "") + hint)
    if not allow_missing:
        raise SystemExit(report + "\n  Fix the tree, or pass --allow-missing-media "
                                  "to run anyway and take error rows for them.")
    print("[warn] " + report)
    print("[warn] --allow-missing-media: those records will be written as error "
          "rows, which this file treats as terminal — re-run them with "
          "--retry-errors once the media is in place.", flush=True)
    return len(seen), len(missing)


def jsonable(o):
    """Last-resort encoder: values that slipped into ``frame_alloc`` become plain
    Python rather than killing a run that has already paid for the generation."""
    tolist = getattr(o, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass
    item = getattr(o, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(o)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m inprocess.run",
        description="Run the in-process selection arms over one subset. Invoke as "
                    "a module from the repo root; the package imports itself "
                    "absolutely, so running the file by path fails to import.")
    ap.add_argument("--subset", required=True,
                    help="question JSON (a top-level list of records)")
    ap.add_argument("--video-root", required=True,
                    help="root the records' video_i paths are relative to; must "
                         "preserve the dataset's own subdirectories")
    ap.add_argument("--model", required=True,
                    help="full HF id (e.g. Qwen/Qwen2.5-VL-7B-Instruct, "
                         "OpenGVLab/InternVL3-8B), or a short alias in "
                         f"{sorted(QWEN_ALIASES) + sorted(INTERNVL_ALIASES)}. One "
                         "backend per invocation: the two need different "
                         "environments.")
    ap.add_argument("--strict-answer-prompt", type=int, choices=(0, 1), required=True,
                    help="REQUIRED, no default: 1 selects the answer-hygiene "
                         "prompt (every legal letter enumerated, options declared "
                         "exhaustive), 0 the original. It rewrites the prompt of "
                         "nearly every record, so rows made with different values "
                         "CANNOT be pooled. Agree the value with whoever else is "
                         "running these legs; it is recorded per row as "
                         "strict_prompt and is part of the default output name.")
    ap.add_argument("--methods", default=DEFAULT_METHODS,
                    help="comma list; default = the three option-guided arms. "
                         f"Also accepts {BASELINE} (the sequential baseline the "
                         "matched budget is matched to), the classic arms "
                         "centralized / per_stream / temporal_weighted, and the "
                         "text-prior floor blind.")
    ap.add_argument("--total-frames", type=int, default=0,
                    help="centralized/per_stream/" f"{BASELINE}: hold the TOTAL "
                         "frame count per question fixed (split evenly across its "
                         "clips) instead of a flat --nframes per clip; 0 = off")
    ap.add_argument("--montage-kind", default="camera",
                    choices=["camera", "video", "view"],
                    help="centralized montage framing: 'camera' (synced views, "
                         "default), 'video' (independent clips), or 'view'; the "
                         "preamble wording and the burned-in cell labels follow it")
    ap.add_argument("--stream-kind", default="camera",
                    choices=["camera", "video", "view"],
                    help="per_stream: label/phrase clips as synced 'camera' views "
                         "or independent 'video' clips, mirroring --montage-kind")
    ap.add_argument("--weighting", default="duration",
                    choices=["duration", "even"],
                    help="temporal_weighted: split the budget by clip duration "
                         "('duration') or evenly ('even', the budget-matched "
                         "control; rows record method=temporal_even)")
    ap.add_argument("--perception-max-new-tokens", type=int, default=1024,
                    help="per_stream: token cap for each per-view perception call "
                         "(1024 truncates thinking backends mid-<think>; raise "
                         "for those)")
    ap.add_argument("--nframes", type=int, default=8,
                    help="frames per clip for the sequential baseline, and the "
                         "per-clip term of the matched budget (nframes x K)")
    ap.add_argument("--passes", type=int, default=4,
                    help="sampled generations per question; the std is taken over "
                         "these, with the selection held fixed")
    ap.add_argument("--seeds", default="1,2,3,4",
                    help="comma seeds, one per pass; must cover --passes")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--budget", type=int, default=0,
                    help="selection arms: TOTAL frames per question. 0 (the "
                         "default) = MATCHED, i.e. nframes x K, the sequential "
                         "arm's own budget. A positive value is an absolute "
                         "budget and breaks matching — say so when reporting.")
    ap.add_argument("--sel-tau", type=float, default=0.0,
                    help="option-union arms: absolute per-option similarity cutoff "
                         "(scorer-specific); 0 = quantile mode")
    ap.add_argument("--sel-tau-q", type=float, default=0.85,
                    help="option-union arms: per-option quantile used when "
                         "--sel-tau is 0 — a frame/clip passes an option when it "
                         "is in that option's top (1-q) fraction")
    ap.add_argument("--n-queries", type=int, default=4,
                    help="query_search: visual search phrases per question")
    ap.add_argument("--segments-keep", type=int, default=4,
                    help="segment_select: top-K segments kept per clip "
                         "(straight top-K over the per-option score matrix, "
                         "descending). 0 = AUTO: K = --seg-pool // "
                         "(frames_per_segment x n_streams), clamped to "
                         "[1, min(16, segments_per_video)]")
    ap.add_argument("--seg-pool", type=int, default=128,
                    help="segment_select with --segments-keep 0: pooled-frame "
                         "target the auto top-K fills with whole segments, "
                         "split evenly across the record's streams (128 with "
                         "8-frame segments: 4 streams -> 4 segments/clip, "
                         "16+ -> 1). Auto K never exceeds segments_per_video, "
                         "so reaching 16 on few-stream records needs a larger "
                         "segment split")
    ap.add_argument("--dedup-tau", type=float, default=0.95,
                    help="segment_select: near-duplicate cosine cutoff in "
                         "(0, 1]. 1 = dedup OFF; 0 is INVALID (fails fast) — "
                         "NOT the --sel-tau '0 = off' convention. Static "
                         "cameras collapse hard at 0.95: budget-parity sweeps "
                         "must pass 1")
    ap.add_argument("--query-max-new-tokens", type=int, default=256,
                    help="query_search: token cap for the phrase-writing call")
    ap.add_argument("--max-new-tokens", type=int, default=8192,
                    help="token cap for the answer call")
    ap.add_argument("--internvl-max-tiles", type=int, default=1,
                    help="InternVL tiles per image. Keep 1 for the selection arms: "
                         "an image tiles while a video frame does not, so anything "
                         "higher inflates their visual budget and the arms refuse "
                         "to construct.")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="direct-answer prompt: no <think> trace requested. There "
                         "is no model-side switch — the visible reasoning is "
                         "produced BY the prompt. Pair with a low --temperature. "
                         "Recorded per row and part of the default output name.")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N records after sharding (smoke test)")
    ap.add_argument("--chunk", type=int, default=0,
                    help="NUMBER OF SHARDS, not shard size — sharding is strided "
                         "(data[offset::chunk]), so this must equal the array "
                         "width or each shard silently runs a fraction of the "
                         "questions and still writes a normal-looking summary")
    ap.add_argument("--offset", type=int, default=0,
                    help="this shard's index in [0, chunk)")
    ap.add_argument("--out", default=None,
                    help="JSONL to append to (default: results/<dataset>_<model>_"
                         "strict<0|1>_reasoning<0|1>[_shard<i>].jsonl). Sharded "
                         "array tasks must each get their own file.")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-run the keys whose rows carry an error, REMOVING "
                         "those rows from the output file first so the retry "
                         "replaces them. Without this a failed key stays failed: "
                         "the file holds exactly one row per key, which is what "
                         "lets two operators' files be concatenated.")
    ap.add_argument("--allow-missing-media", action="store_true",
                    help="downgrade the missing-clip preflight to a warning and "
                         "take error rows for the affected records")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="permit appending to a file whose rows carry a different "
                         "backend, dataset, run-defining knob or media-provenance "
                         "stamp (default: refuse — "
                         "the resume key cannot tell those rows apart, so the run "
                         "would skip work and label another protocol's data with "
                         "this run's summary)")
    return ap


def resolve_strict_prompt(flag):
    """The flag is authoritative; an exported value that disagrees is fatal.

    The prompt version is read off the module global at generation time, so
    setting it here reaches every arm. The environment variable remains the
    library-level switch the package README documents — this entry point just
    refuses to let a stale export and an explicit flag disagree silently, since
    the disagreement is invisible in the output but changes every prompt.
    """
    raw = os.environ.get("STRICT_ANSWER_PROMPT")
    want = bool(flag)
    if raw is not None and (raw == "1") != want:
        raise SystemExit(
            f"STRICT_ANSWER_PROMPT={raw} is exported but --strict-answer-prompt "
            f"{flag} was passed. The flag is this runner's only authority over the "
            "prompt version, and the two disagreeing means one of them is not what "
            "you meant: unset the variable, or export the value you are passing. "
            "Whichever you settle on must match the operator you are pooling rows "
            "with.")
    qa_json.STRICT_ANSWER_PROMPT = want
    return want


def main():
    args = build_parser().parse_args()
    strict_prompt = resolve_strict_prompt(args.strict_answer_prompt)
    reasoning = not args.no_reasoning

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise SystemExit("--methods is empty")
    if len(set(methods)) != len(methods):
        raise SystemExit(f"--methods repeats an arm: {methods}. Every row would "
                         "be attempted twice under one key.")
    parsed = {m: parse_method(m) for m in methods}      # fails fast on a typo
    kinds = {m: k for m, (k, _) in parsed.items()}
    scorers = {s for _, s in parsed.values()}

    if args.budget < 0:
        raise SystemExit(f"--budget {args.budget}: use 0 for the matched "
                         "nframes x K budget, or a positive frame count.")
    if args.passes < 1:
        raise SystemExit(f"--passes {args.passes}: need at least one pass.")
    if args.budget and BASELINE in kinds:
        print(f"[note] --budget {args.budget} does not apply to {BASELINE}; it "
              f"always shows --nframes {args.nframes} frames per clip.")
    if "temporal_weighted" in kinds.values() and args.budget == 0:
        raise SystemExit(
            "temporal_weighted does not implement the matched --budget 0 "
            "convention — at 0 it would allocate nothing and answer blind. "
            "Pass an explicit --budget (the arm's historic value is 64).")
    if "internvl" in args.model.lower() and args.internvl_max_tiles != 1 \
            and any(k in SELECTION_KINDS for k in kinds.values()):
        raise SystemExit(
            f"--internvl-max-tiles {args.internvl_max_tiles} with a selection arm: "
            "selected frames are sent as images, which tile, while the sequential "
            "arm's video frames do not — the matched-budget comparison would be a "
            "tokenization artifact. Run those legs with 1, and any higher-tile "
            "montage leg separately.")
    if VICLIP_SCORER in scorers:
        viclip_precheck()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()][:args.passes]
    if len(seeds) < args.passes:
        raise SystemExit(f"need >= {args.passes} seeds, got {seeds}")

    with open(args.subset) as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise SystemExit(f"{args.subset}: expected a top-level list of records")
    subset_sha = sha256_of(args.subset)
    n_all = len(data)
    if not 0 <= args.offset < max(args.chunk, 1):
        raise SystemExit(f"--offset {args.offset} outside [0, {max(args.chunk, 1)}) "
                         f"for --chunk {args.chunk} (chunk is the SHARD COUNT)")
    if args.chunk > 1:
        data = data[args.offset::args.chunk]
    if args.limit:
        data = data[:args.limit]
    if not data:
        raise SystemExit("no records selected (check --chunk/--offset/--limit)")
    ids = [rec.get("id") for rec in data]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{args.subset}: record ids are not unique within this "
                         "shard, so rows could not be keyed. Fix the subset.")

    dataset, run_id, node = run_identity(args.subset)
    # Rows record backend.name, which is the HF id's BASENAME — an alias has to be
    # resolved the same way the factory resolves it, or the guards below compare
    # an alias against a model name and refuse every append.
    model_tag = (QWEN_ALIASES.get(args.model) or INTERNVL_ALIASES.get(args.model)
                 or args.model).rstrip("/").split("/")[-1]

    knobs = {
        "dataset": dataset,
        "backend": model_tag,
        "subset_sha256": subset_sha,
        "strict_prompt": strict_prompt,
        "reasoning": reasoning,
        "temperature": args.temperature,
        "seeds": ",".join(str(s) for s in seeds),
        "nframes": args.nframes,
        "budget": args.budget,      # requested; the effective one, and whether it
                                    # was matched, land in the row's frame_alloc
        "sel_tau": args.sel_tau,
        "sel_tau_q": args.sel_tau_q,
        "n_queries": args.n_queries,
        "query_max_new_tokens": args.query_max_new_tokens,
        "max_new_tokens": args.max_new_tokens,
        "internvl_max_tiles": args.internvl_max_tiles,
        "frame_candidates": FRAME_CANDIDATES,
        "sel_thumbs": SEL_THUMBS,
        "cell_px": CELL_PX,
        "frame_floor": FRAME_FLOOR,
        "clip_floor": CLIP_FLOOR,
        "segments_per_video": SEGMENTS_PER_VIDEO,
        "segments_keep": args.segments_keep,
        "frames_per_segment": FRAMES_PER_SEGMENT,
        "dedup_tau": args.dedup_tau,
        "seg_pool": args.seg_pool,
        "total_frames": args.total_frames,
        "montage_kind": args.montage_kind,
        "stream_kind": args.stream_kind,
        "weighting": args.weighting,
        "perception_max_new_tokens": args.perception_max_new_tokens,
    }

    shard_tag = f"_shard{args.offset}" if args.chunk > 1 else ""
    out = args.out or os.path.join(
        _REPO_ROOT, "results",
        f"{dataset}_{model_tag}_strict{int(strict_prompt)}"
        f"_reasoning{int(reasoning)}{shard_tag}.jsonl")
    if args.out and args.chunk > 1:
        print("[warn] --out with --chunk: every shard must write its own file, or "
              "concurrent appends interleave and tear each other's lines.")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    counts, errored, prior, prior_remaps = scan_output(out)
    dup = describe_duplicates(counts)
    if dup:
        raise SystemExit(
            f"refusing to use {out}: it already holds more than one row per key.\n"
            f"  {dup}"
            "  Rows are counted, not deduplicated, by everything downstream, so a "
            "duplicated key inflates a denominator silently. Keep the row you want "
            "and delete the rest, or start a fresh --out.")
    stray = []
    for knob in IDENTITY_KNOBS:
        others = sorted((v for v in prior.get(knob, ()) if v != knobs[knob]), key=str)
        if others:
            stray.append(f"  {knob}: file holds {others}; this run is "
                         f"{knobs[knob]!r}")
    if stray:
        detail = (f"{out} already holds rows this run disagrees with:\n"
                  + "\n".join(stray) + "\n"
                  "  The resume key cannot tell those rows apart, so this run "
                  "would skip work it has not done and write a summary labelled "
                  "with ITS knobs over THEIR data.")
        if not args.allow_mixed:
            raise SystemExit(
                "refusing to append: " + detail
                + "\n  Give each protocol its own --out (the default name already "
                  "carries the model and the prompt version), or pass "
                  "--allow-mixed if you genuinely intend one mixed file.")
        print("[warn] --allow-mixed: " + detail, flush=True)

    # media provenance, checked separately from the identity knobs because it is
    # per RECORD, not per launch: rows written before the .avi remux carry no
    # media_remap stamp and decoded the wrong frames, so resume/append must not
    # pool them with remuxed rows under a reused --out. A new or empty file, or
    # a non-MEVA file (every stamp None), is always compatible.
    this_remap = {media_remap(rec) for rec in data} - {None}
    prior_stamped = prior_remaps - {"unstamped", None}
    mixed_media = ((this_remap and "unstamped" in prior_remaps)
                   or (prior_stamped and prior_stamped != this_remap))
    if mixed_media:
        detail = (f"{out} holds rows with media_remap "
                  f"{sorted(str(v) for v in prior_remaps)}; this run would stamp "
                  f"{sorted(str(v) for v in this_remap) or ['None']}\n"
                  "  (rows without the stamp predate the MEVA remux and decoded "
                  "the wrong frames).")
        if not args.allow_mixed:
            raise SystemExit(
                "refusing to append: " + detail
                + "\n  Give this run a fresh --out, or pass --allow-mixed if you "
                  "genuinely intend one mixed file.")
        print("[warn] --allow-mixed: " + detail, flush=True)

    # temporal_weighted requires video records too (it allocates by clip
    # duration and refuses stills), so it counts toward the preflight's
    # no-video-slot refusal alongside the selection kinds
    need_video = any(k in SELECTION_KINDS or k == "temporal_weighted"
                     for k in kinds.values())
    if all(k == "blind" for k in kinds.values()):
        n_clips, n_missing = 0, 0       # text-only: nothing to resolve
    else:
        n_clips, n_missing = preflight_media(data, args.video_root, need_video,
                                             args.allow_missing_media)

    # What each launched name will RECORD on its rows: temporal_weighted under
    # the even weighting writes method='temporal_even' (make_method allows the
    # rename), and every name-keyed consumer below — the retry set and the
    # health filter — must use the recorded name or that leg's rows are
    # invisible to it.
    recorded = {m: ("temporal_even"
                    if kinds[m] == "temporal_weighted" and args.weighting == "even"
                    else m) for m in methods}
    planned = {(dataset, rec.get("id"), recorded[mname], model_tag, pi)
               for mname in methods for rec in data
               for pi in range(1, len(seeds) + 1)}
    if args.retry_errors:
        dropped = drop_error_rows(out, errored & planned)
        print(f"--retry-errors: removed {dropped} error row(s) from {out}; they "
              "will be re-run and replaced.", flush=True)
        counts, errored, _, _ = scan_output(out)
    done = set(counts)      # a row is done whether or not it carries an error

    # The summary is the completion signal, so a stale one from an earlier run of
    # this file must not survive a run that dies half way.
    sumpath = os.path.splitext(out)[0] + "_summary.json"
    if os.path.exists(sumpath):
        os.remove(sumpath)

    shown_budget = "matched (nframes x K)" if args.budget == 0 else args.budget
    print(f"subset={args.subset} sha256={subset_sha[:12]} records={len(data)}/{n_all} "
          f"clips={n_clips} missing_clips={n_missing}")
    print(f"methods={methods} model={args.model} passes={args.passes} seeds={seeds} "
          f"temp={args.temperature}")
    print(f"nframes={args.nframes} budget={shown_budget} sel_tau={args.sel_tau} "
          f"sel_tau_q={args.sel_tau_q} n_queries={args.n_queries} "
          f"internvl_max_tiles={args.internvl_max_tiles}")
    print(f"total_frames={args.total_frames} montage_kind={args.montage_kind} "
          f"stream_kind={args.stream_kind} weighting={args.weighting}")
    print(f"segments_keep={args.segments_keep} seg_pool={args.seg_pool} "
          f"dedup_tau={args.dedup_tau}")
    print(f"reasoning={int(reasoning)} strict_prompt={int(strict_prompt)} "
          f"dataset={dataset} run_id={run_id} node={node} "
          f"allow_avi={int(qa_json.ALLOW_AVI)}")
    print(f"video_root={args.video_root}\nout={out} (rows already present: "
          f"{len(done)}, of them errors: {len(errored)})", flush=True)

    # Everything that goes on every row and is not already on the Result: what a
    # row has to carry for someone else's rows to pool with it without knowing
    # anything about the launch environment.
    stamp = {k: v for k, v in knobs.items() if k not in ("backend", "temperature")}
    stamp.update(run_id=run_id, node=node, model=args.model,
                 subset=os.path.basename(args.subset))

    try:
        from tqdm import tqdm
    except ImportError:                 # progress bars are not worth a dependency
        def tqdm(it, **kw):
            return it

    backend = make_backend(args.model, nframes=args.nframes,
                           internvl_max_tiles=args.internvl_max_tiles)
    if backend.name != model_tag:
        raise SystemExit(
            f"the backend records itself as '{backend.name}' but the resume keys "
            f"and the output name were built from '{model_tag}'. Pass the full HF "
            "id rather than an alias.")

    written = 0
    with open(out, "a") as fh:
        for mname in methods:
            kind, scorer = parsed[mname]
            method = make_method(mname, kind, scorer, backend, args)
            # RECORD-MAJOR: all passes of one question run consecutively, so the
            # arm's keep-last-record selection cache is reused across passes. The
            # selection is deterministic, so this is what makes the multi-pass std
            # measure the answer stage — and what makes the leg affordable.
            jobs = [(rec, pi, sd)
                    for rec in data
                    for pi, sd in enumerate(seeds, 1)
                    if (dataset, rec.get("id"), method.name, backend.name, pi) not in done]
            print(f"[{method.name}/{backend.name}] {len(jobs)} of "
                  f"{len(data) * len(seeds)} rows to run", flush=True)
            for rec, pass_idx, seed in tqdm(jobs, desc=f"{method.name}/{backend.name}"):
                res = method.answer(rec, args.video_root, seed=seed)
                if res is None:
                    raise SystemExit(
                        f"{method.name} returned no row for id={rec.get('id')}. "
                        "Every supported arm answers every record; a skipped "
                        "record would be retried on every resume forever.")
                res.pass_idx = pass_idx
                row = res.to_dict()
                # .avi records decode from their remuxed .mp4 sibling
                # (qa_json.resolve_media); stamp the provenance so rows from
                # before the remux (wrapped frames) never pool with these.
                # Blind rows stamp None: the arm never reads media.
                row["media_remap"] = (None if kind == "blind"
                                      else media_remap(rec))
                row.update(stamp)
                # flushed per row: a job killed at the wall clock keeps every
                # question it has already paid for
                fh.write(json.dumps(row, ensure_ascii=False, default=jsonable) + "\n")
                fh.flush()
                written += 1

    # Read the file back rather than accumulate in memory, so a resumed run
    # summarizes the whole leg and not just the part it happened to run.
    rows = list(iter_rows(out))
    dup = describe_duplicates(Counter(row_key(r) for r in rows))
    if dup:
        raise SystemExit(
            f"{out} ended up with duplicated keys — {dup}"
            f"  Every row is safely on disk ({len(rows)} of them) and nothing has "
            "been lost, but a summary over duplicated keys would double-count, so "
            "none was written. De-duplicate the file, keeping the last row per "
            "key, and re-run to summarize.")
    print(format_summary(rows))
    with open(sumpath, "w") as fh:
        json.dump(summarize_by_method_backend_passes(rows), fh, indent=2,
                  default=jsonable)
    # Health is reported for THIS leg's rows, not for whatever else a mixed file
    # may hold: a leg is only poolable at its full row count with no error rows.
    mine = [r for r in rows if r.get("method") in set(recorded.values())
            and all(r.get(k, knobs[k]) == knobs[k] for k in IDENTITY_KNOBS)]
    expected = len(data) * len(methods) * len(seeds)
    errors = sum(1 for r in mine if r.get("error"))
    state = "COMPLETE" if len(mine) == expected and not errors else "INCOMPLETE"
    print(f"\nsummary -> {sumpath}")
    print(f"leg {state}: wrote {written} row(s) this run, {len(mine)} of "
          f"{expected} expected for this leg, {errors} carrying an error.")
    if state != "COMPLETE":
        print("  A leg is only poolable at the expected row count with no errors. "
              "Re-run to continue where this stopped; add --retry-errors to "
              "replace the error rows once their cause is fixed.")


if __name__ == "__main__":
    main()
