#!/usr/bin/env python3
# Ported from Wavy-Hec/MultiCam analysis/letter_floors.py @ 8f138c80f12ddb923200ffcdcf69828a1039105a
# Deltas from 8f138c8 (imports, docstrings and path constants only; every
# statistic is byte-identical to the fork):
#   - sibling imports rewritten to this repo's layout: normalize comes from
#     scripts/data/crossview_question_types.py (the same single source of truth
#     build_crossview.py uses) and loo_floor from evaluation/evidence_class_split.py
#     (byte-identical to the fork's analysis/evidence_class_accuracy.loo_floor).
#   - the annotations root is this repo's data/crossview-release-annotations/...
#     and is overridable with --ann-root, as scripts/data/crossview_question_types.py
#     already does; the fork hardcodes its own checkout's root.
#   - run examples, the --group-by help example and the `script` field in the
#     report name this repo's paths; docstring references to the fork's
#     convert_crossview.py name this repo's scripts/data/build_crossview.py.
"""Answer-letter bias statistics for any CrossView-style question pool.

1/n_options is the wrong baseline wherever the answer key is skewed. On the
MEVA pool it is very skewed: a guesser that sees only the task type and answers
that task's most common letter scores far above chance, so any arm that merely
tracks the key's skew can look like it found visual signal. This script measures
the skew itself — where it lives (release vs our compile step), how much of it a
letter-only guesser can convert into accuracy, and what in the option text makes
it guessable (fixed distractors, shared option sets, an identity ordering that
sits in a fixed slot).

Two input shapes, normalized to one item schema
    {id, type, gold_letter, options: [(letter, text)], question, source, extra}:
  --subset PATH        a harness subset JSON (options as a list of "A. text",
                       `answer` a letter, `task_type`/`question_type` present).
                       `type` = task_type, which is what loo_floor conditions on.
  --release PATH...    one or more raw release QA JSONs (options a dict
                       letter->text). `type` = question_type. Items are parsed
                       through crossview_question_types.normalize(), the repo's
                       single source of truth for question/options/answer.

Items whose answer is not one of the item's own option letters (nuScenes'
open-ended counting/summarization, plus a couple of double-encoded items whose
options never got recovered) are SKIPPED AND COUNTED, per file and per reason —
never silently dropped.

Reported overall and per type: n; gold-letter counts and percentages; chance
(mean over items of 1/n_options); the in-sample modal-letter accuracy; the
leave-one-out task-conditioned modal-letter floor; and a pool-wide LOO floor
that ignores type (one modal letter for the whole pool). Per type it also dumps
the distinct option texts per letter position with counts (fixed distractors and
shared option sets), how many items carry an option text that is never gold
anywhere in that type, and — for permutation ("event ordering") types — whether
the gold ordering is the identity I -> II -> III [-> IV] and which letter
positions the identity ordering occupies.

--group-by regroups the per-item LOO hits by an external label map (evidence
class, or any record field): the guesser still learns PER TYPE over the whole
pool — it cannot see the group — but is scored only on the group's questions.
That is the same restriction loo_floor(records, restrict_ids) applies, so a
class split stays readable against its own floor.

Side-effect free: writes only the paths named on the command line (--json), and
--help writes nothing. Both sibling imports below were checked to have no
top-level side effects (constants + defs only, real work under a __main__
guard), so loo_floor is imported rather than reimplemented; loo_hits() below
mirrors it per item and is cross-checked against the imported function on every
run (the check is recorded in the output as `loo_crosscheck`).

Run from the repo root (no GPU, no videos needed). Nothing defaults to a
pool: every path is given on the command line.
  python3 evaluation/letter_floors.py \
      --subset data/subsets/crossview_meva1033_subset.json \
      --group-by data/subsets/meva_evidence_labels.json --group-alias meva1033 \
      --converter-check --json letter_floors_meva1033.json
  python3 evaluation/letter_floors.py --release <ann-root>/meva/qa_*.json \
      --ann-root <ann-root> --json letter_floors_meva_release.json
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# Works both as `python evaluation/letter_floors.py` and as
# `python -m evaluation.letter_floors`: the repo root carries the evaluation
# package, scripts/data the release parser (a flat sibling import there, the
# same one scripts/data/build_crossview.py uses).
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "data"))

# Both are import-safe: crossview_question_types and evidence_class_split
# define only constants and functions at module level and do their work in
# main() under `if __name__ == "__main__"`. loo_floor is THE published floor
# (evaluation/evidence_class_split.py) — imported so the two can never drift.
from crossview_question_types import normalize            # noqa: E402
from evaluation.evidence_class_split import loo_floor     # noqa: E402

# Default annotations root, as scripts/data/crossview_question_types.py
# defines it; --ann-root overrides. Only --release input and --converter-check
# read it, so a --subset run needs no annotations checkout at all.
ANN = os.path.join(REPO, "data", "crossview-release-annotations",
                   "crossview-release", "annotations", "multi-cam-dataset")
# superseded pre-regeneration duplicate; same question_type as qa_best_camera.json,
# so including it would double-count that type (crossview_question_types.py drops
# it the same way). --include-legacy overrides.
LEGACY_FILES = {"meva/qa_best_camera_pre_regenerate.json"}

OPT_RE = re.compile(r"^\s*([A-Z])\s*[.):]\s*(.*)$", re.S)
ROMAN_RE = re.compile(r"^[IVX]+$")
ARROW_RE = re.compile(r"\s*(?:->|-->|→|=>)\s*")
ROMAN_VAL = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}


# ----------------------------------------------------------------- loading ---
def _mkitem(iid, typ, gold, options, question, source, extra):
    return {"id": iid, "type": typ, "gold_letter": gold, "options": options,
            "question": question, "source": source, "extra": extra}


def _finish(items, skipped, iid, typ, gold, options, question, source, extra, where):
    """Append the item, or record why it was skipped. Never drops silently."""
    reason = None
    if options is None:
        reason = "open_ended_no_options"
    elif not options:
        reason = "mcq_shaped_but_zero_options"
    elif not gold:
        reason = "missing_answer"
    elif gold not in {L for L, _ in options}:
        reason = "answer_not_an_option_letter"
    if reason:
        skipped.append({"id": iid, "type": typ, "file": where, "reason": reason,
                        "answer": gold, "n_options": len(options or [])})
        return
    items.append(_mkitem(iid, typ, gold, options, question, source, extra))


def load_harness(path):
    """Harness subset JSON -> (items, skipped, per_file)."""
    recs = json.load(open(path))
    items, skipped = [], []
    where = os.path.basename(path)
    for r in recs:
        options = []
        bad = False
        for o in (r.get("options") or []):
            m = OPT_RE.match(str(o))
            if not m:
                bad = True
                break
            options.append((m.group(1).strip().upper(), m.group(2).strip()))
        if bad:
            skipped.append({"id": r.get("id"), "type": r.get("task_type"),
                            "file": where, "reason": "unparseable_option_string",
                            "answer": r.get("answer"), "n_options": len(r.get("options") or [])})
            continue
        extra = {k: v for k, v in r.items()
                 if k not in ("options", "question", "answer") and not k.startswith("video_")}
        _finish(items, skipped, r.get("id"),
                r.get("task_type") or r.get("question_type") or "?",
                str(r.get("answer") or "").strip().upper()[:1],
                options, r.get("question") or "", r.get("source") or "?", extra, where)
    n_ok = sum(1 for i in items)
    per_file = {where: {"n_items": len(recs), "n_scored": n_ok,
                        "n_skipped": len(recs) - n_ok, "kind": "harness_subset",
                        "types": dict(Counter(i["type"] for i in items))}}
    return items, skipped, per_file


def load_release(paths, include_legacy=False):
    """Raw release QA JSONs -> (items, skipped, per_file). id = '<file>#<idx>',
    the same index label_evidence_class.py joins harness records on."""
    items, skipped, per_file = [], [], {}
    for path in paths:
        rel = os.path.relpath(os.path.abspath(path), ANN)
        source = rel.split(os.sep)[0] if os.sep in rel else "?"
        raw = json.load(open(path))
        if rel.replace(os.sep, "/") in LEGACY_FILES and not include_legacy:
            probe = [normalize(it, source) for it in raw]
            per_file[rel] = {
                "n_items": len(raw), "n_scored": 0, "n_skipped": 0,
                "kind": "mcq" if any(p["is_mcq"] for p in probe) else "open_ended",
                "status": "excluded (superseded pre-regeneration duplicate; "
                          "--include-legacy to keep)",
                "types": dict(Counter(p["question_type"] for p in probe))}
            continue
        before, n_sk = len(items), len(skipped)
        kinds = Counter()
        for idx, it in enumerate(raw):
            rec = normalize(it, source)
            kinds["mcq" if rec["is_mcq"] else "open_ended"] += 1
            options = (sorted(rec["options"].items(), key=lambda kv: kv[0])
                       if rec["options"] is not None else None)
            extra = {"flags": rec["flags"], "num_cameras": rec["num_cameras"],
                     "release_index": idx, "file": rel,
                     "video_id": it.get("video_id") or it.get("slot")
                     or it.get("task_name") or "na"}
            _finish(items, skipped, f"{rel}#{idx}", rec["question_type"],
                    str(rec["answer"] or "").strip().upper()[:1],
                    options, rec["question"], source, extra, rel)
        per_file[rel] = {
            "n_items": len(raw), "n_scored": len(items) - before,
            "n_skipped": len(skipped) - n_sk,
            "kind": ("mcq" if kinds["mcq"] and not kinds["open_ended"]
                     else "open_ended" if kinds["open_ended"] and not kinds["mcq"]
                     else "mixed"),
            "types": dict(Counter(i["type"] for i in items[before:]))}
    return items, skipped, per_file


# ------------------------------------------------------------------ floors ---
def loo_hits(items, by_type=True):
    """Per-item leave-one-out modal-letter hit, {item_id: bool}.

    Mirrors loo_floor() in evaluation/evidence_class_split.py EXACTLY, including
    its tie-break — `max(sorted(c), key=lambda k: c[k])` scans the letters in
    ascending order and max keeps the FIRST maximum, so a tie goes to the
    alphabetically smallest letter. (clip_scorer_gate.modal_letter_floor breaks
    ties the other way, `max(c.items(), key=lambda kv: (kv[1], kv[0]))`; do not
    mix them.) Only reimplemented because loo_floor returns a pooled percentage,
    not per-item hits; check_loo() asserts the two agree on every run.

    by_type=False pools every item into one bucket: the pool-wide floor that
    ignores type.
    """
    by = defaultdict(Counter)
    for it in items:
        by[it["type"] if by_type else "_POOL_"][it["gold_letter"]] += 1
    hits = {}
    for it in items:
        c = by[it["type"] if by_type else "_POOL_"].copy()
        g = it["gold_letter"]
        c[g] -= 1
        guess = max(sorted(c), key=lambda k: c[k])
        hits[it["id"]] = (guess == g)
    return hits


def pct(hits, ids=None):
    sel = [v for k, v in hits.items() if ids is None or k in ids]
    return 100.0 * sum(sel) / len(sel) if sel else None


def as_records(items):
    """Shim for the imported loo_floor(), which reads task_type/answer/id."""
    return [{"id": it["id"], "task_type": it["type"], "answer": it["gold_letter"]}
            for it in items]


def check_loo(items, hits):
    """Cross-check the per-item mirror against the published loo_floor()."""
    imported = loo_floor(as_records(items))
    local = pct(hits)
    ok = imported is not None and local is not None and abs(imported - local) < 1e-9
    if not ok:
        raise SystemExit(f"loo_hits() disagrees with the imported loo_floor(): "
                         f"{local} vs {imported} — the floor definition drifted")
    return {"imported_loo_floor": round(imported, 6),
            "local_loo_hits": round(local, 6), "match": True}


# ------------------------------------------------------------------- stats ---
def letter_block(items, hits, pool_hits):
    n = len(items)
    counts = Counter(it["gold_letter"] for it in items)
    letters = sorted({L for it in items for L, _ in it["options"]} | set(counts))
    ch = sum(1.0 / len(it["options"]) for it in items) / n if n else None
    modal = counts.most_common(1)[0][0] if counts else None
    ids = {it["id"] for it in items}
    return {
        "n": n,
        "gold_counts": {L: counts.get(L, 0) for L in letters},
        "gold_pct": {L: round(100.0 * counts.get(L, 0) / n, 2) for L in letters} if n else {},
        "n_options_hist": dict(sorted(Counter(len(it["options"]) for it in items).items())),
        "chance_pct": round(100.0 * ch, 2) if ch is not None else None,
        "modal_letter": modal,
        "modal_letter_in_sample_pct": round(100.0 * counts.get(modal, 0) / n, 2) if n else None,
        "loo_floor_task_conditioned_pct": round(pct(hits, ids), 2) if n else None,
        "loo_floor_pool_wide_pct": round(pct(pool_hits, ids), 2) if n else None,
    }


def overall_block(items, hits, pool_hits):
    """Overall = the per-type guesser scored on every item, plus the type-blind one."""
    b = letter_block(items, hits, pool_hits)
    n = len(items)
    per_type_modal = {}
    for t, group in groupby_type(items).items():
        c = Counter(i["gold_letter"] for i in group)
        per_type_modal[t] = c.most_common(1)[0][0]
    b["modal_letter_in_sample_pct_task_conditioned"] = round(
        100.0 * sum(1 for i in items if i["gold_letter"] == per_type_modal[i["type"]]) / n, 2
    ) if n else None
    b["modal_letter_per_type"] = per_type_modal
    return b


def groupby_type(items):
    out = defaultdict(list)
    for it in items:
        out[it["type"]].append(it)
    return dict(out)


def option_text_block(items, top=6):
    """Distinct option texts per letter position, plus never-gold coverage."""
    per_letter = defaultdict(Counter)
    for it in items:
        for L, t in it["options"]:
            per_letter[L][t] += 1
    gold_texts = {dict(it["options"])[it["gold_letter"]] for it in items}
    n_never_gold_items = sum(
        1 for it in items if any(t not in gold_texts for _, t in it["options"]))
    never_gold_texts = {t for c in per_letter.values() for t in c} - gold_texts
    sets = Counter(tuple(t for _, t in it["options"]) for it in items)
    # A guesser that has seen the same OPTION SET before and answers its modal
    # gold letter — the shared-option-set analogue of the task-conditioned floor,
    # same leave-one-out and same tie-break. Scored only on items whose option set
    # is shared with at least one other item (a singleton set carries no evidence
    # and would degenerate to "always A").
    set_gold = defaultdict(Counter)
    for it in items:
        set_gold[tuple(t for _, t in it["options"])][it["gold_letter"]] += 1
    set_hit = set_tot = singletons = 0
    for it in items:
        key = tuple(t for _, t in it["options"])
        c = set_gold[key].copy()
        c[it["gold_letter"]] -= 1
        if sum(c.values()) <= 0:
            singletons += 1
            continue
        set_tot += 1
        set_hit += (max(sorted(c), key=lambda k: c[k]) == it["gold_letter"])
    positions = {}
    for L in sorted(per_letter):
        c = per_letter[L]
        tot = sum(c.values())
        positions[L] = {
            "n_items": tot,
            "n_distinct_texts": len(c),
            "n_texts_used_once": sum(1 for v in c.values() if v == 1),
            "top_texts": [{"text": t[:160], "n": v, "pct": round(100.0 * v / tot, 2),
                           "ever_gold": t in gold_texts}
                          for t, v in c.most_common(top)],
            "n_never_gold_texts": sum(1 for t in c if t not in gold_texts),
            "n_items_with_never_gold_text_here": sum(
                v for t, v in c.items() if t not in gold_texts),
        }
    return {
        "per_letter_position": positions,
        "n_distinct_gold_texts": len(gold_texts),
        "n_never_gold_texts": len(never_gold_texts),
        "n_items_with_a_never_gold_option": n_never_gold_items,
        "pct_items_with_a_never_gold_option": round(
            100.0 * n_never_gold_items / len(items), 2) if items else None,
        "n_distinct_full_option_sets": len(sets),
        "option_set_loo": {
            "n_items_in_shared_sets": set_tot,
            "n_items_in_singleton_sets": singletons,
            "loo_floor_pct_on_shared_sets": (round(100.0 * set_hit / set_tot, 2)
                                             if set_tot else None)},
        "top_option_sets": [{"options": [t[:60] for t in k], "n": v}
                            for k, v in sets.most_common(3) if v > 1],
    }


def parse_perm(text):
    """['I','III','II'] for 'I -> III -> II', else None."""
    toks = [t.strip().upper() for t in ARROW_RE.split(text.strip()) if t.strip()]
    if len(toks) < 2 or not all(ROMAN_RE.match(t) and t in ROMAN_VAL for t in toks):
        return None
    return toks


def ordering_block(items):
    """Identity-ordering analysis; None when this type is not permutation-shaped."""
    parsed = 0
    ident_letters = Counter()
    gold_is_ident = 0
    no_ident_option = 0
    gold_perm_pos = Counter()
    for it in items:
        perms = {L: parse_perm(t) for L, t in it["options"]}
        if not all(perms.values()):
            continue
        parsed += 1
        gold_perm = perms[it["gold_letter"]]
        gold_perm_pos[" -> ".join(gold_perm)] += 1
        ident = sorted(gold_perm, key=lambda r: ROMAN_VAL[r])
        hit = [L for L, p in sorted(perms.items()) if p == ident]
        if not hit:
            no_ident_option += 1
            continue
        for L in hit:
            ident_letters[L] += 1
        if it["gold_letter"] in hit:
            gold_is_ident += 1
    if not parsed or parsed < 0.5 * len(items):
        return None
    return {
        "n_permutation_items": parsed,
        "n_items": len(items),
        "identity_letter_positions": dict(sorted(ident_letters.items())),
        "n_without_identity_option": no_ident_option,
        "n_gold_is_identity": gold_is_ident,
        "pct_gold_is_identity": round(100.0 * gold_is_ident / parsed, 2),
        "top_gold_permutations": [{"perm": k, "n": v}
                                  for k, v in gold_perm_pos.most_common(6)],
    }


# ---------------------------------------------------------------- grouping ---
def item_join_key(it):
    """(question_type, release index) key, as label_evidence_class.py builds it."""
    e = it.get("extra") or {}
    qt, orig = e.get("question_type"), e.get("orig_id")
    if qt and isinstance(orig, str) and "#" in orig:
        return f"{qt}#{orig.rsplit('#', 1)[1]}"
    return None


def resolve_labels(spec, items, alias=None, field="evidence_class"):
    """-> ({item_id: label}, meta). spec is 'record:<field>' or a JSON path."""
    if spec.startswith("record:"):
        f = spec.split(":", 1)[1]
        lab = {}
        for it in items:
            v = (it.get("extra") or {}).get(f, it.get(f))
            if v is not None:
                lab[it["id"]] = str(v)
        return lab, {"kind": "record_field", "field": f}
    data = json.load(open(spec))
    if isinstance(data, dict) and isinstance(data.get("labels"), list):
        rows = data["labels"]
        aliases = sorted({a for r in rows for a in (r.get("ids") or {})})
        keys = {it["id"]: item_join_key(it) for it in items}
        cand = []
        for a in ([alias] if alias else aliases):
            by = {r["ids"][a]: r for r in rows if a in (r.get("ids") or {})}
            cover = sum(1 for it in items if it["id"] in by)
            if not cover:
                continue
            bad = sum(1 for it in items
                      if keys[it["id"]] and it["id"] in by
                      and by[it["id"]].get("join_key") != keys[it["id"]])
            cand.append((a, cover, bad, by))
        if not cand:
            raise SystemExit(f"--group-by {spec}: no alias in {aliases} covers this pool")
        verified = [c for c in cand if c[2] == 0]
        pick = verified or cand
        best = max(c[1] for c in pick)
        pick = [c for c in pick if c[1] == best]
        maps = [{it["id"]: c[3][it["id"]].get(field)
                 for it in items if it["id"] in c[3]} for c in pick]
        if len(pick) > 1 and any(m != maps[0] for m in maps[1:]):
            raise SystemExit(
                f"--group-by {spec}: aliases {[c[0] for c in pick]} disagree on this "
                f"pool — pass --group-alias to say which one")
        a, cover, bad, by = pick[0]
        return maps[0], {"kind": "label_file", "path": spec, "alias": a, "field": field,
                         "aliases_available": aliases,
                         "aliases_equivalent_here": [c[0] for c in pick],
                         "join_key_verified": bool(verified),
                         "n_covered": cover, "n_uncovered": len(items) - cover}
    if isinstance(data, list):
        by = {str(r.get("id")): r.get(field) for r in data if isinstance(r, dict)}
        lab = {it["id"]: by[str(it["id"])] for it in items if str(it["id"]) in by}
        return lab, {"kind": "list_of_records", "path": spec, "field": field,
                     "n_covered": len(lab), "n_uncovered": len(items) - len(lab)}
    if isinstance(data, dict):
        lab = {it["id"]: data[str(it["id"])] for it in items if str(it["id"]) in data}
        return lab, {"kind": "flat_map", "path": spec,
                     "n_covered": len(lab), "n_uncovered": len(items) - len(lab)}
    raise SystemExit(f"--group-by {spec}: unrecognized JSON shape")


def group_block(items, hits, pool_hits, labels):
    by = defaultdict(list)
    for it in items:
        by[str(labels.get(it["id"], "UNLABELED"))].append(it)
    def gkey(k):
        try:
            return (0, float(k), "")
        except ValueError:
            return (1, 0.0, k)
    out = {}
    for g in sorted(by, key=gkey):
        group = by[g]
        ids = {i["id"] for i in group}
        counts = Counter(i["gold_letter"] for i in group)
        out[g] = {
            "n": len(group),
            "types": dict(sorted(Counter(i["type"] for i in group).items())),
            "gold_counts": dict(sorted(counts.items())),
            "chance_pct": round(100.0 * sum(1.0 / len(i["options"]) for i in group)
                                / len(group), 2),
            "loo_floor_task_conditioned_pct": round(pct(hits, ids), 2),
            "loo_floor_pool_wide_pct": round(pct(pool_hits, ids), 2),
        }
    return out


# -------------------------------------------------------- converter check ----
def converter_check(items):
    """Join every harness record back to its release item and compare the option
    order (letter -> text) and the gold letter. Decides whether the letter bias
    is in the release or was introduced by our compile step.

    Join key: label_evidence_class.py takes int(orig_id.rsplit('#')[1]) as the
    index into the per-question_type release file (QA_FILES); this repo's
    scripts/data/build_crossview.py built orig_id as f"{video_id|slot|task_name|'na'}#{idx}" over the same file,
    so the prefix is checked too.
    """
    qa_files = {"temporal": "qa_temporal.json",
                "event_ordering": "qa_event_ordering.json",
                "spatial": "qa_spatial.json",
                "counting": "qa_counting.json",
                "camera": "qa_best_camera.json",
                "summarization": "qa_summarization.json"}
    cache, res = {}, Counter()
    mismatches = []
    for it in items:
        e = it["extra"]
        src, qt, orig = e.get("source"), e.get("question_type"), e.get("orig_id")
        if src != "meva" or qt not in qa_files or not isinstance(orig, str):
            res["unjoinable"] += 1
            mismatches.append({"id": it["id"], "why": "no meva/question_type/orig_id"})
            continue
        if qt not in cache:
            cache[qt] = json.load(open(os.path.join(ANN, "meva", qa_files[qt])))
        idx = int(orig.rsplit("#", 1)[1])
        pool = cache[qt]
        if idx >= len(pool):
            res["index_out_of_range"] += 1
            mismatches.append({"id": it["id"], "why": "index out of range", "key": orig})
            continue
        rel = normalize(pool[idx], "meva")
        res["joined"] += 1
        why = []
        if (rel["question"] or "").strip() != (it["question"] or "").strip():
            why.append("question_text")
        natural = (pool[idx].get("video_id") or pool[idx].get("slot")
                   or pool[idx].get("task_name") or "na")
        if orig.rsplit("#", 1)[0] != str(natural):
            why.append("orig_id_prefix")
        rel_opts = (sorted((rel["options"] or {}).items(), key=lambda kv: kv[0]))
        if [(L, t.strip()) for L, t in rel_opts] != [(L, t.strip()) for L, t in it["options"]]:
            why.append("options")
        if str(rel["answer"] or "").strip().upper()[:1] != it["gold_letter"]:
            why.append("gold_letter")
        if why:
            res["mismatch"] += 1
            for w in why:
                res["mismatch_" + w] += 1
            if len(mismatches) < 20:
                mismatches.append({"id": it["id"], "key": orig, "why": why})
        else:
            res["identical"] += 1
    return {"n_checked": len(items), "counts": dict(sorted(res.items())),
            "first_mismatches": mismatches[:20],
            "verdict": ("identical — the letter skew is in the release, not in our "
                        "compile step" if res["mismatch"] == 0 and res["identical"] == len(items)
                        else "MISMATCHES — see first_mismatches")}


# ---------------------------------------------------------------- markdown ---
def md_table(header, rows):
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def to_markdown(rep):
    L = []
    p = rep["pool"]
    L.append(f"# Answer-letter floors — {p['name']}")
    L.append("")
    L.append(f"{p['n_scored']} scored item(s) from {len(p['sources'])} file(s); "
             f"{p['n_skipped']} skipped (not a single option letter).")
    L.append("")
    L.append("## Files")
    L.append(md_table(["file", "kind", "items", "scored", "skipped", "types"],
                      [[f, v["kind"], v["n_items"], v["n_scored"], v["n_skipped"],
                        ", ".join(f"{k}:{n}" for k, n in sorted(v["types"].items()))
                        or (v.get("status", ""))]
                       for f, v in sorted(rep["per_file"].items())]))
    L.append("")
    if rep["skipped"]["by_reason"]:
        L.append("Skipped, by reason: "
                 + ", ".join(f"`{k}` {v}" for k, v in sorted(rep["skipped"]["by_reason"].items()))
                 + ".")
        L.append("")
    letters = sorted(rep["overall"]["gold_counts"])
    rows = []
    for name, b in [("**overall**", rep["overall"])] + \
                   [(t, rep["by_type"][t]["letters"]) for t in sorted(rep["by_type"])]:
        rows.append([name, b["n"]]
                    + [f'{b["gold_counts"].get(L, 0)} ({b["gold_pct"].get(L, 0):.1f}%)'
                       for L in letters]
                    + [f'{b["chance_pct"]:.2f}',
                       f'{b["modal_letter"]} {b["modal_letter_in_sample_pct"]:.2f}',
                       f'{b["loo_floor_task_conditioned_pct"]:.2f}',
                       f'{b["loo_floor_pool_wide_pct"]:.2f}'])
    L.append("## Gold-letter distribution and floors")
    L.append(md_table(["pool / type", "n"] + letters
                      + ["chance %", "modal (in-sample) %", "LOO floor %", "LOO pool-wide %"],
                      rows))
    L.append("")
    L.append(f"Task-conditioned in-sample modal accuracy over the whole pool: "
             f"{rep['overall']['modal_letter_in_sample_pct_task_conditioned']:.2f}% "
             f"(per-type modal letters: "
             f"{', '.join(f'{k}={v}' for k, v in sorted(rep['overall']['modal_letter_per_type'].items()))}).")
    L.append("")
    L.append("## Option text per letter position")
    rows = []
    for t in sorted(rep["by_type"]):
        o = rep["by_type"][t]["option_texts"]
        for Ltr, v in sorted(o["per_letter_position"].items()):
            top = v["top_texts"][0] if v["top_texts"] else None
            rows.append([t, Ltr, v["n_distinct_texts"], v["n_never_gold_texts"],
                         f'{top["pct"]:.1f}% "{top["text"][:48]}"' if top else ""])
    L.append(md_table(["type", "letter", "distinct texts", "never-gold texts",
                       "most common text at this position"], rows))
    L.append("")
    rows = []
    for t in sorted(rep["by_type"]):
        o = rep["by_type"][t]["option_texts"]
        sl = o["option_set_loo"]
        rows.append([t, rep["by_type"][t]["letters"]["n"],
                     o["n_distinct_full_option_sets"],
                     f'{o["n_items_with_a_never_gold_option"]} '
                     f'({o["pct_items_with_a_never_gold_option"]:.1f}%)',
                     o["n_distinct_gold_texts"],
                     (f'{sl["loo_floor_pct_on_shared_sets"]:.2f} (n={sl["n_items_in_shared_sets"]})'
                      if sl["loo_floor_pct_on_shared_sets"] is not None else "—")])
    L.append(md_table(["type", "n", "distinct option sets", "items with a never-gold option",
                       "distinct gold texts", "option-set LOO floor %"], rows))
    L.append("")
    ords = {t: rep["by_type"][t]["ordering"] for t in sorted(rep["by_type"])
            if rep["by_type"][t].get("ordering")}
    if ords:
        L.append("## Identity ordering (I -> II -> III [-> IV])")
        L.append(md_table(["type", "permutation items", "gold is identity",
                           "identity sits at"],
                          [[t, o["n_permutation_items"],
                            f'{o["n_gold_is_identity"]} ({o["pct_gold_is_identity"]:.2f}%)',
                            ", ".join(f"{k}:{v}" for k, v in o["identity_letter_positions"].items())
                            or "—"]
                           for t, o in ords.items()]))
        L.append("")
    if rep.get("group_by"):
        g = rep["group_by"]
        m = g["meta"]
        desc = (f"`{m['path']}` field `{m.get('field')}`"
                + (f", alias `{m['alias']}`" if m.get("alias") else "")
                + (" (join_key verified)" if m.get("join_key_verified") else "")
                + (f", {m['n_uncovered']} unlabelled" if m.get("n_uncovered") else "")
                if m["kind"] != "record_field" else f"record field `{m['field']}`")
        L.append(f"## Regrouped LOO hits — {desc}")
        L.append("")
        L.append("The guesser still learns per type over the whole pool; only the "
                 "scoring is restricted to the group.")
        L.append("")
        L.append(md_table(["group", "n", "chance %", "LOO floor %", "LOO pool-wide %",
                           "gold counts"],
                          [[k, v["n"], f'{v["chance_pct"]:.2f}',
                            f'{v["loo_floor_task_conditioned_pct"]:.2f}',
                            f'{v["loo_floor_pool_wide_pct"]:.2f}',
                            " ".join(f"{a}:{b}" for a, b in v["gold_counts"].items())]
                           for k, v in g["groups"].items()]))
        L.append("")
    if rep.get("converter_check"):
        c = rep["converter_check"]
        L.append("## Converter check (harness record vs release item)")
        L.append(f"{c['n_checked']} checked — "
                 + ", ".join(f"{k} {v}" for k, v in c["counts"].items())
                 + f". **{c['verdict']}**")
        L.append("")
    return "\n".join(L)


# -------------------------------------------------------------------- main ---
def build_report(items, skipped, per_file, name, sources, args):
    hits = loo_hits(items, by_type=True)
    pool_hits = loo_hits(items, by_type=False)
    by_type = {}
    for t, group in sorted(groupby_type(items).items()):
        by_type[t] = {"letters": letter_block(group, hits, pool_hits),
                      "option_texts": option_text_block(group, args.top_texts),
                      "ordering": ordering_block(group)}
    rep = {
        "script": "evaluation/letter_floors.py",
        "pool": {"name": name, "kind": args.kind, "sources": sources,
                 "n_scored": len(items), "n_skipped": len(skipped)},
        "per_file": per_file,
        "skipped": {"n": len(skipped),
                    "by_reason": dict(sorted(Counter(s["reason"] for s in skipped).items())),
                    "by_file_reason": dict(sorted(Counter(
                        f"{s['file']}::{s['reason']}" for s in skipped).items())),
                    "examples": skipped[:20]},
        "overall": overall_block(items, hits, pool_hits),
        "by_type": by_type,
        "loo_crosscheck": check_loo(items, hits),
        "definitions": {
            "chance_pct": "mean over items of 100/n_options",
            "modal_letter_in_sample_pct": "share of the type's most common gold letter",
            "loo_floor_task_conditioned_pct":
                "loo_floor() of evaluation/evidence_class_split.py: a guesser that "
                "knows only the task type answers that type's most common letter, "
                "with the graded item removed from the counts; ties to the "
                "alphabetically smallest letter",
            "loo_floor_pool_wide_pct":
                "same, but one modal letter for the whole pool (type ignored)",
        },
    }
    if args.group_by:
        labels, meta = resolve_labels(args.group_by, items, args.group_alias,
                                      args.group_field)
        rep["group_by"] = {"spec": args.group_by, "meta": meta,
                           "groups": group_block(items, hits, pool_hits, labels)}
    return rep


def main():
    global ANN
    ap = argparse.ArgumentParser(
        description="Answer-letter bias statistics for a CrossView-style pool.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--subset", metavar="PATH",
                     help="harness subset JSON (options as a list of 'A. text')")
    src.add_argument("--release", nargs="+", metavar="PATH",
                     help="raw release QA JSON(s) (options a dict letter->text); "
                          "globs are expanded")
    ap.add_argument("--name", default=None, help="pool name for the report")
    ap.add_argument("--include-legacy", action="store_true",
                    help=f"keep files normally excluded as superseded duplicates "
                         f"({', '.join(sorted(LEGACY_FILES))})")
    ap.add_argument("--group-by", metavar="SPEC",
                    help="regroup the per-item LOO hits: a JSON label file "
                         "(e.g. data/subsets/meva_evidence_labels.json, or a flat "
                         "{id: label} map) or 'record:<field>' (e.g. "
                         "record:orig_num_cameras). The guesser still learns per "
                         "type over the whole pool.")
    ap.add_argument("--group-alias", default=None,
                    help="which ids{} alias of a label file to join on (evidence "
                         "labels carry one per subset)")
    ap.add_argument("--group-field", default="evidence_class",
                    help="label field to read from the label file (default "
                         "evidence_class)")
    ap.add_argument("--converter-check", action="store_true",
                    help="join every harness record back to its MEVA release item "
                         "(via orig_id) and compare option order and gold letter")
    ap.add_argument("--top-texts", type=int, default=6,
                    help="how many option texts to list per letter position")
    ap.add_argument("--json", metavar="OUT", default=None,
                    help="write the full report here (the only file this script writes)")
    ap.add_argument("--quiet", action="store_true", help="suppress the markdown tables")
    ap.add_argument("--ann-root", default=ANN,
                    help="CrossView release annotations root (multi-cam-dataset); "
                         "read only by --release (for the per-file labels) and "
                         "--converter-check")
    args = ap.parse_args()
    ANN = args.ann_root

    if args.subset:
        args.kind = "harness_subset"
        items, skipped, per_file = load_harness(args.subset)
        sources = [args.subset]
        name = args.name or os.path.basename(args.subset)
    else:
        args.kind = "release"
        paths = []
        for p in args.release:
            paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
        if not paths:
            ap.error("--release matched no files")
        items, skipped, per_file = load_release(paths, args.include_legacy)
        sources = paths
        name = args.name or os.path.commonprefix([os.path.basename(os.path.dirname(p))
                                                  for p in paths]) or "release"
    if not items:
        ap.error("no scorable items in the given pool")

    rep = build_report(items, skipped, per_file, name, sources, args)
    if args.converter_check:
        if args.kind != "harness_subset":
            ap.error("--converter-check needs --subset")
        rep["converter_check"] = converter_check(items)

    if not args.quiet:
        print(to_markdown(rep))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print(f"\nwrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
