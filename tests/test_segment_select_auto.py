# Ported from Wavy-Hec/MultiCam bench/tests/test_segment_select_auto.py
# @ b5f666ca990458059cd072048625ccfe629b2737.
# Deliberate delta vs source: the fork's _stmt method name is not exercised
# (this port carries no _stmt arm); the pools are read from data/subsets/
# (byte-identical to the fork's analysis/ copies) and the method grammar from
# inprocess.run.
"""CPU gate for the _auto query rule (segment_select.auto_query).

Pure CPU — no backend, no scorer model, no video. Runnable either way:

    python -m tests.test_segment_select_auto
    pytest tests/test_segment_select_auto.py
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import pytest                                                     # noqa: E402

from inprocess.harnesses.clip_select import (                     # noqa: E402
    option_texts, statement_texts)
from inprocess.harnesses.segment_select import (                  # noqa: E402
    auto_query, generic_option_table, uninformative_option)
from inprocess.run import (                                       # noqa: E402
    SEGMENT_QUERY_MODES, SEGMENT_SELECT_RE, parse_method)


def _rec(task, question, options):
    return {"task_type": task, "question": question, "options": options}


def test_method_names():
    for name, tag, qmode, kind in [
            ("segment_select_auto", None, "_auto", "segment_auto"),
            ("segment_select_siglip_auto", "siglip", "_auto", "segment_auto"),
            ("segment_select_viclip_auto", "viclip", "_auto", "segment_auto"),
            ("segment_select_viclip_opt", "viclip", "_opt", "segment_opt"),
            ("segment_select_siglip", "siglip", None, "segment"),
            ("segment_select", None, None, "segment")]:
        m = SEGMENT_SELECT_RE.match(name)
        assert m and m.group("tag") == tag and m.group("qmode") == qmode, name
        assert parse_method(name)[0] == kind, name
    assert SEGMENT_QUERY_MODES["_auto"] == "auto"
    assert SEGMENT_QUERY_MODES["_opt"] == "options"
    assert SEGMENT_QUERY_MODES[None] == "question"
    # the suffix is a query mode, not a scorer tag, and cannot be doubled
    for bad in ("segment_select_auto_opt", "segment_select_opt_auto",
                "segment_select_auto_auto"):
        assert SEGMENT_SELECT_RE.match(bad) is None, bad


def test_statements_win():
    r = _rec("X-Event-Ordering", "Order these: I. A man opens a door. II. A car stops. III. A dog runs.",
             ["A. I -> II -> III", "B. III -> II -> I"])
    q, mode, dropped = auto_query(r, frozenset())
    assert mode == "statements" and len(q) == 3 and dropped == {}


def test_uninformative_options():
    g = frozenset({("T", "they occurred simultaneously")})
    assert uninformative_option("Video 3", "T", g) == "clip_ref"
    assert uninformative_option("II -> I -> III", "T", g) == "clip_ref"
    assert uninformative_option("3.2", "T", g) == "number"
    assert uninformative_option("Left", "T", g) == "nonvisual_word"
    assert uninformative_option("They occurred simultaneously", "T", g) == "generic"
    assert uninformative_option("Video 6, Video 4, Video 2", "T", g) == "clip_ref_list"
    assert uninformative_option("Video 1 and Video 5", "T", g) == "clip_ref_list"
    assert uninformative_option("Video 1, Video 2, and Video 3.", "T", g) == "clip_ref_list"
    assert uninformative_option("C -> B -> A", "T", g) == "letter_order"
    assert uninformative_option("A, C, B", "T", g) == "letter_order"
    # the letter test is case-sensitive and needs a separator: words survive
    assert uninformative_option("a, b", "T", g) is None
    assert uninformative_option("A", "T", g) is None
    assert uninformative_option("A car, a bus", "T", g) is None
    assert uninformative_option("Video 1 shows a red car", "T", g) is None
    assert uninformative_option("Bicycle", "T", g) is None
    assert uninformative_option("The person opens the trunk", "T", g) is None
    # the generic test is off without a pool table
    assert uninformative_option("They occurred simultaneously", "T", None) is None
    assert uninformative_option("", "T", g) == "empty"


def test_partial_and_fallback():
    g = frozenset({("T", "they occurred simultaneously"), ("T", "cannot be determined")})
    r = _rec("T", "Which happened first: X or Y?",
             ["A. The person opening a door occurred first", "B. The car stopping occurred first",
              "C. They occurred simultaneously", "D. Cannot be determined"])
    q, mode, dropped = auto_query(r, g)
    assert mode == "options" and len(q) == 2 and dropped == {"generic": 2}
    r = _rec("T", "Which video shows the most people?", ["A. Video 1", "B. Video 2", "C. Video 3"])
    q, mode, dropped = auto_query(r, g)
    assert mode == "question" and q == r["question"] and dropped == {"clip_ref": 3}


def test_generic_option_table_thresholds():
    """An option is generic on >= half of a type's questions, judged only for
    types with at least AUTO_GENERIC_MIN_N records."""
    big = [_rec("T", f"q{i}", ["A. Same text", f"B. Unique {i}"]) for i in range(20)]
    assert generic_option_table(big) == frozenset({("T", "same text")})
    small = [_rec("S", f"q{i}", ["A. Same text", f"B. Unique {i}"]) for i in range(19)]
    assert generic_option_table(small) == frozenset()
    half = ([_rec("H", f"q{i}", ["A. Shared", "B. x"]) for i in range(10)]
            + [_rec("H", f"r{i}", ["A. Other", "B. y"]) for i in range(10)])
    assert generic_option_table(half) == frozenset({("H", "shared"), ("H", "other"),
                                                    ("H", "x"), ("H", "y")})


def _pool(name):
    path = os.path.join(REPO, "data", "subsets", name)
    if not os.path.exists(path):
        pytest.skip(f"pool {path} is not in this checkout")
    with open(path) as fh:
        return json.load(fh)


def _modes(records):
    g = generic_option_table(records)
    out = {}
    for r in records:
        _, mode, _ = auto_query(r, g)
        out.setdefault(r["task_type"], {}).setdefault(mode, 0)
        out[r["task_type"]][mode] += 1
    return out


def test_crossview_pools():
    meva = _modes(_pool("crossview_meva1033_subset.json"))
    assert meva["CrossView-MEVA-Event-Ordering"] == {"statements": 297}
    assert meva["CrossView-MEVA-Spatial"] == {"question": 431}
    assert meva["CrossView-MEVA-Temporal"] == {"options": 305}
    ego = _modes(_pool("crossview_egoexo500.json"))
    assert ego["CrossView-EgoExo4D-Event-Ordering"] == {"statements": 250}
    assert ego["CrossView-EgoExo4D-Temporal"] == {"options": 250}


def test_nuscenes_pool():
    """nuScenes enumerates its events three ways — "I.", "(I)" and "(A)"
    (options "C -> B -> A") — and all 250 must reach the statements."""
    recs = _pool("crossview_nuscenes1497_subset.json")
    ns = _modes(recs)
    assert ns["CrossView-nuScenes-Event-Ordering"] == {"statements": 250}
    assert sum(sum(v.values()) for v in ns.values()) == 1497
    by_id = {r["id"]: r for r in recs}
    a = statement_texts(by_id[0])                      # "(A) ... (B) ... (C) ..."
    assert len(a) == 3 and a[0].startswith("A dark-colored sedan")
    assert not any(t.startswith("(") for t in a)
    i = statement_texts(by_id[9])                      # "(I) ... (IV) ..."
    assert len(i) == 4 and i[0].startswith("A white utility truck")
    # one statement per label the options permute
    for r in recs:
        if r["task_type"] == "CrossView-nuScenes-Event-Ordering":
            labels = set(re.findall(r"\b([A-J]|I{1,3}|IV|V|VI{1,3})\b",
                                    " ".join(option_texts(r))))
            assert len(statement_texts(r)) == len(labels), r["id"]


def test_parenthesised_enumerators_never_override_the_dotted_form():
    r = _rec("X", "Order: I. A man (A) waves. II. A car (B) stops. Which is correct?",
             ["A. I -> II", "B. II -> I"])
    assert statement_texts(r) == ["A man (A) waves.", "A car (B) stops."]
    # an INLINE "(A) ... (B)" is a label inside a sentence, not an event list:
    # the parenthesised form counts only at the start of a line
    inline = _rec("X", "Is the red car nearer to camera (A) than to camera (B)?",
                  ["A. The red car is nearer to the first camera",
                   "B. The red car is nearer to the second camera"])
    assert statement_texts(inline) == []
    assert auto_query(inline, frozenset())[1] == "options"
    assert statement_texts(_rec("X", "The narrator (I) said that (II) was wrong.", [])) == []
    # events "(I) (II) (III)" plus lettered choices: the EVENTS are returned
    both = _rec("X", "Order the events:\n(I) a man walks\n(II) a dog barks\n(III) a car "
                     "stops\nChoices:\n(A) first\n(B) second\n(C) third\n(D) fourth", [])
    assert statement_texts(both) == ["a man walks", "a dog barks", "a car stops"]
    # a lone "(A)" or an out-of-sequence pair is not an enumeration
    assert statement_texts(_rec("X", "Is the car (A) near the bus (C)?", [])) == []
    assert statement_texts(_rec("X", "Compare (B) the car and (A) the bus.", [])) == []


def test_meva_temporal_keeps_the_two_events():
    recs = _pool("crossview_meva1033_subset.json")
    g = generic_option_table(recs)
    for r in recs:
        if r["task_type"] == "CrossView-MEVA-Temporal":
            q, mode, dropped = auto_query(r, g)
            assert mode == "options" and len(q) == 2 and dropped == {"generic": 2}, r["id"]


def test_mvueval_pool():
    m = _modes(_pool("mvueval_qa.json"))
    assert m["MVU-ICL"] == {"question": 164}
    assert m["MVU-Comparison"] == {"options": 135}
    # clip-reference LISTS ("Video 6, Video 4, Video 2") are not visual queries:
    # these records score the question instead
    assert m["MVU-TR"] == {"question": 336, "options": 37}
    assert m["MVU-KIR"] == {"question": 163, "options": 118}
    assert sum(sum(v.values()) for v in m.values()) == 1824


def main():
    fails = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as e:                                # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    print("FAILED" if fails else "ALL PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
