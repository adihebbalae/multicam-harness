#!/usr/bin/env python3
"""CPU-only gate: evidence_class_split refuses rows from a foreign pool.

Audit finding (Aug 31 audit, section 4 item 12): the join in
evaluation.evidence_class_split.split() keyed strictly on the bare integer
record id — a row from a different pool (All-Angles, MVU-Eval) whose id
happens to coincide with a labeled MEVA id was joined to that MEVA label and
scored against MEVA floors, and the script exited 0. The fix requires the
row's task_type to match the label's task_type before joining.

No GPU / no model load, no dependency on the real MEVA data files: builds a
tiny synthetic labels blob and rows in memory.

Usage (from repo root):
  python -m tests.evidence_class_split_pool_guard
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from evaluation.evidence_class_split import class_map, split  # noqa: E402

ALIAS = "testpool"

LABELS = {
    "labels": [
        {"ids": {ALIAS: 0}, "evidence_class": "C4", "task_type": "CrossView-MEVA-Event-Ordering"},
        {"ids": {ALIAS: 1}, "evidence_class": "C2", "task_type": "CrossView-MEVA-Spatial"},
    ]
}


def _row(id_, task_type, pass_idx=1, correct=True, error=None):
    return {"id": id_, "task_type": task_type, "pass_idx": pass_idx,
            "correct": correct, "error": error}


def test_genuine_rows_join_and_score():
    cmap, task_types = class_map(LABELS, ALIAS)
    rows = [_row(0, "CrossView-MEVA-Event-Ordering"),
            _row(1, "CrossView-MEVA-Spatial")]
    out, overall = split(rows, cmap, task_types)
    assert overall == 100.0, f"expected 100.0 overall, got {overall}"
    assert set(out) == {"C4", "C2"}, f"unexpected classes: {set(out)}"
    print("genuine MEVA rows join and score: OK")


def test_foreign_pool_row_is_refused():
    cmap, task_types = class_map(LABELS, ALIAS)
    # id=0 collides with a labeled MEVA id, but its task_type belongs to a
    # different pool entirely (mirrors an All-Angles AAB row keyed the same
    # as a MEVA event_ordering row).
    rows = [_row(0, "AAB-Attribute-Identification")]
    try:
        split(rows, cmap, task_types)
    except SystemExit as e:
        msg = str(e)
        assert "id=0" in msg, f"message should name the row id: {msg!r}"
        assert "AAB-Attribute-Identification" in msg, (
            f"message should name the row's task_type: {msg!r}")
        assert "CrossView-MEVA-Event-Ordering" in msg, (
            f"message should name the pool the labels describe: {msg!r}")
        print("foreign-pool colliding id is refused, not averaged: OK")
        return
    raise AssertionError(
        "split() did not raise SystemExit for a foreign-pool colliding id")


def test_missing_label_still_refused():
    cmap, task_types = class_map(LABELS, ALIAS)
    rows = [_row(999, "CrossView-MEVA-Event-Ordering")]
    try:
        split(rows, cmap, task_types)
    except SystemExit as e:
        assert "id=999" in str(e)
        print("row with no label at all is still refused: OK")
        return
    raise AssertionError(
        "split() did not raise SystemExit for an unlabeled id")


def main():
    test_genuine_rows_join_and_score()
    test_foreign_pool_row_is_refused()
    test_missing_label_still_refused()
    print("evidence_class_split_pool_guard: all checks passed")


if __name__ == "__main__":
    main()
