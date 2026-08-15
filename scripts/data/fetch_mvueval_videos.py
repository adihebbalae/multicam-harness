#!/usr/bin/env python3
# New file (not a port): the fork downloaded the MVU-Eval release by hand, so
# there is no upstream source for this one. docs/PORTING.md's byte-faithfulness
# rule covers ported bodies; nothing here is ported.
"""Fetch the MVU-Eval release (question file + the clips a subset needs).

MVU-Eval is a public HF dataset — no license gate, no credentials — so unlike
the Ego-Exo4D and AgiBot fetchers this one really does download. The release
holds ~4.9k mp4 clips; a full pull is large and usually unnecessary, so the
default fetches only the clips referenced by one subset.

Repository layout (repo_id MVU-Eval-Team/MVU-Eval-Data, repo_type "dataset"):
  MVU_Eval_QAs.json          the questions -> what scripts/data/build_mvueval.py reads
  mvu_eval_config.csv        a second copy of the answer key (see the warning below)
  <name>.mp4                 most clips, flat at the repo root
  temp_ordering-videos/      four subdirectories hold the rest
  temp_grounding-videos/
  temp_cap_filling-videos/
  video_editing-video/
The record `video_i` fields are these repo-relative paths verbatim, so whatever
directory you pass as --root here is exactly what the harness wants as
--video-root. Nothing is renamed, moved or transcoded.

WARNING -- two answer keys. The release ships the ground truth twice, in
MVU_Eval_QAs.json and in mvu_eval_config.csv, and they disagree on 92 of the 1,824
questions (5.04%). build_mvueval.py reads the JSON. If you score against the CSV instead
you will get different numbers on those 92 for reasons that have
nothing to do with the harness, so pick one key and say which one you used.

Run from the repo root (needs internet; no GPU):
  # 1. question file only -- enough to build the pool and read it
  python3 scripts/data/fetch_mvueval_videos.py --qa-only
  # 2. the clips the committed dev subset needs (a few hundred, not all ~4.9k)
  python3 scripts/data/fetch_mvueval_videos.py --subset data/subsets/mvueval_dev_subset.json
  # 3. the whole release, for a full-pool run
  python3 scripts/data/fetch_mvueval_videos.py --all
  # 4. confirm what is on disk without downloading anything
  python3 scripts/data/fetch_mvueval_videos.py --subset data/subsets/mvueval_dev_subset.json --check
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

if REPO not in sys.path:
    sys.path.insert(0, REPO)

from inprocess.dataloaders.qa_json import MAX_SLOTS  # noqa: E402

REPO_ID = "MVU-Eval-Team/MVU-Eval-Data"
QA_FILES = ["MVU_Eval_QAs.json", "mvu_eval_config.csv"]
DEFAULT_ROOT = os.path.join(REPO, "data", "mvueval")
DEFAULT_SUBSET = os.path.join(REPO, "data", "subsets", "mvueval_dev_subset.json")


def clips_of(subset_json):
    """Every distinct video_i path a subset's records reference, in sorted order."""
    records = json.load(open(subset_json, encoding="utf-8"))
    paths = set()
    for rec in records:
        for i in range(1, MAX_SLOTS + 1):
            vp = rec.get(f"video_{i}")
            if vp:
                paths.add(vp)
    return sorted(paths)


def report_presence(root, paths):
    """Split paths into present/missing under root. Returns the missing ones."""
    missing = [p for p in paths if not os.path.exists(os.path.join(root, p))]
    print(f"  under {root}: {len(paths) - len(missing)}/{len(paths)} present")
    for p in missing[:10]:
        print(f"    missing: {p}")
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="where the release lands; pass this same path to the "
                         "harness as --video-root")
    ap.add_argument("--subset", default=DEFAULT_SUBSET,
                    help="fetch only the clips this subset's records reference")
    ap.add_argument("--all", action="store_true",
                    help="fetch every clip in the release, ignoring --subset")
    ap.add_argument("--qa-only", action="store_true",
                    help="fetch only the question file and the config csv")
    ap.add_argument("--check", action="store_true",
                    help="report what is already on disk; download nothing")
    args = ap.parse_args()

    if args.all and args.qa_only:
        raise SystemExit("--all and --qa-only are mutually exclusive")

    if args.qa_only:
        wanted, patterns = list(QA_FILES), list(QA_FILES)
    elif args.all:
        wanted, patterns = list(QA_FILES), None
    else:
        clips = clips_of(args.subset)
        print(f"subset {args.subset}: {len(clips)} distinct clips")
        wanted = QA_FILES + clips
        patterns = list(wanted)

    if args.check:
        print("checking what is already on disk (no download):")
        missing = report_presence(args.root, wanted)
        raise SystemExit(1 if missing else 0)

    from huggingface_hub import snapshot_download

    print(f"downloading {REPO_ID} -> {args.root}")
    print("  scope:", "entire release" if patterns is None
          else f"{len(patterns)} files")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                      local_dir=args.root, allow_patterns=patterns)

    print("done. verifying:")
    missing = report_presence(args.root, wanted)
    if missing:
        raise SystemExit("some expected files did not land -- rerun to resume")
    print(f"\nnext:\n"
          f"  python3 scripts/data/build_mvueval.py --qa-json {os.path.join(args.root, 'MVU_Eval_QAs.json')}\n"
          f"  then run with --video-root {args.root}")


if __name__ == "__main__":
    main()
