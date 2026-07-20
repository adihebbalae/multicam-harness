#!/usr/bin/env python3
# Ported from Wavy-Hec/CVBench analysis/download_videos.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Fetch CVBench videos from HuggingFace (Dongyh35/CVBench).

The HF repo stores all videos in a single 40 GB archive `CVBench.zip` whose
internal paths are exactly `<id>/<file>.mp4` -- matching the QA references in
CVBench.json. The QA metadata also lives at data/test-00000-of-00001.parquet.

Modes:
  subset (default) : extract ONLY the videos listed in --videos-file directly
                     from the remote zip via HTTP range requests (no 40 GB pull).
                     Ideal for the qualitative Stage-A pass (~101 videos, a few GB).
  full             : download the whole CVBench.zip and extract everything.
                     Use for the Stage-B full eval.
  verify           : given a QA json, count how many referenced videos are missing
                     under the target dir.

Target dir defaults to data/CVBench/ (the CVBench --video-root).

Examples:
  python3 scripts/data/download_videos.py subset
  python3 scripts/data/download_videos.py verify --qa data/subsets/subset.json
  python3 scripts/data/download_videos.py full
"""
import argparse
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_TARGET = os.path.join(REPO, "data", "CVBench")
DEFAULT_VIDEOS_FILE = os.path.join(REPO, "data", "subsets", "subset_videos.txt")
REPO_ID = "Dongyh35/CVBench"
ZIP_NAME = "CVBench.zip"


def _auth_headers():
    from huggingface_hub import get_token
    tok = get_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _zip_url():
    from huggingface_hub import hf_hub_url
    return hf_hub_url(REPO_ID, ZIP_NAME, repo_type="dataset")


def qa_video_paths(qa_path):
    with open(qa_path) as f:
        data = json.load(f)
    paths = []
    for r in data:
        for i in range(1, 5):
            v = r.get(f"video_{i}")
            if v:
                paths.append(v)
    return sorted(set(paths))


def cmd_subset(args):
    from remotezip import RemoteZip
    with open(args.videos_file) as f:
        wanted = [l.strip() for l in f if l.strip()]
    todo = [w for w in wanted if not os.path.exists(os.path.join(args.target, w))]
    print(f"{len(wanted)} videos requested; {len(wanted) - len(todo)} already present; "
          f"{len(todo)} to fetch -> {args.target}")
    if not todo:
        return
    os.makedirs(args.target, exist_ok=True)
    fetched, missing, total_mb = 0, [], 0.0
    with RemoteZip(_zip_url(), headers=_auth_headers()) as z:
        names = set(z.namelist())
        for i, w in enumerate(todo, 1):
            if w not in names:
                missing.append(w)
                print(f"  [{i}/{len(todo)}] MISSING in archive: {w}")
                continue
            z.extract(w, args.target)
            sz = os.path.getsize(os.path.join(args.target, w)) / 1e6
            total_mb += sz
            fetched += 1
            print(f"  [{i}/{len(todo)}] {w}  ({sz:.1f} MB)")
    print(f"\nFetched {fetched} videos ({total_mb/1000:.2f} GB). Missing: {len(missing)}")
    if missing:
        print("  missing:", *missing, sep="\n  ")


def cmd_full(args):
    from huggingface_hub import hf_hub_download
    print(f"Downloading {ZIP_NAME} (~40 GB) ...")
    zip_path = hf_hub_download(REPO_ID, ZIP_NAME, repo_type="dataset",
                               local_dir=args.zip_dir)
    print(f"Extracting -> {args.target}")
    os.makedirs(args.target, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.target)
    print("Done.")


def cmd_verify(args):
    wanted = qa_video_paths(args.qa)
    missing = [w for w in wanted if not os.path.exists(os.path.join(args.target, w))]
    print(f"QA file: {args.qa}")
    print(f"  unique videos referenced: {len(wanted)}")
    print(f"  present: {len(wanted) - len(missing)}")
    print(f"  MISSING: {len(missing)}")
    for w in missing[:25]:
        print("   -", w)
    if len(missing) > 25:
        print(f"   ... and {len(missing) - 25} more")
    raise SystemExit(0 if not missing else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("subset")
    s.add_argument("--videos-file", default=DEFAULT_VIDEOS_FILE)
    s.add_argument("--target", default=DEFAULT_TARGET)
    s.set_defaults(func=cmd_subset)

    f = sub.add_parser("full")
    f.add_argument("--target", default=DEFAULT_TARGET)
    f.add_argument("--zip-dir", default=os.path.join(REPO, "data"))
    f.set_defaults(func=cmd_full)

    v = sub.add_parser("verify")
    v.add_argument("--qa", default=os.path.join(REPO, "data", "subsets", "subset.json"))
    v.add_argument("--target", default=DEFAULT_TARGET)
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
