#!/usr/bin/env python3
# Ported from Wavy-Hec/CVBench analysis/fetch_meva_videos.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Download the MEVA videos referenced by a CrossView subset into the release root.

MEVA is public (CC-BY-4.0) on the open S3 bucket `mevadata-public-01`, served over
plain HTTPS (no AWS account / credentials needed). Ground-camera clips live at
  drops-123-r13/<date>/<hour>/<name>.avi
but the hour sub-dir does not map cleanly from the filename, so for each needed
date we LIST drops-123-r13/<date>/ and match clips by basename.

The release QA paths are like `videos/meva/mp4_resized/<date>/<hour>/<slot>/<name>.EXT`.
Source files are `.avi`. If the subset was built with `--meva-video-ext avi`
(recommended) the .avi is saved directly. If the path ends in `.mp4`, the .avi is
transcoded with ffmpeg (must be installed).

Run (no GPU; needs internet), from the repo root:
  python3 scripts/data/fetch_meva_videos.py --subset data/subsets/crossview_subset.json
  python3 scripts/data/fetch_meva_videos.py --subset data/subsets/crossview_subset.json --limit 2   # smoke
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_ROOT = os.path.join(REPO, "data", "crossview-release-annotations", "crossview-release")
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def meva_dest_paths(subset_path):
    import json
    out = []
    for r in json.load(open(subset_path)):
        for i in range(1, 5):
            vp = r.get(f"video_{i}")
            if vp and "/meva/" in vp:
                out.append(vp)
    # dedupe, keep order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def stem_and_date(dest_path):
    base = os.path.basename(dest_path)
    stem = base.rsplit(".", 1)[0]          # strip .avi/.mp4
    date = stem.split(".")[0]              # YYYY-MM-DD
    return stem, date


def list_date(bucket, prefix, date):
    """Return {stem: key} for all .avi clips under <prefix>/<date>/ (paginated)."""
    out, token = {}, None
    base = f"https://{bucket}.s3.amazonaws.com/"
    while True:
        q = {"list-type": "2", "prefix": f"{prefix}/{date}/"}
        if token:
            q["continuation-token"] = token
        url = base + "?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall(f"{S3_NS}Contents"):
            key = c.find(f"{S3_NS}Key").text
            if key.endswith(".avi"):
                out[os.path.basename(key)[:-4]] = key
        if (root.findtext(f"{S3_NS}IsTruncated") or "false") == "true":
            token = root.findtext(f"{S3_NS}NextContinuationToken")
        else:
            break
    return out


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dest)
    return os.path.getsize(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=os.path.join(REPO, "data", "subsets", "crossview_subset.json"))
    ap.add_argument("--release-root", default=DEFAULT_ROOT)
    ap.add_argument("--bucket", default="mevadata-public-01")
    ap.add_argument("--prefix", default="drops-123-r13")
    ap.add_argument("--limit", type=int, default=0, help="only fetch first N (smoke)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dests = meva_dest_paths(args.subset)
    if args.limit:
        dests = dests[: args.limit]
    print(f"{len(dests)} MEVA videos referenced by {os.path.basename(args.subset)}")

    have_ffmpeg = shutil.which("ffmpeg") is not None
    date_cache, done, miss, transcode_needed, mb = {}, 0, [], [], 0.0
    for n, vp in enumerate(dests, 1):
        dest = os.path.join(args.release_root, vp)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            done += 1; continue
        stem, date = stem_and_date(vp)
        if date not in date_cache:
            date_cache[date] = list_date(args.bucket, args.prefix, date)
        key = date_cache[date].get(stem)
        if not key:
            miss.append(vp); print(f"  [{n}/{len(dests)}] MISSING on S3: {stem}.avi"); continue
        url = f"https://{args.bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"
        if args.dry_run:
            print(f"  [{n}/{len(dests)}] would fetch {key} -> {vp}"); continue
        if vp.endswith(".avi"):
            sz = download(url, dest)
        else:  # .mp4 dest -> need transcode
            if not have_ffmpeg:
                transcode_needed.append(vp)
                print(f"  [{n}/{len(dests)}] need ffmpeg to make .mp4 (or rebuild subset "
                      f"with --meva-video-ext avi): {vp}")
                continue
            tmp = dest + ".avi"
            download(url, tmp)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, dest], check=True)
            os.remove(tmp)
            sz = os.path.getsize(dest)
        mb += sz / 1e6
        print(f"  [{n}/{len(dests)}] {vp}  ({sz/1e6:.1f} MB)")

    fetched = len(dests) - done - len(miss) - len(transcode_needed)
    print(f"\nMEVA videos: {len(dests)} referenced | already had: {done} | "
          f"downloaded: {fetched} ({mb/1000:.2f} GB) | "
          f"missing-on-s3: {len(miss)} | need-ffmpeg: {len(transcode_needed)}")
    if transcode_needed:
        print("  -> install ffmpeg (conda install -c conda-forge ffmpeg) "
              "or rebuild the subset with --meva-video-ext avi to skip transcoding.")
    sys.exit(1 if (miss or transcode_needed) else 0)


if __name__ == "__main__":
    main()
