#!/usr/bin/env python3
# Ported from Wavy-Hec/CVBench analysis/fetch_egoexo_videos.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Fetch the Ego-Exo4D videos referenced by a CrossView subset into the release root.

Unlike MEVA, Ego-Exo4D is LICENSE-GATED: there is no anonymous HTTPS bucket. You must
  1. accept the license at https://ego-exo4d-data.org/  -> get AWS-style credentials,
  2. `pip install ego4d` (provides the `egoexo` CLI downloader),
  3. fetch the metadata part once to get `takes.json` (the take_name -> take_uid map):
        egoexo -o /tmp/egoexo_meta --parts metadata
This script does NOT download anything itself (it can't carry your license). It:
  * extracts the distinct Ego-Exo4D *takes* the subset needs (deduped across questions),
  * maps take_name -> take_uid via your `takes.json` (so the download is restricted to
    exactly the needed takes, not all ~5k),
  * prints the precise `egoexo` command to run (or runs it with --run if egoexo is on PATH),
  * with --check, reports which expected video files are present/missing under the release
    root so you can confirm before launching the eval.

The release QA paths are `videos/ego-exo4d/takes/<take>/frame_aligned_videos/downscaled/448/<cam>.mp4`,
which is exactly the layout `egoexo --parts downscaled_takes/448 -o <release_root>/videos/ego-exo4d`
produces, so no moving/transcoding is needed (decord reads these .mp4s).

Run from the repo root (no GPU; the actual download needs internet + your Ego-Exo4D credentials):
  # 1. see what's needed + the command to run
  python3 scripts/data/fetch_egoexo_videos.py --subset data/subsets/crossview_egoexo_subset.json \
      --takes-json /tmp/egoexo_meta/takes.json
  # 2. (optionally) let this script run egoexo for you
  python3 scripts/data/fetch_egoexo_videos.py --subset data/subsets/crossview_egoexo_subset.json \
      --takes-json /tmp/egoexo_meta/takes.json --run
  # 3. confirm the files landed where the harness expects them
  python3 scripts/data/fetch_egoexo_videos.py --subset data/subsets/crossview_egoexo_subset.json --check
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_ROOT = os.path.join(REPO, "data", "crossview-release-annotations", "crossview-release")


def egoexo_dest_paths(subset_path):
    """Deduped <=4 release-relative video paths for Ego-Exo4D questions in the subset."""
    seen, uniq = set(), []
    for r in json.load(open(subset_path)):
        for i in range(1, 5):
            vp = r.get(f"video_{i}")
            if vp and "/ego-exo4d/" in vp and vp not in seen:
                seen.add(vp)
                uniq.append(vp)
    return uniq


def take_of(vp):
    """take name from .../takes/<take>/frame_aligned_videos/..."""
    parts = vp.split("/")
    return parts[parts.index("takes") + 1] if "takes" in parts else None


def load_name_to_uid(takes_json):
    """Map take_name -> take_uid from an Ego-Exo4D takes.json (list or dict form)."""
    data = json.load(open(takes_json))
    rows = data.values() if isinstance(data, dict) else data
    mapping = {}
    for t in rows:
        name = t.get("take_name") or t.get("name")
        uid = t.get("take_uid") or t.get("uid")
        if name and uid:
            mapping[name] = uid
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=os.path.join(REPO, "data", "subsets", "crossview_egoexo_subset.json"))
    ap.add_argument("--release-root", default=DEFAULT_ROOT)
    ap.add_argument("--takes-json", default=None,
                    help="Ego-Exo4D takes.json (from `egoexo --parts metadata`); needed to "
                         "map take names to the --uids the downloader expects")
    ap.add_argument("--parts", default="downscaled_takes/448",
                    help="egoexo --parts value; 448 matches the release's downscaled layout")
    ap.add_argument("--release", default="v2",
                    help="egoexo --release value; v2 is the current public release (the CLI "
                         "default), passed explicitly to be future-proof")
    ap.add_argument("--s3-profile", default=None,
                    help="AWS profile holding your Ego-Exo4D credentials (from `aws configure "
                         "--profile <name>`); omit to use the 'default' profile")
    ap.add_argument("--run", action="store_true",
                    help="actually invoke egoexo (requires it on PATH + your credentials)")
    ap.add_argument("--check", action="store_true",
                    help="only report which expected files are present/missing, then exit")
    args = ap.parse_args()

    dests = egoexo_dest_paths(args.subset)
    takes = sorted({t for t in (take_of(vp) for vp in dests) if t})
    out_dir = os.path.join(args.release_root, "videos", "ego-exo4d")
    print(f"{len(dests)} Ego-Exo4D video files across {len(takes)} takes "
          f"referenced by {os.path.basename(args.subset)}")

    # --check: verify presence under the release root (use before launching the eval)
    if args.check:
        missing = [vp for vp in dests
                   if not (os.path.exists(os.path.join(args.release_root, vp))
                           and os.path.getsize(os.path.join(args.release_root, vp)) > 0)]
        print(f"present: {len(dests) - len(missing)} | missing: {len(missing)}")
        for vp in missing[:20]:
            print("  MISSING:", vp)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        sys.exit(1 if missing else 0)

    # map take names -> uids (download is restricted to these uids)
    uids = None
    if args.takes_json:
        name2uid = load_name_to_uid(args.takes_json)
        uids, unmapped = [], []
        for t in takes:
            (uids.append(name2uid[t]) if t in name2uid else unmapped.append(t))
        if unmapped:
            print(f"WARNING: {len(unmapped)} take(s) not found in takes.json "
                  f"(stale metadata?): {unmapped[:5]}")
        print(f"mapped {len(uids)}/{len(takes)} takes to UIDs")
    else:
        print("\nNo --takes-json given. Fetch the metadata once, then re-run with it:")
        print("  egoexo -o /tmp/egoexo_meta --parts metadata")
        print("  python3 scripts/data/fetch_egoexo_videos.py --subset",
              args.subset, "--takes-json /tmp/egoexo_meta/takes.json")

    cmd = ["egoexo", "-o", out_dir, "--parts", args.parts, "--release", args.release, "-y"]
    if args.s3_profile:
        cmd += ["--s3_profile", args.s3_profile]
    if uids:
        cmd += ["--uids", *uids]
    print("\nDownload command (needs your Ego-Exo4D license/credentials):")
    head = cmd if len(cmd) < 14 else cmd[: cmd.index("--uids") + 1] + [f"... {len(uids)} uids"]
    print("  " + " ".join(head))

    if args.run:
        if not uids:
            print("\nFATAL: --run needs --takes-json to resolve UIDs.", file=sys.stderr)
            sys.exit(2)
        print(f"\nRunning egoexo for {len(uids)} takes -> {out_dir} ...")
        subprocess.run(cmd, check=True)
        print("Done. Re-run with --check to confirm the files the harness needs are present.")


if __name__ == "__main__":
    main()
