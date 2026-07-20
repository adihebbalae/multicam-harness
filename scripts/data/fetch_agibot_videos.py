#!/usr/bin/env python3
# Ported from Wavy-Hec/CVBench analysis/fetch_agibot_videos.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Incrementally fetch AgiBot World videos for the CrossView benchmark.

AgiBot videos live in per-task .tar shards (2-287 GB each) inside the GATED HF dataset
`agibot-world/AgiBotWorld-Beta` (CC BY-NC-SA 4.0, NON-COMMERCIAL). Individual episode
mp4s are NOT separately addressable on HF, so the smallest fetch unit is a whole task
tar. This helper therefore fetches **cheapest tars first** up to a `--budget-gb` cap,
extracts only the needed head_color/hand_*_color.mp4 into the release layout, and
**deletes each tar** (delete-as-you-go -> peak disk ~= one tar). It is idempotent:
re-run later with a bigger budget to extend coverage; episodes already present are
skipped. Pair with `build_crossview.py --require-local <root>` to build a runnable
MEVA+AgiBot subset from whatever has been fetched so far.

Prereqs (one-time, only you can do these):
  - accept the gate at https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta
    (instant click-through; needs an HF account)
  - have an HF token on this machine:  export HF_TOKEN=hf_xxx   (or run `hf auth login`)

Usage from the repo root (no GPU; the tree listing is public, the tar download needs your token):
  python3 scripts/data/fetch_agibot_videos.py                 # plan + full cost, fetch nothing
  python3 scripts/data/fetch_agibot_videos.py --budget-gb 10 --run   # grab cheapest tars <=~10 GB
  python3 scripts/data/fetch_agibot_videos.py --budget-gb 100 --run  # later: get more (skips local)
  python3 scripts/data/fetch_agibot_videos.py --check         # coverage report
"""
import argparse
import json
import os
import sys
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_ROOT = os.path.join(REPO, "data", "crossview-release-annotations", "crossview-release")
REPO_ID = "agibot-world/AgiBotWorld-Beta"
TREE_API = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main/observations/"
RESOLVE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"
TREE_CACHE = os.path.join(REPO, "data", ".agibot_tree_cache.json")
REL_PREFIX = "videos/agibot/"   # release-relative paths start here; tar members drop it


def hf_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for p in (os.path.expanduser("~/.cache/huggingface/token"),
              os.path.expanduser("~/.huggingface/token")):
        if os.path.exists(p):
            return open(p).read().strip()
    return None


def agibot_targets(qa_path):
    """episode-key (task, ep) -> sorted list of release-relative video paths needed."""
    targets = {}
    for r in json.load(open(qa_path)):
        if r.get("source") != "agibot":
            continue
        for i in range(1, 5):
            vp = r.get(f"video_{i}")
            if not vp or "/agibot/" not in vp:
                continue
            p = vp.split("/")        # videos/agibot/observations/<task>/<ep>/videos/<cam>.mp4
            key = (p[3], p[4])
            targets.setdefault(key, set()).add(vp)
    return {k: sorted(v) for k, v in targets.items()}


def load_tree(tasks, refresh=False):
    """task_id -> [(tar_path, size_bytes)] from the public HF tree API (cached)."""
    cache = {}
    if os.path.exists(TREE_CACHE) and not refresh:
        cache = json.load(open(TREE_CACHE))
    missing = [t for t in tasks if t not in cache]
    for n, t in enumerate(missing, 1):
        try:
            with urllib.request.urlopen(TREE_API + t, timeout=60) as f:
                entries = json.load(f)
            cache[t] = [[e["path"], e.get("size", 0)] for e in entries
                        if e["path"].endswith(".tar")]
        except Exception as e:
            print(f"  tree API error for task {t}: {e}", file=sys.stderr)
            cache[t] = []
        if n % 20 == 0:
            print(f"  listed {n}/{len(missing)} tasks...")
    if missing:
        json.dump(cache, open(TREE_CACHE, "w"))
    return cache


def tar_for_episode(tars, ep):
    """Return (tar_path, size) whose <lo>-<hi>.tar range brackets episode ep, else None."""
    ep = int(ep)
    for path, size in tars:
        base = os.path.basename(path)[:-4]
        if "-" in base:
            lo, hi = base.split("-", 1)
            if lo.isdigit() and hi.isdigit() and int(lo) <= ep <= int(hi):
                return path, size
    return None


def have_all(paths, root):
    return all(os.path.exists(os.path.join(root, p)) and os.path.getsize(os.path.join(root, p)) > 0
               for p in paths)


def plan(targets, tree, root):
    """Group needed episodes by their bracketing tar; return tars not yet satisfied."""
    by_tar = {}          # tar_path -> {"size":int, "need":[(rel_paths...)], "all_local":bool}
    unmatched = []
    for (task, ep), rels in targets.items():
        if have_all(rels, root):
            continue                                   # episode already fetched
        hit = tar_for_episode(tree.get(task, []), ep)
        if not hit:
            unmatched.append((task, ep))
            continue
        path, size = hit
        slot = by_tar.setdefault(path, {"size": size, "need": []})
        slot["need"].append(rels)
    return by_tar, unmatched


def stream_download(url, dest, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got, mark = 0, 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got - mark > (2 << 30):                 # progress every ~2 GB
                mark = got
                pct = f" ({100*got/total:.0f}%)" if total else ""
                print(f"      ...{got/1e9:.1f} GB{pct}")
    os.replace(tmp, dest)
    return os.path.getsize(dest)


def extract_needed(tar_path, needed_rels, root):
    """Extract only the needed `<cam>.mp4` members into <root>/videos/agibot/..."""
    wanted = {p[len(REL_PREFIX):] for p in needed_rels}   # tar member = rel minus videos/agibot/
    out_base = os.path.join(root, REL_PREFIX.rstrip("/"))
    got = 0
    with tarfile.open(tar_path, "r") as tf:
        members = {m.name.lstrip("./"): m for m in tf.getmembers() if m.isfile()}
        for member in wanted:
            m = members.get(member)
            if m is None:                               # tolerate a leading dir in member names
                m = next((mm for nm, mm in members.items() if nm.endswith(member)), None)
            if m is None:
                print(f"      WARN: {member} not found inside tar")
                continue
            m.name = member                             # extract at the expected relative path
            tf.extract(m, out_base)
            got += 1
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=os.path.join(REPO, "data", "subsets", "crossview_agibot_qa.json"),
                    help="QA/subset json listing AgiBot questions to cover (default: full "
                         "AgiBot pool, so coverage can grow over successive runs)")
    ap.add_argument("--release-root", default=DEFAULT_ROOT)
    ap.add_argument("--budget-gb", type=float, default=0,
                    help="stop after this many GB of tars (cheapest first). 0 = plan only / "
                         "with --run, no cap (the full pull)")
    ap.add_argument("--max-tars", type=int, default=0, help="also cap the number of tars")
    ap.add_argument("--scratch", default=os.path.join(DEFAULT_ROOT, "..", "agibot_tar_scratch"),
                    help="where to stage the big tars (must be on a roomy fs, NOT /tmp)")
    ap.add_argument("--refresh-tree", action="store_true", help="re-list the HF tree (ignore cache)")
    ap.add_argument("--run", action="store_true", help="actually download+extract+delete")
    ap.add_argument("--check", action="store_true", help="coverage report only, then exit")
    args = ap.parse_args()

    targets = agibot_targets(args.qa)
    tasks = sorted({t for t, _ in targets}, key=int)
    present = sum(1 for rels in targets.values() if have_all(rels, args.release_root))
    print(f"{len(targets)} AgiBot episodes referenced by {os.path.basename(args.qa)} "
          f"across {len(tasks)} tasks | already local: {present}")

    if args.check:
        sys.exit(0 if present == len(targets) else 1)

    print("listing per-task tar shards from Hugging Face (public)...")
    tree = load_tree(tasks, refresh=args.refresh_tree)
    by_tar, unmatched = plan(targets, tree, args.release_root)
    if not by_tar:
        print("Nothing to fetch — all referenced episodes are already local.")
        sys.exit(0)
    if unmatched:
        print(f"WARNING: {len(unmatched)} episodes had no bracketing tar (check task ids): "
              f"{unmatched[:3]}")

    order = sorted(by_tar.items(), key=lambda kv: kv[1]["size"])   # cheapest first
    total_gb = sum(v["size"] for _, v in order) / 1e9
    print(f"\n{len(order)} tars remaining to fully cover this QA set (~{total_gb:.0f} GB total).")

    # apply budget / count caps
    selected, cum = [], 0.0
    for path, info in order:
        gb = info["size"] / 1e9
        if args.budget_gb and cum + gb > args.budget_gb and selected:
            break
        if args.max_tars and len(selected) >= args.max_tars:
            break
        selected.append((path, info))
        cum += gb
    eps_covered = sum(len(info["need"]) for _, info in selected)
    print(f"this run: {len(selected)} tars (~{cum:.1f} GB) -> +{eps_covered} episodes")
    for path, info in selected:
        print(f"  {info['size']/1e9:6.1f} GB  {path}  (+{len(info['need'])} eps)")

    if not args.run:
        print("\n(plan only — re-run with --run to download+extract+delete)")
        sys.exit(0)

    token = hf_token()
    if not token:
        print("\nFATAL: no HF token. `export HF_TOKEN=hf_...` or `hf auth login`, and accept the "
              f"gate at https://huggingface.co/datasets/{REPO_ID}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.scratch, exist_ok=True)
    done = 0
    for path, info in selected:
        dest = os.path.join(args.scratch, os.path.basename(path))
        needed = [rel for rels in info["need"] for rel in rels]
        print(f"\n[{done+1}/{len(selected)}] {path}  ({info['size']/1e9:.1f} GB)")
        try:
            stream_download(RESOLVE + path, dest, token)
            n = extract_needed(dest, needed, args.release_root)
            print(f"      extracted {n} mp4s")
        finally:
            if os.path.exists(dest):
                os.remove(dest)                          # delete-as-you-go
        done += 1

    still = sum(1 for rels in targets.values() if not have_all(rels, args.release_root))
    print(f"\nDone. {len(targets)-still}/{len(targets)} episodes now local "
          f"({still} remaining). Rebuild a runnable subset with:")
    print(f"  python3 scripts/data/build_crossview.py --sources meva,agibot --meva-video-ext avi "
          f"--require-local {args.release_root} --out-subset data/subsets/crossview_meva_agibot_subset.json ...")


if __name__ == "__main__":
    main()
