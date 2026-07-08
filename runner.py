# Ported from Wavy-Hec/CVBench bench/run_bench.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""Experiment loop for run_vqa.py: the resume filter, the backend load-once loop
over methods -> records -> passes (answer, append JSONL, flush), and the
summary/report tail.

Split minimally out of the fork's ``bench/run_bench.py`` ``main()``: run_vqa.py
owns the argparse block and calls ``run(args, METHODS, CLIP_SELECT_RE,
make_backend, make_method)``; ``run`` is the remainder of ``main()`` from the
``data = json.load(...)`` line onward, byte-identical apart from the default
results path being rooted at ``results/`` instead of the fork's module dir.
"""
import json
import os

from evaluation import scoring as metrics


def load_done(path):
    """Completed (id, method, backend, pass_idx) keys for resume."""
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                done.add((r.get("id"), r.get("method"), r.get("backend"), r.get("pass_idx")))
    return done


def run(args, METHODS, CLIP_SELECT_RE, make_backend, make_method):
    data = json.load(open(args.subset))
    if args.chunk and args.chunk > 1:
        data = data[args.offset::args.chunk]
    if args.limit:
        data = data[: args.limit]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for m in methods:
        if m not in METHODS and not CLIP_SELECT_RE.match(m):
            raise SystemExit(f"unknown method '{m}'. Known: {list(METHODS)} "
                             f"or clip_select[_<scorer>]_top<m>")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()][: args.passes]
    if len(seeds) < args.passes:
        raise SystemExit(f"need >= {args.passes} seeds, got {seeds}")

    out = args.out or os.path.join(
        "results",
        f"bench_{os.path.splitext(os.path.basename(args.subset))[0]}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = load_done(out)
    print(f"subset={args.subset} n={len(data)} methods={methods} backends={backends} "
          f"passes={args.passes} seeds={seeds} temp={args.temperature}")
    print(f"video_root={args.video_root}\nout={out} (already done: {len(done)})", flush=True)

    from tqdm import tqdm
    with open(out, "a") as fh:
        for b in backends:
            backend = make_backend(b, nframes=args.nframes,
                                   internvl_max_tiles=args.internvl_max_tiles)  # loads the model once
            for mname in methods:
                method = make_method(mname, backend, args)
                # process all passes of a record consecutively so the centralized
                # montage cache (and fixed frames) are reused across passes.
                # Resume must key on method.name (what rows record), not mname:
                # e.g. WEIGHTING=even runs under mname 'temporal_weighted' but
                # writes method='temporal_even'.
                jobs = [(rec, pi, sd) for rec in data
                        for pi, sd in enumerate(seeds, 1)
                        if (rec["id"], method.name, backend.name, pi) not in done]
                for rec, pass_idx, seed in tqdm(jobs, desc=f"{mname}/{backend.name}"):
                    res = method.answer(rec, args.video_root, seed=seed)
                    res.pass_idx = pass_idx
                    fh.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
                    fh.flush()

    rows = [json.loads(l) for l in open(out) if l.strip()]
    print(metrics.format_summary(rows))
    sumpath = out.replace(".jsonl", "_summary.json")
    json.dump(metrics.summarize_by_method_backend_passes(rows), open(sumpath, "w"), indent=2)
    print(f"\nsummary -> {sumpath}")
