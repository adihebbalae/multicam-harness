# Multi-view understanding legs — run note

`python -m inprocess.run` runs the three option-guided selection arms
(`frame_select_siglip_optu`, `clip_select_viclip_optu`, `query_search_siglip`)
over `data/subsets/mvueval_qa.json`. Both sites run the same legs — same arms,
same models, same protocol — on disjoint shards, so a backed-up queue at one does
not hold the result up; the rows are then concatenated, which they only survive if
every knob below matches.

## Commands

Run from the repo root, as a module: the package imports itself absolutely, so
`python inprocess/run.py` puts the wrong directory on `sys.path` and dies before
argparse. `--subset` and the default `--out` are repo-root-relative.

```bash
export VICLIP_DIR=<local copy of the OpenGVLab/ViCLIP download>   # clip_select_viclip_optu only
# leave STRICT_ANSWER_PROMPT unset: the flag below is what sets it, and an
# exported value that disagrees with the flag is a hard exit

# Qwen leg — three arms, one process
python -m inprocess.run \
    --subset data/subsets/mvueval_qa.json \
    --video-root <root the records' video_i paths hang off> \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --methods frame_select_siglip_optu,clip_select_viclip_optu,query_search_siglip \
    --strict-answer-prompt 1 \
    --no-reasoning --temperature 0.1 --passes 4 --seeds 1,2,3,4 \
    --nframes 8 --budget 0 --sel-tau 0 --sel-tau-q 0.85 \
    --n-queries 4 --query-max-new-tokens 256 --max-new-tokens 8192 \
    --internvl-max-tiles 1

# InternVL leg — same flags, other backend, run from the older-transformers
# environment (its remote code does not load under the newer one)
python -m inprocess.run \
    --subset data/subsets/mvueval_qa.json \
    --video-root <root the records' video_i paths hang off> \
    --model OpenGVLab/InternVL3-8B \
    --methods frame_select_siglip_optu,clip_select_viclip_optu,query_search_siglip \
    --strict-answer-prompt 1 \
    --no-reasoning --temperature 0.1 --passes 4 --seeds 1,2,3,4 \
    --nframes 8 --budget 0 --sel-tau 0 --sel-tau-q 0.85 \
    --n-queries 4 --query-max-new-tokens 256 --max-new-tokens 8192 \
    --internvl-max-tiles 1
```

`--strict-answer-prompt 1` is the value the legs already queued here were
launched with; it is not a default the runner chose, and it is the one knob still
open (see below). Everything else above is the target config verbatim — the
`--methods`, `--budget`, `--sel-*`, `--n-queries`, token and tile values happen to
equal the runner's defaults, and are written out anyway so the leg does not depend
on a default staying put.

One process will not finish this subset in a normal wall clock. Shard it —
`--chunk` is the NUMBER OF SHARDS, not the shard size, and each shard gets its own
output file under the default `--out` naming:

```bash
python -m inprocess.run <every flag from the leg above> \
    --chunk <array width> --offset $SLURM_ARRAY_TASK_ID
```

Use the same `--chunk` at both sites and split the `--offset` values between us
(disjoint halves, agreed before launch). Same shard run twice means duplicate
`(dataset, id, method, backend, pass_idx)` keys in the pooled file, and everything
downstream counts rows rather than de-duplicating them.

## Knobs that must be identical at both sites

Every one of these is stamped on every row, and the runner refuses to append to a
file whose rows disagree with the run about any of them — except the arm name,
which is part of the resume key instead.

| Knob | Value |
|---|---|
| subset | `data/subsets/mvueval_qa.json`, sha256 `5030c22aa91026922c57970c2883adc8d73a8548c2c0f56a590efe11bd1053e4` (stamped per row as `subset_sha256`) |
| methods | `frame_select_siglip_optu,clip_select_viclip_optu,query_search_siglip` — the name is recorded verbatim and is part of the resume key |
| models | `Qwen/Qwen2.5-VL-7B-Instruct`, `OpenGVLab/InternVL3-8B` (full HF ids, not aliases) |
| `--strict-answer-prompt` | **still open — agree before launching**, see below |
| reasoning | off (`--no-reasoning`) |
| `--temperature` / `--passes` / `--seeds` | `0.1` / `4` / `1,2,3,4` |
| `--nframes` / `--budget` | `8` / `0` (matched, nframes x K) |
| `--sel-tau` / `--sel-tau-q` | `0` / `0.85` (quantile mode) |
| `--n-queries` / `--query-max-new-tokens` | `4` / `256` |
| `--max-new-tokens` | `8192` |
| `--internvl-max-tiles` | `1` |
| selection geometry | pinned as module constants in `inprocess/run.py` (candidates per clip, thumbnails per clip, cell px, per-clip floors) and stamped per row — nothing to pass, but a leg run from an edited copy of that file will not pool |

`--strict-answer-prompt {0,1}` picks the answer-hygiene prompt version (v2
enumerates every legal letter and declares the options exhaustive). On this subset
it rewrites essentially every question, so rows made with different values are two
series, not one — the flag is required with no default so neither of us can pick a
side by omission. The queued legs here use `1`; `0` is equally defensible and may yet be
resubmitted that way.
**Confirm the value with me before launching**, pass the same number at both
sites, and check the `strict_prompt` field on the rows before pooling. It is also
in the default output filename, so two prompt versions cannot land in one file by
accident.

`inprocess/README.md` documents `STRICT_ANSWER_PROMPT` in the environment as the
library-level switch, and it still is for direct API use. For these legs the flag
is authoritative: the runner sets the module global from it and hard-exits if an
exported value disagrees, rather than letting a stale shell export change every
prompt invisibly.

## Traps

- **`--budget 0` is the matched budget** (nframes x K, the sequential arm's own),
  not "no budget"; a positive value is an absolute budget and breaks the
  comparison. The runner passes `budget` explicitly to all three arms and then
  re-reads what each one stored, because the three classes do not name `budget` in
  their own signatures — it reaches the parent through `**kw`, whose default is a
  flat per-question frame count. An arm constructed without it gets an unmatched leg
  with no error and nothing to read it against. At the command line the default
  is already `0`, so passing `--budget 0` is belt-and-braces rather than a
  rescue. Nothing in the arm files needs changing; the entry point just never
  leaves it to the default.
- **Do not flatten the media tree.** A large minority of this subset's video
  references carry a subdirectory component — some of them several levels deep,
  with distinct clips sharing one basename — so `--video-root` must be the
  directory those relative paths hang off, not a dump of every clip into one
  place. A flat dump loses exactly those records, and would silently alias the
  colliding ones; the preflight names them and refuses before the model loads.
- **A wrong root does not crash the arms** — they raise per record and the row is
  written with an `error`, so a leg can be all error rows and still exit cleanly.
  The preflight catches it up front, and distinguishes "wrong root" from
  "flattened tree". Error rows are terminal on resume (one row per key), so a
  partially failed leg is resumed with `--retry-errors`, which removes those rows
  and replaces them — do not delete the file, the good rows are in it.
- **The ViCLIP checkpoint is the full model**, plus the BPE vocab and the four
  class files, under `VICLIP_DIR`. The larger `.pth` on the same release is
  vision-encoder-only and the loader deliberately ignores it, so grabbing the
  biggest file gives a scorer that cannot embed text. The repo is gated:
  `huggingface-cli login`, then
  `huggingface-cli download OpenGVLab/ViCLIP --local-dir $VICLIP_DIR`; leave
  `VICLIP_SIZE` unset. The runner verifies all of it before loading a model.
- **The two `*_siglip*` arms need their scorer too.**
  `frame_select_siglip_optu` and `query_search_siglip` load
  `google/siglip-so400m-patch14-384` lazily, inside the per-record prepare step,
  where a failure is caught into an error row — so an absent copy would otherwise
  cost a whole leg and still exit zero. The runner resolves it up front alongside
  the ViCLIP check; fetch it with
  `huggingface-cli download google/siglip-so400m-patch14-384`.
- **Equal frames is not equal tokens on the Qwen backend.** The frame-level and
  query-search arms send their selection as still images, already sized to mirror
  that backend's per-item video pixel caps — but it merges video frames pairwise
  along time (`temporal_patch_size` 2) and does not merge still images, so an
  equal frame count is still not an equal token count against the sequential arm.
  (The clip-level arm hands back whole clips and is not affected.) Read
  `input_tokens` and `video_tokens` off the rows before calling any arm cheaper.
  On InternVL at one tile images and video frames tokenize alike, which is the
  whole reason the arms refuse to construct above one tile — and the runner
  refuses the leg before they get the chance.
- **The subset encodes one of the dataset's two conflicting answer keys.** Score
  against its own `answer` field, never against a key rebuilt from the release;
  the stamped `subset_sha256` is the check that both sites used the same one.
- **Question ids collide across the subsets in this repo**, so the resume key and
  the append guard both include the dataset (the subset filename). Keeping the
  default `--out` naming is what keeps a relaunch from resuming over another
  subset's rows.

## What to send back

- The raw JSONL, unaggregated — every shard file as written, one row per
  `(dataset, id, method, backend, pass_idx)`, with `frame_alloc`, tokens and the
  stamped knobs intact. Do not de-duplicate, filter or merge them first; the rows
  carry `run_id`, `node` and `subset_sha256`, which is what makes the two sites'
  files concatenate.
- The `*_summary.json` the runner writes next to each JSONL when it reaches the
  end. It is written whether or not the leg completed, so treat its absence as
  "the job died", not its presence as "the job is good".
- The runner's final line per leg. It prints `leg COMPLETE` only at the full
  expected row count with no error rows; anything else means the leg is not
  poolable yet, so say which shards are short rather than sending them silently.
