# CVBench text-only analysis files

The analysis outputs behind [`docs/cvbench_text_only.md`](../../docs/cvbench_text_only.md):
Qwen3.8-27B answering CVBench's 1000 questions with no video, as a thinking leg (`_q38bm`, effort
medium) and a no-thinking control (`_q38nt`), each in both option orders (O1 = released, O2 =
reversed). Section numbers below are the write-up's. The rows are in [`../legs/`](../legs/)
(`cvbench_full_runnable_subset{,_o2}_cvbench_{q38bm,q38nt}.jsonl.gz`), the question records in
[`../../data/subsets/`](../../data/subsets/) (`cvbench_full_runnable_subset.json` for O1,
`cvbench_full_runnable_subset_o2.json` for O2).

| file | holds |
|---|---|
| [`cvbench_text_answerability_q38bm.json`](cvbench_text_answerability_q38bm.json) | the analysis of the thinking leg: each order's tables, the both-orders label and its nulls, the with-video cross-tab, the 45-question slice and every question's label |
| [`cvbench_text_answerability_q38nt.json`](cvbench_text_answerability_q38nt.json) | the same for the no-thinking control |
| [`text_answerable_ids.json`](text_answerable_ids.json) | all 1000 questions with both legs' seeds right per order and the text-answerable flags (section 3.2) |
| [`fig_cvbench_data.json`](fig_cvbench_data.json) | every number plotted in the 12 figures (section 2) |

The two analysis JSONs were written by `analysis/cvbench_text_answerability.py` and
`fig_cvbench_data.json` by `analysis/make_cvbench_text_figs.py`, both at Wavy-Hec/MultiCam commit
`6c377c0`. `text_answerable_ids.json` is built from the two analysis JSONs (each question's task,
format and k values, and the flags derived from them) and from the option texts of the two subsets
in `data/subsets/` (the `o2_position_dependent` flags); no script in either repository writes it.

## The two analysis JSONs

`cvbench_text_answerability_q38bm.json` (thinking) and `cvbench_text_answerability_q38nt.json` (no
thinking) have the same layout. Percentages are rounded half-up from counts, which sit beside them
(`n_correct`, `n_rows`). Top-level keys:

| key | holds |
|---|---|
| `provenance` | `argv`, `script` (path and sha256), `script_git_blob`, `seed` (0), `permutations` (100,000), `generator`, `numpy`, `draws_per_block`, `allow_partial`, and every input file with its sha256: `subsets` (O1, O2), `rows` (O1, O2), `sighted`, `slices` |
| `O1`, `O2` | one option order each: `files`, `backend`, `decode` (the rows' decode stamp), `passes_per_question`, `torn_lines`, `duplicate_keys`, `partial_questions`, `overall`, `per_task`, `text_answerable_split`, `video_ref_options`, `hygiene`, `reparse`, `questions` |
| `both_orders` | the both-orders label (text-answerable in O1 and in O2): `n_questions`, `labels`, `expected_both_at_chance`, `bias_null` (the independent null), `bias_null_paired` (the paired null, the baseline the write-up uses), `per_task`, `by_format` (`lettered`, `yes/no`) |
| `sighted` | the Era-1 InternVL3-8B with-video arms split by this leg's O1 label: `fulldur:temporal_weighted/InternVL3-8B` (the best arm) and `perstream:per_stream/InternVL3-8B` |
| `slices` | `era1_45`, the 45-question Era-1 slice of section 4.3, laid out like `overall` (O1), with the O2 values under `O2` |

The fields a reader needs most:

- **`overall` and each `per_task` entry**: `n_questions`, `n_rows`, `n_correct`, `accuracy`
  (`mean`, `std` = population SD over the 4 passes, `n_passes`), `per_pass`, `per_pass_counts`,
  `chance`, `key_floor_insample`, `key_floor_loo`, `longest_option`, `abstained_rows`, `error_rows`,
  `labels` (k of 4 passes right: `text-answerable` 3 or 4, `unstable` 1 or 2, `not` 0),
  `expected_text_answerable_at_chance` and `bias_null`.
- **A null block** (`bias_null`, `bias_null_paired`): `observed_*`, `null_mean_*`, `null_95_*` (the
  central 95% range), `p_*` = (1 + draws ≥ observed) / (B + 1), stored to 5 decimals, so the floor
  1/100,001 reads `1e-05`; `B`, `draws_ge_observed_*` (the exact count), `seed`, `generator`. The
  `per_task` and `by_format` entries of `both_orders` hold `both` / `one` / `neither` counts with
  `expected_both_at_chance`, `null_mean_both`, `null_95_both`, `p_both` and their `_paired` twins.
- **`questions`** (per order, keyed by question id): `task`, `fmt` (`ABCD`, `AB` or `YN`), `gold`,
  `preds` (the four parsed answers), `k` (passes right), `n` (usable passes), `label`, `kind`,
  `chance` (1 / number of options), `video_ref_options`, and `stratum_default_without_q`, the modal
  prediction of the question's stratum (task × answer space) computed without it. `kind` is set on
  text-answerable questions only: `stratum-default` when the question's majority answer is that
  prediction, `departs-from-default` when it is not, `n/a` in a stratum of fewer than 5 other
  questions; `text_answerable_split` counts the three.
- **`video_ref_options`** (per order): the 148 questions whose options only name clips (section 3.2,
  sanity check), scored like `overall` but without a null; the per-question flag of the same name
  marks them.
- **`hygiene`**: rows, mean and max output tokens, GPU hours, a count of rows per `batch_n`, and
  unclosed thinking traces. **`reparse`**: rows whose stored answer differs from a parse of the
  reply after the last `</think>`.
- **`sighted` entries**: `accuracy_on_blind_questions`, `correct_on_blind_questions`,
  `by_blind_label` (with-video accuracy per O1 label), `blind_not_but_sighted_ge_0.75` and
  `blind_text_answerable_but_sighted_never` (the two counts after the tables in section 3.3).

**The provenance paths refer to Wavy-Hec/MultiCam's tree.** Every path in `provenance` (and in
`O1.files` / `O2.files`) is relative to a MultiCam checkout, as the analysis ran there:

- `script`, `script_git_blob`: `analysis/cvbench_text_answerability.py` as of `6c377c0`, by sha256
  and by git blob id.
- `subsets`: MultiCam's `analysis/cvbench_full_runnable_subset{,_o2}.json`. Their sha256 match the
  copies in `data/subsets/`.
- `rows`: the raw result shards in MultiCam's `bench/results/` (result rows are not in git). The
  files in `../legs/` are those shards merged, one per leg and order, with the export's `leg` and
  `source_commit` fields added, so their sha256 differ from the ones listed here. `source_commit` is
  the MultiCam commit the export was made from (`f4ba98d`, which registers these legs for the
  export), not the commit that ran the legs (`a477d80`).
- `sighted`, `slices`: the Era-1 with-video rows and the 45-question slice file, local copies in
  MultiCam's working tree that are not in git. Their paths (and the matching `argv` values) begin
  with `scratchpad/` and `docs/`, untracked folders of that working tree: they do not point into
  this repository's own `docs/` or into the public MultiCam repository. The Era-1 rows are not
  public. The slice file is byte-identical to `analysis/subset.json` at MultiCam commit `27c96b1`.
- `argv`: the command as run; its `--json` value is where the file was written in MultiCam's tree.

## `text_answerable_ids.json`

The per-question list of section 3.2, for use as a deletion set. Top-level keys:

| key | holds |
|---|---|
| `note` | what the file holds, with its caveats; its "handoff" is the write-up, which keeps the same section numbers |
| `source` | the two analysis JSONs behind its k values and flags, named by their path in MultiCam's working tree (an output directory, not in git); this folder holds the same files |
| `counts` | for `all` (1000 questions), `lettered` (864) and `yes/no` (136): `thinking`, `no_thinking`, `both_modes`, `either_mode` and `questions` |
| `o2_position_dependent` | the ids of the 48 questions flagged below |
| `questions` | all 1000 questions, keyed by id (`cvb-0` to `cvb-999`) |

Every entry of `questions` has these fields:

| field | meaning |
|---|---|
| `task` | the CVBench task type (15 types) |
| `format` | `lettered` (options A–D, or A–B) or `yes/no` |
| `thinking`, `no_thinking` | one object per leg: `O1_k` and `O2_k`, the seeds right (0 to 4 of 4) in each option order, and `both_orders`, true when both are 3 or more (text-answerable in both orders) |
| `both_modes` | `both_orders` is true in both legs (235 questions) |
| `either_mode` | `both_orders` is true in at least one leg (424 questions): the conservative deletion set for video claims |
| `o2_position_dependent` | true for the 48 questions whose O2 order changes an option's meaning (section 7): 47 lettered questions whose "All of the above", "None of the above" or "None of above" option moves from D in O1 to A in O2, and `cvb-89`, whose gold "Both A and C." lands at A |

Filter on `format` to keep the lettered questions (338 in `either_mode`): on a Yes/No question the
answer is a word, so reversing the options cannot catch a Yes/No habit, and the Yes/No passes are
not above the model's own habit (section 3.2). The no-thinking leg's option orders are coupled
(section 5, Seeds): its per-order `O1_k` and `O2_k` are valid per question, but its `both_orders`
flags are not those of independent orders.

## `fig_cvbench_data.json`

Every number the figure script plotted, including figure 11's bin counts; `bm` is the thinking leg
and `nt` the no-thinking control. Accuracies are stored unrounded next to their counts; the
baselines taken from the analysis JSONs (`chance`, `key_insample`, `key_loo`, `longest`) keep that
file's rounding. Top-level keys, by the figures that use them:

| key | holds | figures |
|---|---|---|
| `bm`, `nt` | per leg: `acc` (the mean of the two orders), `acc_O1_O2`, `n_correct`, `rows`, `bias_null_acc`, `bias_null_correct`, `both` (the both-orders split) | 1, 9, 10 |
| `chance`, `choices` | `chance`, the mean of 1 / number of options over all 1000 questions, and `choices`, the option counts behind it (`all`) and behind `longest_chance` (`longest_option`) | 1, 9, 10 |
| `key_insample`, `key_loo` | the answer-key guesser (O1), in-sample and leave-one-out; only the in-sample value is plotted | 1, 9 |
| `longest`, `longest_n`, `longest_chance` | the longest-option rule, the 863 lettered questions it is scored on, and their chance | 1 |
| `sighted` | the eight Era-1 InternVL3-8B with-video arms on the full 1000: `acc`, `std`, `rows`, `n_correct` | 1, 9, 10 |
| `stitch_input_tokens` | each stitch arm's lowest and highest input tokens per row | 10 |
| `tasks` | per task: `n`, both legs' accuracy, the thinking-minus-no-thinking `delta` with `delta_se`, `delta_exact` (an exact fraction) and `delta_counts` (each leg's correct answers and rows), chance, the in-sample key guesser, with-video accuracy, the both-orders split (`both_split`: both, one, neither, n) and share (`both_share`, a fraction), the number of questions expected to pass both orders under pure guessing (`chance_both`, a count; figure 4 divides it by `n`), and the independent and paired nulls with their p values | 2, 3, 4, 7 |
| `both_overall`, `both_null_paired` | the thinking leg's both-orders split over all 1000 (both, one, neither, total) and its paired null (`mean`, `lo`, `hi`, `p`) | 3 |
| `delta_all`, `delta_all_se`, `delta_all_exact`, `delta_all_counts` | thinking minus no thinking overall, in points, with its SE, the exact fraction, and each leg's correct answers out of 8,000 | 7 |
| `hist` | questions with 0 to 4 of 4 seeds right, O1: `bm`, `nt` and the with-video best arm (`sighted`) | 5 |
| `sighted_by_label` | with-video accuracy split by the thinking leg's both-orders label | 6 |
| `letters`, `yesno` | answer shares per option order: letters on the 864 lettered questions, "Yes" on the 136 Yes/No questions, for the key (`gold`) and both legs | 8 |
| `tokens`, `tokens_hist` | the thinking leg's output length per answer (reasoning plus reply): count, mean, median and max; 100-token bins up to 2,500 and the count above | 11 |
| `order_matrix` | the thinking leg's questions by seeds right in O1 (rows, 0 to 4) and in O2 (columns) | 12 |
