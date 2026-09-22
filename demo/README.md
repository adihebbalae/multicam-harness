# Four-camera demo showcase

## What this is

A replay of one recorded run, packaged so anyone can open it in a browser.

Five questions from the CrossView MEVA set, each with four camera views of the same
scene. For every question a segment selector scores short windows of each camera and
keeps the best ones, then InternVL3-8B is asked the question with only the kept
frames. Two selectors are included: SigLIP so400m, which scores segments against the
question text, and ViCLIP-opt, which scores them against the four answer options. The answers are single pass, direct answers, temperature 0.1. The run
was recorded on 2026-09-16.

Nothing in this folder computes anything. The page reads the recorded result rows,
the figures and the video clips from disk and plays them back.

## How to run

```bash
cd demo
python3 -m http.server 8765
```

Then open http://localhost:8765/demo_ui.html.

Pick a question on the left, or use the left and right arrow keys. Then use the five
buttons, or the number keys 1 to 5:

1. Play streams. The 2x2 camera wall for that question, with the scored segment
   strips and the annotated event windows.
2. Kept segments. The still figure of what the selector kept and which frames went
   to the model.
3. Answer. The answer card: the letter each arm gave, the gold letter, and two
   timing columns (Answer call and Selection).
4. Reasoning. A reasoning trace for that question, read from a result row.
5. Reset.

The SigLIP / ViCLIP switch at the top swaps every asset: figures, wall clips, answer
card and traces. The caption strip above the stage always names the file on screen.

### The case study

The first entry in the rail, **Case**, is not one of the five demo questions. It is
question 954 of the full 1,033-question MEVA pool, the two-slide example from the
motivation deck: every camera on its own gets it wrong, all four together get it
right. Selecting it switches the buttons to three:

1. Each camera alone. The 2x2 wall with each camera's four answers stamped on its
   tile, and under it one row per camera: what its annotated event was, the four
   letters it gave when it was the only input (8 evenly spaced frames), and the
   count. Press 1 again for the deck's figure of the exact 8 frames each camera
   received.
2. All four together. The same wall with the multi-camera answer in a banner, and
   under it one row per recorded run on this question: the winning run first
   (SigLIP selection, 96 frames), then every other run in file order, including
   the two that had frames inside both event windows and were still wrong. Press
   2 again for the figure of the 96 frames selection sent and where they fall.
3. Reset.

Above the case sits **How it works**: one click puts the pipeline figure from the
deck on the stage, and the rail lists the six stages with the segment count and
frame budget read from the manifest. The wall clips play about 42 s of footage at
2x as a 21 s loop; the clip is the raw footage over the events with the recorded
answers stamped on, not what the model saw. The second press of each step shows
that.

Every letter, count and caption in case mode is read from `figs_case/case_q954.json`;
nothing is typed into the page. The record was written by
`scripts/render_case_contrast.py` from the recorded rows in `multicam_results/legs`
and the release annotations, with four passes at temperature 0.1, which the banner
says while the case is selected. Two lines travel with the case wherever it is
shown: the disclosure (27 of the 602 multi-camera questions behave like this in at
least one of five runs; a right single-camera answer turning wrong is more common)
and the counter-evidence (two runs had frames inside both event windows and were
still wrong). The case does not show that the model was right because it saw both
events; it shows the question type. See `docs/figs/demo_2026-09-22/case_vetting.md`
for how 954 was picked and what the rows can and cannot support.

## What each folder holds

- `demo_ui.html` - the replay page. One file, no dependencies, no build step.
- `figs_demo/` - the SigLIP set. Per question a selection figure and an answer
  figure, plus `summary.png` and `manifest.json`, which holds every number the page
  and the figures draw.
- `figs_demo/trace_q1/`, `figs_demo/trace_q860/` - two questions re-rendered with a
  reasoning leg included, so the answer figure there carries a trace excerpt.
- `figs_demo/walls/` - the camera-wall clips: one MP4 per question, a snapshot PNG
  per question, `walls_all.mp4` with all five joined, and `walls_manifest.json`.
- `figs_demo_viclip/` and `figs_demo_viclip/walls/` - the same two sets rendered from
  the ViCLIP run.
- `figs_case/` - the case study: `case_q954.json` (every value the case mode shows),
  the two deck figures `case1_q954.png` and `case2_q954.png`, the two wall clips
  `wall_case1_q954.mp4` and `wall_case2_q954.mp4` with their `_mid.png` stills, and
  `how_it_works.png`, the pipeline figure from the deck. The same record and case
  figures live in `docs/figs/demo_2026-09-22/` for the deck.
- `rows/` - the eight result files, one JSON object per answered question, their
  eight summary files, and the five-question subset that was run.
- `scripts/` - the five scripts that produced the assets in this folder.

## Results on these five questions

Single pass, direct answers, InternVL3-8B. Gold letters are in the subset file.
Question ids are the columns.

| Arm | 860 | 749 | 251 | 1 | 130 | n |
|---|---|---|---|---|---|---|
| Blind, no video | A ok | A ok | C ok | D wrong | A wrong | 3/5 |
| All four cameras, 96 frames | A ok | A ok | B wrong | C ok | D wrong | 3/5 |
| SigLIP segments per camera, 96 | A ok | A ok | C ok | C ok | A wrong | 4/5 |
| SigLIP segments global, 96 | A ok | A ok | C ok | C ok | A wrong | 4/5 |
| ViCLIP segments per camera, 96 | A ok | A ok | C ok | C ok | A wrong | 4/5 |
| ViCLIP segments global, 96 | A ok | A ok | C ok | C ok | A wrong | 4/5 |

The two selectors give the same five letters. They do not keep the same frames.

Cost, per question, measured in the same rows: about 3.3 s for the answer call plus
about 50 s of segment scoring with SigLIP, about 42 s with ViCLIP. Most of the
selection time is frame decoding, not the scorer. The first question of the ViCLIP
run took 73 s to score. The all-cameras arm does its own frame decode and that decode
is not timed, so its 4.8 s answer call is not a like-for-like speed number. Selection
buys accuracy here, not speed.

## Benchmark context

The five questions above are a demo, not a measurement. The numbers below are
accuracy on the full 1,033-question MEVA pool, same model:

| Condition | Accuracy |
|---|---|
| Chance, four options | 25.00 |
| Model, no video | 34.75 |
| All cameras, 96 frames | 43.85 |
| ViCLIP segments, 96 frames | 48.31 |
| SigLIP segments, 96 frames | 49.10 |
| Text-only letter floor | 54.70 |

The letter floor is what a guesser scores that sees only the question type, never a
frame and never the question text, and answers that type's most common letter. It is
measured leave one out over the same pool. Nothing sighted clears it. The skew it
exploits is a known artifact of the question release, and it is why these accuracies
are read as a ladder between the sighted arms rather than as absolute skill.

The five demo questions were chosen. They were picked to show selection helping and
failing visibly, so their 4/5 is not an estimate of anything.

## Two things to be honest about

1. The blind arm is right on three of the five. It sees no video at all. A single
   pass at temperature 0.1 answers the usual gold letter of each question type, and
   on these five that lands correctly three times. The evidence for selection here is
   the all-cameras miss on question 251 and where the kept frames land, not the blind
   score.
2. The reasoning traces behind button 4 come from a separate reasoning run. The
   scored answers in the table are the direct-answer run. The traces are shown as
   illustration. Several of them argue from priors about how events usually order
   rather than from what is in the frames.

## Regenerating

The scripts in `scripts/` read the rows and the manifests in this folder, so the
figures can be re-rendered from what is here. Re-rendering the wall clips or the
frame thumbnails cannot: those decode the MEVA video, which is not in this
repository.

- `render_demo.py` writes the figures and `manifest.json` from the result rows. It
  resolves video paths through the MultiCam checkout's `bench.reuse`, and falls back
  to a byte-identical local replica under `--no-bench-reuse`. MEVA clips are spelled
  `.avi` in the records and an `.avi` must never reach the decoder, so the scripts
  require the remuxed `.mp4` sibling of every clip and fail loudly if one is missing.
- `make_demo_walls.py` renders the 2x2 wall clips. It draws only numbers already in
  `manifest.json` and decodes video for the tiles. It needs ffmpeg.
- `demo_live_tail.py` prints one line per answer from a result file, for a terminal
  shot during a recording.
- `render_case_contrast.py` writes `case_q<ID>.json` and the two case figures into
  `--out`, from the recorded legs and the release annotations; `--vet` also writes
  the candidate table `case_vetting.md` there (the one checked in under
  `docs/figs/demo_2026-09-22/` was written with that directory as `--out`).
  `render_case_wall.py` reads that record and renders the two case wall clips (GIF
  for the deck, MP4 for this page) and their stills. Both are CPU only and take the
  same `--out`; from the repo root, under the cvbench env:

  ```bash
  ~/anaconda3/envs/cvbench/bin/python demo/scripts/render_case_contrast.py \
      --release-root ~/MultiCam/crossview-release-annotations/crossview-release \
      --qid 954 --out demo/figs_case
  ~/anaconda3/envs/cvbench/bin/python demo/scripts/render_case_wall.py \
      --release-root ~/MultiCam/crossview-release-annotations/crossview-release \
      --qid 954 --out demo/figs_case
  ```

  The GIFs are not kept under `figs_case/` (6 MB each); delete them after the render.
- The event windows drawn on the figures come from the CrossView release annotation
  tree, which the renderer expects beside the checkout.
- The camera `path` fields in the manifests point into that release tree. Those files
  are not in this repository, so the paths are there for provenance and will not
  resolve here. The page never follows them; it plays the clips in `walls/`.
- The rows are the recorded ones. The only change made when copying them here is that
  the compute node field was blanked. No result, latency or token count was touched.

## Licence and attribution

The wall clips, the snapshots and the frame thumbnails in the figures contain frames
from the MEVA dataset, the Multiview Extended Video with Activities dataset, released
under the Creative Commons Attribution 4.0 International licence (CC-BY-4.0). Carry
this attribution line with any reuse of those frames:

> Contains frames from the MEVA dataset (Multiview Extended Video with Activities),
> https://mevadata.org, licensed CC-BY-4.0.

The questions, the options, the gold answers and the event annotations come from the
CrossView release, which builds its multi-camera questions on MEVA. Anyone reusing
these questions should cite CrossView and keep the MEVA attribution above.

The scripts, the page and the figures in this folder are part of this repository and
carry its licence.
