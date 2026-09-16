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
- `rows/` - the eight result files, one JSON object per answered question, their
  eight summary files, and the five-question subset that was run.
- `scripts/` - the three scripts that produced the assets in this folder.

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
