<!-- Labels and counts computed from the MEVA release annotations by
     scripts/data/label_evidence_class.py (ported from Wavy-Hec/CVBench
     analysis/label_evidence_class.py @ 04165739e1049bd05be150b9b250cb7e4a7edbec);
     floors by evaluation/evidence_class_split.py. Written 2026-08-31. -->
# Evidence-locality taxonomy — where the answer lives

Five classes, one axis: how much of the camera network, and how much of the
timeline, the evidence for the correct answer occupies. The class is a property
of the *question* (its verification metadata), not of any model or harness, so
it is assigned once, mechanically, and every result row ever produced on the
pool inherits it for free.

## 1. The five classes and the answer contract

| Class | Definition | An answer is grounded when it can cite | Reachable by |
|---|---|---|---|
| **C1** — 1 frame, 1 camera | One moment in one feed decides it | one `(camera, t)` | any arm that delivers the right frame of the right view |
| **C2** — n frames, 1 camera | One feed, observed over a window | one `(camera, [t0, t1])` | per-view sequential frames; a single-view arm suffices |
| **C3** — 1 timestamp, m cameras | The same moment seen from several feeds | `({cameras}, t)` | time-synchronized packaging (a montage at *t*); sequential per-view sampling can miss the moment on one view |
| **C4** — n frames, m cameras | Extended observation across several feeds | `({cameras}, {[t0, t1]})`, one window per camera | all-view arms; selection arms must keep at least one segment per evidence camera |
| **C5** — n frames, all cameras | The whole network must be cleared (negative evidence) | every camera over the window | all-view arms only; any arm that drops a camera cannot certify the answer |

"1 frame" is read as **one timestamp**, not one decoded frame: event durations in
the harness pool run 0.47–158.90 s (median 3.13 s; 1 of 610 measured events is
under half a second), so a literal single-frame class is empty in this data
while a single-moment class is populated. The exact frame is recoverable per
event from the release's per-camera fps if the strict reading is ever wanted.

**Consistency rules** that make the assignment reproducible:

- The class is the *minimum* evidence that decides the question — read from the
  release's `verification` block (`requires_cameras`, per-event camera and
  `start_sec`/`end_sec`), never from the question text.
- Distractors count. A spatial question whose wrong options quantify over the
  window ("they stay near each other", "they cross paths") cannot be falsified
  from one frame, so it is C2 even when the correct option is visible in one.
- "All cameras" (C5) is reserved for questions whose answer needs *negative*
  evidence — a count "across all camera views", a scene characterization, "did
  any sensor record X". Positive evidence on m cameras is C4, however many m.
- A class split is only readable against its own floor (section 3): the classes
  inherit different task mixes and different answer-letter priors.

## 2. Mechanical assignment (rules v1, 2026-08-31)

`scripts/data/label_evidence_class.py` joins every benchmark record back to its
release item via `(question_type, orig_id-index)`, refuses on any question-text
mismatch, and applies:

| Release `question_type` | Class | Basis |
|---|---|---|
| spatial | C2 | single required camera; the distractors quantify over the window |
| temporal | C4 | two interval events on two distinct cameras (asserted on every item) |
| event_ordering | C4 | 3–4 events over >1 distinct camera (a single-camera item would degrade to C2; none exists) |
| counting | C5 | "across all camera views" phrasing; the positive-evidence reading (C2/C4 by `requires_cameras`) is recorded per item |
| summarization | C5 | scene characterization over every feed |
| camera (first entrance) | **AMBIGUOUS(C1\|C5)** | positive evidence is one entrance moment on one camera (C1); verifying it means scanning every feed for an earlier appearance (C5) |

Two definitional rulings are pending and change nothing above:

1. C1/C3 defined by timestamp (recommended, see section 1) or by literal frame.
2. Which reading the 495 camera/first-entrance questions take — C1 (positive
   evidence) or C5 (verification set). The label is a one-line change either
   way. Independently, those questions' options name physical camera IDs that
   the harness's "Video i" relabeling destroys, so a relabel fix precedes any
   run on them.

## 3. What the pools contain

Harness pool (`data/subsets/crossview_meva1033_subset.json`, 1,033 questions —
temporal, event_ordering, spatial; the three MCQ types the converter reads):

| Class | Questions | Task-conditioned LOO floor |
|---|---:|---:|
| C1 | 0 | — |
| C2 | 431 (all spatial) | 48.03 |
| C3 | 0 | — |
| C4 | 602 (305 temporal + 297 event_ordering) | 59.47 |
| C5 | 0 | — |
| all | 1,033 | 54.70 |

Full release (2,105 MEVA questions): the same 431 C2 and 602 C4, plus 577 C5
(289 counting, open-numeric; 288 summarization, free text — both need an MCQ
conversion before the harness can score them) and 495 camera questions awaiting
ruling 2. C3 is structurally empty: no release question asks about one moment
across feeds. The nearest material is the 177 of 305 temporal pairs whose two
events are within 2 s of each other and carry a "they occurred simultaneously"
distractor — seeds for a C3 template, not C3 questions.

The floor is the leave-one-out task-conditioned modal-letter guesser: for each
question, answer the most common gold letter among the *other* questions of its
task type. Restricted to a class it still learns per task over the whole pool —
it cannot see the class — but is scored only on that class's questions. It is
the text-only bar a sighted arm has to clear; the per-question chance level is
25.00 (every question has four options) and the pool-wide modal letter scores
36.21.

**So the benchmark as run so far lives entirely in C2 and C4.** Three of the
five classes are unmeasured — which is the finding that defines the
question-bank work (`docs/dod_question_bank.md`): C5 by conversion of what the
release already has, C1 and C3 by templates instantiated from the verification
metadata.

## 4. Files and regeneration

```bash
# labels: (question_type#index) -> class, basis, cameras, event spans, harness id
python3 scripts/data/label_evidence_class.py            # -> data/subsets/meva_evidence_labels.json
# the curated bank (reads the labels)
python3 scripts/data/make_question_bank.py              # -> data/subsets/dod_question_bank.json
# any leg's rows, split by class against the per-class floors
python -m evaluation.evidence_class_split results/<leg>_shard*.jsonl
```

Both data scripts need only the release annotations (`qa_*.json` under the
crossview annotations root in `configs/datasets.yaml`) — no videos, no GPU. The
labels file keys on the release index, carries the benchmark id per subset
alias, and records the rules version; a row whose id has no label stops the
split rather than being averaged in.
