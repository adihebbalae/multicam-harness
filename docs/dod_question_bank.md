<!-- Rendered from data/subsets/dod_question_bank.json (scripts/data/make_question_bank.py,
     ported from Wavy-Hec/CVBench analysis/make_question_bank.py @ bb372e450a25fc1b0e5aca6df8234768dc8e7abb).
     Every question below is verbatim from the MEVA release annotations. Written 2026-08-31. -->
# DOD-relevant question bank — MEVA

A curated list of example questions for fixed-camera surveillance scenarios,
each assigned to one of the five evidence-locality classes of
`docs/evidence_taxonomy.md` (where the answer's evidence lives: one timestamp or
a window; one camera, several, or all) and to an operational theme. Every entry
is a real question from the MEVA release, chosen deterministically (the first
items of each source after a stable sort — never sampled), so the bank can be
rebuilt and audited. Nothing here is invented: where a class has no release
question yet, the bank lists the template that would fill it and the metadata
slots that instantiate it.

## 1. What the footage supports

MEVA is the DOD-relevant core of the CrossView release: IARPA surveillance
footage from a fixed camera network at four sites (school, hospital, bus depot,
admin building; 906 / 505 / 463 / 231 release questions), 1–16 cameras per
question (median 7), with a closed vocabulary of **34 machine-readable activity
labels** on every grounded event. Every release question carries a
`verification` block — activity, camera ID, start/end seconds, actor IDs,
clothing-based descriptions — which is what makes mechanical class assignment
and template instantiation possible without new human annotation.

| Theme | Activity labels in the release |
|---|---|
| perimeter / entry-exit monitoring | person_enters_scene_through_structure, person_exits_scene_through_structure, person_opens_facility_door, person_closes_facility_door |
| vehicle activity | vehicle_starts, vehicle_stops, vehicle_turns_left, vehicle_turns_right, vehicle_reverses, vehicle_makes_u_turn, vehicle_drops_off_person, vehicle_picks_up_person, person_enters_vehicle, person_exits_vehicle, person_opens_vehicle_door, person_closes_vehicle_door, person_opens_trunk, person_closes_trunk |
| object transfer / carry | person_picks_up_object, person_puts_down_object, person_carries_heavy_object, person_transfers_object, person_loads_vehicle, person_unloads_vehicle |
| interaction / communications posture | person_talks_to_person, person_talks_on_phone, person_texts_on_phone, person_embraces_person, hand_interacts_with_person, person_reads_document, person_purchases |
| posture / movement | person_stands_up, person_sits_down, person_rides_bicycle |

Also supported: activity counting (289 release questions) and cross-camera
tracking / timeline reconstruction (the temporal and event-ordering questions
link the same actor across cameras; the release marks the link's validation
source per item).

**Not supported — do not promise these:** weapons / threat / contraband; abandoned objects; loitering / dwell time; crowd density; identity (faces, plates, names); camera-to-camera geometry (no calibration in release); intent / anomaly detection.
No faces, plates or names exist in the annotations — subjects are described by
clothing and hair only — and the release ships no camera calibration.

## 2. Templates by class

Thirteen question templates in operational language. "Exists" counts release
questions that already instantiate the template; "new" templates are
instantiable from the verification metadata (camera, time, activity, actor
description) but have no release question yet.

| Class | Template | Status |
|---|---|---|
| C1 | attribute snapshot at a named sensor + time | new — slots exist in every verification block |
| C1 | presence check for a described subject on one sensor | new — slots exist |
| C2 | single-sensor activity recognition over a window | new (431 spatial questions are the C2 material that runs today) |
| C2 | vehicle maneuver sequence on one sensor | new |
| C2 | single-camera activity count | 52 single-camera counting questions exist (open-numeric; MCQ conversion needed) |
| C3 | first-detection sensor for a described subject | 495 first-entrance questions exist, pending the C1/C5 ruling and the camera-ID relabel fix |
| C3 | concurrent coverage: which sensors have eyes on the subject at t | new — 177 near-simultaneous temporal pairs are the seeds |
| C3 | simultaneity check across the network | new — same seeds |
| C4 | cross-sensor precedence (which event happened first) | 305 temporal questions run today |
| C4 | incident timeline reconstruction across sensors | 297 event-ordering questions run today |
| C4 | re-ID handoff: where does the subject reappear, doing what | new — the temporal pairs' `same_person` / actor ids instantiate it |
| C5 | network-wide activity census | 289 counting questions exist (open-numeric; bin the count into A–D) |
| C5 | scene characterization across all feeds | 288 summarization questions exist (free text; MCQ via the scene_type / top_activity fields) |
| C5 | network-wide negative assertion (did ANY sensor record X) | new — instantiable from the per-slot activity inventory |

Status vocabulary used on the entries below: **runnable** (in the harness pool
today), **needs_conversion** (exists in the release but open-numeric or free
text), **pending_ruling** (the first-entrance questions), **seed** (C3-adjacent
temporal pairs with ~1 s gaps).

## 3. The curated entries (20 questions)


### C2 — n frames, 1 camera

**spatial#0** · runnable · theme: single-sensor observation  
slot `2018-03-05.13-10.school` · evidence cameras G424 · harness id 297

> In the footage, how close does the person in black pants with indigo shoes get to the person wearing a dark indigo top with dark gray pants and dark gray shoes?
> A. They approach and stay near each other (within a few meters) / B. They stay at a moderate distance (5-15 meters apart) / C. They remain far apart (more than 15 meters) / D. They walk past each other, swapping positions (cross paths)
> **Answer: A**

**spatial#1** · runnable · theme: single-sensor observation  
slot `2018-03-05.13-15.bus` · evidence cameras G340 · harness id 298

> In the footage, how close do the black-haired person in a navy top and black pants (wearing a hat and carrying a bag) and the black-haired person dressed all in black (also wearing a hat and carrying a bag) get to each other?
> A. They approach and stay near each other (within a few meters) / B. They stay at a moderate distance (5-15 meters apart) / C. They remain far apart (more than 15 meters) / D. They walk past each other, swapping positions (cross paths)
> **Answer: A**

**spatial#2** · runnable · theme: single-sensor observation  
slot `2018-03-05.13-15.bus` · evidence cameras G340 · harness id 299

> In the footage, how close do the black-haired person dressed all in black with a hat and a bag get to the black-haired person in a blue top, black pants, and black shoes who is also carrying a bag?
> A. They approach and stay near each other (within a few meters) / B. They stay at a moderate distance (5-15 meters apart) / C. They remain far apart (more than 15 meters) / D. They walk past each other, swapping positions (cross paths)
> **Answer: A**

**counting#12** · needs_conversion · theme: interaction / communications posture  
slot `2018-03-05.14-10.bus` · evidence cameras G506 · activities: person_talks_to_person

> Across all camera views in this time window, how many separate instances are there of someone talking to another person?
> **Answer: 2**
>
> _single-camera counting under the positive-evidence reading; open-numeric today_


### C3 seeds — near-simultaneous cross-camera pairs

**temporal#21** · seed · theme: concurrent cross-sensor coverage  
slot `2018-03-07.11-00.school` · evidence cameras G421, G424 · activities: person_purchases, person_enters_vehicle · harness id 749

> Between these two moments, which happened first: the person with charcoal hair in a charcoal top and navy pants, wearing dark gray shoes and a hat, making a purchase, or the person with blue hair in a blue top and navy pants, wearing black shoes and a scarf, getting into a vehicle?
> A. The person with charcoal hair, wearing a charcoal top and navy pants, dark gray shoes, with a hat, purchasing occurred first / B. The person with blue hair, wearing a blue top and navy pants, black shoes, with a scarf, entering a vehicle occurred first / C. They occurred simultaneously / D. Cannot be determined
> **Answer: A**
>
> _events 1.0s apart with a 'simultaneously' option — the nearest thing to a same-moment cross-camera question in the release_

**temporal#85** · seed · theme: concurrent cross-sensor coverage  
slot `2018-03-09.10-10.school` · evidence cameras G339, G639 · activities: person_talks_to_person, vehicle_stops · harness id 813

> Between these two moments, which happened first: a person with teal hair in a navy top and dark teal pants talking to someone, or a vehicle coming to a stop?
> A. The person with teal hair, wearing a navy top and dark teal pants, talking to a person occurred first / B. The vehicle stopping occurred first / C. They occurred simultaneously / D. Cannot be determined
> **Answer: A**
>
> _events 1.0s apart with a 'simultaneously' option — the nearest thing to a same-moment cross-camera question in the release_

**temporal#103** · seed · theme: concurrent cross-sensor coverage  
slot `2018-03-11.11-15.school` · evidence cameras G328, G424 · activities: person_closes_vehicle_door, person_enters_vehicle · harness id 831

> Between these two moments, which happened first: a person in a navy top and black pants with black shoes, carrying a bag, closing a vehicle door, or a person with indigo hair in an indigo top and blue pants with navy shoes and a scarf getting into a vehicle?
> A. The person, wearing a navy top and black pants, black shoes, carrying a bag, closing a vehicle door occurred first / B. The person with indigo hair, wearing an indigo top and blue pants, navy shoes, with a scarf, entering a vehicle occurred first / C. They occurred simultaneously / D. Cannot be determined
> **Answer: A**
>
> _events 1.0s apart with a 'simultaneously' option — the nearest thing to a same-moment cross-camera question in the release_


### C4 — n frames, m cameras

**temporal#0** · runnable · theme: perimeter / entry-exit monitoring  
slot `2018-03-05.13-10.school` · evidence cameras G300, G419 · activities: person_exits_scene_through_structure, person_talks_on_phone · harness id 728

> Which happened first: the person with indigo hair wearing an olive top, dark indigo pants, indigo shoes, and a hat leaving through a structure, or the black-haired person in a patterned green top, navy pants, indigo shoes, and a hat talking on the phone?
> A. The person with indigo hair, wearing an olive top and dark indigo pants, indigo shoes, with a hat, exiting a scene through a structure occurred first / B. The person with black hair, wearing a patterned green top and navy pants, indigo shoes, with a hat, talking on a phone occurred first / C. They occurred simultaneously / D. Cannot be determined
> **Answer: A**

**temporal#1** · runnable · theme: perimeter / entry-exit monitoring  
slot `2018-03-05.13-10.school` · evidence cameras G300, G420 · activities: person_opens_facility_door, person_enters_scene_through_structure · harness id 729

> Which happened first: the black-haired person in a dark green top with indigo pants and indigo shoes, wearing a scarf, walking in through a structure, or someone opening a facility door?
> A. The person with black hair, wearing a dark green top and indigo pants, indigo shoes, with a scarf, entering a scene through a structure occurred first / B. The person opening a facility door occurred first / C. They occurred simultaneously / D. Cannot be determined
> **Answer: B**

**temporal#2** · runnable · theme: vehicle activity  
slot `2018-03-05.13-15.bus` · evidence cameras G331, G506 · activities: person_closes_trunk, person_sits_down · harness id 730

> Between these two moments, which happened first: the person in a blue top and blue pants closing a trunk, or the black-haired person dressed in black with black shoes, wearing a hat and carrying a bag, sitting down?
> A. The person, wearing a blue top and blue pants, closing a trunk occurred first / B. The person with black hair, wearing a black top and black pants, black shoes, with a hat, carrying a bag, sitting down occurred first / C. They occurred simultaneously / D. Cannot be determined
> **Answer: A**

**event_ordering#0** · runnable · theme: timeline reconstruction  
slot `2018-03-05.13-10.bus` · evidence cameras G331, G340 · activities: person_picks_up_object, person_enters_scene_through_structure, vehicle_turns_right, person_puts_down_object · harness id 0

> Put the following moments in the order they happened: I. A vehicle makes a right turn II. A black-haired person in a black top, navy pants, and navy shoes—wearing a hat and carrying a bag—walks into view through a doorway III. A navy-haired person in a navy top and black pants—wearing a hat and scarf and carrying a bag—sets an object down IV. A blue-haired person in a navy top and black pants with black shoes—wearing a scarf and carrying a bag—picks an object up Which sequence is correct?
> A. III -> I -> II -> IV / B. IV -> I -> II -> III / C. IV -> II -> I -> III / D. II -> IV -> I -> III
> **Answer: C**

**event_ordering#1** · runnable · theme: timeline reconstruction  
slot `2018-03-05.13-10.school` · evidence cameras G300, G328, G420, G424 · activities: person_texts_on_phone, person_enters_scene_through_structure, vehicle_reverses, vehicle_starts · harness id 1

> Put these four moments in the order they happened: I. A black-haired person in a dark green top, indigo pants, and indigo shoes, wearing a scarf, walks into view through a doorway II. A vehicle begins moving III. A black-haired person in an olive top, black pants, and navy shoes, wearing a hat, is texting on a phone IV. A vehicle reverses Which sequence is correct?
> A. II -> IV -> I -> III / B. III -> IV -> I -> II / C. III -> I -> IV -> II / D. I -> III -> IV -> II
> **Answer: C**

**event_ordering#2** · runnable · theme: timeline reconstruction  
slot `2018-03-05.13-15.bus` · evidence cameras G331, G340, G506 · activities: person_talks_on_phone, person_transfers_object, person_picks_up_object, person_transfers_object · harness id 2

> Put the following observed moments in the order they happened: I. A person in a navy top and navy pants hands an object to someone II. A black-haired person in a navy top, black pants, and black shoes, carrying a bag, hands an object to someone III. A black-haired person wearing a hat, dressed in a black top and black pants with navy shoes, carrying a bag, talks on the phone IV. A vehicle picks up an object Which sequence is correct?
> A. II -> IV -> I -> III / B. III -> IV -> I -> II / C. III -> I -> IV -> II / D. I -> III -> IV -> II
> **Answer: C**


### C1 or C5 (ruling pending) — first-entrance camera questions

**camera#0** · pending_ruling · theme: perimeter / entry-exit monitoring  
slot `2018-03-05.13-10.school` · evidence cameras G336, G300, G328, G339, G419, G420, G421, G423, G424 · activities: person_enters_scene_through_structure

> Which camera shows the first appearance of a black-haired person wearing a dark plum top, a dark crimson skirt, and pink shoes as they enter the scene?
> A. Camera G424 / B. Camera G419 / C. Camera G423 / D. Camera G336
> **Answer: D**
>
> _positive evidence = one entrance moment in one camera (C1); verification scans all feeds (C5); options name physical camera IDs — the harness relabel fix is prerequisite_

**camera#1** · pending_ruling · theme: perimeter / entry-exit monitoring  
slot `2018-03-05.13-10.school` · evidence cameras G300, G328, G336, G339, G419, G420, G421, G423, G424 · activities: person_enters_scene_through_structure

> Which camera shows the first appearance of a black-haired person wearing a dark green top, indigo pants, indigo shoes, and a scarf as they enter the scene?
> A. Camera G300 / B. Camera G423 / C. Camera G421 / D. Camera G419
> **Answer: A**
>
> _positive evidence = one entrance moment in one camera (C1); verification scans all feeds (C5); options name physical camera IDs — the harness relabel fix is prerequisite_

**camera#2** · pending_ruling · theme: perimeter / entry-exit monitoring  
slot `2018-03-05.13-10.school` · evidence cameras G336, G300, G328, G339, G419, G420, G421, G423, G424 · activities: person_enters_scene_through_structure

> Which camera shows the first appearance of a person wearing a pink top and a dark plum dress, black shoes, and a scarf, carrying a bag as they enter the scene?
> A. Camera G421 / B. Camera G419 / C. Camera G336 / D. Camera G328
> **Answer: C**
>
> _positive evidence = one entrance moment in one camera (C1); verification scans all feeds (C5); options name physical camera IDs — the harness relabel fix is prerequisite_


### C5 — n frames, all cameras

**counting#0** · needs_conversion · theme: network-wide activity census  
slot `2018-03-05.13-10.bus` · evidence cameras G331, G340 · activities: person_talks_to_person

> Across all camera views in this time window, how many separate instances are there of someone talking to another person?
> **Answer: 3**

**counting#1** · needs_conversion · theme: network-wide activity census  
slot `2018-03-05.13-10.hospital` · evidence cameras G341, G436 · activities: person_exits_scene_through_structure

> Across all camera feeds in this time window, how many times does a person exit the camera’s view through a doorway?
> **Answer: 2**

**counting#2** · needs_conversion · theme: network-wide activity census  
slot `2018-03-05.13-10.school` · evidence cameras G328, G336, G339, G421, G424 · activities: vehicle_stops

> Across all camera views in this time window, how many times do you see a person come to a stop?
> **Answer: 3**

**summarization#0** · needs_conversion · theme: scene characterization / pattern of life  
slot `2018-03-05.13-10.bus` · evidence cameras G331, G340

> Across both of the 2 camera feeds in this time slot, which option most accurately summarizes the overall scene?
> **Answer: A pedestrian-dominant scene across 2 cameras, primarily featuring sitting down**
>
> _free-text today; MCQ via scene_type/top_activity distractor swap_


## 4. Rebuilding and extending

```bash
python3 scripts/data/label_evidence_class.py     # labels the pool from the release annotations
python3 scripts/data/make_question_bank.py       # -> data/subsets/dod_question_bank.json
```

The class rules live in the labeler only, so a bank rebuilt after a rule change
re-labels itself. Extending the bank, in order of cost:

1. **C5 by conversion** — bin the 289 counting answers (range 2–10) into four
   options and swap the summarization `scene_type` / `top_activity` fields for
   distractors; both are mechanical and give the harness 577 C5 questions.
2. **C3 / C1 by ruling** — the 495 first-entrance questions become C1 or C5
   material once the reading is fixed and the camera-ID relabel is in place.
3. **New templates** — instantiate from the verification metadata, then inspect
   the clips by hand before any question enters a scored pool: a template
   instance is unverified until someone has watched it.
