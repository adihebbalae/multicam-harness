# Ported from Wavy-Hec/CVBench Video-R1/src/eval_thinking.py @ f65d6e043014b6e9090c32dec4893ebc14fa4320
# Ported from Wavy-Hec/CVBench bench/reuse.py @ f65d6e043014b6e9090c32dec4893ebc14fa4320
"""QA-record loading/prompting helpers shared by every harness.

Record schema contract (one JSON object per question, in a top-level list):
  id                : stable unique question id (int or str, e.g. "cvb-123")
  task_type         : task label used for per-task accuracy breakdowns
  question          : question text (options NOT included)
  options           : list of option strings, e.g. ["A. ...", ...] or ["Yes.", "No."]
  answer            : ground-truth letter (MC) or Yes/No
  source            : dataset source tag (e.g. "meva", "ego-exo4d", "agibot", "cvbench")
  question_type     : source question type (e.g. "temporal", "cvbench_full")
  orig_num_cameras  : number of cameras/videos in the original question (pre-cap)
  cap_answer_safe   : True if the view cap provably kept the answer-bearing views
  orig_id           : id of the question in its source annotation file
  video_1..video_13 : video paths RELATIVE to --video-root (unused slots absent/None)
Optional extras:
  image_1..image_13 : still-image view paths, for multi-view IMAGE records; a
                      record carries video_i OR image_i, never both
  pair_idx          : paired-question twin id (All-Angles-Bench consistency metric)
  dropped_cameras   : number of views dropped by the view cap
  temporal_level    : CVBench temporal-logic level (1 = reference, 2 = complex)

There is deliberately NO default video root constant here: the root lives in
configs/datasets.yaml and is passed in as an ordinary parameter (--video-root).
"""
import os

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', "
    "'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought "
    "expressions. It's encouraged to include self-reflection or verification in the "
    "reasoning process. Provide your detailed reasoning between the <think> and </think> "
    "tags, and then give your final answer between the <answer> and </answer> tags."
)


# Widest supported multiple-choice letter range and slot count. Records carry
# up to MAX_SLOTS video_i (or image_i) fields; option letters are always the
# per-record prefix of LETTERS_ALL — never the blanket range (a bare \b[A-M]\b
# would match prose words like "I"/"A" in the tag-missing fallback).
LETTERS_ALL = "ABCDEFGHIJKLM"
MAX_SLOTS = 13


def letters_for(options):
    """Option letters for one record's options list ("ABCD" for 4 options).
    Malformed single-string option rows fall back to the classic ABCD."""
    n = len(options or [])
    return LETTERS_ALL[:n] if 2 <= n <= len(LETTERS_ALL) else "ABCD"


def num_videos(rec):
    return sum(1 for i in range(1, MAX_SLOTS + 1) if rec.get(f"video_{i}"))


def video_paths(rec, video_root):
    out = []
    for i in range(1, MAX_SLOTS + 1):
        v = rec.get(f"video_{i}")
        if v:
            out.append(os.path.normpath(os.path.join(video_root, v)))
    return out


def num_images(rec):
    return sum(1 for i in range(1, MAX_SLOTS + 1) if rec.get(f"image_{i}"))


def image_paths(rec, image_root):
    """Ordered view-image paths of a still-image record (image_1..image_N).
    Order is semantically load-bearing: question text references "View k"."""
    out = []
    for i in range(1, MAX_SLOTS + 1):
        v = rec.get(f"image_{i}")
        if v:
            out.append(os.path.normpath(os.path.join(image_root, v)))
    return out


def _letters_phrase(letters):
    """Human phrasing of the letter range: "A, B, C, or D" (byte-identical to
    the original 4-option prompt), "A or B", "A, B, or C", and an elided
    "A, B, ..., or J" once enumerating would bloat the prompt."""
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    if len(letters) <= 6:
        return ", ".join(letters[:-1]) + f", or {letters[-1]}"
    return f"{letters[0]}, {letters[1]}, ..., or {letters[-1]}"


def build_messages(rec, video_root, nframes, no_video=False):
    options = rec["options"]
    is_yesno = all(o.strip().strip(".").lower() in ("yes", "no") for o in options)
    # still-image records (image_1..N view images) share the text scaffold but
    # speak of "views" and attach image items instead of video clips
    is_image = num_images(rec) > 0
    unit = "views" if is_image else "videos"
    if is_yesno:
        option_prompt = ("Select the best answer to the following yes-no question based on "
                         f"all the listed {unit}.")
        post = "Provide only the single word (Yes or No) within the <answer> </answer> tags."
    else:
        letters = letters_for(options)
        option_prompt = ("Select the best answer to the following multiple-choice question "
                         f"based on all the listed {unit}.")
        post = (f"Provide only the single option letter ({_letters_phrase(letters)}) "
                "within the <answer> </answer> tags.")

    question = rec["question"] + "\n" + "\n".join(options)
    full_prompt = option_prompt + "\n" + QUESTION_TEMPLATE.format(Question=question) + "\n" + post

    # interleave a text marker before each clip/view; with no_video (blind
    # baseline) keep the prompt text identical but attach zero visual input
    content = []
    if not no_video:
        if is_image:
            for k, ip in enumerate(image_paths(rec, video_root), 1):
                content.append({"type": "text", "text": f"View {k}:"})
                content.append({"type": "image", "image": ip})
        else:
            for k, vp in enumerate(video_paths(rec, video_root), 1):
                content.append({"type": "text", "text": f"Video {k}:"})
                content.append({"type": "video", "video": vp, "nframes": nframes})
    content.append({"type": "text", "text": full_prompt})
    return [{"role": "user", "content": content}], is_yesno


def letters_of(rec):
    """The record's multiple-choice letter set ("ABCD" for classic 4-option),
    for threading into parse_choice/gt_choice."""
    return letters_for(rec.get("options"))


def is_yesno(options):
    """Same predicate build_messages() uses to pick MC vs yes/no parsing."""
    return all(o.strip().strip(".").lower() in ("yes", "no") for o in options)
