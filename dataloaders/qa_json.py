# Ported from Wavy-Hec/CVBench Video-R1/src/eval_thinking.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
# Ported from Wavy-Hec/CVBench bench/reuse.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
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
  cap_answer_safe   : True if the <=4-video cap provably kept the answer-bearing views
  orig_id           : id of the question in its source annotation file
  video_1..video_4  : video paths RELATIVE to --video-root (unused slots absent/None)
Optional extras:
  dropped_cameras   : number of views dropped by the <=4 cap
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


def num_videos(rec):
    return sum(1 for i in range(1, 5) if rec.get(f"video_{i}"))


def video_paths(rec, video_root):
    out = []
    for i in range(1, 5):
        v = rec.get(f"video_{i}")
        if v:
            out.append(os.path.normpath(os.path.join(video_root, v)))
    return out


def build_messages(rec, video_root, nframes, no_video=False):
    options = rec["options"]
    is_yesno = all(o.strip().strip(".").lower() in ("yes", "no") for o in options)
    if is_yesno:
        option_prompt = ("Select the best answer to the following yes-no question based on "
                         "all the listed videos.")
        post = "Provide only the single word (Yes or No) within the <answer> </answer> tags."
    else:
        option_prompt = ("Select the best answer to the following multiple-choice question "
                         "based on all the listed videos.")
        post = "Provide only the single option letter (A, B, C, or D) within the <answer> </answer> tags."

    question = rec["question"] + "\n" + "\n".join(options)
    full_prompt = option_prompt + "\n" + QUESTION_TEMPLATE.format(Question=question) + "\n" + post

    # interleave a text marker before each video clip; with no_video (blind
    # baseline) keep the prompt text identical but attach zero visual input
    content = []
    if not no_video:
        for k, vp in enumerate(video_paths(rec, video_root), 1):
            content.append({"type": "text", "text": f"Video {k}:"})
            content.append({"type": "video", "video": vp, "nframes": nframes})
    content.append({"type": "text", "text": full_prompt})
    return [{"role": "user", "content": content}], is_yesno


def is_yesno(options):
    """Same predicate build_messages() uses to pick MC vs yes/no parsing."""
    return all(o.strip().strip(".").lower() in ("yes", "no") for o in options)
