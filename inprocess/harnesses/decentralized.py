# Ported from Wavy-Hec/CVBench bench/methods/per_stream.py @ f65d6e043014b6e9090c32dec4893ebc14fa4320
"""A1 - Distributed "1-VLM-per-stream": each clip gets an independent
perception pass; an aggregator pass then reasons over the K text descriptions.

Latency is reported two ways: ``perception_latency_serial_s`` (sum of the
per-stream calls, as actually run here) and ``perception_latency_par_s``
(``max`` over them, the estimate under a truly parallel distributed system).
"""
from inprocess.harnesses.base import Method, Result, result_fields
from inprocess.dataloaders.qa_json import (build_messages, image_paths, letters_of,
                                 num_images, video_paths)
from inprocess.evaluation.scoring import extract_think, gt_choice, parse_choice

PERCEPTION_PROMPT = (
    "You are looking at ONE {unit} only. Describe what is visible that is relevant to the "
    "question below: any people and their appearance, their movements/trajectory, and the "
    "timestamps at which events occur. If nothing relevant is visible, say so plainly. "
    "Do NOT try to answer the question yet.\n\nQuestion: {q}"
)
# Still-image variant: several categories (relative direction, camera pose) are
# inherently joint across views — the per-view pass must describe rather than
# guess, and state plainly what a single view cannot determine.
PERCEPTION_PROMPT_IMAGE = (
    "You are looking at ONE camera view (View {k}) of a scene that is also seen by "
    "other cameras. Describe what is visible that is relevant to the question below: "
    "people and objects, their positions relative to this camera (left/right, "
    "near/far), and the apparent camera viewpoint. If the question cannot be judged "
    "from this view alone, describe what IS visible and say plainly what this view "
    "cannot determine. Do NOT try to answer the question yet.\n\nQuestion: {q}"
)
AGGREGATE_PREFIX = ("You are given independent descriptions of each {unit}. Reason over ALL "
                    "of them together to answer.\n\n")
# 'camera' keeps the original MEVA phrasing byte-identical; 'video' mirrors
# centralized's montage_kind fix for sets whose questions say "Video k" —
# calling independent clips "camera views" there is a documented ~5-pt
# labeling artifact. 'view' is the still-image multi-view labeling (forced for
# image records regardless of --stream-kind, matching their question text).
STREAM_KINDS = {"camera": ("Camera", "camera view"), "video": ("Video", "video clip"),
                "view": ("View", "camera view")}


class PerStreamMethod(Method):
    name = "per_stream"

    def __init__(self, backend, nframes=8, max_new_tokens=8192, temperature=0.0,
                 perception_max_new_tokens=1024, stream_kind="camera",
                 total_frames=0):
        super().__init__(backend, nframes=nframes, max_new_tokens=max_new_tokens,
                         temperature=temperature)
        self.perception_max_new_tokens = perception_max_new_tokens
        self._label, self._unit = STREAM_KINDS[stream_kind]
        self.stream_kind = stream_kind
        # total_frames > 0: one TOTAL frame budget per question split evenly
        # across the per-view perception calls (the mentor's fixed-budget
        # protocol), instead of a flat nframes per view
        self.total_frames = total_frames

    def answer(self, rec, video_root, seed=None) -> Result:
        f = result_fields(rec)
        letters = letters_of(rec)
        is_image = num_images(rec) > 0
        if is_image:
            paths = image_paths(rec, video_root)
            label, unit = STREAM_KINDS["view"]
        else:
            paths = video_paths(rec, video_root)
            label, unit = self._label, self._unit
        # reuse build_messages to get the exact text-only answer scaffold + yes/no flag
        base_msgs, yn = build_messages(rec, video_root, self.nframes, no_video=True)
        base_text = base_msgs[0]["content"][0]["text"]
        gold = gt_choice(rec["answer"], yn, letters=letters)

        per_view = None
        if self.total_frames and not is_image and paths:
            from inprocess.harnesses.uniform import allocate_frames
            per_view = allocate_frames([1] * len(paths), budget=self.total_frames, floor=1)

        descs, lat_per, in_tok, vid_tok, out_tok, calls = [], [], 0, 0, 0, 0
        try:
            # --- per-stream perception ---
            for k, vp in enumerate(paths, 1):
                if is_image:
                    visual = {"type": "image", "image": vp}
                    prompt = PERCEPTION_PROMPT_IMAGE.format(k=k, q=rec["question"])
                else:
                    nf = per_view[k - 1] if per_view else self.nframes
                    visual = {"type": "video", "video": vp, "nframes": nf}
                    prompt = PERCEPTION_PROMPT.format(unit=unit, q=rec["question"])
                msg = [{"role": "user", "content": [
                    {"type": "text", "text": f"{label} {k}:"},
                    visual,
                    {"type": "text", "text": prompt},
                ]}]
                g = self.backend.generate(msg, max_new_tokens=self.perception_max_new_tokens,
                                          seed=seed, temperature=self.temperature)
                descs.append(f"{label} {k}:\n{g.text.strip()}")
                lat_per.append(g.latency_s)
                in_tok += g.input_tokens; vid_tok += g.video_tokens; out_tok += g.output_tokens
                calls += 1

            # --- aggregator (text-only reasoning over the descriptions) ---
            agg_text = (AGGREGATE_PREFIX.format(unit=unit)
                        + "\n\n".join(descs) + "\n\n" + base_text)
            agg_msg = [{"role": "user", "content": [{"type": "text", "text": agg_text}]}]
            g = self.backend.generate(agg_msg, max_new_tokens=self.max_new_tokens,
                                      seed=seed, temperature=self.temperature)
            calls += 1
            in_tok += g.input_tokens; out_tok += g.output_tokens
            pred = parse_choice(g.text, yn, letters=letters)

            serial = sum(lat_per) + g.latency_s
            return Result(
                **f, method=self.name, backend=self.backend.name,
                prediction=pred, gold=gold,
                correct=(pred.strip().upper() == gold.strip().upper()),
                abstained=(pred == ""),
                seed=seed, temperature=self.temperature,
                latency_s=serial,
                perception_latency_serial_s=sum(lat_per),
                perception_latency_par_s=(max(lat_per) if lat_per else None),
                aggregate_latency_s=g.latency_s,
                input_tokens=in_tok, video_tokens=vid_tok, output_tokens=out_tok,
                num_model_calls=calls,
                # aggregator trace + the per-view descriptions it reasoned over,
                # so blind-perception vs bad-aggregation failures are separable
                response_text=g.text, think=extract_think(g.text),
                frame_alloc=({"perception_texts": descs} if per_view is None else
                             {"perception_texts": descs, "total_frames": self.total_frames,
                              "per_view": per_view}),
            )
        except Exception as e:
            return Result(**f, method=self.name, backend=self.backend.name,
                          prediction="", gold=gold, correct=False, abstained=True,
                          seed=seed, temperature=self.temperature,
                          num_model_calls=calls, error=f"{type(e).__name__}: {e}")
