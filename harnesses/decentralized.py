# Ported from Wavy-Hec/CVBench bench/methods/per_stream.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""A1 - Distributed "1-VLM-per-stream": each clip gets an independent
perception pass; an aggregator pass then reasons over the K text descriptions.

Latency is reported two ways: ``perception_latency_serial_s`` (sum of the
per-stream calls, as actually run here) and ``perception_latency_par_s``
(``max`` over them, the estimate under a truly parallel distributed system).
"""
from harnesses.base import Method, Result, result_fields
from dataloaders.qa_json import build_messages, video_paths
from evaluation.scoring import parse_choice, gt_choice

PERCEPTION_PROMPT = (
    "You are looking at ONE {unit} only. Describe what is visible that is relevant to the "
    "question below: any people and their appearance, their movements/trajectory, and the "
    "timestamps at which events occur. If nothing relevant is visible, say so plainly. "
    "Do NOT try to answer the question yet.\n\nQuestion: {q}"
)
AGGREGATE_PREFIX = ("You are given independent descriptions of each {unit}. Reason over ALL "
                    "of them together to answer.\n\n")
# 'camera' keeps the original MEVA phrasing byte-identical; 'video' mirrors
# centralized's montage_kind fix for CVBench, where questions say "Video k"
# and calling independent clips "camera views" is the known ~5-pt labeling
# artifact (see analysis/ cvbench flip result, job 58000).
STREAM_KINDS = {"camera": ("Camera", "camera view"), "video": ("Video", "video clip")}


class PerStreamMethod(Method):
    name = "per_stream"

    def __init__(self, backend, nframes=8, max_new_tokens=8192, temperature=0.0,
                 perception_max_new_tokens=1024, stream_kind="camera"):
        super().__init__(backend, nframes=nframes, max_new_tokens=max_new_tokens,
                         temperature=temperature)
        self.perception_max_new_tokens = perception_max_new_tokens
        self._label, self._unit = STREAM_KINDS[stream_kind]
        self.stream_kind = stream_kind

    def answer(self, rec, video_root, seed=None) -> Result:
        f = result_fields(rec)
        paths = video_paths(rec, video_root)
        # reuse build_messages to get the exact text-only answer scaffold + yes/no flag
        base_msgs, yn = build_messages(rec, video_root, self.nframes, no_video=True)
        base_text = base_msgs[0]["content"][0]["text"]
        gold = gt_choice(rec["answer"], yn)

        descs, lat_per, in_tok, vid_tok, out_tok, calls = [], [], 0, 0, 0, 0
        try:
            # --- per-stream perception ---
            for k, vp in enumerate(paths, 1):
                msg = [{"role": "user", "content": [
                    {"type": "text", "text": f"{self._label} {k}:"},
                    {"type": "video", "video": vp, "nframes": self.nframes},
                    {"type": "text", "text": PERCEPTION_PROMPT.format(unit=self._unit,
                                                                      q=rec["question"])},
                ]}]
                g = self.backend.generate(msg, max_new_tokens=self.perception_max_new_tokens,
                                          seed=seed, temperature=self.temperature)
                descs.append(f"{self._label} {k}:\n{g.text.strip()}")
                lat_per.append(g.latency_s)
                in_tok += g.input_tokens; vid_tok += g.video_tokens; out_tok += g.output_tokens
                calls += 1

            # --- aggregator (text-only reasoning over the descriptions) ---
            agg_text = (AGGREGATE_PREFIX.format(unit=self._unit)
                        + "\n\n".join(descs) + "\n\n" + base_text)
            agg_msg = [{"role": "user", "content": [{"type": "text", "text": agg_text}]}]
            g = self.backend.generate(agg_msg, max_new_tokens=self.max_new_tokens,
                                      seed=seed, temperature=self.temperature)
            calls += 1
            in_tok += g.input_tokens; out_tok += g.output_tokens
            pred = parse_choice(g.text, yn)

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
            )
        except Exception as e:
            return Result(**f, method=self.name, backend=self.backend.name,
                          prediction="", gold=gold, correct=False, abstained=True,
                          seed=seed, temperature=self.temperature,
                          num_model_calls=calls, error=f"{type(e).__name__}: {e}")
