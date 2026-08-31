# Ported from Wavy-Hec/CVBench bench/methods/blind.py @ f16922294861940df6a648f83edc3acd798158e0
"""BLIND baseline: the exact production prompt scaffold with ZERO visual input
(``build_messages(no_video=True)``). Whatever it scores is the text-prior floor
every sighted arm must clear; the per-question delta vs ``cvbench_native`` is
the "do the pixels help" measurement.
"""
from inprocess.harnesses.base import Method, Result, result_fields
from inprocess.dataloaders.qa_json import build_messages, letters_of
from inprocess.evaluation.scoring import extract_think, gt_choice, parse_choice


class BlindMethod(Method):
    name = "blind"

    def answer(self, rec, video_root, seed=None) -> Result:
        messages, yn = build_messages(rec, video_root, self.nframes, no_video=True, reasoning=self.reasoning)
        f = result_fields(rec)
        letters = letters_of(rec)
        gold = gt_choice(rec["answer"], yn, letters=letters)
        try:
            g = self.backend.generate(messages, max_new_tokens=self.max_new_tokens,
                                      seed=seed, temperature=self.temperature)
            pred = parse_choice(g.text, yn, letters=letters, options=rec.get('options'))
            return Result(
                **f, method=self.name, backend=self.backend.name,
                prediction=pred, gold=gold,
                correct=(pred.strip().upper() == gold.strip().upper()),
                abstained=(pred == ""),
                seed=seed, temperature=self.temperature,
                latency_s=g.latency_s,
                input_tokens=g.input_tokens, video_tokens=g.video_tokens,
                output_tokens=g.output_tokens, num_model_calls=1,
                response_text=g.text, think=extract_think(g.text),
            )
        except Exception as e:
            return Result(**f, method=self.name, backend=self.backend.name,
                          prediction="", gold=gold, correct=False, abstained=True,
                          seed=seed, temperature=self.temperature,
                          num_model_calls=1, error=f"{type(e).__name__}: {e}")
