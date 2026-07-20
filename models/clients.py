# Ported from Wavy-Hec/CVBench bench/methods/base.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
# Ported from Wavy-Hec/CVBench bench/backends/qwen.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
# Ported from Wavy-Hec/CVBench bench/backends/internvl.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
# Ported from Wavy-Hec/CVBench Video-R1/src/eval_thinking.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
# Ported from Wavy-Hec/CVBench bench/run_bench.py @ 480d6f41cddddc7efea9a09b79134811740ba17a
"""VLM backends for the multi-camera harness.

Merges the fork's backend code into one module: the ``Backend``/``GenOut``
abstraction (``bench/methods/base.py``), the shared HF ``load_model`` loader
(``Video-R1/src/eval_thinking.py``), ``QwenBackend`` (``bench/backends/qwen.py``),
``InternVL3Backend`` (``bench/backends/internvl.py``), and the ``make_backend``
factory plus backend alias dicts (``bench/run_bench.py``).

QwenBackend (Qwen3-VL-* Thinking/Instruct) mirrors the inference path in
``eval_thinking.py`` exactly (same ``process_vision_info`` call with
``return_video_metadata=True`` so the ``nframes`` sampling is preserved, same
``<|video_pad|>`` token accounting) so that the centralized method reproduces
the existing harness's numbers.

InternVL3Backend mirrors ``QwenBackend``'s interface (``name`` +
``generate(messages, max_new_tokens, *, seed, temperature) -> GenOut``) so the
same Methods drive it. The chat-style ``messages`` content is flattened into
InternVL's ``model.chat(...)`` convention: each ``{"type":"image"}`` (a montage)
and each frame of a ``{"type":"video"}`` becomes an ``<image>`` placeholder
backed by a tile in ``num_patches_list``; a text-only message (the per-stream
aggregator) runs with ``pixel_values=None``.

IMPORTANT: run this backend under the ``internvl`` conda env (transformers 4.48.3).
``cvbench``'s newer transformers breaks the InternVL3 remote code.

Preprocessing helpers (build_transform / dynamic_preprocess / load_image /
load_video) are copied from ``lmms-eval/lmms_eval/models/internvl2.py`` so we
don't depend on the (absent) ``internvl`` training package.

Laziness note: the fork imports ``bench.backends.internvl`` only inside
``make_backend``'s InternVL branch so the ``cvbench`` env never touches the
InternVL code path's dependencies at module import time. With both backends
merged into this module, that boundary is preserved by keeping the
InternVL-only third-party imports (torchvision, numpy, PIL, decord,
transformers ``AutoModel``/``AutoTokenizer``) function-local: importing
``models.clients`` needs only ``torch``. QwenBackend's ``qwen_vl_utils``
import stays inside its constructor, exactly as in the fork.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Backend abstraction (fork: bench/methods/base.py)
# ---------------------------------------------------------------------------

@dataclass
class GenOut:
    """One VLM generation call's output + accounting."""
    text: str
    input_tokens: int
    video_tokens: int
    output_tokens: int
    latency_s: float


class Backend:
    """A VLM that turns a chat ``messages`` list into text + token/latency stats.

    ``generate`` accepts a ``seed`` and ``temperature`` so the 4-pass protocol
    can vary the decoding sampling while holding the (deterministic) frames fixed.
    """
    name = "backend"

    def generate(self, messages, max_new_tokens, *, seed=None, temperature=0.0) -> GenOut:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HF loader (fork: Video-R1/src/eval_thinking.py)
# ---------------------------------------------------------------------------

def load_model(model_path, dtype):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as AutoVLM
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoVLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    ).eval()
    return model, processor


# ---------------------------------------------------------------------------
# Qwen backend (fork: bench/backends/qwen.py)
# ---------------------------------------------------------------------------

class QwenBackend(Backend):
    def __init__(self, model_path, dtype="bfloat16"):
        self.model_path = model_path
        self.name = model_path.rstrip("/").split("/")[-1]
        dt = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.model, self.processor = load_model(model_path, dt)
        # Qwen3-VL uses 16px patches, Qwen2/2.5-VL 14px.
        self.patch_size = getattr(self.processor.image_processor, "patch_size", None) or 14
        from qwen_vl_utils import process_vision_info
        self._pvi = process_vision_info
        self._vid_id = self.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    def generate(self, messages, max_new_tokens, *, seed=None, temperature=0.0) -> GenOut:
        proc = self.processor
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = self._pvi(
            messages, return_video_kwargs=True, return_video_metadata=True,
            image_patch_size=self.patch_size)
        if video_inputs is not None:
            video_inputs, video_metadata = map(list, zip(*video_inputs))
        else:
            video_metadata = None
        inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                      video_metadata=video_metadata, do_resize=False, padding=True,
                      return_tensors="pt", **video_kwargs).to(self.model.device)
        total = int(inputs.input_ids.shape[1])
        vid = int((inputs.input_ids[0] == self._vid_id).sum())
        do_sample = temperature is not None and temperature > 0
        if do_sample and seed is not None:
            torch.manual_seed(seed)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=0.9)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = self.model.generate(**inputs, **gen_kwargs)
        dt = time.perf_counter() - t0
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        out = proc.batch_decode(trimmed, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)[0]
        return GenOut(text=out, input_tokens=total, video_tokens=vid,
                      output_tokens=int(trimmed.shape[1]), latency_s=dt)


# alias -> HF model id (must be cached locally; HF_HUB_OFFLINE=1 on this cluster)
QWEN_ALIASES = {
    "qwen3vl": "Qwen/Qwen3-VL-8B-Thinking",
    "qwen3vl-instruct": "Qwen/Qwen3-VL-8B-Instruct",
    # "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",  # not cached offline; download first
}


# ---------------------------------------------------------------------------
# InternVL3 backend (fork: bench/backends/internvl.py)
#
# IMPORTANT: run this backend under the ``internvl`` conda env (transformers 4.48.3).
# ``cvbench``'s newer transformers breaks the InternVL3 remote code.
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set((i, j) for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(im) for im in images])


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    import numpy as np
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                     for idx in range(num_segments)])


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32,
               frame_indices=None):
    from PIL import Image
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    # A method may pass explicit ``frame_indices`` to override frame choice;
    # every current method leaves it None and gets the uniform get_index grid,
    # so this is a no-op for centralized / temporal_* / cvbench_native / *_select.
    if frame_indices is None:
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        frame_index = int(min(max(0, int(frame_index)), max_frame))   # clamp into range
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pv = torch.stack([transform(t) for t in tiles])
        num_patches_list.append(pv.shape[0])
        pixel_values_list.append(pv)
    return torch.cat(pixel_values_list), num_patches_list


class InternVL3Backend(Backend):
    # honors a per-video ``frame_indices`` override (see load_video); a method
    # that relies on explicit indices can assert this flag so it never silently
    # collapses to uniform sampling on a backend that ignores the key.
    consumes_frame_indices = True

    def __init__(self, model_path="OpenGVLab/InternVL3-8B", num_frame=8, max_tiles=1,
                 device="cuda:0"):
        from transformers import AutoModel, AutoTokenizer
        self.model_path = model_path
        self.name = model_path.rstrip("/").split("/")[-1]
        self.num_frame = num_frame
        self.max_tiles = max_tiles
        self.device = device

        # InternVL remote code calls .item() on a torch.linspace during __init__;
        # route device-less linspace to CPU while from_pretrained constructs the
        # model (harmless no-op on transformers 4.48.3; required on >=5).
        _orig_linspace = torch.linspace

        def _cpu_linspace(*args, **kwargs):
            kwargs.setdefault("device", "cpu")
            return _orig_linspace(*args, **kwargs)

        torch.linspace = _cpu_linspace
        try:
            self.model = AutoModel.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                trust_remote_code=True, device_map=device).eval()
        finally:
            torch.linspace = _orig_linspace

        # Without flash-attn the remote code forces eager attention, which OOMs on
        # multi-image prompts; the LLM picks its kernel from config at runtime.
        lm_cfg = getattr(getattr(self.model, "language_model", None), "config", None)
        if lm_cfg is not None and getattr(lm_cfg, "_attn_implementation", None) == "eager":
            lm_cfg._attn_implementation = "sdpa"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.num_image_token = getattr(self.model, "num_image_token", 256)

    def _flatten(self, content):
        """messages content list -> (question_str, pixel_values|None, num_patches_list|None)."""
        parts, pixel_chunks, npl = [], [], []
        for item in content:
            t = item.get("type")
            if t == "text":
                parts.append(item["text"])
            elif t == "image":
                px = load_image(item["image"], input_size=448, max_num=self.max_tiles)
                pixel_chunks.append(px)
                npl.append(px.shape[0])
                parts.append("<image>\n")
            elif t == "video":
                nf = item.get("nframes", self.num_frame)
                px, sub = load_video(item["video"], num_segments=nf, max_num=1, input_size=448,
                                     frame_indices=item.get("frame_indices"))
                pixel_chunks.append(px)
                npl.extend(sub)
                parts.append("".join(f"Frame{i+1}: <image>\n" for i in range(len(sub))))
        question = "".join(parts)
        if pixel_chunks:
            pixel_values = torch.cat(pixel_chunks, dim=0).to(torch.bfloat16).to(self.device)
            return question, pixel_values, npl
        return question, None, None

    def generate(self, messages, max_new_tokens, *, seed=None, temperature=0.0) -> GenOut:
        content = messages[0]["content"]
        question, pixel_values, npl = self._flatten(content)
        do_sample = temperature is not None and temperature > 0
        if do_sample and seed is not None:
            torch.manual_seed(seed)
        gen_cfg = dict(num_beams=1, max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            gen_cfg.update(temperature=temperature, top_p=0.9)
        t0 = time.perf_counter()
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, question, gen_cfg,
                                       num_patches_list=npl, history=None, return_history=True)[0]
        dt = time.perf_counter() - t0

        # best-effort token accounting (not directly comparable to Qwen's <|video_pad|>):
        # text tokens + the IMG_CONTEXT expansion (num_image_token per tile).
        text_tokens = int(self.tokenizer(question, return_tensors="pt").input_ids.shape[1])
        img_tokens = self.num_image_token * (sum(npl) if npl else 0)
        out_tokens = int(self.tokenizer(response, return_tensors="pt").input_ids.shape[1])
        return GenOut(text=response, input_tokens=text_tokens + img_tokens,
                      video_tokens=img_tokens, output_tokens=out_tokens, latency_s=dt)


# ---------------------------------------------------------------------------
# Backend factory + alias dicts (fork: bench/run_bench.py)
# ---------------------------------------------------------------------------

# alias -> HF id (cached locally; runs under the `internvl` conda env, NOT cvbench,
# because cvbench's transformers breaks the InternVL3 remote code).
INTERNVL_ALIASES = {"internvl3": "OpenGVLab/InternVL3-8B"}


def make_backend(alias, nframes=8, internvl_max_tiles=1):
    if alias in QWEN_ALIASES:
        return QwenBackend(QWEN_ALIASES[alias])
    if alias in INTERNVL_ALIASES:
        return InternVL3Backend(INTERNVL_ALIASES[alias], num_frame=nframes,
                                max_tiles=internvl_max_tiles)
    if "/" in alias:  # raw HF id
        if "internvl" in alias.lower():
            return InternVL3Backend(alias, num_frame=nframes, max_tiles=internvl_max_tiles)
        return QwenBackend(alias)
    raise SystemExit(
        f"unknown backend '{alias}'. Known: {list(QWEN_ALIASES) + list(INTERNVL_ALIASES)}.")
