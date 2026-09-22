# Ported from Wavy-Hec/CVBench bench/methods/viclip_scorer.py @ 967fe27;
# the preprocessing-departure and text-cap notes additionally ported from
# @ e9ea59088dd286ab3609a61cfaeb8591d1fe11d6.
# Deliberate delta vs source: the two departures are stated as mechanisms
# without the fork's measured score delta or its dated internal audit
# reference, and without naming the fork's registered legs — both unpublished.
# VICLIP_SIDE (the square input side _tube resizes to, read by
# segment_select's thumbnail decode) additionally ported from
# Wavy-Hec/MultiCam bench/methods/viclip_scorer.py
# @ b5f666ca990458059cd072048625ccfe629b2737.
"""ViCLIP (video-native CLIP) as a clip-level scorer for the selection arms.

Unlike CLIP/SigLIP — which embed each thumbnail independently — ViCLIP embeds
an 8-frame clip JOINTLY (spatiotemporal attention over the frame tube), so its
similarity can react to motion and event order, not just static content. It is
also the retriever inside the lab's VSeek-R1 agent, so a result scored with it
speaks directly to that pipeline's premise.

The model is NOT a transformers checkpoint. It loads from a local download of
https://huggingface.co/OpenGVLab/ViCLIP (gated: `huggingface-cli login`, then
`huggingface-cli download OpenGVLab/ViCLIP --local-dir $VICLIP_DIR`), which
ships the class files (viclip.py / viclip_text.py / viclip_vision.py /
simple_tokenizer.py + BPE vocab) next to the .pth weights. Those files use
package-relative imports, so a flat download cannot be imported directly; we
mount VICLIP_DIR as a synthetic package instead, which works for both the flat
HF layout and a cloned InternVideo `viclip/` package directory.

Preprocessing is done here, NOT via the repo's `frames2tensor`: that helper
expects cv2-style BGR arrays and flips them, while every frame in this harness
is RGB (decord/PIL). Feeding RGB through it would swap channels silently and
degrade every score with no error. Verified against the InternVideo reference:
resize to 224, ImageNet mean/std, stack to [1, T, C, H, W], T == 8. Two
deliberate departures, neither of which changes a ranking (and changing them
now would make already-scored ViCLIP legs unpoolable): the resize filter is
PIL's default (bicubic, antialiased) rather than cv2 INTER_LINEAR (a small,
common offset on every score), and the callers sample the 8 tube frames at
centred positions with repeat-padding instead of the demo's front-anchored
step=len//8 (closer to the model's training-time sampler).
The text encoder's context is 32 tokens (max_txt_l), silently truncated —
not CLIP's 77.
"""
import importlib.util
import os
import sys
import types

import numpy as np

VICLIP_DIR = os.environ.get("VICLIP_DIR", os.path.expanduser("~/models/ViCLIP"))
VICLIP_SIZE = os.environ.get("VICLIP_SIZE", "l")
# ONLY the full-model checkpoint. The repo's other .pth
# (ViCLIP-L_InternVid-FLT-10M.pth, 4.1 GB) is the vision-encoder-only file
# referenced by viclip_vision.py — feeding it to ViCLIP(pretrain=...) dies on
# a missing ['model'] key, so it must never be picked up as a fallback.
_CKPTS = ("ViClip-InternVid-10M-FLT.pth",)
VICLIP_NFRAMES = 8            # the tube length ViCLIP was trained with
VICLIP_SIDE = 224             # the square input side _tube resizes every frame to
_V_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_V_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_bundle = None


def _load_module(pkg_dir):
    """Import the ViCLIP class files from ``pkg_dir`` under a synthetic package
    so their relative imports resolve in the flat HF layout too."""
    sub = os.path.join(pkg_dir, "viclip")
    if os.path.isdir(sub):                       # cloned InternVideo package
        pkg_dir = sub
    pkg = types.ModuleType("viclip_pkg")
    pkg.__path__ = [pkg_dir]
    sys.modules.setdefault("viclip_pkg", pkg)
    vocab = os.path.join(pkg_dir, "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(vocab):
        raise FileNotFoundError(
            f"{vocab} missing — the tokenizer needs the BPE vocab next to "
            "simple_tokenizer.py; re-download OpenGVLab/ViCLIP in full (no "
            "--include filters)")
    mods = {}
    for name in ("simple_tokenizer", "viclip_text", "viclip_vision", "viclip"):
        path = os.path.join(pkg_dir, f"{name}.py")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing — VICLIP_DIR must hold the full "
                "OpenGVLab/ViCLIP download (code files AND weights)")
        spec = importlib.util.spec_from_file_location(f"viclip_pkg.{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"viclip_pkg.{name}"] = mod
        # the GitHub package uses relative imports (resolved via viclip_pkg),
        # but the HF flat copy uses ABSOLUTE ones (`from simple_tokenizer
        # import ...`), so register each module under its bare name too —
        # exec order (tokenizer first) guarantees dependencies exist by then
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


def ensure_viclip(device="cuda:0"):
    """(model, tokenizer, device), loaded once per process. Fails loud with the
    download recipe when the checkpoint is absent — a missing scorer must never
    degrade to unguided selection (the pre-fix silent-fallback lesson)."""
    global _bundle
    if _bundle is not None:
        return _bundle
    if VICLIP_SIZE != "l":
        raise SystemExit(
            f"VICLIP_SIZE={VICLIP_SIZE}: only 'l' is supported — the HF repo "
            "ships the ViT-L/14 checkpoint only, and a size mismatch dies in "
            "a strict state-dict load at job start.")
    ckpt = next((os.path.join(VICLIP_DIR, c) for c in _CKPTS
                 if os.path.exists(os.path.join(VICLIP_DIR, c))), None)
    if ckpt is None:
        raise FileNotFoundError(
            f"ViCLIP checkpoint not found under VICLIP_DIR={VICLIP_DIR}.\n"
            "Download it once (needs a logged-in HF account; the repo is "
            "gated for anonymous access):\n"
            "  huggingface-cli login\n"
            f"  huggingface-cli download OpenGVLab/ViCLIP --local-dir {VICLIP_DIR}")
    mods = _load_module(VICLIP_DIR)
    tokenizer = mods["simple_tokenizer"].SimpleTokenizer()
    # the HF flat copy hard-codes ViT-L/14 and takes no `size` kwarg (the
    # GitHub package version does); the VICLIP_SIZE guard above keeps the two
    # in agreement
    model = mods["viclip"].ViCLIP(tokenizer=tokenizer, pretrain=ckpt)
    import torch
    model = model.to(torch.device(device)).eval()
    _bundle = (model, tokenizer, device)
    return _bundle


def _tube(frames_pil, device):
    """Exactly VICLIP_NFRAMES RGB PIL frames -> [1, T, C, H, W] float tensor
    with ViCLIP's normalization. Callers must pad/repeat to 8 themselves so the
    temporal sampling policy stays visible at the call site."""
    import torch
    if len(frames_pil) != VICLIP_NFRAMES:
        raise ValueError(f"ViCLIP needs exactly {VICLIP_NFRAMES} frames, "
                         f"got {len(frames_pil)}")
    arrs = []
    for im in frames_pil:
        a = np.asarray(im.convert("RGB").resize((VICLIP_SIDE, VICLIP_SIDE)),
                       dtype=np.float32)
        arrs.append((a / 255.0 - _V_MEAN) / _V_STD)
    x = np.stack(arrs)                            # [T, H, W, C]
    x = np.transpose(x, (0, 3, 1, 2))[None]       # [1, T, C, H, W]
    return torch.from_numpy(x).float().to(device)


def viclip_option_scores(frames_pil, texts, device="cuda:0", text_cache=None):
    """Cosine similarity of ONE clip (8 RGB PIL frames, embedded jointly)
    against each text in ``texts``. Returns np.ndarray [len(texts)].

    Pass one ``text_cache`` dict per question so the K clips of a record reuse
    each option's embedding instead of re-encoding it K times. Both sides are
    re-normalized defensively (idempotent if the model already did)."""
    import torch
    model, tokenizer, dev = ensure_viclip(device)
    cache = text_cache if text_cache is not None else {}
    with torch.no_grad():
        v = model.get_vid_features(_tube(frames_pil, dev))
        v = v / v.norm(dim=-1, keepdim=True)
        feats = []
        for t in texts:
            f = model.get_text_features(t, tokenizer, cache)
            feats.append(f / f.norm(dim=-1, keepdim=True))
        t_emb = torch.cat(feats, dim=0)
        sims = (v @ t_emb.T).float().cpu().numpy()
    return sims[0]
