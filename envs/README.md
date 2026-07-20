# Environments

Two conda envs are required — a transformers version conflict means one env cannot run both backends:

| Env | Backend(s) | Create |
|---|---|---|
| `cvbench` | Qwen3-VL (qwen3vl / qwen3vl-instruct), CLIP/SigLIP scorers | `conda env create -f envs/cvbench.yml` |
| `internvl` | InternVL3-8B (its remote code breaks under cvbench's newer transformers) | `conda env create -f envs/internvl.yml` |

These are cluster-exported lockfiles (`--no-builds`); exact pins may need loosening on other platforms. `decord` and `qwen_vl_utils` install via the pip section — decord wheels can be finicky on non-x86_64 or older glibc; if `pip install decord` fails, build from source or use `eva-decord`.

Models are loaded in-process from the local HF cache. Pre-download weights on a login node (compute nodes may run with `HF_HUB_OFFLINE=1`).
