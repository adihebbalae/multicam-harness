"""Dollar-cost estimation for benchmark runs (new code, not ported).

Reads a price sheet (``configs/model_prices.yaml``) mapping a model id to
per-million-token USD rates and converts already-aggregated token counts into
an estimated dollar cost. Locally hosted checkpoints carry ``null`` rates, in
which case ``estimate_usd`` returns ``None`` (no estimate) rather than 0.

Expected YAML shape::

    Qwen/Qwen3-VL-8B-Thinking:
      input: null    # USD per 1M input tokens (null = no price known)
      output: null   # USD per 1M output tokens

Token aggregation itself lives in ``evaluation.scoring``; this module only
prices token totals produced elsewhere.
"""


def load_model_prices(path="configs/model_prices.yaml"):
    """Load the model price sheet; returns ``{}`` if the file is missing or empty."""
    import os

    import yaml

    if not os.path.exists(path):
        return {}
    with open(path) as f:
        prices = yaml.safe_load(f)
    return prices or {}


def _lookup(prices, model_id):
    """Price entry for ``model_id``: exact key match first, then a match on the
    HF id's last path component so callers may pass either the full id
    ("Org/Model-X") or the backend's short ``name`` ("Model-X")."""
    if model_id in prices:
        return prices[model_id]
    tail = str(model_id).rstrip("/").split("/")[-1]
    for key, entry in prices.items():
        if str(key).rstrip("/").split("/")[-1] == tail:
            return entry
    return None


def estimate_usd(input_tokens, output_tokens, model_id, prices):
    """Estimated USD for one generation (or an aggregated run) at the sheet's
    per-1M-token rates. Returns ``None`` when the model has no entry, either
    rate is ``null``, or a token count is unknown."""
    entry = _lookup(prices or {}, model_id)
    if not isinstance(entry, dict):
        return None
    rate_in = entry.get("input")
    rate_out = entry.get("output")
    if rate_in is None or rate_out is None:
        return None
    if input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000
