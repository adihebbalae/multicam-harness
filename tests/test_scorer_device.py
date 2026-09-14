"""scorer_device(): where the CLIP/SigLIP scorer is placed.

Pure CPU — no backend, no scorer model, no video, no GPU. Runnable either way:

    python -m tests.test_scorer_device
    pytest tests/test_scorer_device.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from inprocess.harnesses.clip_select import (                     # noqa: E402
    SCORER_DEVICE_ENV, scorer_device)


class _Backend:
    def __init__(self, device):
        self.device = device


class _NoDevice:
    pass


def _clear():
    os.environ.pop(SCORER_DEVICE_ENV, None)


def test_defaults_to_the_backend_device():
    _clear()
    assert scorer_device(_Backend("cuda:3")) == "cuda:3"


def test_defaults_to_cuda0_when_the_backend_names_no_device():
    _clear()
    assert scorer_device(_NoDevice()) == "cuda:0"


def test_env_overrides_the_backend_device():
    os.environ[SCORER_DEVICE_ENV] = "cuda:1"
    try:
        assert scorer_device(_Backend("cuda:0")) == "cuda:1"
        assert scorer_device(_NoDevice()) == "cuda:1"
    finally:
        _clear()


def test_empty_env_is_not_an_override():
    """An exported-but-empty variable is how a shell passes "unset", and it must
    not place the scorer on "". `or` rather than a plain `in os.environ` check."""
    os.environ[SCORER_DEVICE_ENV] = ""
    try:
        assert scorer_device(_Backend("cuda:2")) == "cuda:2"
    finally:
        _clear()


if __name__ == "__main__":
    ran = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            ran += 1
            print(f"ok  {name}")
    print(f"\n{ran} passed")
