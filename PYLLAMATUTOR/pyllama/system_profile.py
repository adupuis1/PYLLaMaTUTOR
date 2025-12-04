"""Simple system profiling and preset selection."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore


@dataclass
class SystemProfile:
    cores: int
    ram_gb: float
    has_gpu: bool
    has_cuda: bool
    has_mps: bool


def detect_system() -> tuple[SystemProfile, bool]:
    ok = True
    cores = os.cpu_count() or 1
    ram_gb = 4.0
    try:
        if psutil:
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        else:
            pages = os.sysconf("SC_PHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            ram_gb = pages * size / (1024 ** 3)
    except Exception:
        ok = False
    has_gpu = False
    has_cuda = False
    has_mps = False
    try:
        if torch:
            has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
            has_cuda = bool(torch.cuda.is_available())
            has_gpu = bool(has_cuda or has_mps)
            # Default to CPU if only MPS is present (to avoid flaky MPS paths)
            if has_mps and not has_cuda:
                os.environ["LLAMA_DEVICE"] = os.environ.get("LLAMA_DEVICE", "cpu")
    except Exception:
        ok = False
    return SystemProfile(cores=cores, ram_gb=ram_gb, has_gpu=has_gpu, has_cuda=has_cuda, has_mps=has_mps), ok


def build_presets(profile: SystemProfile, ok: bool) -> Dict[str, Dict[str, object]]:
    # Defaults if detection failed
    if not ok:
        return {
            "light": {"device": "cpu", "max_tokens": 256, "threads": 2, "note": "Fallback defaults"},
            "balanced": {"device": "cpu", "max_tokens": 512, "threads": 4, "note": "Fallback defaults"},
            "overkill": {"device": "gpu", "max_tokens": 768, "threads": 6, "note": "Fallback defaults"},
        }

    def mk(device: str, max_tokens: int, threads: int, note: str) -> Dict[str, object]:
        return {"device": device, "max_tokens": max_tokens, "threads": threads, "note": note}

    presets: Dict[str, Dict[str, object]] = {}
    # Force CPU if only MPS is present
    device_for_perf = "gpu" if profile.has_cuda else ("cpu" if profile.has_mps and not profile.has_cuda else "auto")
    # light
    presets["light"] = mk(
        "cpu",
        256,
        max(1, min(4, profile.cores)),
        "For minimal impact on system resources.",
    )
    # balanced
    presets["balanced"] = mk(
        device_for_perf if device_for_perf != "auto" else ("gpu" if profile.has_gpu else "cpu"),
        512,
        max(1, min(8, profile.cores)),
        "Balanced speed vs resource use (prefers CUDA; MPS falls back to CPU).",
    )
    # overkill
    presets["overkill"] = mk(
        device_for_perf if device_for_perf != "auto" else ("gpu" if profile.has_gpu else "cpu"),
        1024,
        max(1, min(12, profile.cores)),
        "Max performance; may impact responsiveness.",
    )
    return presets
