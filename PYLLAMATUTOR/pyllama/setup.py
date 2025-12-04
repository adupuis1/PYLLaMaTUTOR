"""Automated environment setup for the AI Tutor project.

This script installs required Python dependencies and downloads LLaMA models
(using the llama-stack CLI) in priority order so the teacher/user can run a
single quickstart command.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

# Lazy imports to avoid requiring llama-models before installation
download_mod = None
sku_list_mod = None
safety_mod = None

ROOT = Path(__file__).parent
REQUIREMENTS = ROOT / "requirements.txt"

# Default (single) model target (Llama 3.2 1B Instruct, lightweight)
DEFAULT_MODEL_IDS = [
    "Llama3.2-1B-Instruct",
]

# Custom download URL for Llama 3.2 (lightweight) provided by the user (used when available)
CUSTOM_LLAMA_URL = os.getenv(
    "LLAMA_CUSTOM_URL",
    "https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiNnlxZmZqbWpsNTNma3F0Z2cyZjFocGlkIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NDk3MTE1NX19fV19&Signature=QdwcZPTINg6Pn3L6OP9IazkDq67MmQX9nFvLFD9%7EhFIFvebgHfUIC0qOwFuBGHKQJwIophc7ToGr0kcjL3i6JRrkmfduRmNdmDs4RHJ7Vtg0nvK%7Ep8oPERiKitxHTrQreJWYkP7bfv2YS67cmqYVDGOWb%7E1nGUdvTwbYlEml6JmA4UEMqApVxSbIGEw%7EjMfIvok926PjrFb94IvQNpyW-QIiNLNOFyV9lgD6pbDBa9KSFA6PjTusmQSnmjfzzrmLXqAy5734k4Lwnm3EMuKEVDuhmrveiyzI9b1hfsQmxJXL1taVIskubF-BvwAvHV2YJ9Ggx0zUSo-pg1CLKKR2pg__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1338187817534618",
)

DEFAULT_MODEL_DIR = Path.home() / ".llama" / "checkpoints" / "Llama3.2-1B-Instruct"


def run(cmd: List[str], desc: str) -> bool:
    print(f"[setup] {desc}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[setup] Failed ({desc}): {exc}")
        return False


def ensure_dependencies() -> None:
    # Install project requirements
    if REQUIREMENTS.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)], "install requirements")
    # Ensure llama-models CLI is present/up-to-date
    run([sys.executable, "-m", "pip", "install", "-U", "llama-models"], "install/update llama-models")


def get_model_ids() -> List[str]:
    env_models = os.getenv("LLAMA_MODEL_IDS")
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]
    return DEFAULT_MODEL_IDS


def download_models(models: Iterable[str]) -> None:
    # Use llama-models download utilities directly (avoids broken CLI import in some versions)
    global download_mod, sku_list_mod, safety_mod
    try:
        if download_mod is None:
            import llama_models.cli.download as download_mod  # type: ignore
        if sku_list_mod is None:
            import llama_models.sku_list as sku_list_mod  # type: ignore
        if safety_mod is None:
            import llama_models.cli.safety_models as safety_mod  # type: ignore
    except Exception as exc:
        print(f"[setup] Failed to import llama-models helpers: {exc}")
        return

    for model_id in models:
        try:
            model = sku_list_mod.resolve_model(model_id)
            if model is None:
                print(f"[setup] Model not found: {model_id}")
                continue
            # Prefer the safety maps if present
            pg_model_map = safety_mod.prompt_guard_model_sku_map()
            pg_info_map = safety_mod.prompt_guard_download_info_map()
            if model_id in pg_model_map:
                model = pg_model_map[model_id]
                info = pg_info_map[model_id]
            else:
                info = sku_list_mod.llama_meta_net_info(model)

            meta_url = CUSTOM_LLAMA_URL
            if not meta_url:
                print(f"[setup] No META URL provided for {model_id}")
                continue

            download_mod._meta_download(
                model=model,
                model_id=model_id,
                meta_url=meta_url,
                info=info,
                max_concurrent_downloads=3,
            )
            print(f"[setup] Model ready: {model_id}")
            return
        except Exception as exc:
            print(f"[setup] Download failed for {model_id}: {exc}")

    print("[setup] No models could be downloaded; the app will fall back to placeholder responses.")


def setup_environment() -> None:
    print("[setup] Starting environment setup...")
    ensure_dependencies()
    DEFAULT_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    download_models(get_model_ids())
    print("[setup] Setup complete.")


if __name__ == "__main__":
    setup_environment()
