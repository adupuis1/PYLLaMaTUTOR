# AI Tutor for Introductory Python Programming using LLM

A beginner-friendly Python tutor with a simple PyQt chat UI. Ask Python questions, paste small code snippets, or request practice exercises; the tutor replies in a fixed four-section format (Concept Explanation, Code Example, Practice Exercise, Feedback) powered by an open-source LLM (e.g., LLaMA).

## Requirements
- Python 3.9+
- PyQt5
- LLM client libraries (pick what you use):
  - `transformers` / `torch` for local or hosted open-source models
  - `tiktoken` (optional, for token counting if compatible) or another tokenizer

## Installation
1. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application (One-Command Quickstart)
- Recommended: run `python3 app.py` to auto-install deps, download the first available LLaMA model (lightweight → heavier), and launch the GUI.
- If you prefer manual startup after setup, run `python3 gui.py`.

## LLM Configuration (Open-Source Required)
- Target model: Llama 3.2 1B Instruct (lightweight). Override via `LLAMA_MODEL_IDS` env (comma-separated) if you have a different variant.
- Custom download URL baked in (can override with `LLAMA_CUSTOM_URL`):  
  `https://llama3-2-lightweight.llamameta.net/*?Policy=...&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1338187817534618`  
- Install Meta’s LLaMA tooling:  
  - `pip install llama-models -U`  
  - List available models: `llama-model list` (add `--show-all` for older versions)  
  - Download Llama 3.2 1B Instruct (uses the custom URL automatically in `setup.py`):  
    `llama-model download --source meta --model-id Llama3.2-1B-Instruct --meta-url <your URL>`  
- Default local path checked by the app: `~/.llama/checkpoints/Llama3.2-1B-Instruct` (override with `LLAMA_LOCAL_PATH`).  
- If no model is reachable, the app falls back to a placeholder response so the UI still works.
- Native inference uses the Meta checkpoint (not transformers); dependencies include `fairscale` and `torchvision` for the Llama 3.2 loader.

## Usage Notes
- The launcher shows a loading screen while dependencies are checked, the system is scanned for presets (GPU/Metal detection included), and the model is warmed up; the chat window only opens once the model is ready.
- Type a question or paste code, then click **Send** (or press **Ctrl+Enter**).
- The tutor always responds with the four sections: Concept Explanation, Code Example, Practice Exercise, Feedback.
- For debugging, include the error message you see; the tutor will highlight likely issues and offer a fix.
- UI tips: a busy bar appears while the model runs; “You” labels are green, “Tutor” labels are red, and section headers are bolded for readability. Footer shows model, device, tokens, cost, and last response time. The model name is shown above the chat once loaded. Code blocks are lightly syntax tinted.
- Config tab: choose hardware presets (light/balanced/overkill) derived from a system scan at launch; adjust max new tokens, device (auto/cpu/gpu with CPU fallback if GPU/MPS fails), quantization, and CPU threads (with guidance). A Rescan button runs detection in the background. App icon and logo come from `images/`.
- Quick prompts: curated lists (sorting algorithms, data types, beginner). Picking one pre-fills the input.
- Evaluation tab: shows totals/averages computed across all saved evaluations (tokens, cost, avg response time, prompt/completion token sums) with a response-time column per row. The evaluation table is persisted to `~/.pyllama/chat_history.json`, reloaded at startup, and includes a red “Wipe saved evaluations” button to clear persisted stats; current-session rows are tinted using the system alternate background. Chat text itself is not restored—only evaluation rows are replayed.

## Token and Cost Reporting
- The GUI footer shows session token count and an approximate cost estimate based on the configured model pricing in `model_client.py` (update to match your deployment). Token counts come from the model tokenizer when available, with a heuristic fallback if not.

## Limitations
- The tutor may occasionally be incorrect or incomplete; verify important code.
- The tutor does not execute code—it reasons about it.
- Signed model URLs expire; if the bundled URL is stale, set `LLAMA_LOCAL_PATH` or a fresh URL.

## Repository Contents
- `AGENT.md` – behavior/design spec for the tutor agent.
- `gui.py` – PyQt chat interface.
- `agent.py` – wraps LLM calls, enforces the four-section format, tracks session tokens.
- `model_client.py` – LLM client wrapper with token counting and pricing stubs.
- `requirements.txt` – Python dependencies.
- `README.md` – this file with setup and usage instructions.
- `reports/report.md` – ~3 page design/evaluation write-up.
- `images/` – logo and app icon assets.
- `LICENSE` – MIT license for source code; Meta Llama license applies to model weights.

## How This Meets the Project Requirements
- **Tokenization & cost**: Tokens and approximate cost are shown per session in the footer, using the model tokenizer when available (heuristic fallback otherwise).
- **Prompt engineering**: The agent enforces a fixed four-section format (Concept, Code, Exercise, Feedback) via a system prompt; quick prompts demonstrate curated intents.
- **Evaluation**: Evaluation tab tracks timing/tokens aggregated across all saved evaluations (persisted to `~/.pyllama/chat_history.json`), includes per-row response time, a red wipe control, and tints current-session rows; quick prompts are available but no automatic eval runs. Chat history is not replayed on startup.
