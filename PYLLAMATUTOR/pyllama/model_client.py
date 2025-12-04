"""LLM client wrapper for open-source LLaMA models with graceful fallback."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional
    tiktoken = None  # type: ignore

# Optional transformers imports; only used if installed
try:  # pragma: no cover - optional import path
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
except Exception:  # pragma: no cover - optional
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

# Native Llama loader (llama-models)
try:  # pragma: no cover - optional import path
    from llama_models.llama3.generation import Llama3
    from llama_models.llama3.chat_format import ChatFormat, RawMessage
    from llama_models.datatypes import QuantizationMode
except Exception:  # pragma: no cover - optional
    Llama3 = None  # type: ignore
    ChatFormat = None  # type: ignore
    RawMessage = None  # type: ignore
    QuantizationMode = None  # type: ignore


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelClient:
    """Lightweight wrapper around open-source LLaMA models or a safe placeholder."""

    DEFAULT_MODEL_PRIORITY = [
        "meta-llama/Llama-3.2-1B-Instruct",
    ]

    def __init__(self, model: Optional[str] = None, max_new_tokens: int = 512, quantization: Optional[str] = None) -> None:
        warnings.filterwarnings(
            "ignore",
            message="torch.set_default_tensor_type\\(\\) is deprecated",
            category=UserWarning,
        )
        # Allow override via env or constructor; fall back to priority list
        env_models = os.getenv("LLAMA_MODEL_IDS")
        if model:
            self.model_candidates = [model]
        elif env_models:
            self.model_candidates = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            self.model_candidates = self.DEFAULT_MODEL_PRIORITY

        # Allow specifying a local path to the downloaded model to avoid re-downloads
        local_path = os.getenv("LLAMA_LOCAL_PATH")
        if local_path:
            self.model_candidates = [local_path] + self.model_candidates
        else:
            default_local = os.path.join(
                os.path.expanduser("~"), ".llama", "checkpoints", "Llama3.2-1B-Instruct"
            )
            self.model_candidates = [default_local] + self.model_candidates

        self.max_new_tokens = max_new_tokens
        self._pipe = None
        self._llama3_native = None
        self._active_model: Optional[str] = None
        self._device_used: Optional[str] = None
        self._initialized = False
        # Simple illustrative pricing (per 1M tokens) for cost reporting
        self._pricing = {"prompt": 0.20 / 1_000_000, "completion": 0.20 / 1_000_000}
        self._device_pref = os.getenv("LLAMA_DEVICE", "auto")
        self._quantization_pref = (quantization or os.getenv("LLAMA_QUANTIZATION", "none")).lower()
        if self._quantization_pref not in ("none", "fp8", "int4"):
            self._quantization_pref = "none"
        # Defer heavy init until first generate to avoid blocking UI thread

    def _ensure_initialized(self) -> None:
        if self._initialized and (self._pipe or self._llama3_native):
            return
        self._init_pipeline()
        # Mark initialized only if a backend is ready; otherwise allow retries.
        self._initialized = bool(self._pipe or self._llama3_native)

    def _init_pipeline(self) -> None:
        if pipeline and AutoTokenizer and AutoModelForCausalLM:
            for candidate in self.model_candidates:
                try:
                    self._pipe = pipeline(
                        "text-generation",
                        model=candidate,
                        tokenizer=candidate,
                        device_map="auto",
                    )
                    self._active_model = candidate
                    self._device_used = "auto"
                    break
                except Exception:
                    self._pipe = None
                    self._active_model = None
                    continue

        # Fallback: try native Llama3 loader if transformers pipeline did not initialize
        if self._pipe is None:
            if Llama3 is None:
                try:
                    from llama_models.llama3.generation import Llama3 as _L3  # type: ignore
                    from llama_models.llama3.chat_format import ChatFormat as _CF, RawMessage as _RM  # type: ignore
                    globals()["Llama3"] = _L3
                    globals()["ChatFormat"] = _CF
                    globals()["RawMessage"] = _RM
                except Exception as exc:
                    print(f"[model_client] Native imports unavailable: {exc}")
                    return

        if self._pipe is None and Llama3 is not None and torch is not None:
            self._init_native()

    def _init_native(self, device_override: Optional[str] = None) -> None:
        for candidate in self.model_candidates:
            try:
                # Ensure single-process env vars for torch.distributed init
                os.environ.setdefault("RANK", "0")
                os.environ.setdefault("WORLD_SIZE", "1")
                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("MASTER_PORT", "29500")
                # Decide device
                if device_override:
                    device = device_override
                elif self._device_pref == "cpu":
                    device = "cpu"
                elif self._device_pref == "gpu":
                    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                else:
                    device = (
                        "cuda"
                        if torch.cuda.is_available()
                        else ("mps" if torch.backends.mps.is_available() else "cpu")
                    )

                if QuantizationMode is None:
                    quant_mode = None
                elif self._quantization_pref == "int4":
                    quant_mode = QuantizationMode.int4_mixed
                elif self._quantization_pref == "fp8":
                    quant_mode = QuantizationMode.fp8_mixed
                else:
                    quant_mode = None

                self._llama3_native = Llama3.build(
                    ckpt_dir=candidate,
                    max_seq_len=2048,
                    max_batch_size=1,
                    device=device,
                    quantization_mode=quant_mode,
                )
                self._active_model = candidate
                self._device_used = device
                break
            except Exception as exc:
                print(f"[model_client] Native Llama3 load failed for {candidate}: {exc}")
                # If torchao or similar optional deps are missing, stop retrying native
                if "torchao" in str(exc):
                    break
                self._llama3_native = None
                self._active_model = None
                continue

    def _token_encoder(self):
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model(self._active_model or "cl100k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        encoder = self._token_encoder()
        if encoder is None:
            # Heuristic fallback: ~4 chars per token
            return max(1, len(text) // 4)
        return len(encoder.encode(text))

    def _estimate_cost(self, usage: TokenUsage) -> float:
        return usage.prompt_tokens * self._pricing["prompt"] + usage.completion_tokens * self._pricing["completion"]

    @property
    def active_device(self) -> str:
        return self._device_used or "unknown"

    def ensure_ready(self) -> None:
        self._ensure_initialized()

    def generate(self, messages: List[Dict[str, str]]) -> Dict[str, object]:
        """Generate text from an LLM or return a placeholder if unavailable."""
        prompt = "\n".join(m.get("content", "") for m in messages)
        prompt_tokens = self.count_tokens(prompt)

        self._ensure_initialized()
        if not (self._pipe or self._llama3_native):
            # Retry once more in case an earlier failure left us uninitialized
            self._initialized = False
            self._ensure_initialized()

        if self._pipe:
            try:
                output = self._pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.7)
                text = output[0]["generated_text"]
                completion_tokens = max(1, self.count_tokens(text) - prompt_tokens)
                usage = TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                return {"text": text, "usage": usage, "cost": self._estimate_cost(usage)}
            except Exception as exc:
                print(f"[model_client] transformers generation failed: {exc}")
                pass

        if self._llama3_native:
            try:
                if ChatFormat is None or RawMessage is None:
                    raise RuntimeError("Chat formatter unavailable")
                formatter = ChatFormat(self._llama3_native.tokenizer)
                user_msg = RawMessage(role="user", content=prompt)
                model_input = formatter.encode_dialog_prompt([user_msg])
                pieces: List[str] = []
                # generate yields lists of GenerationResult
                for step in self._llama3_native.generate(
                    [model_input],
                    max_gen_len=self.max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                ):
                    for res in step:
                        if res.source == "output" and not res.ignore_token:
                            pieces.append(res.text)
                text = "".join(pieces).replace("<|eot_id|>", "").strip()
                completion_tokens = max(1, self.count_tokens(text))
                usage = TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                return {"text": text, "usage": usage, "cost": self._estimate_cost(usage)}
            except Exception as exc:
                print(f"[model_client] native generation failed: {exc}")
                # Try CPU fallback once if we were on GPU/MPS
                if self._device_used and self._device_used != "cpu":
                    self._llama3_native = None
                    self._device_used = None
                    self._active_model = None
                    self._init_native(device_override="cpu")
                    if self._llama3_native:
                        try:
                            formatter = ChatFormat(self._llama3_native.tokenizer)
                            user_msg = RawMessage(role="user", content=prompt)
                            model_input = formatter.encode_dialog_prompt([user_msg])
                            pieces = []
                            for step in self._llama3_native.generate(
                                [model_input],
                                max_gen_len=self.max_new_tokens,
                                temperature=0.7,
                                top_p=0.9,
                            ):
                                for res in step:
                                    if res.source == "output" and not res.ignore_token:
                                        pieces.append(res.text)
                            text = "".join(pieces).replace("<|eot_id|>", "").strip()
                            completion_tokens = max(1, self.count_tokens(text))
                            usage = TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                            return {"text": text, "usage": usage, "cost": self._estimate_cost(usage)}
                        except Exception as exc2:
                            print(f"[model_client] cpu retry failed: {exc2}")
                            pass

        # Offline/placeholder response keeps UI usable
        placeholder = (
            "### Concept Explanation\n"
            "This is a placeholder because the open-source LLaMA model is not available. Summarize the user's request and explain the relevant Python idea.\n\n"
            "### Code Example\n"
            "Provide a short Python snippet related to the question.\n\n"
            "### Practice Exercise\n"
            "Offer one beginner exercise.\n\n"
            "### Feedback\n"
            "Share one suggestion for improvement or what to check next."
        )
        completion_tokens = self.count_tokens(placeholder)
        usage = TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        return {"text": placeholder, "usage": usage, "cost": self._estimate_cost(usage)}
