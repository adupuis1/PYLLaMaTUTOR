"""Tutor agent logic enforcing structured responses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from model_client import ModelClient, TokenUsage

REQUIRED_SECTIONS = [
    "### Concept Explanation",
    "### Code Example",
    "### Practice Exercise",
    "### Feedback",
]


@dataclass
class AgentResponse:
    text: str
    usage: TokenUsage
    cost: float
    session_tokens: int
    session_cost: float


class TutorAgent:
    """Beginner-friendly Python tutor using an open-source LLM."""

    def __init__(self, model: str = "llama-3-8b-instruct", max_new_tokens: int = 512, quantization: str | None = None) -> None:
        self.client = ModelClient(model=model, max_new_tokens=max_new_tokens, quantization=quantization)
        self.session_tokens = 0
        self.session_cost = 0.0

    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a patient AI tutor for introductory Python programming. "
            "Keep answers concise and beginner-friendly. Always respond with four sections in this exact order: "
            "### Concept Explanation, ### Code Example, ### Practice Exercise, ### Feedback. "
            "If a section does not apply, write 'Not applicable for this question.' in that section."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

    def _ensure_required_sections(self, text: str) -> str:
        missing = [section for section in REQUIRED_SECTIONS if section not in text]
        if not missing:
            return text
        parts = [text.rstrip(), ""]
        for section in missing:
            parts.append(section)
            parts.append("Not applicable for this question.")
            parts.append("")
        return "\n".join(parts).rstrip()

    def ask(self, user_text: str) -> AgentResponse:
        messages = self._build_messages(user_text)
        result = self.client.generate(messages)
        raw_text: str = result["text"]
        usage: TokenUsage = result["usage"]
        cost: float = float(result.get("cost", 0.0))

        self.session_tokens += usage.total_tokens
        self.session_cost += cost

        cleaned = self._ensure_required_sections(raw_text)

        return AgentResponse(
            text=cleaned,
            usage=usage,
            cost=cost,
            session_tokens=self.session_tokens,
            session_cost=self.session_cost,
        )
