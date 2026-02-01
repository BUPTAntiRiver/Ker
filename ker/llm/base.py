"""Base interfaces for LLM clients."""

from __future__ import annotations

from typing import Protocol

from .types import ChatMessage


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def generate_reply(self, messages: list[ChatMessage]) -> str:
        """Generate a reply for the provided chat messages."""


class NullLLMClient:
    """Fallback LLM client when no API key is configured."""

    def generate_reply(self, messages: list[ChatMessage]) -> str:
        return (
            "LLM is not configured. Set DEEPSEEK_API_KEY to enable responses."
        )
