"""Types for LLM messaging."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    """Single chat message for an LLM conversation."""

    role: str
    content: str
