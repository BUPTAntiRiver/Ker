"""Agent logic for Ker."""

from __future__ import annotations

from dataclasses import dataclass

from .llm import ChatMessage, LLMClient


BEHAVIORAL_CONSTRAINTS = (
    "You are Ker, an AI learning companion. You are strictly non-autonomous. "
    "Do not set goals, plan tasks, or take actions on behalf of the user. "
    "Only respond to explicit user input. Offer explanations, hints, or optional "
    "suggestions. Avoid manipulation or pressure."
)


@dataclass
class KerAgent:
    """User-driven agent that only responds to explicit input."""

    llm_client: LLMClient

    def handle_user_message(self, user_message: str, screen_context: str | None = None) -> str:
        """Handle a single user message and return the response."""

        cleaned = user_message.strip()
        if not cleaned:
            return ""

        messages = [
            ChatMessage(role="system", content=BEHAVIORAL_CONSTRAINTS),
        ]

        if screen_context:
            messages.append(
                ChatMessage(
                    role="system",
                    content=f"Screen context (user-provided):\n{screen_context}",
                )
            )

        messages.append(ChatMessage(role="user", content=cleaned))
        return self.llm_client.generate_reply(messages)
