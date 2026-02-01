"""Console chat UI for early development and testing."""

from __future__ import annotations

from dataclasses import dataclass

from ..agent import KerAgent


@dataclass
class ConsoleChatUI:
    """Minimal console-based chat interface."""

    prompt: str = "You> "

    def start(self, agent: KerAgent) -> None:
        """Start a simple REPL-style chat session."""

        print("Ker console chat started. Type /exit to quit.")
        while True:
            try:
                user_input = input(self.prompt)
            except EOFError:
                print("\nSession ended.")
                return

            if not user_input.strip():
                continue
            if user_input.strip().lower() in {"/exit", "/quit"}:
                print("Session ended.")
                return

            response = agent.handle_user_message(user_input)
            if response:
                print(f"Ker> {response}")
