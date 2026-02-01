"""Voice I/O interfaces for Ker."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VoiceIO:
    """Placeholder for voice input/output handling."""

    def start_listening(self) -> None:
        """Start listening for user voice input."""
        pass

    def stop_listening(self) -> None:
        """Stop listening for user voice input."""
        pass
