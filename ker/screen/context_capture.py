"""Screen context capture utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScreenContextProvider:
    """Placeholder for screen context capture with explicit user consent."""

    def capture(self) -> str:
        """Capture screen context when explicitly requested by the user."""
        return ""
