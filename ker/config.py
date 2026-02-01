"""Configuration loading for Ker."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class KerConfig:
    """Runtime configuration for Ker."""

    deepseek_api_key: str | None
    deepseek_base_url: str
    model: str
    temperature: float

    @classmethod
    def from_env(cls) -> "KerConfig":
        """Load configuration from environment variables."""

        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("KER_MODEL", "deepseek-chat"),
            temperature=float(os.getenv("KER_TEMPERATURE", "0.4")),
        )
