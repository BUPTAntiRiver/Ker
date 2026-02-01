"""DeepSeek API client."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable
from urllib import request, error

from .types import ChatMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeepSeekClient:
    """Minimal DeepSeek API client using OpenAI-compatible chat endpoints."""

    api_key: str
    base_url: str
    model: str
    temperature: float = 0.4
    timeout_seconds: int = 30

    def generate_reply(self, messages: list[ChatMessage]) -> str:
        payload = {
            "model": self.model,
            "messages": [self._message_to_dict(message) for message in messages],
            "temperature": self.temperature,
        }
        endpoint = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                return self._extract_content(body)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8") if exc.fp else str(exc)
            logger.warning("DeepSeek API error: %s", detail)
            return "The LLM request failed. Check API configuration or try again."
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected DeepSeek client error: %s", exc)
            return "The LLM request failed due to a network error."

    @staticmethod
    def _message_to_dict(message: ChatMessage) -> dict[str, str]:
        return {"role": message.role, "content": message.content}

    @staticmethod
    def _extract_content(raw_body: str) -> str:
        response = json.loads(raw_body)
        choices = response.get("choices", [])
        if not choices:
            return "No response was returned by the LLM."
        message = choices[0].get("message", {})
        return message.get("content", "")
