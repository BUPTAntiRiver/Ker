"""Application wiring for Ker."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

from .agent import KerAgent
from .config import KerConfig
from .llm import DeepSeekClient, NullLLMClient
from .ui import ConsoleChatUI
from .screen import ScreenContextProvider
from .voice import VoiceIO, VoiceOutputConfig
from .voice.voice_io import VoiceIOConfig

logger = logging.getLogger(__name__)


class KerApp:
    """Top-level application that wires Ker components together."""

    def __init__(self) -> None:
        config = KerConfig.from_env()
        if config.deepseek_api_key:
            llm_client = DeepSeekClient(
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
                model=config.model,
                temperature=config.temperature,
            )
        else:
            llm_client = NullLLMClient()

        self.agent = KerAgent(llm_client=llm_client)
        self.voice = VoiceIO(
            config=VoiceIOConfig.from_env(),
            output_config=VoiceOutputConfig.from_env(),
        )
        self.ui = ConsoleChatUI(on_response=self.voice.speak_streaming)
        self.screen = ScreenContextProvider()
        self.transcript_logger = TranscriptLogger.from_env()

    def run(self) -> None:
        """Run the application."""
        self._configure_voice_mode()
        if _console_enabled():
            logger.info("Ker starting in console + voice mode.")
            self.ui.start(self.agent)
            return

        logger.info("Ker starting in voice-only mode.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Ker stopped.")

    def _configure_voice_mode(self) -> None:
        def handle_voice_input(transcript: str) -> None:
            self.transcript_logger.log_user(transcript)
            if _should_print_voice_transcript():
                print(f"You> {transcript}")
            response = self.agent.handle_user_message(transcript)
            if response:
                if _should_print_voice_output():
                    print(f"Ker> {response}")
                self.transcript_logger.log_ker(response)
                self.voice.speak_streaming(response)

        self.voice.on_transcript = handle_voice_input
        self.voice.start_listening()


def _should_print_voice_output() -> bool:
    value = os.getenv("KER_VOICE_TEXT_OUTPUT", "true")
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _should_print_voice_transcript() -> bool:
    value = os.getenv("KER_VOICE_TEXT_INPUT", "true")
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _console_enabled() -> bool:
    value = os.getenv("KER_CONSOLE_ENABLED", "false")
    return value.strip().lower() in {"1", "true", "yes", "y"}


@dataclass
class TranscriptLogger:
    path: str | None

    @classmethod
    def from_env(cls) -> "TranscriptLogger":
        return cls(path=os.getenv("KER_TRANSCRIPT_LOG"))

    def log_user(self, text: str) -> None:
        self._write("USER", text)

    def log_ker(self, text: str) -> None:
        self._write("KER", text)

    def _write(self, role: str, text: str) -> None:
        if not self.path:
            return
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{timestamp}] {role}: {text}\n"
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            logger.warning("Failed to write transcript log to %s", self.path)


def main() -> None:
    """Entry point for running Ker."""
    log_level = os.getenv("KER_LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")
    KerApp().run()
