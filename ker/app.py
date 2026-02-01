"""Application wiring for Ker."""

from __future__ import annotations

import logging

from .agent import KerAgent
from .config import KerConfig
from .llm import DeepSeekClient, NullLLMClient
from .ui import ConsoleChatUI
from .screen import ScreenContextProvider
from .voice import VoiceIO

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
        self.voice = VoiceIO()
        self.ui = ConsoleChatUI(on_response=self.voice.speak_streaming)
        self.screen = ScreenContextProvider()

    def run(self) -> None:
        """Run the application."""

        logger.info("Ker starting in console mode.")
        self.ui.start(self.agent)


def main() -> None:
    """Entry point for running Ker."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    KerApp().run()
