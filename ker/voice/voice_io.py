"""Voice I/O interfaces for Ker.

This module provides an always-listening microphone loop that emits
transcribed user speech via callbacks. It is intentionally reactive and
does not initiate any action on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import logging
import os
import queue
import re
import threading
import time
from typing import Callable, Iterable, Protocol

logger = logging.getLogger(__name__)


class VoiceBackend(Protocol):
    """Protocol for a speech-to-text backend."""

    def listen_forever(
        self,
        on_transcript: Callable[[str], None],
        stop_event: threading.Event,
    ) -> None:
        """Start a blocking listen loop until stop_event is set."""


class TextToSpeechBackend(Protocol):
    """Protocol for a text-to-speech backend."""

    def speak(self, text: str) -> None:
        """Speak the provided text."""


@dataclass(frozen=True)
class VoiceIOConfig:
    """Configuration for voice input handling."""

    listen_timeout: float = 1.0
    phrase_time_limit: float = 8.0
    energy_threshold: int | None = None
    pause_threshold: float = 0.8
    device_index: int | None = None


@dataclass(frozen=True)
class VoiceOutputConfig:
    """Configuration for text-to-speech output."""

    rate: int | None = None
    volume: float | None = None
    voice_id: str | None = None
    voice_name: str | None = None
    voice_gender: str | None = "female"

    @classmethod
    def from_env(cls) -> "VoiceOutputConfig":
        """Load voice output settings from environment variables."""

        rate_value = os.getenv("KER_VOICE_RATE")
        volume_value = os.getenv("KER_VOICE_VOLUME")
        voice_gender = os.getenv("KER_VOICE_GENDER", "female")

        return cls(
            rate=int(rate_value) if rate_value else None,
            volume=float(volume_value) if volume_value else None,
            voice_id=os.getenv("KER_VOICE_ID"),
            voice_name=os.getenv("KER_VOICE_NAME"),
            voice_gender=voice_gender,
        )


class SpeechRecognitionBackend:
    """SpeechRecognition-based backend using the default microphone."""

    def __init__(self, config: VoiceIOConfig) -> None:
        if importlib.util.find_spec("speech_recognition") is None:
            raise RuntimeError(
                "speech_recognition is not installed. "
                "Install it to enable voice input."
            )

        import speech_recognition as sr  # type: ignore

        self._sr = sr
        self._config = config

    def listen_forever(
        self,
        on_transcript: Callable[[str], None],
        stop_event: threading.Event,
    ) -> None:
        recognizer = self._sr.Recognizer()
        if self._config.energy_threshold is not None:
            recognizer.energy_threshold = self._config.energy_threshold
        recognizer.pause_threshold = self._config.pause_threshold

        microphone = self._sr.Microphone(device_index=self._config.device_index)
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            logger.info("Voice input: microphone initialized.")

        while not stop_event.is_set():
            try:
                with microphone as source:
                    audio = recognizer.listen(
                        source,
                        timeout=self._config.listen_timeout,
                        phrase_time_limit=self._config.phrase_time_limit,
                    )
                transcript = recognizer.recognize_google(audio)
                cleaned = transcript.strip()
                if cleaned:
                    on_transcript(cleaned)
            except self._sr.WaitTimeoutError:
                continue
            except self._sr.UnknownValueError:
                continue
            except self._sr.RequestError as exc:
                logger.warning("Voice input request error: %s", exc)
                time.sleep(1.0)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Voice input error: %s", exc)
                time.sleep(1.0)


class Pyttsx3TTSBackend:
    """pyttsx3-based text-to-speech backend."""

    def __init__(self, config: VoiceOutputConfig) -> None:
        if importlib.util.find_spec("pyttsx3") is None:
            raise RuntimeError("pyttsx3 is not installed. Install it for TTS.")

        import pyttsx3  # type: ignore

        self._engine = pyttsx3.init()
        if config.rate is not None:
            self._engine.setProperty("rate", config.rate)
        if config.volume is not None:
            self._engine.setProperty("volume", config.volume)
        if config.voice_id is not None:
            self._engine.setProperty("voice", config.voice_id)
        else:
            voice_id = _select_voice_id(
                self._engine.getProperty("voices"),
                preferred_name=config.voice_name,
                preferred_gender=config.voice_gender,
            )
            if voice_id:
                self._engine.setProperty("voice", voice_id)

        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        with self._lock:
            self._engine.say(text)
            self._engine.runAndWait()


@dataclass
class VoiceIO:
    """Always-listening voice input with optional backend support."""

    on_transcript: Callable[[str], None] | None = None
    on_error: Callable[[str], None] | None = None
    config: VoiceIOConfig = field(default_factory=VoiceIOConfig)
    output_config: VoiceOutputConfig = field(default_factory=VoiceOutputConfig)
    backend: VoiceBackend | None = None
    tts_backend: TextToSpeechBackend | None = None

    _thread: threading.Thread | None = field(init=False, default=None)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _listening: bool = field(init=False, default=False)
    _speak_thread: threading.Thread | None = field(init=False, default=None)
    _speak_queue: "queue.Queue[str]" = field(init=False, default_factory=queue.Queue)
    _speak_stop_event: threading.Event = field(init=False, default_factory=threading.Event)

    def start_listening(self) -> None:
        """Start listening for user voice input."""
        if self._listening:
            return

        if self.on_transcript is None:
            self._report_error("No transcript callback is configured.")
            return

        if self.backend is None:
            try:
                self.backend = SpeechRecognitionBackend(self.config)
            except RuntimeError as exc:
                self._report_error(str(exc))
                return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="KerVoiceListener",
            daemon=True,
        )
        self._thread.start()
        self._listening = True
        logger.info("Voice input started.")

    def stop_listening(self) -> None:
        """Stop listening for user voice input."""
        if not self._listening:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._listening = False
        logger.info("Voice input stopped.")

    @property
    def is_listening(self) -> bool:
        """Return whether the voice loop is running."""

        return self._listening

    def speak(self, text: str) -> None:
        """Speak text using the configured TTS backend."""

        cleaned = text.strip()
        if not cleaned:
            return

        if self.tts_backend is None:
            try:
                self.tts_backend = Pyttsx3TTSBackend(self.output_config)
            except RuntimeError as exc:
                self._report_error(str(exc))
                return

        if self._speak_thread and self._speak_thread.is_alive():
            return

        self._speak_thread = threading.Thread(
            target=self._speak_blocking,
            args=(cleaned,),
            name="KerTTS",
            daemon=True,
        )
        self._speak_thread.start()

    def speak_streaming(self, text: str) -> None:
        """Speak text in short chunks to approximate streaming TTS."""

        cleaned = text.strip()
        if not cleaned:
            return

        if self.tts_backend is None:
            try:
                self.tts_backend = Pyttsx3TTSBackend(self.output_config)
            except RuntimeError as exc:
                self._report_error(str(exc))
                return

        for chunk in _split_into_chunks(cleaned):
            self._speak_queue.put(chunk)

        if self._speak_thread and self._speak_thread.is_alive():
            return

        self._speak_stop_event.clear()
        self._speak_thread = threading.Thread(
            target=self._speak_queue_loop,
            name="KerTTSQueue",
            daemon=True,
        )
        self._speak_thread.start()

    def stop_speaking(self) -> None:
        """Stop speaking and clear any queued text."""

        self._speak_stop_event.set()
        while not self._speak_queue.empty():
            try:
                self._speak_queue.get_nowait()
                self._speak_queue.task_done()
            except queue.Empty:
                break

    def _run_loop(self) -> None:
        if self.backend is None:
            return
        try:
            if self.on_transcript is None:
                return
            self.backend.listen_forever(self.on_transcript, self._stop_event)
        except Exception as exc:  # noqa: BLE001
            self._report_error(f"Voice backend failed: {exc}")
        finally:
            self._listening = False

    def _speak_blocking(self, text: str) -> None:
        if self.tts_backend is None:
            return
        try:
            self.tts_backend.speak(text)
        except Exception as exc:  # noqa: BLE001
            self._report_error(f"TTS backend failed: {exc}")

    def _speak_queue_loop(self) -> None:
        if self.tts_backend is None:
            return
        try:
            while not self._speak_stop_event.is_set():
                try:
                    chunk = self._speak_queue.get(timeout=0.2)
                except queue.Empty:
                    if self._speak_queue.empty():
                        break
                    continue
                try:
                    self.tts_backend.speak(chunk)
                finally:
                    self._speak_queue.task_done()
        except Exception as exc:  # noqa: BLE001
            self._report_error(f"TTS backend failed: {exc}")


def _split_into_chunks(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not buffer:
            buffer = sentence
        elif len(buffer) + len(sentence) + 1 <= 180:
            buffer = f"{buffer} {sentence}"
        else:
            chunks.append(buffer)
            buffer = sentence
    if buffer:
        chunks.append(buffer)
    return chunks


def _select_voice_id(
    voices: Iterable[object],
    preferred_name: str | None,
    preferred_gender: str | None,
) -> str | None:
    desired_gender = _normalize_gender(preferred_gender)
    preferred_name_lower = preferred_name.lower() if preferred_name else None

    best_score = 0
    best_id: str | None = None

    for voice in voices:
        voice_id = getattr(voice, "id", None)
        voice_name = getattr(voice, "name", "") or ""
        voice_gender = _normalize_gender(getattr(voice, "gender", None))

        score = 0
        if preferred_name_lower and preferred_name_lower in voice_name.lower():
            score += 3
        if desired_gender and voice_gender == desired_gender:
            score += 2
        if desired_gender == "female" and _looks_female_voice(voice_name):
            score += 1

        if score > best_score and voice_id:
            best_score = score
            best_id = voice_id

    return best_id


def _normalize_gender(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"female", "f", "woman", "women", "girl"}:
        return "female"
    if lowered in {"male", "m", "man", "boy"}:
        return "male"
    if lowered in {"neutral", "none"}:
        return "neutral"
    return lowered or None


def _looks_female_voice(name: str) -> bool:
    lowered = name.lower()
    return any(
        token in lowered
        for token in (
            "female",
            "zira",
            "susan",
            "samantha",
            "victoria",
            "karen",
            "hazel",
            "aria",
            "jenny",
        )
    )

    def _report_error(self, message: str) -> None:
        logger.warning("Voice input unavailable: %s", message)
        if self.on_error:
            self.on_error(message)
