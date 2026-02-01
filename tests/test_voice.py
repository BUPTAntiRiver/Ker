"""Tests for voice utilities."""

from dataclasses import dataclass

from ker.voice.voice_io import _select_voice_id, _split_into_chunks


@dataclass
class DummyVoice:
    id: str
    name: str
    gender: str | None = None


def test_split_into_chunks_keeps_short_sentence() -> None:
    text = "Hello world."
    assert _split_into_chunks(text) == ["Hello world."]


def test_split_into_chunks_splits_long_text() -> None:
    text = " ".join(["Sentence one.", "Sentence two.", "Sentence three."])
    chunks = _split_into_chunks(text)
    assert len(chunks) >= 1
    assert all(chunk.endswith(".") for chunk in chunks)


def test_select_voice_prefers_named_voice() -> None:
    voices = [
        DummyVoice(id="1", name="Generic Male", gender="male"),
        DummyVoice(id="2", name="Aria Pleasant", gender="female"),
    ]
    assert _select_voice_id(voices, preferred_name="Aria", preferred_gender=None) == "2"


def test_select_voice_prefers_female_when_requested() -> None:
    voices = [
        DummyVoice(id="1", name="Generic Male", gender="male"),
        DummyVoice(id="2", name="Kind Voice", gender="female"),
    ]
    assert _select_voice_id(voices, preferred_name=None, preferred_gender="female") == "2"
