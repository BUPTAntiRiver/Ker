"""Tests for voice utilities."""

from ker.voice.voice_io import _split_into_chunks


def test_split_into_chunks_keeps_short_sentence() -> None:
    text = "Hello world."
    assert _split_into_chunks(text) == ["Hello world."]


def test_split_into_chunks_splits_long_text() -> None:
    text = " ".join(["Sentence one.", "Sentence two.", "Sentence three."])
    chunks = _split_into_chunks(text)
    assert len(chunks) >= 1
    assert all(chunk.endswith(".") for chunk in chunks)
