"""Tests for Ker configuration loading."""

import os

from ker.config import KerConfig


def test_config_defaults(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("KER_MODEL", raising=False)
    monkeypatch.delenv("KER_TEMPERATURE", raising=False)

    config = KerConfig.from_env()
    assert config.deepseek_api_key is None
    assert config.deepseek_base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-chat"
    assert config.temperature == 0.4


def test_config_overrides(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.com")
    monkeypatch.setenv("KER_MODEL", "model")
    monkeypatch.setenv("KER_TEMPERATURE", "0.8")

    config = KerConfig.from_env()
    assert config.deepseek_api_key == "key"
    assert config.deepseek_base_url == "https://example.com"
    assert config.model == "model"
    assert config.temperature == 0.8
