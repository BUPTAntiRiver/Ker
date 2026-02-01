"""Tests for Ker agent behavior."""

from ker.agent import KerAgent
from ker.llm.base import NullLLMClient


def test_agent_returns_empty_for_blank_input() -> None:
    agent = KerAgent(llm_client=NullLLMClient())
    assert agent.handle_user_message("   ") == ""


def test_agent_uses_llm_for_user_input() -> None:
    agent = KerAgent(llm_client=NullLLMClient())
    response = agent.handle_user_message("Hello")
    assert "LLM is not configured" in response
