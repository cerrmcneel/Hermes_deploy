"""Unit tests for Ollama client."""

from unittest.mock import MagicMock, patch
import pytest

from hermes_deploy.config import Settings
from hermes_deploy.ollama_client import OllamaClient


@patch("httpx.Client.post")
def test_ollama_empty_response_invariant_12(mock_post: MagicMock):
    """Ensure empty response without tool calls raises ValueError (Invariant 12)."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": "", "tool_calls": None},
        "done": True,
    }
    mock_post.return_value = mock_resp

    client = OllamaClient(settings=Settings(thinking_mode=False))

    with pytest.raises(ValueError, match="Ollama returned empty response"):
        client.chat(messages=[{"role": "user", "content": "hi"}])
