"""Native Ollama client for Hermes Deploy.

Interacts directly with Ollama's native /api/chat endpoint to ensure full control over
parameters like num_ctx, think (reasoning control), and tool calling schemas.
"""

from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from hermes_deploy.config import Settings, get_settings


class ToolCall(BaseModel):
    """Parsed tool call from Ollama response."""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Message object for Ollama chat payload."""
    role: str
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None


class OllamaResponse(BaseModel):
    """Structured response from Ollama /api/chat."""
    message: ChatMessage
    done: bool = True
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None


class OllamaClient:
    """Client for Ollama /api/chat endpoint."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout: float = 120.0,
    ) -> OllamaResponse:
        """Send chat request to Ollama /api/chat endpoint.

        Args:
            messages: List of message dictionaries.
            tools: Optional list of tool definitions.
            timeout: Request timeout in seconds.

        Returns:
            OllamaResponse parsed object.

        Raises:
            ValueError: If empty content is returned without tool calls.
            httpx.HTTPError: On network or HTTP status failure.
        """
        payload: Dict[str, Any] = {
            "model": self.settings.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self.settings.num_ctx,
            },
        }

        # Native reasoning control for Gemma 4 / thinking models
        if not self.settings.thinking_mode:
            payload["think"] = False

        if tools:
            payload["tools"] = tools

        url = f"{self.base_url}/api/chat"
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        message_data = data.get("message", {})
        content = message_data.get("content", "")
        tool_calls = message_data.get("tool_calls")

        # Harness Invariant 12 guard: An empty model response is never a success
        if not content.strip() and not tool_calls:
            raise ValueError(
                f"Ollama returned empty response (content='' and tool_calls=None). "
                f"Ensure thinking_mode parameter is correctly set (current: {self.settings.thinking_mode})."
            )

        return OllamaResponse(
            message=ChatMessage(
                role=message_data.get("role", "assistant"),
                content=content,
                tool_calls=tool_calls,
            ),
            done=data.get("done", True),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
        )
