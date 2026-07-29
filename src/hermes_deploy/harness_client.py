"""Client for posting HITL clarification events to agentic_HITL_harness."""

import logging
from typing import Any, Dict, Optional
import httpx
from pydantic import BaseModel, Field

from hermes_deploy.config import Settings, get_settings

logger = logging.getLogger(__name__)


class HumanInteractionAsk(BaseModel):
    """Payload for asking human operator intervention via the Harness."""
    run_id: str
    question: str
    context: Dict[str, Any] = Field(default_factory=dict)
    options: Optional[list[str]] = None


class HarnessClient:
    """HTTP Client interacting with agentic_HITL_harness."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.harness_base_url.rstrip("/")

    def post_ask(
        self,
        run_id: str,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[list[str]] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Post a HumanInteractionEvent card to the harness queue.

        Args:
            run_id: Unique run ID for tracking.
            question: Prompt/question for operator feed.
            context: Additional metadata or stuck loop signature.
            options: Selectable option strings for the operator.
            timeout: HTTP timeout in seconds.

        Returns:
            API response payload.
        """
        payload = HumanInteractionAsk(
            run_id=run_id,
            question=question,
            context=context or {},
            options=options,
        ).model_dump(mode="json")

        url = f"{self.base_url}/api/ask"
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as e:
            logger.warning(
                f"Could not reach Harness at {url} ({e}). Logging ask locally."
            )
            return {
                "status": "offline_fallback",
                "message": "Harness unreachable, ask logged to console",
                "ask": payload,
            }
