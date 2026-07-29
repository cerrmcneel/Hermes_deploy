"""Stuck loop detection and circuit breaker for Hermes Deploy.

Prevents infinite loops by computing hashes of tool calls and outputs, triggering an
exception or HITL escalation when repeated identical operations occur.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from hermes_deploy.config import Settings, get_settings


class ToolInvocation(BaseModel):
    """Snapshot signature of a tool execution turn."""
    tool_name: str
    arguments_hash: str
    result_hash: str


class StuckLoopException(Exception):
    """Raised when a stuck loop circuit breaker trips."""
    pass


class StuckLoopDetector:
    """Sliding window detector for repetitive tool execution signatures."""

    def __init__(self, threshold: Optional[int] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.threshold = threshold or self.settings.stuck_loop_threshold
        self.history: List[ToolInvocation] = []

    @staticmethod
    def hash_dict(d: Dict[str, Any]) -> str:
        """Compute SHA256 hash of a dictionary."""
        serialized = json.dumps(d, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def hash_str(s: str) -> str:
        """Compute SHA256 hash of a string."""
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def record_turn(self, tool_name: str, arguments: Dict[str, Any], result: str) -> bool:
        """Record a tool execution turn and check if stuck threshold is reached.

        Args:
            tool_name: Executed tool name.
            arguments: Tool arguments.
            result: Execution result output string.

        Returns:
            True if stuck loop threshold is breached, False otherwise.
        """
        args_h = self.hash_dict(arguments)
        res_h = self.hash_str(result)

        inv = ToolInvocation(
            tool_name=tool_name,
            arguments_hash=args_h,
            result_hash=res_h,
        )
        self.history.append(inv)

        if len(self.history) < self.threshold:
            return False

        # Check last N invocations
        recent = self.history[-self.threshold:]

        # Case 1: Identical tool name, arguments, and result across threshold turns
        first = recent[0]
        if all(
            item.tool_name == first.tool_name
            and item.arguments_hash == first.arguments_hash
            and item.result_hash == first.result_hash
            for item in recent
        ):
            return True

        return False

    def reset(self) -> None:
        """Reset history window."""
        self.history.clear()
