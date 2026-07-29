"""Main autonomous task loop runner for Hermes Deploy."""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from hermes_deploy.config import Settings, get_settings
from hermes_deploy.harness_client import HarnessClient
from hermes_deploy.ollama_client import OllamaClient
from hermes_deploy.stuck_loop import StuckLoopDetector, StuckLoopException
from hermes_deploy.tools import SandboxedToolSet, get_tool_schemas

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Hermes Agent, a containerized autonomous coding assistant operating inside a sandboxed workspace environment.

Rules:
1. Always use available tools (read_file, write_file, list_directory, exec_command) to inspect and modify files.
2. Operations outside the workspace directory are strictly forbidden.
3. Be concise and precise. Test your changes using exec_command when relevant.
4. When finished, provide a concise summary of accomplishments.
"""


class LoopResult(BaseModel):
    """Execution result from agent loop."""
    run_id: str
    turns_taken: int
    completed: bool
    final_output: str
    stuck_loop_detected: bool = False
    error: Optional[str] = None


class AgentLoop:
    """Coordinates model inference, tool execution, stuck loop checks, and HITL fallback."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.ollama_client = OllamaClient(self.settings)
        self.harness_client = HarnessClient(self.settings)
        self.tools = SandboxedToolSet(self.settings)
        self.stuck_detector = StuckLoopDetector(settings=self.settings)

    def run(self, prompt: str, run_id: Optional[str] = None) -> LoopResult:
        """Run autonomous task loop for a given prompt/goal.

        Args:
            prompt: Task instruction or goal description.
            run_id: Optional run ID for tracing.

        Returns:
            LoopResult detailing outcome.
        """
        run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tool_schemas = get_tool_schemas()
        self.stuck_detector.reset()

        turns = 0
        while turns < self.settings.max_turns:
            turns += 1
            logger.info(f"[{run_id}] Turn {turns}/{self.settings.max_turns}")

            try:
                response = self.ollama_client.chat(messages=messages, tools=tool_schemas)
            except Exception as e:
                logger.error(f"[{run_id}] Ollama client error: {e}")
                return LoopResult(
                    run_id=run_id,
                    turns_taken=turns,
                    completed=False,
                    final_output="",
                    error=str(e),
                )

            msg = response.message
            assistant_dict: Dict[str, Any] = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_dict["tool_calls"] = msg.tool_calls
            messages.append(assistant_dict)

            # If no tool calls, model finished task
            if not msg.tool_calls:
                logger.info(f"[{run_id}] Task completed by model.")
                return LoopResult(
                    run_id=run_id,
                    turns_taken=turns,
                    completed=True,
                    final_output=msg.content,
                )

            # Execute tool calls
            for tool_call in msg.tool_calls:
                func_data = tool_call.get("function", {})
                name = func_data.get("name", "")
                args = func_data.get("arguments", {})

                # Ensure args is dict if string
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}

                logger.info(f"[{run_id}] Tool call: {name}({args})")
                try:
                    tool_result = self.tools.execute_tool(name, args)
                except Exception as te:
                    tool_result = f"Tool Error: {str(te)}"

                # Record turn in stuck-loop detector
                if self.stuck_detector.record_turn(name, args, tool_result):
                    logger.warning(f"[{run_id}] Stuck loop detected on tool {name}!")
                    # Escalating to Harness HITL queue
                    self.harness_client.post_ask(
                        run_id=run_id,
                        question=f"Stuck loop circuit breaker tripped! Tool '{name}' repeated without progress.",
                        context={"tool": name, "arguments": args, "result": tool_result},
                    )
                    return LoopResult(
                        run_id=run_id,
                        turns_taken=turns,
                        completed=False,
                        final_output="",
                        stuck_loop_detected=True,
                        error=f"Circuit breaker tripped: repeated identical calls for tool '{name}'.",
                    )

                # Append tool response message
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                })

        return LoopResult(
            run_id=run_id,
            turns_taken=turns,
            completed=False,
            final_output="",
            error=f"Reached maximum turn limit ({self.settings.max_turns}).",
        )
