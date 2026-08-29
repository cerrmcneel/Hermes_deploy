"""Sandboxed file and command execution tools for Hermes Deploy.

All filesystem operations and shell commands are strictly bounded to the workspace directory.
"""

import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

from hermes_deploy.config import Settings, get_settings


class ToolExecutionError(Exception):
    """Raised when tool execution fails or violates safety bounds."""
    pass


class SandboxedToolSet:
    """Tool set whose filesystem operations are confined to workspace_dir."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.workspace_dir = self.settings.workspace_dir.resolve()

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative or absolute path ensuring it stays within workspace_dir.

        Args:
            relative_path: The target path.

        Returns:
            Resolved absolute Path object within workspace_dir.

        Raises:
            PermissionError: If target path escapes workspace_dir.
        """
        target = (self.workspace_dir / relative_path).resolve()
        try:
            target.relative_to(self.workspace_dir)
        except ValueError:
            raise PermissionError(
                f"Path access denied: '{relative_path}' points outside workspace bound '{self.workspace_dir}'"
            )
        return target

    def read_file(self, file_path: str) -> str:
        """Read text contents of a file within workspace.

        Args:
            file_path: Path relative to workspace.

        Returns:
            File content string.
        """
        resolved = self.resolve_path(file_path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not resolved.is_file():
            raise IsADirectoryError(f"Path is a directory, not a file: {file_path}")
        return resolved.read_text(encoding="utf-8")

    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file within workspace, creating directories if needed.

        Args:
            file_path: Path relative to workspace.
            content: Text content to write.

        Returns:
            Success confirmation string.
        """
        resolved = self.resolve_path(file_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {file_path}"

    def list_directory(self, dir_path: str = ".") -> str:
        """List contents of a directory within workspace.

        Args:
            dir_path: Directory path relative to workspace.

        Returns:
            Formatted string of directory contents.
        """
        resolved = self.resolve_path(dir_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {dir_path}")

        entries = sorted(resolved.iterdir())
        lines = []
        for entry in entries:
            kind = "DIR " if entry.is_dir() else "FILE"
            rel = entry.relative_to(self.workspace_dir)
            lines.append(f"[{kind}] {rel}")
        return "\n".join(lines) if lines else "(Empty directory)"

    #: Longest a single command may run. 300s, not 60s: the system prompt explicitly
    #: authorises `pip install`, which routinely exceeds a minute, and the model has no
    #: way to raise this - `timeout` is not in the tool schema. A 60s ceiling therefore
    #: guaranteed a tool failure on an action the agent was told it could take.
    DEFAULT_TIMEOUT_S = 300.0

    #: Cap on captured output. Applied ONLY here, never to read_file: truncating a file
    #: read would make the model write back a partial file. A failing pytest traceback,
    #: by contrast, can be tens of thousands of characters and evict the task prompt.
    MAX_OUTPUT_CHARS = 12_000

    def _format_output(self, stdout: str, stderr: str, returncode: int) -> str:
        """Shared stdout/stderr/exit-code formatting, truncated to MAX_OUTPUT_CHARS
        keeping the tail. Used by exec_command and both delegation tools so all three
        subprocess-backed tools read identically to the model."""
        output = stdout
        if stderr:
            output += f"\n[stderr]\n{stderr}"
        output += f"\n[exit_code: {returncode}]"
        if len(output) > self.MAX_OUTPUT_CHARS:
            # Keep the tail: for a failing test run the summary and the assertion
            # are at the end, and the head is usually collection noise.
            kept = output[-self.MAX_OUTPUT_CHARS:]
            dropped = len(output) - self.MAX_OUTPUT_CHARS
            output = f"[truncated: {dropped} earlier chars omitted]\n{kept}"
        return output

    def exec_command(self, command: str, timeout: float | None = None) -> str:
        """Execute a shell command inside workspace_dir.

        Args:
            command: Command string to execute.
            timeout: Timeout in seconds. Defaults to DEFAULT_TIMEOUT_S.

        Returns:
            Combined stdout and stderr, truncated to MAX_OUTPUT_CHARS.
        """
        timeout = self.DEFAULT_TIMEOUT_S if timeout is None else timeout
        # Block obvious system-wide destructive patterns in local execution mode
        forbidden_patterns = [
            "rm -rf /", "rmdir /s /q c:", "rmdir /s /q c:\\", "del /f /s /q c:", "del /f /s /q c:\\",
            "format c:", "mkfs", "dd if="
        ]
        cmd_lower = command.lower()
        for pattern in forbidden_patterns:
            if pattern in cmd_lower:
                return f"Error: Command execution blocked for safety reasons: dangerous command pattern '{pattern}' detected."

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return self._format_output(result.stdout, result.stderr, result.returncode)
        except subprocess.TimeoutExpired:
            return (
                f"Error: Command timed out after {timeout} seconds. Do not simply retry - "
                "either the command needs longer than a tool call allows, or it is waiting "
                "on input. Report it instead."
            )
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def ask_claude(self, prompt: str) -> str:
        """Delegate a prompt to Claude via the `claude` CLI, billed against the
        operator's Claude subscription rather than a metered API key.

        Verified 2026-08-29 through this exact method: `claude setup-token` prints a
        long-lived token but does not persist it anywhere, and a shell `export` in one
        terminal does not reach this process - so the token is read from
        settings.claude_token_file at call time and injected into the subprocess's own
        environment, rather than assumed to already be set.

        shell=True is required on Windows: npm installs `claude` as a `.cmd`/`.ps1`
        shim, and subprocess.run with an argv list and shell=False cannot exec those
        directly - it fails with FileNotFoundError even though the command works fine
        from any real shell. The prompt is passed via stdin (`input=`), never embedded
        in the shell-parsed command string, so shell=True does not turn arbitrary
        model-authored prompt text into a shell-injection surface.

        Args:
            prompt: The prompt or question to send to Claude.

        Returns:
            Claude's text response, or an error string. Never raises.
        """
        env = dict(os.environ)
        token_file = self.settings.claude_token_file
        if token_file.is_file():
            token = token_file.read_text(encoding="utf-8").strip()
            if token:
                env["CLAUDE_CODE_OAUTH_TOKEN"] = token
        try:
            result = subprocess.run(
                "claude -p",
                shell=True,
                input=prompt,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=self.settings.ask_claude_timeout_s,
                env=env,
            )
            return self._format_output(result.stdout, result.stderr, result.returncode)
        except subprocess.TimeoutExpired:
            return f"Error: claude did not respond within {self.settings.ask_claude_timeout_s} seconds."
        except FileNotFoundError:
            return "Error: the `claude` CLI is not installed or not on PATH."
        except Exception as e:
            return f"Error calling claude: {str(e)}"

    def ask_antigravity(self, prompt: str) -> str:
        """Delegate a prompt to Gemini via Antigravity's `agy` CLI.

        Not `antigravity-ide.exe chat` - that opens a GUI window and has no scriptable
        output. `agy -p` is the real headless path, verified 2026-08-29.

        The model is pinned to settings.antigravity_model (operator decision, not a tool
        parameter): `agy --model` also reaches non-Gemini models such as
        claude-sonnet-4-6, and exposing that choice here would make this tool a second,
        overlapping path to Claude alongside ask_claude. This tool is Gemini-only by
        construction - there is no parameter that could route it anywhere else.

        `--model` MUST come before `-p`: `agy` parses `-p` as consuming the next token
        as its value, so `-p --model X` misreads `--model` itself as the prompt and
        drops X and the real prompt entirely - confirmed via agy's own error message.
        `agy.exe` is a real native binary (its own --help banner says so), unlike
        `claude`'s npm shim, so shell=False with an argv list works directly here - no
        stdin workaround needed the way ask_claude requires one.

        Args:
            prompt: The prompt or question to send to Gemini.

        Returns:
            Gemini's text response, or an error string. Never raises.
        """
        try:
            result = subprocess.run(
                ["agy", "--model", self.settings.antigravity_model, "-p", prompt],
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=self.settings.ask_antigravity_timeout_s,
            )
            return self._format_output(result.stdout, result.stderr, result.returncode)
        except subprocess.TimeoutExpired:
            return f"Error: agy did not respond within {self.settings.ask_antigravity_timeout_s} seconds."
        except FileNotFoundError:
            return "Error: the `agy` CLI is not installed or not on PATH."
        except Exception as e:
            return f"Error calling agy: {str(e)}"

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch tool call by name.

        Args:
            name: Tool function name.
            arguments: Tool call arguments dict.

        Returns:
            String output from tool execution.
        """
        if name == "read_file":
            return self.read_file(arguments["file_path"])
        elif name == "write_file":
            return self.write_file(arguments["file_path"], arguments["content"])
        elif name == "list_directory":
            return self.list_directory(arguments.get("dir_path", "."))
        elif name == "exec_command":
            return self.exec_command(arguments["command"])
        elif name == "ask_claude":
            return self.ask_claude(arguments["prompt"])
        elif name == "ask_antigravity":
            return self.ask_antigravity(arguments["prompt"])
        else:
            raise ToolExecutionError(f"Unknown tool name: '{name}'")


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return JSON schemas for Ollama function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read text contents of a file relative to workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative path to file."}
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write text contents to a file relative to workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative path to file."},
                        "content": {"type": "string", "description": "Content string."}
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files and directories within a workspace directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dir_path": {"type": "string", "description": "Relative directory path. Default is '.'"}
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "Execute a shell command inside the workspace directory (e.g. running pytest, executing scripts, or installing missing packages via pip/uv).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command line."}
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_claude",
                "description": "Delegate a prompt to Claude (Anthropic), billed against the operator's Claude subscription. Use when a task explicitly calls for Claude specifically, a second opinion, or a capability better suited to a different model - not as a substitute for your own reasoning on ordinary tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The prompt or question to send to Claude."}
                    },
                    "required": ["prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_antigravity",
                "description": "Delegate a prompt to Gemini via Antigravity. Use when a task explicitly calls for Gemini specifically, a second opinion, or a capability better suited to a different model - not as a substitute for your own reasoning on ordinary tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The prompt or question to send to Gemini."}
                    },
                    "required": ["prompt"],
                },
            },
        },
    ]
