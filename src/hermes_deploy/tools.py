"""Sandboxed file and command execution tools for Hermes Deploy.

All filesystem operations and shell commands are strictly bounded to the workspace directory.
"""

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

    def exec_command(self, command: str, timeout: float = 60.0) -> str:
        """Execute a shell command inside workspace_dir.

        Args:
            command: Command string to execute.
            timeout: Timeout in seconds.

        Returns:
            Combined stdout and stderr.
        """
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
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            output += f"\n[exit_code: {result.returncode}]"
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds."
        except Exception as e:
            return f"Error executing command: {str(e)}"

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
                "description": "Execute a shell command inside the workspace directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command line."}
                    },
                    "required": ["command"],
                },
            },
        },
    ]
