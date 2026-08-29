"""Unit tests for sandboxed tool execution."""

import subprocess
from unittest.mock import patch

import pytest
from pathlib import Path

from hermes_deploy.config import Settings
from hermes_deploy.tools import SandboxedToolSet


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> SandboxedToolSet:
    """Fixture providing SandboxedToolSet pointing to temporary workspace."""
    settings = Settings(workspace_dir=tmp_path)
    return SandboxedToolSet(settings=settings)


def test_path_escape_prevention(tmp_workspace: SandboxedToolSet):
    """Ensure accessing paths outside workspace raises PermissionError."""
    with pytest.raises(PermissionError):
        tmp_workspace.resolve_path("../secret.txt")

    with pytest.raises(PermissionError):
        tmp_workspace.resolve_path("/etc/passwd")


def test_write_and_read_file(tmp_workspace: SandboxedToolSet):
    """Test reading and writing files within workspace."""
    tmp_workspace.write_file("sub/test.txt", "Hello World")
    content = tmp_workspace.read_file("sub/test.txt")
    assert content == "Hello World"


def test_list_directory(tmp_workspace: SandboxedToolSet):
    """Test listing directory contents."""
    tmp_workspace.write_file("file1.txt", "1")
    tmp_workspace.write_file("dir1/file2.txt", "2")

    listing = tmp_workspace.list_directory(".")
    assert "[FILE] file1.txt" in listing
    assert "[DIR ] dir1" in listing


def test_exec_command(tmp_workspace: SandboxedToolSet):
    """Test executing a shell command in workspace."""
    out = tmp_workspace.exec_command("python -c \"print('test output')\"")
    assert "test output" in out
    assert "[exit_code: 0]" in out


def test_exec_command_dangerous_guard(tmp_workspace: SandboxedToolSet):
    """Test blocking dangerous system commands."""
    out = tmp_workspace.exec_command("rmdir /s /q C:\\")
    assert "Command execution blocked for safety reasons" in out


def _fake_result(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ── ask_claude ────────────────────────────────────────────────────────────

def test_ask_claude_reads_token_from_file_and_injects_it(tmp_workspace: SandboxedToolSet, tmp_path: Path):
    """The token from `claude setup-token` is never in ambient env - it must be read
    from the configured file and injected into the subprocess's own environment."""
    token_file = tmp_path / "token.txt"
    token_file.write_text("secret-token-value\n", encoding="utf-8")
    tmp_workspace.settings.claude_token_file = token_file

    with patch("hermes_deploy.tools.subprocess.run") as mock_run:
        mock_run.return_value = _fake_result(stdout="Four.")
        out = tmp_workspace.ask_claude("what is 2+2?")

    assert "Four." in out
    assert "[exit_code: 0]" in out
    call_args, call_kwargs = mock_run.call_args
    # shell=True + input=, not an argv list: npm installs `claude` as a .cmd/.ps1 shim,
    # which subprocess.run cannot exec directly with shell=False - confirmed live.
    assert call_args[0] == "claude -p"
    assert call_kwargs["shell"] is True
    assert call_kwargs["input"] == "what is 2+2?"
    assert call_kwargs["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "secret-token-value"


def test_ask_claude_without_token_file_leaves_env_untouched(
    tmp_workspace: SandboxedToolSet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A missing token file must not crash the call or inject a bogus credential -
    it falls through to whatever the host process's own environment already has."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    tmp_workspace.settings.claude_token_file = tmp_path / "does_not_exist.txt"

    with patch("hermes_deploy.tools.subprocess.run") as mock_run:
        mock_run.return_value = _fake_result(stdout="ok")
        out = tmp_workspace.ask_claude("hello")

    assert "ok" in out
    _, call_kwargs = mock_run.call_args
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in call_kwargs["env"]


def test_ask_claude_timeout_reports_clearly(tmp_workspace: SandboxedToolSet, tmp_path: Path):
    tmp_workspace.settings.claude_token_file = tmp_path / "does_not_exist.txt"
    with patch("hermes_deploy.tools.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)
        out = tmp_workspace.ask_claude("hello")
    assert "did not respond within" in out


def test_ask_claude_missing_cli_reports_clearly(tmp_workspace: SandboxedToolSet, tmp_path: Path):
    tmp_workspace.settings.claude_token_file = tmp_path / "does_not_exist.txt"
    with patch("hermes_deploy.tools.subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()
        out = tmp_workspace.ask_claude("hello")
    assert "not installed" in out


# ── ask_antigravity ──────────────────────────────────────────────────────

def test_ask_antigravity_pins_configured_model(tmp_workspace: SandboxedToolSet):
    """The model is an operator setting, never something the call site chooses.
    --model must precede -p: agy parses -p as consuming the very next token as its
    value, so -p before --model silently eats the flag instead of the prompt -
    confirmed live via agy's own error message."""
    tmp_workspace.settings.antigravity_model = "gemini-3.7-flash-high"
    with patch("hermes_deploy.tools.subprocess.run") as mock_run:
        mock_run.return_value = _fake_result(stdout="hi")
        tmp_workspace.ask_antigravity("hello")
    call_args, _ = mock_run.call_args
    assert call_args[0] == ["agy", "--model", "gemini-3.7-flash-high", "-p", "hello"]


def test_ask_antigravity_schema_has_no_model_parameter():
    """`agy --model` also reaches non-Gemini models including claude-sonnet-4-6. If a
    model parameter were ever exposed here, this tool would become a second, overlapping
    path to Claude alongside ask_claude - the whole point of pinning it in config.py."""
    from hermes_deploy.tools import get_tool_schemas

    schema = next(s for s in get_tool_schemas() if s["function"]["name"] == "ask_antigravity")
    props = schema["function"]["parameters"]["properties"]
    assert "model" not in props


def test_ask_antigravity_missing_cli_reports_clearly(tmp_workspace: SandboxedToolSet):
    with patch("hermes_deploy.tools.subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()
        out = tmp_workspace.ask_antigravity("hello")
    assert "not installed" in out


def test_execute_tool_dispatches_delegation_tools(tmp_workspace: SandboxedToolSet, tmp_path: Path):
    """Exercise the real dispatch path agent_loop uses, not just the bound methods."""
    tmp_workspace.settings.claude_token_file = tmp_path / "does_not_exist.txt"
    with patch("hermes_deploy.tools.subprocess.run") as mock_run:
        mock_run.return_value = _fake_result(stdout="hi")
        out_claude = tmp_workspace.execute_tool("ask_claude", {"prompt": "x"})
        out_antigravity = tmp_workspace.execute_tool("ask_antigravity", {"prompt": "y"})
    assert "hi" in out_claude
    assert "hi" in out_antigravity
