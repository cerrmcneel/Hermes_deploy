"""Unit tests for sandboxed tool execution."""

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
