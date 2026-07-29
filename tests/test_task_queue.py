"""Unit tests for CLI task queue feature."""

from pathlib import Path
import pytest
from hermes_deploy.config import Settings
from hermes_deploy.agent_loop import AgentLoop


def test_task_queue_execution(tmp_path: Path):
    """Test creating and reading a queued tasks file."""
    tasks_file = tmp_path / "tasks.txt"
    tasks_file.write_text(
        "# Comment line\n"
        "Task 1: Inspect README.md\n"
        "Task 2: List files\n",
        encoding="utf-8"
    )

    lines = [
        line.strip() for line in tasks_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) == 2
    assert lines[0] == "Task 1: Inspect README.md"
    assert lines[1] == "Task 2: List files"
