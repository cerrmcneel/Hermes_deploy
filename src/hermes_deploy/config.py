"""Configuration management for Hermes Deploy."""

import os
from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Configuration settings for the agent loop."""

    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    harness_base_url: str = Field(
        default_factory=lambda: os.getenv("HARNESS_BASE_URL", "http://127.0.0.1:8000")
    )
    workspace_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("WORKSPACE_DIR", os.getcwd())).resolve()
    )
    model_id: str = Field(
        default_factory=lambda: os.getenv("MODEL_ID", "gemma4:12b")
    )
    num_ctx: int = Field(
        default_factory=lambda: int(os.getenv("NUM_CTX", "8192"))
    )
    thinking_mode: bool = Field(
        default_factory=lambda: os.getenv("THINKING_MODE", "false").lower() in ("true", "1", "yes")
    )
    stuck_loop_threshold: int = Field(
        default_factory=lambda: int(os.getenv("STUCK_LOOP_THRESHOLD", "3"))
    )
    max_turns: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TURNS", "30"))
    )


def get_settings() -> Settings:
    """Get loaded settings singleton."""
    return Settings()
