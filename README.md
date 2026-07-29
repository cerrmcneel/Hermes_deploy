# Hermes Deploy

Containerized local agent-loop deployment for Hermes, operating as an autonomous execution engine paired with `agentic_HITL_harness`.

## Overview

`Hermes_deploy` runs as an independent, sandboxed container loop (`Option B+`). It connects to the host's Ollama instance over `host.docker.internal:11434` for model inference (`gemma4:12b`) and posts Human-in-the-loop (HITL) clarification cards to `agentic_HITL_harness` at `host.docker.internal:8000`.

## Architecture Features

- **Native Ollama Client**: Custom HTTP transport passing `num_ctx: 8192` and native reasoning controls (`think: false`).
- **Stuck-Loop Detection**: Identifies repetitive tool call signatures and no-op workspace edits, firing hard circuit breakers before burning token/time budgets.
- **Strict Scope Isolation**: Bounded file execution within `/workspace:rw`. Immutable read-only mounts for harness documentation/schemas (`/harness_readonly:ro`).
- **HITL Integration**: Direct posting of `HumanInteractionEvent` cards to the Harness queue when human approval or intervention is required.

## Running Locally

```bash
# Set up Python environment
uv venv
uv pip install -e .

# Run CLI
uv run hermes-loop --prompt "Refactor src/hermes_deploy/config.py"
```

## Running Containerized

```bash
docker compose up --build
```
