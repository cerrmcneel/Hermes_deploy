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
    #: devstral:24b, benchmarked against gemma4:12b and qwen3-coder:30b on this machine
    #: with Hermes's real tool schemas (scripts/bench_models.py):
    #:
    #:   model             gen tok/s   tool probes   verdict
    #:   gemma4:12b             36.1         1 / 4   fast and wrong - read a file by
    #:                                               calling list_directory, invented
    #:                                               `unittest discover` when told pytest
    #:   gemma4:26b             26.5         1 / 4   the MoE (4B active). Fast, honest,
    #:                                               and wrong in EXACTLY the same three
    #:                                               ways as its 12B sibling - see below
    #:   qwen3-coder:30b        21.2         2 / 4   DANGEROUS - emits `<function=...>` as
    #:                                               prose, so the loop reads "no tool
    #:                                               calls" as SUCCESS and reports work it
    #:                                               never did. This one is OLLAMA's bug,
    #:                                               not the model's - see below
    #:   devstral:24b            3.4         4 / 4   slow and correct
    #:
    #: DO NOT re-try a larger gemma4 hoping size fixes tool selection. It was tried.
    #: gemma4:26b makes the identical wrong choices as gemma4:12b, and re-running the
    #: three failing probes 5x under each of {default temp 1, temp 0, temp 0 + thinking
    #: on} scored 0/45 - never once correct, and never once varying:
    #:
    #:   "Read the file AGENTS.md"                    -> list_directory  5/5/5
    #:   "you must use .venv/Scripts/python.exe"      -> list_directory  5/5/5
    #:   "Create a file at .../test_probe.py"         -> exec_command    5/5/5
    #:
    #: That determinism is the finding. This is not a capability gap that more parameters
    #: close, it is a fixed behavioural prior in the gemma4 tool post-training: explore
    #: the directory first, regardless of an instruction naming the exact file. gemma4
    #: remains fine for the harness's VISION agents, which use plain completion and never
    #: pass `tools` - the defect only exists where tool selection matters.
    #:
    #: qwen3-coder's failure is NOT a model defect and NOT fixable from here. Measured on
    #: Ollama 0.32.5, temperature 0, 5 samples per cell, sweeping the write_file probe by
    #: how many tools were offered:
    #:
    #:   1 tool  (write_file)        5/5   native JSON tool_call, parsed correctly
    #:   2 tools (write, read)       0/5   `<function=...>` in content
    #:   3 tools                     0/5   `<function=...>` in content
    #:   4 tools (Hermes's set)      0/5   `<function=...>` in content
    #:
    #: Above ONE tool the model switches to its native XML call format, and Ollama's
    #: compiled `PARSER qwen3-coder` only handles the JSON one. The model is emitting a
    #: well-formed call; the runtime drops it. So there is no schema-trimming workaround -
    #: an agent with one tool is not an agent - and no Modelfile fix either, because
    #: `ollama show --modelfile qwen3-coder:30b` is `TEMPLATE {{ .Prompt }}` with the
    #: logic compiled into Ollama.
    #:
    #: This is the ONE failure here that a different runtime could plausibly fix:
    #: llama.cpp's llama-server carries its own qwen-family tool parsers and can load the
    #: model's own template with --jinja. Retest before adopting; do not assume.
    #:
    #: Correctness over speed was the explicit instruction. devstral is a dense 24B at
    #: ~51% GPU on a 12GB card, so it is genuinely slow; see ollama_timeout_s.
    model_id: str = Field(
        default_factory=lambda: os.getenv("MODEL_ID", "devstral:24b")
    )
    #: HTTP timeout for one /api/chat call. 600s, not 120s.
    #: A turn that calls write_file emits the ENTIRE FILE as output tokens, and
    #: generation is the slow half - measured on gemma4:12b / RTX 3060 at 29.4 tok/s
    #: against ~2000 tok/s for prefill. So a 4000-token file write takes ~136s and blew
    #: the old fixed 120s ceiling; that is exactly how the "add tests to
    #: test_hitl_server.py" task died with "Ollama client error: timed out" mid-write.
    #: 600s covers roughly a 17,000-token generation, far past any sane single write.
    #: 1800s. devstral generates at ~3.4 tok/s (dense 24B, half its layers in RAM), so a
    #: 4000-token file write needs ~20 minutes. 600s was sized for gemma4's 29 tok/s and
    #: would cut devstral off mid-write. Slow is acceptable here; a truncated write is not.
    ollama_timeout_s: float = Field(
        default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT_S", "1800"))
    )
    #: 32768, not 8192. The loop appends every tool result to `messages` and never
    #: trims, so a single read_file of a 16KB source file was ~4000 tokens - half of an
    #: 8192 window. Ollama then silently drops the OLDEST messages, which is the task
    #: prompt, and the model wanders or repeats calls until the turn limit. Raising
    #: max_turns without this just buys more thrashing.
    #: Measured on gemma4:12b / RTX 3060: 8192 -> 7986MB resident, 32768 -> 7718MB.
    #: Four times the window for roughly 116 MiB of GPU, because gemma4's attention
    #: makes KV cache cheap. There is no reason to run this model at 8192.
    num_ctx: int = Field(
        default_factory=lambda: int(os.getenv("NUM_CTX", "32768"))
    )
    thinking_mode: bool = Field(
        default_factory=lambda: os.getenv("THINKING_MODE", "false").lower() in ("true", "1", "yes")
    )
    stuck_loop_threshold: int = Field(
        default_factory=lambda: int(os.getenv("STUCK_LOOP_THRESHOLD", "3"))
    )
    #: A real task is: read the guardrails, read two or three source files, write a
    #: module, write tests, run pytest, fix what failed, re-run. That is comfortably 25+
    #: turns before anything goes wrong. 50 leaves room for one repair cycle without
    #: making a genuinely stuck run expensive - the stuck-loop breaker, not this limit,
    #: is what should stop a spinning agent.
    max_turns: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TURNS", "50"))
    )

    #: Delegation to `claude -p`, verified 2026-08-29 through the real exec_command path.
    #: `claude setup-token` prints a token but does not persist it anywhere - it has to be
    #: read from a file at call time rather than assumed to already be in the environment,
    #: since a shell `export` in one terminal does not reach a separately-launched process.
    #: Default path is outside every repo on purpose: never let a token land somewhere
    #: `git add -A` could pick it up.
    claude_token_file: Path = Field(
        default_factory=lambda: Path(
            os.getenv("CLAUDE_TOKEN_FILE", str(Path.home() / ".claude_code_token"))
        )
    )
    ask_claude_timeout_s: float = Field(
        default_factory=lambda: float(os.getenv("ASK_CLAUDE_TIMEOUT_S", "120"))
    )

    #: Delegation to `agy -p`, Antigravity's real headless CLI - distinct from
    #: `antigravity-ide.exe chat`, which opens a GUI window and has no scriptable output.
    #: Pinned to a single Gemini model on purpose (operator decision, 2026-08-29): the
    #: model is hardcoded in ask_antigravity rather than exposed as a tool parameter, so
    #: Hermes cannot pick `claude-sonnet-4-6` or another non-Gemini model `agy` also
    #: serves - `ask_claude` is the only path to Claude, keeping the two tools' purposes
    #: from overlapping. Needs no token file: `agy` reads Antigravity's existing desktop
    #: login.
    #: gemini-3.7-flash-high, verified live 2026-08-29 (it names itself correctly when
    #: asked). 3.7 is the newest Flash `agy models` offers and is the model the operator
    #: runs as the delegated worker agent in the harness repo, so Hermes's delegation
    #: path and the worker agent are deliberately the same model - one thing to reason
    #: about, not two.
    antigravity_model: str = Field(
        default_factory=lambda: os.getenv("ANTIGRAVITY_MODEL", "gemini-3.7-flash-high")
    )
    ask_antigravity_timeout_s: float = Field(
        default_factory=lambda: float(os.getenv("ASK_ANTIGRAVITY_TIMEOUT_S", "120"))
    )


def get_settings() -> Settings:
    """Get loaded settings singleton."""
    return Settings()
