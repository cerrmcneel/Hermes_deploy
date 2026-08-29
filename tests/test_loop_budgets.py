"""Guards on the loop's budgets.

Two runs failed on these values, and neither failure pointed at its own cause:

  * MAX_TURNS ran out. The visible symptom was the turn limit; the actual cause was
    NUM_CTX=8192. The loop appends every tool result to `messages` and never trims, so a
    single read_file of a 16KB source file was ~4000 tokens - half the window. Ollama then
    drops the OLDEST messages, which is the task prompt, and the model wanders.
  * exec_command timed out. The system prompt authorises `pip install`, which routinely
    takes longer than a minute, and `timeout` is not in the tool schema so the model
    cannot raise it. A 60s ceiling guaranteed failure on an action the agent was told to
    take.
"""

from hermes_deploy.config import Settings
from hermes_deploy.tools import SandboxedToolSet


def test_context_window_fits_more_than_one_source_file():
    """Two 16KB files plus a prompt is ~9000 tokens. 8192 could not hold it."""
    assert Settings().num_ctx >= 32768


def test_turn_budget_allows_a_repair_cycle():
    """Read guardrails, read files, write module, write tests, run pytest, fix, re-run."""
    assert Settings().max_turns >= 50


def test_chat_timeout_outlives_a_large_file_write():
    """A turn calling write_file emits the whole file as output tokens. Measured on
    gemma4:12b / RTX 3060: 29.4 tok/s generating, so a 4000-token write needs ~136s. The
    old fixed 120s ceiling killed the task that had to write a test file, mid-write."""
    assert Settings().ollama_timeout_s >= 300


def test_chat_uses_the_configured_timeout_not_a_hardcoded_one():
    """agent_loop calls chat() without a timeout argument, so the default must come from
    settings - otherwise the configured value is dead."""
    import inspect

    from hermes_deploy.ollama_client import OllamaClient

    assert inspect.signature(OllamaClient.chat).parameters["timeout"].default is None


def test_a_tool_call_written_as_prose_is_not_success():
    """The most dangerous model failure found while benchmarking.

    qwen3-coder:30b emits `<function=write_file> <parameter=...>` as plain text 4 times
    out of 6 instead of a structured tool call, and Ollama's template does not parse it.
    The loop's completion check is `if not msg.tool_calls: return completed=True` - so it
    reported tasks finished having written nothing at all.
    """
    from hermes_deploy.agent_loop import _unparsed_tool_call

    assert _unparsed_tool_call(
        "I'll create the file.  <function=write_file> <parameter=file_path> a.py"
    ) == "<function="
    assert _unparsed_tool_call("<tool_call>{\"name\": \"read_file\"}</tool_call>") == "<tool_call>"

    # Ordinary prose and ordinary code must not trip it.
    assert _unparsed_tool_call("I read the file and it defines three tests.") is None
    assert _unparsed_tool_call("Use `def f(x): return x < 3` for the filter.") is None
    assert _unparsed_tool_call(None) is None
    assert _unparsed_tool_call("") is None


def test_model_and_timeout_are_matched_to_each_other():
    """The chat timeout is sized from the configured model's measured throughput. Change
    the model without revisiting this and writes get truncated mid-file."""
    s = Settings()
    if s.model_id.startswith("devstral"):
        # ~3.4 tok/s measured: a 4000-token write needs ~20 minutes.
        assert s.ollama_timeout_s >= 1200


def test_the_agent_model_is_not_a_gemma4():
    """gemma4 cannot select tools, and this does NOT improve with size.

    Both gemma4:12b and gemma4:26b score 1/4 on the probes in scripts/bench_models.py,
    failing on the SAME three. Re-running those three 5x under each of {temp 1, temp 0,
    temp 0 + thinking on} scored 0/45 - never correct, never varying. It is a fixed prior
    ("list the directory first") that overrides an instruction naming the exact file, not
    a capability gap that a bigger checkpoint closes.

    This guard exists because gemma4 is the obvious thing to reach for: it is the fastest
    model on the box (26-36 tok/s against devstral's 3.4) and the only one with vision.
    Speed is not the axis that was failing. Vision agents in the harness are unaffected -
    they use plain completion and never pass `tools`.
    """
    assert not Settings().model_id.startswith("gemma4"), (
        "gemma4 scored 0/45 on deterministic re-test of tool selection; see config.py"
    )


def test_exec_timeout_outlives_a_pip_install():
    """The system prompt authorises installs; the model cannot raise this itself."""
    assert SandboxedToolSet.DEFAULT_TIMEOUT_S >= 300


def test_command_output_is_capped_but_file_reads_are_not(tmp_path):
    """Truncation is safe for command output and unsafe for file reads: a partial
    read_file would be written back as a partial file."""
    tools = SandboxedToolSet(Settings(workspace_dir=tmp_path))

    big = "x" * (SandboxedToolSet.MAX_OUTPUT_CHARS + 5_000)
    (tmp_path / "big.txt").write_text(big, encoding="utf-8")

    # read_file returns everything, untruncated.
    assert len(tools.read_file("big.txt")) == len(big)

    # exec_command output is capped, and keeps the TAIL where failures report.
    out = tools.exec_command('python -c "print(\'y\' * 40000)"')
    assert len(out) <= SandboxedToolSet.MAX_OUTPUT_CHARS + 200
    assert "truncated" in out
    assert "exit_code" in out, "the tail must survive - it carries the exit code"
