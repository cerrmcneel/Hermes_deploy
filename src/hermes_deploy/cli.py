"""CLI entrypoint for Hermes Deploy task execution loop."""

import argparse
import sys
from rich.console import Console
from rich.panel import Panel

from hermes_deploy.agent_loop import AgentLoop
from hermes_deploy.config import get_settings

console = Console()


def main() -> None:
    """CLI main function."""
    parser = argparse.ArgumentParser(description="Hermes Deploy Autonomous Agent Loop")
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Task goal or prompt for the agent loop",
    )
    parser.add_argument(
        "--tasks-file",
        "-f",
        type=str,
        default=None,
        help="File path containing task prompts (one per line or JSON array)",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=None,
        help="Target workspace directory path for agent operations",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier",
    )
    args = parser.parse_args()

    if not args.prompt and not args.tasks_file:
        parser.error("Either --prompt (-p) or --tasks-file (-f) must be specified.")

    settings = get_settings()
    if args.workspace:
        from pathlib import Path
        settings.workspace_dir = Path(args.workspace).resolve()

    console.print(Panel.fit(
        f"[bold green]Hermes Deploy Loop Initialized[/bold green]\n"
        f"Workspace: [cyan]{settings.workspace_dir}[/cyan]\n"
        f"Model: [magenta]{settings.model_id}[/magenta]\n"
        f"Ollama Base URL: [blue]{settings.ollama_base_url}[/blue]\n"
        f"Harness Base URL: [yellow]{settings.harness_base_url}[/yellow]",
        title="Agent Loop Configuration"
    ))

    # Read tasks into queue
    tasks: list[str] = []
    if args.prompt:
        tasks.append(args.prompt)
    if args.tasks_file:
        from pathlib import Path
        tf = Path(args.tasks_file).resolve()
        if not tf.exists():
            console.print(f"[bold red]Error:[/bold red] Task file not found: {tf}")
            sys.exit(1)
        content = tf.read_text(encoding="utf-8")
        if tf.suffix in (".json", ".jsonl"):
            import json
            data = json.loads(content)
            if isinstance(data, list):
                tasks.extend([str(item) for item in data])
        else:
            lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
            tasks.extend(lines)

    loop = AgentLoop(settings=settings)
    total_tasks = len(tasks)

    for idx, task_prompt in enumerate(tasks, start=1):
        console.print(Panel(
            f"[bold]Task {idx}/{total_tasks}:[/bold] {task_prompt}",
            border_style="cyan"
        ))

        run_id = f"{args.run_id}-task-{idx}" if args.run_id else None
        result = loop.run(task_prompt, run_id=run_id)

        if result.completed:
            console.print(Panel(
                result.final_output,
                title=f"[bold green]Task {idx}/{total_tasks} Completed in {result.turns_taken} Turns[/bold green]",
                border_style="green"
            ))
        else:
            error_msg = result.error or "Unknown failure"
            console.print(Panel(
                f"Error: {error_msg}\nStuck Loop: {result.stuck_loop_detected}",
                title=f"[bold red]Task {idx}/{total_tasks} Failed after {result.turns_taken} Turns[/bold red]",
                border_style="red"
            ))
            if result.stuck_loop_detected:
                console.print("[bold yellow]Queue paused due to circuit breaker event.[/bold yellow]")
                sys.exit(1)


if __name__ == "__main__":
    main()
