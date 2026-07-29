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
        required=True,
        help="Task goal or prompt for the agent loop",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier",
    )
    args = parser.parse_args()

    settings = get_settings()

    console.print(Panel.fit(
        f"[bold green]Hermes Deploy Loop Initialized[/bold green]\n"
        f"Workspace: [cyan]{settings.workspace_dir}[/cyan]\n"
        f"Model: [magenta]{settings.model_id}[/magenta]\n"
        f"Ollama Base URL: [blue]{settings.ollama_base_url}[/blue]\n"
        f"Harness Base URL: [yellow]{settings.harness_base_url}[/yellow]",
        title="Agent Loop Configuration"
    ))

    loop = AgentLoop(settings=settings)
    console.print(f"[bold]Executing Task:[/bold] {args.prompt}\n")

    result = loop.run(args.prompt, run_id=args.run_id)

    if result.completed:
        console.print(Panel(
            result.final_output,
            title=f"[bold green]Task Completed in {result.turns_taken} Turns[/bold green]",
            border_style="green"
        ))
    else:
        error_msg = result.error or "Unknown failure"
        console.print(Panel(
            f"Error: {error_msg}\nStuck Loop: {result.stuck_loop_detected}",
            title=f"[bold red]Task Failed after {result.turns_taken} Turns[/bold red]",
            border_style="red"
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
