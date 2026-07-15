from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

console = Console()
WORKSPACE = Path.cwd().resolve()


def resolve_workspace_path(path: str) -> Path:
    target = (WORKSPACE / path).resolve()

    if target != WORKSPACE and WORKSPACE not in target.parents:
        raise ValueError(f"Path is outside the workspace: {path}")

    return target


@tool
def list_files(path: str = ".") -> str:
    """List files and directories inside the workspace."""
    target = resolve_workspace_path(path)

    if not target.exists():
        return f"Path does not exist: {path}"

    if not target.is_dir():
        return f"Path is not a directory: {path}"

    entries = []

    for item in sorted(target.iterdir()):
        suffix = "/" if item.is_dir() else ""
        entries.append(f"{item.name}{suffix}")

    return "\n".join(entries) or "(empty directory)"


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file inside the workspace."""
    target = resolve_workspace_path(path)

    if not target.exists():
        return f"File does not exist: {path}"

    if not target.is_file():
        return f"Not a file: {path}"

    return target.read_text(
        encoding="utf-8",
        errors="replace",
    )


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a UTF-8 text file inside the workspace."""
    target = resolve_workspace_path(path)

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_text(
        content,
        encoding="utf-8",
    )

    return f"Wrote {len(content)} characters to {path}"


ALLOWED_COMMANDS = {
    "python",
    "pytest",
    "ruff",
    "git",
}


@tool
def run_command(command: str) -> str:
    """
    Execute an allowed command in the workspace.

    Allowed commands: python, pytest, ruff, git.
    """
    parts = command.strip().split()

    if not parts:
        return "Command is empty."

    executable = parts[0]

    if executable not in ALLOWED_COMMANDS:
        return (
            f"Command '{executable}' is not allowed. "
            f"Allowed commands: {sorted(ALLOWED_COMMANDS)}"
        )

    try:
        result = subprocess.run(
            parts,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."

    output = (
        f"Exit code: {result.returncode}\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    # Prevent an enormous command result from consuming the context window.
    return output[-20_000:]


def build_agent():
    model = ChatOpenAI(
        model="gpt-5-mini",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    checkpointer = InMemorySaver()

    return create_agent(
        model=model,
        tools=[
            list_files,
            read_file,
            write_file,
            run_command,
        ],
        system_prompt=f"""
You are a CLI software-engineering agent.

Your workspace is:

{WORKSPACE}

Rules:
- Inspect files before modifying them.
- Do not access files outside the workspace.
- Make the smallest reasonable change.
- Explain important changes concisely.
- Do not claim a command succeeded unless its tool result confirms it.
- Use tools when information must be obtained from the workspace.
""",
        checkpointer=checkpointer,
    )


def run_cli() -> None:
    agent = build_agent()

    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    console.print(
        f"[bold green]CLI Agent[/bold green]\n"
        f"Workspace: {WORKSPACE}\n"
        "Commands: /exit, /new, /clear"
    )

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.")
            break

        command = user_input.strip()

        if not command:
            continue

        if command in {"/exit", "/quit"}:
            break

        if command == "/clear":
            console.clear()
            continue

        if command == "/new":
            thread_id = str(uuid.uuid4())
            config = {
                "configurable": {
                    "thread_id": thread_id,
                }
            }
            console.print("[yellow]Started a new session.[/yellow]")
            continue

        try:
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                },
                config=config,
            )

            final_message = result["messages"][-1]
            content = final_message.content

            console.print("\n[bold magenta]Agent[/bold magenta]")
            console.print(Markdown(str(content)))

        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] " f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run_cli()
