import sys
import shlex
import argparse
import subprocess
from dataclasses import dataclass

from rich.panel import Panel
from rich.console import Console
from rich.prompt import Confirm, Prompt

console = Console()


@dataclass(frozen=True)
class DockerConfiguration:
    image: str = "pynguin-runner:local"
    volume: str = "pynguin-workspace"
    use_ollama: bool = True
    ollama_url: str = "http://127.0.0.1:11434/v1"
    ollama_api_key: str = "ollama"
    ollama_network: str = "host"


def compose_docker_command(repositories: list[str], config: DockerConfiguration) -> list[str]:
    """
    build the Docker command, preserving each repository as one argument.
    :param repositories: list of repository
    :param config: docker configuration
    :return: command as a list
    """
    command = [
        "docker",
        "run",
        "--rm",
        "-it",
        "-v",
        f"{config.volume}:/workspace",
    ]
    if config.use_ollama:
        if config.ollama_network == "host":
            command.extend(["--network", "host"])
        else:
            command.append("--add-host=host.docker.internal:host-gateway")
        command.extend(
            [
                "-e",
                f"OPENAI_BASE_URL={config.ollama_url}",
                "-e",
                f"OPENAI_API_KEY={config.ollama_api_key}",
            ]
        )
    command.extend([config.image, *repositories])
    return command


def check_ollama_connection(config: DockerConfiguration) -> tuple[bool, str]:
    """
    Checks ollama config
    :param config: Docker configuration
    :return: return code and error message
    """
    if not config.use_ollama:
        return True, ""
    command = ["docker", "run", "--rm"]

    if config.ollama_network == "host":
        command.extend(["--network", "host"])
    else:
        command.append("--add-host=host.docker.internal:host-gateway")

    command.extend([
        "--entrypoint", "python", config.image, "-c",
        f"import urllib.request; urllib.request.urlopen('{config.ollama_url}/models', timeout=5)",
    ])

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    error = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown connection error"

    return result.returncode == 0, error


def find_current_repositories(config: DockerConfiguration) -> list[str]:
    """
    List repositories already stored
    :param config: Docker configuration
    :return: list of repos
    """
    command = [
        "docker", "run", "--rm", "-v", f"{config.volume}:/workspace",
        "--entrypoint", "sh", config.image, "-lc",
        "find /workspace/repositories -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' 2>/dev/null | sort",
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        return []

    return [line for line in result.stdout.splitlines() if line]


def ask_repositories(initial: list[str] | None = None, current: list[str] | None = None) -> list[str]:
    """
    Asks the user the URLs repositories, one per prompt.
    :param initial: Already available
    :param current: current workspace repositories
    :return: repo list
    """
    repositories = list(initial or [])
    current = list(current or [])
    if current:
        console.print(f"[green]Available workspace repositories:[/] {', '.join(current)}")
        if not repositories and Confirm.ask("Use the current workspace repositories?", default=True):
            return []

    while not repositories or Confirm.ask("Add another repository?", default=False):
        repository = Prompt.ask("Git repository URL").strip()
        if repository and repository not in repositories:
            repositories.append(repository)
        elif repository in repositories:
            console.print("[yellow]That repository is already in the list.[/]")

    return repositories


def run(repositories: list[str], config: DockerConfiguration, not_run: bool = False) -> int:
    """

    :param repositories:
    :param config:
    :param not_run: Print the command without running Docker
    :return:
    """
    command = compose_docker_command(repositories, config)
    console.print(Panel.fit(shlex.join(command), title="Docker command", style="cyan"))
    if not_run:
        return 0
    connected, error = check_ollama_connection(config)

    if not connected:
        console.print(f"[bold red]Cannot reach Ollama from Docker:[/] {error}")
        console.print(f"[yellow]Checked endpoint:[/] {config.ollama_url}/models using {config.ollama_network} networking")
        return 1

    if not Confirm.ask("Start the container?", default=True):
        return 0
    console.print("[green]Starting Docker container...[/]")

    return subprocess.run(command, check=False).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repositories", nargs="*", help="Git repository URLs")
    parser.add_argument("--image", default="pynguin-runner:local", help="Docker image to run")
    parser.add_argument("--volume", default="pynguin-workspace", help="Docker volume used for /workspace")
    parser.add_argument("--no-ollama", action="store_true", help="Do not configure the Ollama endpoint")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--ollama-api-key", default="ollama")
    parser.add_argument("--ollama-network", choices=["host", "bridge"], default="host", help="Docker network used to reach Ollama")
    parser.add_argument("--not-run", action="store_true", help="Print the command without running Docker")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    console.print(Panel.fit("Pynguin Comparison Launcher", style="bold cyan"))
    config = DockerConfiguration(
        image=args.image,
        volume=args.volume,
        use_ollama=not args.no_ollama,
        ollama_url=args.ollama_url,
        ollama_api_key=args.ollama_api_key,
        ollama_network=args.ollama_network,
    )
    current = find_current_repositories(config)
    repositories = ask_repositories(args.repositories, current)
    if not repositories and not current:
        console.print("[bold red]Error:[/] No repository URLs supplied and no current repositories found.")
        return 1
    try:
        return run(repositories, config, args.dry_run)
    except (OSError, ValueError) as error:
        console.print(f"[bold red]Error:[/] {error}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
