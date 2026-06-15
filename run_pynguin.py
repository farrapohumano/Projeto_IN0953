import os
import sys
import json
import time
import shlex
import argparse
import subprocess
from pathlib import Path

from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from urllib.parse import urlparse

from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field


@dataclass
class RunReport:
    name: str
    repository: str
    source_root: str
    module: str
    algorithm: str
    stop_criteria: str
    stop_value: int
    seed: int
    assertion_generator: str
    started_at: str
    duration_seconds: float
    return_code: int
    output_directory: str
    generated_test_files: int
    command: list[str]
    compile_return_code: int | None = None
    test_return_code: int | None = None
    branch_coverage: float | None = None
    llm_configuration: dict[str, object] = field(default_factory=dict)

class PynguinRunner:

    ALGORITHMS = ("DYNAMOSA", "MOSA", "MIO", "LLM", "LLMOSA", "RANDOM", "RANDOM_TEST_SUITE_SEARCH")
    ASSERTION_GENERATORS = ("SIMPLE", "MUTATION_ANALYSIS", "CHECKED_MINIMIZING", "LLM")
    STOPPING_CRITERIA = {
        "search-time": "--maximum-search-time",
        "iterations": "--maximum-iterations",
        "test-executions": "--maximum-test-executions",
        "statement-executions": "--maximum-statement-executions",
        "slicing-time": "--maximum-slicing-time",
        "coverage": "--maximum-coverage",
        "coverage-plateau": "--maximum-coverage-plateau",
        "memory": "--maximum-memory",
    }

    def __init__(self, repo_urls: list[str], workspace: Path) -> None:
        self.repo_urls, self.workspace = repo_urls, workspace.resolve()
        self.repositories_dir, self.results_dir = self.workspace / "repositories", self.workspace / "results"
        self.console, self.repositories = Console(), []

    def initialize(self) -> None:
        """
        Initialize the environment to start generating tests
        :return: None
        """
        self.repositories_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(path for path in self.repositories_dir.iterdir() if path.is_dir())
        cloned = [self._clone(repo) for repo in self.repo_urls]

        self.repositories = list(dict.fromkeys([*existing, *cloned]))

        if existing:
            self.console.print(f"[green]Found {len(existing)} existing repository/repositories.[/]")
        if not self.repositories:
            raise ValueError("No repositories supplied and none exist in the workspace")

    def _clone(self, url: str) -> Path:
        """
        Run git clone and clone the repo(s) given
        :param url: repository url
        :return: Path to the cloned repository
        """
        name = Path(urlparse(url).path.rstrip("/")).name.removesuffix(".git")

        if not name:
            raise ValueError(f"Not able to indentify repository name from {url!r}")

        destination = self.repositories_dir / name

        if destination.exists():
            self.console.print(f"[yellow]Using existing repository:[/] {destination}"); return destination

        self.console.print(f"[cyan]Cloning[/] {url}")

        subprocess.run(["git", "clone", "--depth", "1", url, str(destination)], check=True)

        return destination

    def find_requirements_files(self, repo: Path) -> list[Path]:
        """

        :param repo:
        :return:
        """
        ignored = {".git", ".venv", "venv", "build", "dist", "site-packages"}
        return sorted(
            path for path in repo.rglob("requirements.txt")
            if not any(part in ignored or part.startswith(".") for part in path.relative_to(repo).parts)
        )

    def install_requirements(self, repo: Path, requirements_file: Path) -> int:
        """

        :param repo:
        :param requirements_file:
        :return:
        """
        command = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        self.console.print(f"[dim]$ {shlex.join(command)}[/]")
        result = subprocess.run(command, cwd=repo, check=False)
        if result.returncode == 0:
            self.console.print(f"[green]Installed requirements from:[/] {requirements_file}")
        else:
            self.console.print(f"[red]Requirements installation failed with exit code {result.returncode}:[/] {requirements_file}")
        return result.returncode

    def _interactive_install_requirements(self) -> None:
        """

        :return:
        """
        repo = self._choose("Repository", self.repositories)
        files = self.find_requirements_files(repo)
        if not files:
            self.console.print(f"[yellow]No requirements.txt files found in:[/] {repo}")
            return

        self._numbered("Requirements files", [str(path.relative_to(repo)) for path in files])
        selection = Prompt.ask("Requirements file numbers separated by commas, or 'all'", default="1")
        selected = files if selection.lower() == "all" else self._indexes(files, selection)
        if not selected:
            self.console.print("[red]No valid requirements files selected.[/]")
            return

        if not Confirm.ask(f"Install dependencies from {len(selected)} requirements file(s)?", default=True):
            return
        for requirements_file in selected:
            self.install_requirements(repo, requirements_file)

    def install_repository_requirements(self, repo: Path) -> bool:
        """

        :param repo:
        :return:
        """
        files = self.find_requirements_files(repo)
        if not files:
            self.console.print(f"[yellow]No requirements.txt files found in:[/] {repo}")
            return False
        self._numbered("Requirements files", [str(path.relative_to(repo)) for path in files])
        if not Confirm.ask("Install repository requirements before running Pynguin?", default=True):
            return False
        return all(self.install_requirements(repo, path) == 0 for path in files)

    def check_module_imports(self, repo: Path, root: Path, modules: list[str]) -> list[tuple[str, str]]:
        """

        :param repo:
        :param root:
        :param modules:
        :return:
        """
        failures: list[tuple[str, str]] = []
        environment = self._test_environment(root)
        for module in modules:
            command = [sys.executable, "-c", f"import importlib; importlib.import_module({module!r})"]
            result = subprocess.run(command, cwd=repo, env=environment, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "import failed"
                failures.append((module, message))
        return failures

    def ensure_modules_importable(self, repo: Path, root: Path, modules: list[str]) -> bool:
        """

        :param repo:
        :param root:
        :param modules:
        :return:
        """
        failures = self.check_module_imports(repo, root, modules)
        if not failures:
            return True
        for module, message in failures:
            self.console.print(f"[red]Cannot import {module}:[/] {message}")
        if not self.install_repository_requirements(repo):
            return False
        failures = self.check_module_imports(repo, root, modules)
        for module, message in failures:
            self.console.print(f"[red]Still cannot import {module}:[/] {message}")
        return not failures

    def discover_modules(self, root: Path) -> list[str]:
        """
        Try to find the modules from the source root path of the repositories given. It tries to avoid some common names not useful for tests
        :param root: the main path of repository cloned
        :return:
        """
        ignored = {"tests", "test", ".git", ".venv", "venv", "build", "dist"}; modules = []

        for path in root.rglob("*.py"):
            rel = path.relative_to(root)
            if any(p in ignored or p.startswith(".") for p in rel.parts): continue

            if path.name == "__init__.py":
                if len(rel.parts) > 1:
                    modules.append(".".join(rel.parent.parts))
            else:
                modules.append(".".join(rel.with_suffix("").parts))

        return sorted(set(modules))

    def find_source_roots(self, repo: Path) -> list[Path]:
        """
        Try to find the source root of in path of the repo given looking for common folder names
        :param repo: Path of repository
        :return: Path of source root
        """
        return [p for p in (repo / "src", repo / "lib", repo) if p.is_dir() and self.discover_modules(p)]

    def run(self, repo: Path, root: Path, modules: list[str], algorithm: str, stop_criteria: str, stop_value: int, seed: int, assertion_generator: str, llm_configuration: dict[str, object], extra: list[str], compile_tests: bool = False, execute_tests: bool = False) -> list[RunReport]:
        """
        Run docker command with all parameters get from the user.
        :param repo: Path of repository
        :param root: source root of the repository
        :param modules: modules for run the tests
        :param algorithm: algorithm for generate the tests
        :param stop_criteria: the stop method criteria (stop the generation)
        :param stop_value: value for the stop criteria
        :param seed: seed for a reproducible run, also used for comparison
        :param assertion_generator: type of assertion, simple is the default due to possible issues using mutation
        :param llm_configuration: configuration of the large language model
        :param extra: extra commands for pynguin
        :param compile_tests: bool - true will compile the tests after generation
        :param execute_tests: bool - true will execute the tests after generation
        :return: dataclass object of the run RunReport
        """
        reports = []

        for module in modules:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            name = f"{timestamp}_{repo.name}_{module.replace('.', '_')}_{algorithm.lower()}_seed{seed}"
            output = self.results_dir / name / "generated_tests"; output.mkdir(parents=True)

            command = ["pynguin", "--project-path", str(root), "--module-name", module, "--output-path", str(output),
                       "--algorithm", algorithm, self.STOPPING_CRITERIA[stop_criteria], str(stop_value), "--seed", str(seed),
                       "--assertion-generation", assertion_generator, *self._llm_arguments(llm_configuration), *extra]

            self.console.rule(f"[bold cyan]{module}")
            self.console.print(f"[dim]$ {shlex.join(command)}[/]")

            started, timer = datetime.now(timezone.utc).isoformat(), time.monotonic()

            result = subprocess.run(command, cwd=repo, env={**os.environ, "PYNGUIN_DANGER_AWARE": "1"}, check=False)

            generated_files = len(list(output.rglob("test_*.py")))
            successful_generation = result.returncode == 0 and generated_files > 0
            compile_return_code = self.compile_generated_tests(output, repo, root) if compile_tests and successful_generation else None
            test_return_code, branch_coverage = self.execute_generated_tests(output, repo, root, module) if execute_tests and successful_generation else (None, None)

            report = RunReport(name, repo.name, str(root), module, algorithm, stop_criteria, stop_value, seed, assertion_generator, started,
                               round(time.monotonic() - timer, 3), result.returncode, str(output),
                               generated_files, command, compile_return_code,
                               test_return_code, branch_coverage, llm_configuration)

            (output.parent / "report.json").write_text(json.dumps(asdict(report), indent=2) + "\n")
            reports.append(report)
            if not successful_generation:
                self.console.print(
                    f"[yellow]No tests were generated for {module}; continuing with the remaining modules.[/]"
                )

        return reports

    def compile_generated_tests(self, output: Path, repo: Path, root: Path) -> int:
        """
        compiles the tests generated by pynguin
        :param output: path of the tests
        :param repo: repository path
        :param root: source root of the repository
        :return: int of compiled tests
        """
        command = [sys.executable, "-m", "compileall", "-q", str(output)]
        self.console.print(f"[dim]$ {shlex.join(command)}[/]")
        return subprocess.run(command, cwd=repo, env=self._test_environment(root), check=False).returncode

    def execute_generated_tests(self, output: Path, repo: Path, root: Path, module: str) -> tuple[int, float | None]:
        """
        executes the tests generated by pynguin
        :param output: Path of the tests
        :param repo: repository path
        :param root: source root of the repository
        :param module: module name
        :return: return code and the coverage
        """
        coverage_file = output.parent / ".coverage"
        coverage_json = output.parent / "coverage.json"

        command = [sys.executable, "-m", "coverage", "run", "--branch", f"--source={module}",
                   f"--data-file={coverage_file}", "-m", "pytest", str(output), "-q"]

        self.console.print(f"[dim]$ {shlex.join(command)}[/]")
        result = subprocess.run(command, cwd=repo, env=self._test_environment(root), check=False)
        json_command = [sys.executable, "-m", "coverage", "json", f"--data-file={coverage_file}", "-o", str(coverage_json)]

        subprocess.run(json_command, cwd=repo, env=self._test_environment(root), check=False)

        return result.returncode, self._read_branch_coverage(coverage_json)

    @staticmethod
    def _test_environment(root: Path) -> dict[str, str]:
        existing = os.environ.get("PYTHONPATH", "")
        pythonpath = str(root) if not existing else f"{root}{os.pathsep}{existing}"
        return {**os.environ, "PYTHONPATH": pythonpath}

    @staticmethod
    def _read_branch_coverage(path: Path) -> float | None:
        try:
            totals = json.loads(path.read_text())["totals"]
            branches = totals.get("num_branches", 0)
            if not branches:
                return None
            return round(100 * totals.get("covered_branches", 0) / branches, 2)
        except (OSError, KeyError, TypeError, ValueError):
            return None

    def load_reports(self) -> list[RunReport]:
        reports = []
        for path in sorted(self.results_dir.glob("*/report.json")):
            try:
                data = json.loads(path.read_text())
                if "maximum_search_time" in data:
                    data["stop_criteria"] = "search-time"
                    data["stop_value"] = data.pop("maximum_search_time")
                if "stop_criterion" in data:
                    data["stop_criteria"] = data.pop("stop_criterion")
                data.setdefault("assertion_generator", "MUTATION_ANALYSIS")
                data.setdefault("compile_return_code", None)
                data.setdefault("test_return_code", None)
                data.setdefault("branch_coverage", None)
                data.setdefault("llm_configuration", {})
                reports.append(RunReport(**data))
            except (OSError, ValueError, TypeError): self.console.print(f"[yellow]Skipping invalid report:[/] {path}")
        return reports

    def show_reports(self, reports: list[RunReport]) -> None:
        table = Table(title="Pynguin run reports")
        for c in ("#", "Run", "Repository", "Module", "Algorithm", "Stop", "Time", "Exit", "Tests", "Branch"): table.add_column(c)
        for i, r in enumerate(reports, 1):
            table.add_row(str(i), r.name, r.repository, r.module, r.algorithm, f"{r.stop_criteria}={r.stop_value}", f"{r.duration_seconds:.2f}s", str(r.return_code), str(r.generated_test_files), self._percentage(r.branch_coverage))
        self.console.print(table)

    def compare(self, a: RunReport, b: RunReport) -> None:
        table = Table(title="Run comparison"); table.add_column("Metric"); table.add_column(a.name); table.add_column(b.name)
        for metric, left, right in [("Repository", a.repository, b.repository), ("Module", a.module, b.module),
            ("Algorithm", a.algorithm, b.algorithm), ("Assertion generator", a.assertion_generator, b.assertion_generator), ("Seed", a.seed, b.seed), ("Stopping criterion", a.stop_criteria, b.stop_criteria), ("Stopping value", a.stop_value, b.stop_value),
            ("Actual duration", f"{a.duration_seconds:.2f}s", f"{b.duration_seconds:.2f}s"), ("Exit code", a.return_code, b.return_code),
            ("Generated test files", a.generated_test_files, b.generated_test_files),
            ("Compile exit code", self._available(a.compile_return_code), self._available(b.compile_return_code)),
            ("Test exit code", self._available(a.test_return_code), self._available(b.test_return_code)),
            ("Branch coverage", self._percentage(a.branch_coverage), self._percentage(b.branch_coverage)),
            ("LLM model", a.llm_configuration.get("model_name", "N/A"), b.llm_configuration.get("model_name", "N/A")),
            ("LLM temperature", a.llm_configuration.get("temperature", "N/A"), b.llm_configuration.get("temperature", "N/A")),
            ("LLM settings", self._llm_summary(a.llm_configuration), self._llm_summary(b.llm_configuration))]: table.add_row(metric, str(left), str(right))
        self.console.print(table)

    def interactive(self) -> None:
        self.console.print(Panel.fit("Interactive Pynguin Runner", style="bold cyan"))
        while True:
            operation = Prompt.ask("Operation", choices=["run", "requirements", "reports", "compare", "exit"], default="run")
            if operation == "exit": return
            reports = self.load_reports()
            if operation == "requirements":
                self._interactive_install_requirements()
                continue
            if operation == "reports": self.show_reports(reports); continue
            if operation == "compare":
                if len(reports) < 2: self.console.print("[yellow]At least two runs are required.[/]"); continue
                self.show_reports(reports); a = IntPrompt.ask("First report number", default=1); b = IntPrompt.ask("Second report number", default=2)
                if 1 <= a <= len(reports) and 1 <= b <= len(reports): self.compare(reports[a-1], reports[b-1])
                else: self.console.print("[red]Invalid report number.[/]")
                continue
            self._interactive_run()

    def _interactive_run(self) -> None:
        """
        get the information needed from the user to start the run
        :return: None
        """
        repo = self._choose("Repository", self.repositories)

        root = self._choose_source_root(repo)
        if root is None:
            return
        available = self.discover_modules(root)
        self._numbered("Discovered modules", available)

        selection = Prompt.ask("Module numbers separated by commas, or 'all'", default="1")
        modules = available if selection.lower() == "all" else self._indexes(available, selection)
        if not modules: self.console.print("[red]No valid modules selected.[/]"); return
        if not self.ensure_modules_importable(repo, root, modules):
            self.console.print("[red]Run cancelled because selected modules are not importable.[/]")
            return

        algorithm = Prompt.ask("Algorithm", choices=list(self.ALGORITHMS), default="DYNAMOSA")
        stop_criteria = Prompt.ask("Stopping criterion", choices=list(self.STOPPING_CRITERIA), default="search-time")
        default_value = 60 if stop_criteria in {"search-time", "slicing-time"} else 100
        stop_value = IntPrompt.ask(f"Maximum {stop_criteria} value", default=default_value)

        while stop_value <= 0:
            stop_value = IntPrompt.ask("Value must be greater than zero")
        seed = IntPrompt.ask("Random seed", default=1)

        if Confirm.ask("Generate assertions?", default=True):
            assertion_generator = Prompt.ask("Assertion generator", choices=list(self.ASSERTION_GENERATORS), default="SIMPLE")
        else:
            assertion_generator = "NONE"

        llm_configuration = self._ask_llm_configuration(algorithm == "LLMOSA") if algorithm in {"LLM", "LLMOSA"} or assertion_generator == "LLM" else {}
        extra = shlex.split(Prompt.ask("Extra Pynguin arguments", default=""))

        compile_tests = Confirm.ask("Compile generated tests after generation?", default=True)
        execute_tests = Confirm.ask("Run generated tests and measure branch coverage?", default=True)

        summary = f"Run {len(modules)} module(s) with {stop_criteria}={stop_value}?"

        if Confirm.ask(summary, default=True):
            self.show_reports(self.run(repo, root, modules, algorithm, stop_criteria, stop_value, seed, assertion_generator, llm_configuration, extra, compile_tests, execute_tests))

    def _ask_llm_configuration(self, llmosa: bool) -> dict[str, object]:
        """
        Ask llm configuration for LLMOSA, as temperature, plateu-len, etc.
        :param llmosa: Boolean
        :return: dict containing the config given by user
        """
        self.console.rule("[bold cyan]LLM configuration")

        config: dict[str, object] = {
            "model_name": Prompt.ask("LLM model", default="qwen2.5-coder:7b"),
            "temperature": self._ask_probability("Temperature", 0.2),
            "enable_response_caching": Confirm.ask("Enable LLM response caching?", default=True),
        }

        if llmosa:
            config.update({
                "hybrid_initial_population": Confirm.ask("Include LLM tests in the initial population?", default=False),
                "llm_test_case_percentage": self._ask_probability("LLM initial-population percentage", 0.5),
                "call_llm_for_uncovered_targets": Confirm.ask("Call LLM for initially uncovered targets?", default=False),
                "coverage_threshold": self._ask_probability("Coverage threshold for LLM calls", 1.0),
                "call_llm_on_stall_detection": Confirm.ask("Call LLM when coverage stalls?", default=True),
                "max_plateau_len": IntPrompt.ask("Iterations before an LLM stall intervention", default=25),
                "max_llm_interventions": IntPrompt.ask("Maximum LLM interventions", default=1),
            })

        return config

    @staticmethod
    def _llm_arguments(config: dict[str, object]) -> list[str]:
        """
        Get llm argument configurations
        :param config: dict containing the config given by user
        :return: turn the dict into a list to be used on subprocess
        """
        flags = {
            "model_name": "--model-name", "temperature": "--temperature",
            "enable_response_caching": "--enable-response-caching",
            "hybrid_initial_population": "--hybrid-initial-population",
            "llm_test_case_percentage": "--llm-test-case-percentage",
            "call_llm_for_uncovered_targets": "--call-llm-for-uncovered-targets",
            "coverage_threshold": "--coverage-threshold",
            "call_llm_on_stall_detection": "--call-llm-on-stall-detection",
            "max_plateau_len": "--max-plateau-len",
            "max_llm_interventions": "--max-llm-interventions",
        }
        arguments: list[str] = []

        for key, value in config.items():
            arguments.extend([flags[key], str(value).lower() if isinstance(value, bool) else str(value)])

        return arguments

    @staticmethod
    def _llm_summary(config: dict[str, object]) -> str:
        if not config:
            return "N/A"
        return ", ".join(f"{key}={value}" for key, value in config.items() if key not in {"model_name", "temperature"})

    @staticmethod
    def _ask_probability(label: str, default: float) -> float:
        value = FloatPrompt.ask(label, default=default)
        while not 0.0 <= value <= 1.0:
            value = FloatPrompt.ask(f"{label} must be between 0.0 and 1.0", default=default)
        return value

    @staticmethod
    def _available(value: int | None) -> str:
        return "N/A" if value is None else str(value)

    @staticmethod
    def _percentage(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2f}%"

    def _choose_source_root(self, repo: Path) -> Path | None:
        """
        
        :param repo:
        :return:
        """
        roots = self.find_source_roots(repo)
        if roots:
            self._numbered("Detected source roots", [str(root) for root in roots])
            if Confirm.ask("Use one of the detected source roots?", default=True):
                return self._choose("Source root", roots)
        else:
            self.console.print("[yellow]No source roots were detected automatically.[/]")

        self.console.print("[dim]Enter '.' for the repository root, a repository-relative path such as 'src', or an absolute path. Leave blank to cancel.[/]")
        while True:
            raw_path = Prompt.ask("Custom source_root path", default="").strip()
            if not raw_path:
                return None

            candidates = self._source_root_candidates(repo, raw_path)
            for candidate in candidates:
                if not candidate.is_dir() or not self.discover_modules(candidate):
                    continue
                if (candidate / "__init__.py").is_file():
                    self.console.print(
                        f"[red]That path is a Python package, not a source root:[/] {candidate}"
                    )
                    parent = candidate.parent
                    if parent.is_dir() and self.discover_modules(parent):
                        self.console.print(f"[yellow]Use its parent as source_root:[/] {parent}")
                    continue
                self.console.print(f"[green]Using custom source root:[/] {candidate}")
                return candidate

            attempted = ", ".join(str(candidate) for candidate in candidates)
            self.console.print(f"[red]No valid Python source root found. Checked:[/] {attempted}")

    @staticmethod
    def _source_root_candidates(repo: Path, raw_path: str) -> list[Path]:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return [path.resolve()]

        candidates = [(repo / path).resolve()]
        if path.parts and path.parts[0] == repo.name:
            repositories_relative = (repo.parent / path).resolve()
            if repositories_relative not in candidates:
                candidates.insert(0, repositories_relative)
        return candidates

    def _choose(self, title: str, values: list[Path]) -> Path:
        self._numbered(title, [str(v) for v in values]); index = IntPrompt.ask(f"{title} number", default=1)
        while not 1 <= index <= len(values): index = IntPrompt.ask(f"Choose 1 to {len(values)}")
        return values[index-1]

    def _numbered(self, title: str, values: list[str]) -> None:
        table = Table(title=title); table.add_column("#"); table.add_column("Value")
        for i, value in enumerate(values, 1): table.add_row(str(i), value)
        self.console.print(table)

    @staticmethod
    def _indexes(values: list, selection: str) -> list:
        result = []
        for raw in selection.split(","):
            try:
                i = int(raw.strip())
                if 1 <= i <= len(values): result.append(values[i-1])
            except ValueError: pass
        return list(dict.fromkeys(result))

def main() -> int:
    """
    Get arguments from command line and starts the docker.
    :return: exit code
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repositories", nargs="*", help="Git repository URLs; existing workspace repositories are also loaded")
    parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("RUNNER_WORKSPACE", "/workspace")))
    args = parser.parse_args(); runner = PynguinRunner(args.repositories, args.workspace)

    try:
        runner.initialize()
        runner.interactive()
        return 0
    except (subprocess.CalledProcessError, OSError, ValueError) as error:
        runner.console.print(f"[bold red]Error:[/] {error}")
        return 1
    except KeyboardInterrupt:
        runner.console.print("\n[yellow]Stopped.[/]")
        return 130 # code of keyboard interrupt

if __name__ == "__main__":
    sys.exit(main())
