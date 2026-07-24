"""Run Hermes Agent on Terminal-Bench 2.1 through Harbor and local Docker."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Sequence
from urllib.parse import urlsplit

from hermes_cli.benchmarks.swebench_verified import (
    BenchmarkError,
    atomic_write_json,
    hermes_source_identity,
    utc_now,
)
from hermes_cli.benchmarks.tracing import (
    TraceRun,
    create_trace_run,
    finalize_trace_run,
)
from hermes_cli.benchmarks.tracing.harbor import (
    HarborTraceHarness,
)


BENCHMARK = "terminal-bench-2.1"
DATASET = "terminal-bench/terminal-bench-2-1"
OFFICIAL_TASK_COUNT = 89
DEFAULT_MODEL = "openrouter/qwen/qwen3-coder-next"
DEFAULT_OUTPUT_DIR = ".benchmark-runs/terminal-bench-2.1"
DEFAULT_ENVIRONMENT = "docker"
DEFAULT_HERMES_VERSION = "play"
DEFAULT_HERMES_REPOSITORY = "https://github.com/NousResearch/hermes-agent.git"
DEFAULT_MAX_TASKS = 1
DEFAULT_ATTEMPTS = 1
DEFAULT_CONCURRENCY = 1
DEFAULT_INFRASTRUCTURE_RETRIES = 0
DEFAULT_COORDINATOR_BUDGET = 24
DEFAULT_PHASE_BUDGETS = {"navigator": 10, "patcher": 18, "reviewer": 12}
DEFAULT_DELEGATION_MODE = "native"
DEFAULT_NATIVE_SUBAGENT_COUNT = len(DEFAULT_PHASE_BUDGETS)
DEFAULT_NATIVE_SUBAGENT_BUDGET = (
    sum(DEFAULT_PHASE_BUDGETS.values()) // DEFAULT_NATIVE_SUBAGENT_COUNT
)
DEFAULT_TEMPERATURE = 0.1
DEFAULT_API_MAX_RETRIES = 1
AGENT_TOPOLOGY = "supervisor-delegation"
AGENT_IMPORT_PATH = "hermes_cli.benchmarks.terminalbench_harbor:BenchmarkHermes"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_DIR = REPO_ROOT / ".benchmark-traces"
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
LOG_TAIL_LINES = 200


@dataclass(frozen=True)
class Options:
    task_names: tuple[str, ...]
    max_tasks: int | None
    attempts: int
    concurrency: int
    max_retries: int
    model: str
    hermes_version: str
    hermes_repository: str
    hermes_commit: str
    environment: str
    output_dir: Path
    run_id: str
    harbor_bin: str
    upload: bool
    public: bool
    leaderboard: bool
    dry_run: bool
    trace_dir: Path | None


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    jobs_dir: Path
    manifest: Path
    stdout: Path
    stderr: Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def default_run_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    return f"terminal-bench-2.1-{re.sub(r'[:+.]', '-', stamp)}"


def default_model(env: dict[str, str] | None = None) -> str:
    source = env if env is not None else os.environ
    if source.get("HERMES_BENCH_MODEL"):
        return source["HERMES_BENCH_MODEL"]
    if source.get("OPENROUTER_MODEL"):
        model = source["OPENROUTER_MODEL"]
        return model if model.startswith("openrouter/") else f"openrouter/{model}"
    return DEFAULT_MODEL


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def default_hermes_repository() -> str:
    remote = _git_output("config", "--get", "remote.origin.url")
    match = re.fullmatch(r"git@github\.com:(.+)", remote)
    if match:
        return f"https://github.com/{match.group(1)}"
    return remote or DEFAULT_HERMES_REPOSITORY


def default_hermes_commit() -> str:
    return _git_output("rev-parse", "HEAD")


def valid_hermes_repository(repository: str) -> bool:
    parsed = urlsplit(repository)
    return bool(
        parsed.scheme == "https"
        and parsed.hostname
        and parsed.path not in {"", "/"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Hermes Agent on Terminal-Bench 2.1 with Harbor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hermes-terminalbench --dry-run
  hermes-terminalbench --task-id task-name --run-id qwen-next-smoke
  hermes-terminalbench --all-tasks --run-id qwen-next-full
  hermes-terminalbench --leaderboard --concurrency 4
""",
    )
    parser.add_argument("--task-id", "--task-name", action="append", dest="task_names")
    parser.add_argument(
        "--max-tasks",
        "--n-limit",
        type=_positive_int,
        default=DEFAULT_MAX_TASKS,
        dest="max_tasks",
        help="Maximum tasks after filtering; defaults to one",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Remove the smoke task limit without enabling leaderboard upload",
    )
    parser.add_argument(
        "--attempts",
        "--n-attempts",
        type=_positive_int,
        default=DEFAULT_ATTEMPTS,
        dest="attempts",
    )
    parser.add_argument(
        "--concurrency",
        "--num-workers",
        type=_positive_int,
        default=DEFAULT_CONCURRENCY,
        dest="concurrency",
    )
    parser.add_argument(
        "--max-retries",
        type=_non_negative_int,
        default=DEFAULT_INFRASTRUCTURE_RETRIES,
    )
    parser.add_argument("--model", default=default_model())
    parser.add_argument(
        "--hermes-version",
        default=DEFAULT_HERMES_VERSION,
        help="Hermes branch or tag installed by Harbor; defaults to play",
    )
    parser.add_argument(
        "--hermes-repository",
        default=default_hermes_repository(),
        help="HTTPS Git repository from which Harbor installs Hermes",
    )
    parser.add_argument(
        "--hermes-commit",
        default=default_hermes_commit(),
        help="Exact Hermes commit installed from the selected branch",
    )
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--harbor-bin", default="harbor")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Run all 89 tasks with at least five attempts and public upload",
    )
    parser.add_argument("--dry-run", action="store_true")
    tracing = parser.add_mutually_exclusive_group()
    tracing.add_argument(
        "--trace-dir",
        type=Path,
        default=DEFAULT_TRACE_DIR,
        help=(
            "Override the benchmark-trace/v1 output base "
            f"(default: {DEFAULT_TRACE_DIR})"
        ),
    )
    tracing.add_argument(
        "--no-trace",
        action="store_const",
        const=None,
        dest="trace_dir",
        help="Disable benchmark tracing for this run",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Options:
    raw_args = list(argv if argv is not None else sys.argv[1:])
    parser = build_parser()
    args = parser.parse_args(raw_args)
    task_names = tuple(args.task_names or ())
    max_tasks = args.max_tasks

    max_tasks_was_set = "--max-tasks" in raw_args or "--n-limit" in raw_args
    attempts_was_set = "--attempts" in raw_args or "--n-attempts" in raw_args
    if args.all_tasks and max_tasks_was_set:
        parser.error("--all-tasks cannot be combined with --max-tasks/--n-limit")
    if args.all_tasks:
        max_tasks = None

    upload = args.upload
    public = args.public
    attempts = args.attempts
    if args.leaderboard:
        if task_names or max_tasks_was_set:
            parser.error("--leaderboard must run the complete official dataset")
        if attempts_was_set and attempts < 5:
            parser.error("--leaderboard requires at least five attempts per task")
        max_tasks = None
        attempts = max(attempts, 5)
        upload = True
        public = True

    if public and not upload:
        parser.error("--public requires --upload")
    if not args.model.startswith("openrouter/") or args.model.endswith("/"):
        parser.error("--model must use openrouter/<model> format")
    if not args.hermes_version.strip():
        parser.error("--hermes-version cannot be empty")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.hermes_version):
        parser.error("--hermes-version must be a simple branch or tag name")
    if not valid_hermes_repository(args.hermes_repository):
        parser.error("--hermes-repository must be a credential-free HTTPS Git URL")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.hermes_commit):
        parser.error("--hermes-commit must be a full 40-character Git commit")
    if not args.environment.strip():
        parser.error("--environment cannot be empty")
    if not args.harbor_bin.strip():
        parser.error("--harbor-bin cannot be empty")
    if not SAFE_RUN_ID.fullmatch(args.run_id) or args.run_id in {".", ".."}:
        parser.error(
            "--run-id may contain only letters, numbers, dots, underscores, and hyphens"
        )

    return Options(
        task_names=task_names,
        max_tasks=max_tasks,
        attempts=attempts,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        model=args.model,
        hermes_version=args.hermes_version,
        hermes_repository=args.hermes_repository,
        hermes_commit=args.hermes_commit.lower(),
        environment=args.environment,
        output_dir=args.output_dir,
        run_id=args.run_id,
        harbor_bin=args.harbor_bin,
        upload=upload,
        public=public,
        leaderboard=args.leaderboard,
        dry_run=args.dry_run,
        trace_dir=args.trace_dir,
    )


def build_paths(options: Options) -> RunPaths:
    root = options.output_dir
    if not root.is_absolute():
        root = REPO_ROOT / root
    run_dir = root.resolve() / "runs" / options.run_id
    return RunPaths(
        run_dir=run_dir,
        jobs_dir=run_dir / "harbor-jobs",
        manifest=run_dir / "manifest.json",
        stdout=run_dir / "harbor.stdout.log",
        stderr=run_dir / "harbor.stderr.log",
    )


def build_harbor_command(
    options: Options,
    jobs_dir: Path,
    *,
    trace_run: TraceRun | None = None,
    harbor_version: str = "unknown",
) -> list[str]:
    agent_kwargs = [
        f"version={options.hermes_version}",
        f"repository={options.hermes_repository}",
        f"commit={options.hermes_commit}",
    ]
    if trace_run is not None:
        agent_kwargs.extend((
            f"trace_root={trace_run.root}",
            f"trace_run_id={trace_run.id}",
            f"trace_created_at={trace_run.created_at}",
            f"trace_benchmark={trace_run.benchmark}",
            f"evaluation_workers={options.concurrency}",
            f"benchmark_retries={options.max_retries}",
            f"harbor_version={harbor_version}",
        ))
    command = [
        options.harbor_bin,
        "run",
        "--dataset",
        DATASET,
        "--agent",
        AGENT_IMPORT_PATH,
        "--model",
        options.model,
        *[
            value
            for agent_kwarg in agent_kwargs
            for value in ("--agent-kwarg", agent_kwarg)
        ],
        "--env",
        options.environment,
        "--n-attempts",
        str(options.attempts),
        "--n-concurrent",
        str(options.concurrency),
        "--max-retries",
        str(options.max_retries),
        "--jobs-dir",
        str(jobs_dir),
        "--job-name",
        options.run_id,
    ]
    for task_name in options.task_names:
        command.extend(["--include-task-name", task_name])
    if options.max_tasks is not None:
        command.extend(["--n-tasks", str(options.max_tasks)])
    if options.upload:
        command.append("--upload")
    if options.public:
        command.append("--public")
    return command


def benchmark_process_env(env: dict[str, str] | None = None) -> dict[str, str]:
    source = dict(env if env is not None else os.environ)
    python_paths = [str(REPO_ROOT)]
    if source.get("PYTHONPATH"):
        python_paths.append(source["PYTHONPATH"])
    source["PYTHONPATH"] = os.pathsep.join(python_paths)
    source["HARBOR_TELEMETRY"] = source.get("HARBOR_TELEMETRY", "off")
    return source


def _captured(command: Sequence[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BenchmarkError(f"Could not run {command[0]}: {exc}") from exc


def preflight(options: Options) -> str:
    if (
        shutil.which(options.harbor_bin) is None
        and not Path(options.harbor_bin).is_file()
    ):
        raise BenchmarkError(f"Harbor is not executable: {options.harbor_bin}")
    harbor = _captured([options.harbor_bin, "--version"], 10)
    if harbor.returncode != 0:
        detail = harbor.stderr.strip() or harbor.stdout.strip()
        raise BenchmarkError(f"Could not execute Harbor: {detail}")

    if options.environment == "docker":
        docker = shutil.which("docker")
        if docker is None:
            raise BenchmarkError(
                "Docker is required for the default Harbor environment"
            )
        info = _captured([docker, "info", "--format", "{{.ServerVersion}}"], 15)
        if info.returncode != 0:
            detail = info.stderr.strip() or info.stdout.strip()
            raise BenchmarkError(f"Docker is installed but not running: {detail}")

    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise BenchmarkError("OPENROUTER_API_KEY is required for inference")
    return harbor.stdout.strip() or harbor.stderr.strip() or "unknown"


def _stream_pipe(
    source: IO[str], console: IO[str], log_file: IO[str], tail: list[str]
) -> None:
    for chunk in iter(source.readline, ""):
        console.write(chunk)
        console.flush()
        log_file.write(chunk)
        log_file.flush()
        tail.append(chunk)
        del tail[:-LOG_TAIL_LINES]


def run_streaming(command: Sequence[str], paths: RunPaths) -> int:
    paths.stdout.touch(mode=0o600)
    paths.stderr.touch(mode=0o600)
    os.chmod(paths.stdout, 0o600)
    os.chmod(paths.stderr, 0o600)
    with (
        paths.stdout.open("w", encoding="utf-8") as stdout_log,
        paths.stderr.open("w", encoding="utf-8") as stderr_log,
    ):
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=benchmark_process_env(),
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise BenchmarkError("Harbor subprocess did not expose output streams")
        stdout_tail: list[str] = []
        stderr_tail: list[str] = []
        threads = [
            threading.Thread(
                target=_stream_pipe,
                args=(process.stdout, sys.stdout, stdout_log, stdout_tail),
                daemon=True,
            ),
            threading.Thread(
                target=_stream_pipe,
                args=(process.stderr, sys.stderr, stderr_log, stderr_tail),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        def forward(signum: int, _frame: Any) -> None:
            if process.poll() is None:
                process.send_signal(signum)

        previous = {
            signum: signal.signal(signum, forward)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            return process.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            for thread in threads:
                thread.join()


def manifest(
    options: Options,
    paths: RunPaths,
    command: Sequence[str],
    *,
    trace_run: TraceRun | None = None,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "benchmark": BENCHMARK,
        "dataset": DATASET,
        "officialTaskCount": OFFICIAL_TASK_COUNT,
        "officialRunner": "harbor",
        "model": options.model,
        "temperature": DEFAULT_TEMPERATURE,
        "agent": AGENT_IMPORT_PATH,
        "agentTopology": AGENT_TOPOLOGY,
        "agentSequence": ["coordinator", *DEFAULT_PHASE_BUDGETS],
        "coordinatorBudget": DEFAULT_COORDINATOR_BUDGET,
        "delegationMode": DEFAULT_DELEGATION_MODE,
        "nativeSubagentBudget": DEFAULT_NATIVE_SUBAGENT_BUDGET,
        "nativeSubagentCount": DEFAULT_NATIVE_SUBAGENT_COUNT,
        "peerPhaseBudgetReference": DEFAULT_PHASE_BUDGETS,
        "apiMaxRetries": DEFAULT_API_MAX_RETRIES,
        "hermesVersion": options.hermes_version,
        "hermesRepository": options.hermes_repository,
        "hermesCommit": options.hermes_commit,
        "hermesSource": hermes_source_identity(),
        "environment": options.environment,
        "taskNames": list(options.task_names),
        "maxTasks": options.max_tasks,
        "allTasks": options.max_tasks is None,
        "attempts": options.attempts,
        "concurrency": options.concurrency,
        "maxRetries": options.max_retries,
        "upload": options.upload,
        "public": options.public,
        "leaderboard": options.leaderboard,
        "command": list(command),
        "jobsDir": str(paths.jobs_dir),
        "stdoutPath": str(paths.stdout),
        "stderrPath": str(paths.stderr),
        **({"traceDir": str(trace_run.root)} if trace_run is not None else {}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(argv)
    paths = build_paths(options)
    command = build_harbor_command(options, paths.jobs_dir)
    if options.dry_run:
        print(json.dumps(command))
        return 0

    source_identity = hermes_source_identity()
    if options.trace_dir is not None and (
        source_identity.get("dirty") is not False
        or source_identity.get("commit") != options.hermes_commit
    ):
        print(
            "Preflight failed: tracing requires the exact clean Hermes checkout "
            "selected by --hermes-commit",
            file=sys.stderr,
        )
        return 1

    try:
        harbor_version = preflight(options)
    except BenchmarkError as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 1

    trace_run = (
        create_trace_run(
            options.trace_dir,
            benchmark=BENCHMARK,
            framework="hermes",
        )
        if options.trace_dir is not None
        else None
    )
    command = build_harbor_command(
        options,
        paths.jobs_dir,
        trace_run=trace_run,
        harbor_version=harbor_version,
    )
    paths.jobs_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        **manifest(options, paths, command, trace_run=trace_run),
        "harborVersion": harbor_version,
        "startedAt": utc_now(),
        "status": "running",
    }
    atomic_write_json(paths.manifest, run_manifest)

    def finalize_trace() -> None:
        if trace_run is None:
            return
        expected_count = (
            len(options.task_names)
            if options.task_names
            else options.max_tasks
            if options.max_tasks is not None
            else OFFICIAL_TASK_COUNT
        )
        try:
            finalize_trace_run(
                trace_run,
                HarborTraceHarness(
                    jobs_dir=paths.jobs_dir,
                    job_name=options.run_id,
                    selected_instance_ids=(
                        tuple(options.task_names) if options.task_names else None
                    ),
                    expected_instance_count=expected_count,
                    expected_attempts_per_instance=options.attempts,
                    selection_strategy=(
                        "explicit_ids"
                        if options.task_names
                        else "ordered_window"
                        if options.max_tasks is not None
                        else "full_dataset"
                    ),
                ),
            )
        except Exception as exc:
            print(
                "[trace] Hermes trace run index could not be finalized; "
                f"benchmark outputs remain valid: {type(exc).__name__}",
                file=sys.stderr,
            )

    try:
        exit_code = run_streaming(command, paths)
    except BaseException:
        finalize_trace()
        atomic_write_json(
            paths.manifest,
            {**run_manifest, "finishedAt": utc_now(), "status": "failed"},
        )
        raise

    finalize_trace()
    atomic_write_json(
        paths.manifest,
        {
            **run_manifest,
            "finishedAt": utc_now(),
            "exitCode": exit_code,
            "status": "completed" if exit_code == 0 else "failed",
        },
    )
    print(f"Harbor artifacts: {paths.jobs_dir}")
    print(f"Run manifest: {paths.manifest}")
    if trace_run is not None:
        print(f"Benchmark traces: {trace_run.root}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
