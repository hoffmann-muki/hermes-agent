"""Local-Docker SWE-bench Verified runner for Hermes Agent.

Inference and official evaluation are intentionally separate operations.  The
inference controller never imports the SWE-bench harness and never sends gold
patches or hidden tests to an agent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import httpx


BENCHMARK = "swe-bench-verified"
DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DATASET_CONFIG = "default"
DATASET_SPLIT = "test"
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_SMOKE_INSTANCE_ID = "scikit-learn__scikit-learn-13439"
DEFAULT_MODEL = "openrouter/qwen/qwen3-coder-next"
DEFAULT_IMAGE_TEMPLATE = "docker.io/swebench/sweb.eval.x86_64.{repo}_1776_{name}:latest"
DEFAULT_OUTPUT_DIR = ".benchmark-runs/swe-bench-verified"
DEFAULT_DOCKER_PLATFORM = "linux/amd64"
DEFAULT_AGENT_TIMEOUT_SECONDS = 30 * 60
DEFAULT_SETUP_TIMEOUT_SECONDS = 10 * 60
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 60 * 60
DEFAULT_INFERENCE_WORKERS = 1
DEFAULT_EVALUATION_WORKERS = 1
DEFAULT_INFRASTRUCTURE_RETRIES = 0
DEFAULT_ATTEMPTS = 1
DEFAULT_API_MAX_RETRIES = 1
DEFAULT_CODING_CONTEXT = "off"
DEFAULT_COORDINATOR_BUDGET = 24
DEFAULT_PHASE_BUDGETS = {"navigator": 10, "patcher": 18, "reviewer": 12}
DEFAULT_AGENT_SEQUENCE = ("coordinator", *DEFAULT_PHASE_BUDGETS)
REQUIRED_SWEBENCH_VERSION = "4.1.0"
DATASET_PAGE_SIZE = 100
DATASET_FETCH_ATTEMPTS = 3
WORKER_SHUTDOWN_GRACE_SECONDS = 30
MANIFEST_SCHEMA_VERSION = 1
SWEBENCH_VERSION_SENTINEL = "__HERMES_SWEBENCH_VERSION__="

SAFE_DATASET_FIELDS = frozenset({
    "repo",
    "instance_id",
    "base_commit",
    "problem_statement",
    "hints_text",
    "difficulty",
})


class BenchmarkError(RuntimeError):
    """A user-actionable benchmark configuration or execution error."""


class ControllerTerminated(KeyboardInterrupt):
    def __init__(self, signum: int) -> None:
        super().__init__(f"benchmark controller received signal {signum}")
        self.signum = signum


@contextmanager
def defer_cleanup_signals():
    """Record SIGINT/SIGTERM while an already-started attempt is torn down."""
    deferred: list[int] = []

    def record(signum: int, _frame: Any) -> None:
        deferred.append(signum)

    previous = {
        signum: signal.signal(signum, record)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield deferred
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@dataclass(frozen=True)
class SweBenchRow:
    repo: str
    instance_id: str
    base_commit: str
    problem_statement: str
    hints_text: str | None = None
    difficulty: str | None = None

    def public_dict(self) -> dict[str, str]:
        result = {
            "repo": self.repo,
            "instance_id": self.instance_id,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
        }
        if self.hints_text is not None:
            result["hints_text"] = self.hints_text
        if self.difficulty is not None:
            result["difficulty"] = self.difficulty
        return result


@dataclass(frozen=True)
class InferenceOptions:
    run_id: str
    output_dir: Path
    instance_ids: tuple[str, ...]
    max_instances: int
    offset: int
    include_hints: bool
    model: str
    image_template: str
    docker_platform: str
    agent_timeout_seconds: int
    setup_timeout_seconds: int
    restart: bool
    dry_run: bool
    source_identity: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    predictions: Path
    manifest: Path
    summary: Path
    instances: Path
    evaluation_manifest: Path


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    return f"swe-verified-{stamp}"


def docker_task_id(run_id: str, instance_id: str) -> str:
    """Build a unique label-safe task id that Docker never needs to truncate."""
    validate_run_id(run_id)
    validate_instance_id(instance_id)
    digest = sha256_text(f"{run_id}\0{instance_id}")[:16]
    return f"hermes-swe-{instance_id[:32]}-{digest}"


def atomic_write_text(path: Path, content: str, *, mode: int = 0o600) -> None:
    """Atomically replace *path* with UTF-8 *content*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def encode_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def parse_swebench_row(value: Any) -> SweBenchRow:
    """Project an upstream row onto the public fields inference may consume."""
    if not isinstance(value, dict):
        raise BenchmarkError("SWE-bench dataset row must be an object")

    def required(name: str) -> str:
        item = value.get(name)
        if not isinstance(item, str) or not item.strip():
            raise BenchmarkError(f"SWE-bench dataset row has invalid {name!r}")
        return item

    instance_id = required("instance_id")
    validate_instance_id(instance_id)
    base_commit = required("base_commit")
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", base_commit):
        raise BenchmarkError(
            f"SWE-bench instance {instance_id} has an unsafe base commit"
        )
    hints = value.get("hints_text")
    difficulty = value.get("difficulty")
    return SweBenchRow(
        repo=required("repo"),
        instance_id=instance_id,
        base_commit=base_commit,
        problem_statement=required("problem_statement"),
        hints_text=hints if isinstance(hints, str) else None,
        difficulty=difficulty if isinstance(difficulty, str) else None,
    )


def validate_instance_id(instance_id: str) -> None:
    parts = instance_id.split("__")
    if (
        len(parts) != 2
        or not all(parts)
        or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts)
    ):
        raise BenchmarkError(f"Invalid SWE-bench instance id: {instance_id}")


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise BenchmarkError(
            "Run id must contain only letters, numbers, dots, underscores, and hyphens"
        )


def official_image(instance_id: str, template: str = DEFAULT_IMAGE_TEMPLATE) -> str:
    validate_instance_id(instance_id)
    repo, name = instance_id.split("__")
    image = (
        template
        .replace("{instance_id}", instance_id)
        .replace("{repo}", repo)
        .replace("{name}", name)
        .replace("{arch}", "x86_64")
        .lower()
    )
    if re.search(r"\{[^}]+\}", image):
        raise BenchmarkError(f"Unsupported placeholder in image template: {template}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9./:@_-]*", image):
        raise BenchmarkError(
            f"Image template produced an unsafe image reference: {image}"
        )
    return image


def canonical_model(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        raise BenchmarkError("Model cannot be empty")
    if normalized.startswith("openrouter/"):
        return normalized
    return f"openrouter/{normalized}"


def provider_model(model: str) -> str:
    normalized = canonical_model(model)
    return normalized.removeprefix("openrouter/")


def validate_docker_platform(platform: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", platform):
        raise BenchmarkError(f"Invalid Docker platform: {platform}")


def _redact_secret(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [_redact_secret(item, secret) for item in value]
    if isinstance(value, dict):
        return {key: _redact_secret(item, secret) for key, item in value.items()}
    return value


def build_prompt(row: SweBenchRow, include_hints: bool) -> str:
    lines = [
        "Resolve this SWE-bench Verified issue using Hermes Agent.",
        "",
        "You are running inside the official SWE-bench task image at /testbed.",
        "Edit repository files directly; do not merely describe a patch.",
        "Do not seek or use gold patches, hidden tests, or benchmark answer artifacts.",
        "Do not modify tests or benchmark metadata unless the issue explicitly requires it.",
        "",
        "The benchmark coordinator must call the fresh foreground roles exactly once",
        "and in this order: navigator, patcher, reviewer. It then reconciles their",
        "reports and leaves the final source changes in the shared worktree.",
        "",
        "## Repository",
        "Worktree: /testbed",
        f"Repo: {row.repo}",
        f"Base commit: {row.base_commit}",
        f"Instance id: {row.instance_id}",
    ]
    if row.difficulty:
        lines.append(f"Difficulty: {row.difficulty}")
    lines.append("")
    if include_hints and row.hints_text and row.hints_text.strip():
        lines.extend(["## Hints", row.hints_text.strip(), ""])
    lines.extend([
        "## Issue",
        row.problem_statement.strip(),
        "",
        "## Completion requirements",
        "- Leave the final source changes in the worktree.",
        "- Run relevant lightweight verification when feasible.",
        "- Inspect the final diff before answering.",
        "- Summarize changed files, verification commands, and residual risk.",
    ])
    return "\n".join(lines)


class DatasetRowsClient:
    """Small retrying client for Hugging Face's public rows endpoint."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or httpx.Client(timeout=60)
        self._owns_client = client is None
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> DatasetRowsClient:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def page(self, offset: int, length: int) -> list[SweBenchRow]:
        params = {
            "dataset": DATASET_NAME,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "offset": offset,
            "length": length,
        }
        for attempt in range(1, DATASET_FETCH_ATTEMPTS + 1):
            try:
                response = self._client.get(DATASET_ROWS_URL, params=params)
            except httpx.HTTPError as exc:
                if attempt == DATASET_FETCH_ATTEMPTS:
                    raise BenchmarkError(
                        f"Could not fetch SWE-bench rows: {exc}"
                    ) from exc
                self._sleep(attempt)
                continue
            if response.status_code == 200:
                try:
                    payload = response.json()
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(
                        "SWE-bench dataset response is not valid JSON"
                    ) from exc
                raw_rows = payload.get("rows") if isinstance(payload, dict) else None
                if not isinstance(raw_rows, list):
                    raise BenchmarkError("SWE-bench dataset response has no rows array")
                result = []
                for item in raw_rows:
                    if not isinstance(item, dict) or "row" not in item:
                        raise BenchmarkError(
                            "SWE-bench dataset response contains a malformed row"
                        )
                    result.append(parse_swebench_row(item["row"]))
                return result
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt == DATASET_FETCH_ATTEMPTS:
                raise BenchmarkError(
                    f"Could not fetch SWE-bench rows ({response.status_code}): "
                    f"{response.text[:500]}"
                )
            self._sleep(attempt)
        raise AssertionError("unreachable")

    def select(
        self,
        instance_ids: Sequence[str],
        *,
        offset: int,
        max_instances: int,
    ) -> list[SweBenchRow]:
        if not instance_ids:
            selected: list[SweBenchRow] = []
            page_offset = offset
            while len(selected) < max_instances:
                length = min(DATASET_PAGE_SIZE, max_instances - len(selected))
                rows = self.page(page_offset, length)
                selected.extend(rows)
                if len(rows) < length:
                    break
                page_offset += length
            return selected[:max_instances]
        if len(set(instance_ids)) != len(instance_ids):
            raise BenchmarkError("Duplicate --instance-id values are not allowed")
        wanted = set(instance_ids)
        found: dict[str, SweBenchRow] = {}
        page_offset = 0
        while len(found) < len(wanted):
            rows = self.page(page_offset, DATASET_PAGE_SIZE)
            if not rows:
                break
            for row in rows:
                if row.instance_id in wanted:
                    found[row.instance_id] = row
            page_offset += DATASET_PAGE_SIZE
        missing = [
            instance_id for instance_id in instance_ids if instance_id not in found
        ]
        if missing:
            raise BenchmarkError(
                f"Could not find SWE-bench instance(s): {', '.join(missing)}"
            )
        return [found[instance_id] for instance_id in instance_ids]


def build_paths(options: InferenceOptions) -> RunPaths:
    run_dir = options.output_dir.resolve() / "runs" / options.run_id
    return RunPaths(
        run_dir=run_dir,
        predictions=run_dir / "predictions.jsonl",
        manifest=run_dir / "prediction-manifest.json",
        summary=run_dir / "summary.json",
        instances=run_dir / "instances.jsonl",
        evaluation_manifest=run_dir / "evaluation-manifest.json",
    )


def run_command(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise BenchmarkError(f"Command timed out: {' '.join(args)}") from exc
    except OSError as exc:
        raise BenchmarkError(f"Could not run {args[0]}: {exc}") from exc
    return CommandResult(
        args=tuple(args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _docker_cli_environment() -> dict[str, str]:
    allowed = {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "TZ",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "XDG_RUNTIME_DIR",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def _evaluation_environment() -> dict[str, str]:
    env = _docker_cli_environment()
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def require_docker(*, env: dict[str, str] | None = None) -> dict[str, Any]:
    if shutil.which("docker") is None:
        raise BenchmarkError("Docker is required for local SWE-bench evaluation")
    result = run_command(
        ["docker", "info", "--format", "{{json .ServerVersion}}"],
        timeout=30,
        env=_docker_cli_environment() if env is None else env,
    )
    if result.returncode != 0:
        raise BenchmarkError(f"Docker daemon is unavailable: {result.stderr.strip()}")
    return {"serverVersion": result.stdout.strip().strip('"')}


def ensure_image(image: str, platform: str, timeout: int) -> dict[str, Any]:
    env = _docker_cli_environment()
    inspected = run_command(["docker", "image", "inspect", image], timeout=30, env=env)
    pulled = False
    if inspected.returncode != 0:
        pull = run_command(
            ["docker", "pull", "--platform", platform, image],
            timeout=timeout,
            env=env,
        )
        if pull.returncode != 0:
            raise BenchmarkError(f"Could not pull {image}: {pull.stderr.strip()}")
        pulled = True
        inspected = run_command(
            ["docker", "image", "inspect", image], timeout=30, env=env
        )
    if inspected.returncode != 0:
        raise BenchmarkError(f"Could not inspect Docker image {image}")
    payload = json.loads(inspected.stdout)
    item = payload[0] if isinstance(payload, list) and payload else {}
    return {
        "image": image,
        "id": item.get("Id"),
        "repoDigests": item.get("RepoDigests") or [],
        "pulled": pulled,
    }


def hermes_version() -> str:
    try:
        return importlib.metadata.version("hermes-agent")
    except importlib.metadata.PackageNotFoundError:
        return "source"


def hermes_source_identity() -> dict[str, Any]:
    """Bind resumable artifacts to the exact committed or dirty Hermes source."""
    root = Path(__file__).resolve().parents[2]
    try:
        revision = run_command(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            timeout=30,
            env=_docker_cli_environment(),
        )
        diff = run_command(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--binary",
                "--no-ext-diff",
                "HEAD",
                "--",
            ],
            timeout=30,
            env=_docker_cli_environment(),
        )
        untracked = run_command(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            timeout=30,
            env=_docker_cli_environment(),
        )
    except BenchmarkError:
        revision = diff = untracked = None
    if (
        not revision
        or not diff
        or not untracked
        or not all(item.returncode == 0 for item in (revision, diff, untracked))
    ):
        digest = hashlib.sha256()
        for path in (
            Path(__file__),
            Path(__file__).with_name("swebench_verified_worker.py"),
        ):
            try:
                digest.update(path.read_bytes())
            except OSError:
                digest.update(str(path).encode())
        return {
            "commit": None,
            "dirty": None,
            "fingerprint": f"package:{hermes_version()}:{digest.hexdigest()}",
        }

    digest = hashlib.sha256()
    commit = revision.stdout.strip()
    digest.update(commit.encode())
    digest.update(diff.stdout.encode())
    untracked_paths = sorted(item for item in untracked.stdout.split("\0") if item)
    for relative in untracked_paths:
        digest.update(b"\0path\0")
        digest.update(relative.encode())
        path = root / relative
        try:
            if path.is_file() and not path.is_symlink():
                digest.update(b"\0content\0")
                digest.update(path.read_bytes())
        except OSError:
            digest.update(b"\0unreadable\0")
    return {
        "commit": commit,
        "dirty": bool(diff.stdout or untracked_paths),
        "fingerprint": digest.hexdigest(),
    }


def _source_identity_for_options(options: InferenceOptions) -> dict[str, Any]:
    return validate_source_identity(options.source_identity or hermes_source_identity())


def validate_source_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "commit",
        "dirty",
        "fingerprint",
    }:
        raise BenchmarkError("Hermes source identity has an invalid schema")
    if value["commit"] is not None and not isinstance(value["commit"], str):
        raise BenchmarkError("Hermes source identity has an invalid commit")
    if value["dirty"] is not None and not isinstance(value["dirty"], bool):
        raise BenchmarkError("Hermes source identity has an invalid dirty state")
    if not isinstance(value["fingerprint"], str) or not value["fingerprint"]:
        raise BenchmarkError("Hermes source identity has an invalid fingerprint")
    return {
        "commit": value["commit"],
        "dirty": value["dirty"],
        "fingerprint": value["fingerprint"],
    }


def require_unchanged_source(options: InferenceOptions) -> None:
    expected = _source_identity_for_options(options)
    if hermes_source_identity() != expected:
        raise BenchmarkError(
            "Hermes source changed after this run started; start a new run id"
        )


def prediction_model_name(model: str) -> str:
    return f"hermes-agent@{hermes_version()}:{canonical_model(model)}"


def _worker_environment(hermes_home: Path) -> dict[str, str]:
    env = _docker_cli_environment()
    if "OPENROUTER_API_KEY" in os.environ:
        env["OPENROUTER_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    env["HERMES_HOME"] = str(hermes_home)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _terminate_worker(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=WORKER_SHUTDOWN_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _wait_for_worker(
    process: subprocess.Popen[Any],
    runtime_path: Path,
    *,
    setup_timeout_seconds: int,
    agent_timeout_seconds: int,
) -> tuple[bool, str | None]:
    setup_deadline = time.monotonic() + setup_timeout_seconds
    runtime: dict[str, Any] | None = None
    while process.poll() is None:
        runtime = _load_json_if_present(runtime_path)
        if runtime and runtime.get("agentStartedAt"):
            break
        if time.monotonic() >= setup_deadline:
            _terminate_worker(process)
            return False, "setup timeout"
        time.sleep(0.2)
    if process.poll() is not None:
        return False, None

    inference_deadline = time.monotonic() + agent_timeout_seconds
    while process.poll() is None:
        runtime = _load_json_if_present(runtime_path)
        if runtime and runtime.get("agentCompletedAt"):
            break
        if time.monotonic() >= inference_deadline:
            _terminate_worker(process)
            return True, "agent timeout"
        time.sleep(0.2)
    if process.poll() is not None:
        return False, None

    teardown_deadline = time.monotonic() + setup_timeout_seconds
    while process.poll() is None:
        if time.monotonic() >= teardown_deadline:
            _terminate_worker(process)
            return False, "teardown timeout"
        time.sleep(0.2)
    return False, None


def _capture_patch_from_container(
    container_id: str, base_commit: str
) -> tuple[str, list[str], str | None]:
    if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
        return "", [], "invalid benchmark container id"
    script = (
        "git config --global --add safe.directory /testbed && "
        "git -C /testbed add -A -- . && "
        f"git -C /testbed diff --cached --binary --no-color {base_commit} -- ."
    )
    try:
        patch = run_command(
            [
                "docker",
                "exec",
                "--workdir",
                "/testbed",
                container_id,
                "bash",
                "-c",
                script,
            ],
            timeout=120,
            env=_docker_cli_environment(),
        )
        names = run_command(
            [
                "docker",
                "exec",
                "--workdir",
                "/testbed",
                container_id,
                "bash",
                "-c",
                f"git -C /testbed diff --cached --name-only -z {base_commit} -- .",
            ],
            timeout=30,
            env=_docker_cli_environment(),
        )
    except BenchmarkError as exc:
        return "", [], str(exc)
    errors = [
        result.stderr.strip() or result.stdout.strip() or fallback
        for result, fallback in (
            (patch, "patch capture failed"),
            (names, "changed-path capture failed"),
        )
        if result.returncode != 0
    ]
    if errors:
        return "", [], "; ".join(errors)[-2000:]
    return patch.stdout, [item for item in names.stdout.split("\0") if item], None


def _restart_container_for_capture(container_id: str) -> str | None:
    """Quiesce every timed-out exec process before controller-side capture."""
    if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
        return "invalid benchmark container id"
    try:
        result = run_command(
            ["docker", "restart", "--time", "0", container_id],
            timeout=60,
            env=_docker_cli_environment(),
        )
    except BenchmarkError as exc:
        return str(exc)
    if result.returncode == 0:
        return None
    return result.stderr.strip() or result.stdout.strip() or "container restart failed"


def _restore_container_ownership(container_id: str) -> str | None:
    if (
        not re.fullmatch(r"[0-9a-f]{12,64}", container_id)
        or not hasattr(os, "getuid")
        or not hasattr(os, "getgid")
    ):
        return None
    try:
        result = run_command(
            [
                "docker",
                "exec",
                "--user",
                "root",
                container_id,
                "chown",
                "-R",
                f"{os.getuid()}:{os.getgid()}",
                "/root",
                "/workspace",
            ],
            timeout=120,
            env=_docker_cli_environment(),
        )
    except BenchmarkError as exc:
        return str(exc)
    if result.returncode == 0:
        return None
    output = result.stderr.strip() or result.stdout.strip()
    remaining = [
        line for line in output.splitlines() if "Read-only file system" not in line
    ]
    if output and not remaining:
        return None
    return ("\n".join(remaining) or "ownership handoff failed")[-2000:]


def _remove_container(container_id: str | None) -> str | None:
    if not container_id or not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
        return None
    try:
        result = run_command(
            ["docker", "rm", "-f", container_id],
            timeout=30,
            env=_docker_cli_environment(),
        )
    except BenchmarkError as exc:
        return str(exc)
    if result.returncode == 0 or "no such container" in result.stderr.lower():
        return None
    return result.stderr.strip() or result.stdout.strip() or "container removal failed"


def _find_benchmark_container(task_id: str) -> str | None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,63}", task_id):
        return None
    try:
        result = run_command(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                "label=hermes-agent=1",
                "--filter",
                f"label=hermes-task-id={task_id}",
            ],
            timeout=30,
            env=_docker_cli_environment(),
        )
    except BenchmarkError:
        return None
    if result.returncode != 0:
        return None
    return next(
        (
            item
            for item in result.stdout.splitlines()
            if re.fullmatch(r"[0-9a-f]{12,64}", item)
        ),
        None,
    )


def _cleanup_sandbox_directory(
    sandbox_path: Path, image: str, platform: str
) -> str | None:
    """Remove run-scoped bind mounts, repairing root ownership if necessary."""
    try:
        if not sandbox_path.exists():
            return None
        if sandbox_path.is_symlink():
            return "refusing to clean a symlinked benchmark sandbox"
    except OSError as exc:
        return f"Could not inspect benchmark sandbox: {exc}"
    try:
        shutil.rmtree(sandbox_path)
        return None
    except OSError:
        pass
    if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
        return "benchmark sandbox contains files the host user cannot remove"
    try:
        result = run_command(
            [
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "--network",
                "none",
                "--platform",
                platform,
                "--user",
                "root",
                "--entrypoint",
                "chown",
                "-v",
                f"{sandbox_path}:/sandbox",
                image,
                "-R",
                f"{os.getuid()}:{os.getgid()}",
                "/sandbox",
            ],
            timeout=120,
            env=_docker_cli_environment(),
        )
    except BenchmarkError as exc:
        return str(exc)
    if result.returncode != 0:
        return result.stderr.strip() or result.stdout.strip() or "sandbox chown failed"
    try:
        shutil.rmtree(sandbox_path)
    except OSError as exc:
        return f"Could not remove benchmark sandbox: {exc}"
    return None


def _redact_worker_logs(paths: Sequence[Path], secret: str) -> list[str]:
    errors = []
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            redacted = content.replace(secret, "[REDACTED]") if secret else content
            if redacted != content:
                atomic_write_text(path, redacted)
        except BaseException as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
    return errors


def _run_worker(
    options: InferenceOptions,
    row: SweBenchRow,
    instance_dir: Path,
    image: str,
    image_metadata: dict[str, Any],
) -> dict[str, Any]:
    instance_dir.mkdir(parents=True, exist_ok=True)
    worker_home = instance_dir / "hermes-home"
    request_path = instance_dir / "worker-request.json"
    result_path = instance_dir / "worker-result.json"
    runtime_path = instance_dir / "runtime.json"
    stdout_path = instance_dir / "worker.stdout.log"
    stderr_path = instance_dir / "worker.stderr.log"
    task_id = docker_task_id(options.run_id, row.instance_id)
    sandbox_path = worker_home / "sandboxes" / "docker" / task_id
    for path in (
        sandbox_path / "home" / ".hermes" / "skills",
        sandbox_path / "workspace",
    ):
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
    prompt = build_prompt(row, options.include_hints)
    atomic_write_text(instance_dir / "prompt.md", prompt)
    atomic_write_json(
        request_path,
        {
            "schemaVersion": 1,
            "benchmark": BENCHMARK,
            "row": row.public_dict(),
            "prompt": prompt,
            "includeHints": options.include_hints,
            "model": canonical_model(options.model),
            "image": image,
            "imageMetadata": image_metadata,
            "dockerPlatform": options.docker_platform,
            "taskId": task_id,
            "resultPath": str(result_path),
            "runtimePath": str(runtime_path),
            "hermesHome": str(worker_home),
            "agentTimeoutSeconds": options.agent_timeout_seconds,
            "attempt": 1,
            "maxInfrastructureRetries": DEFAULT_INFRASTRUCTURE_RETRIES,
            "sourceIdentity": _source_identity_for_options(options),
        },
    )

    command = [
        sys.executable,
        "-m",
        "hermes_cli.benchmarks.swebench_verified_worker",
        "--request",
        str(request_path),
    ]
    started_at = utc_now()
    secret = os.environ.get("OPENROUTER_API_KEY", "")
    process: subprocess.Popen[Any] | None = None
    timed_out = False
    timeout_error = None
    returncode = None
    controller_error: BaseException | None = None

    def remember_controller_error(exc: BaseException) -> None:
        nonlocal controller_error
        if controller_error is None:
            controller_error = exc

    for log_path in (stdout_path, stderr_path):
        atomic_write_text(log_path, "")
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        try:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=_worker_environment(worker_home),
                stdout=stdout_handle,
                stderr=stderr_handle,
                stdin=subprocess.DEVNULL,
                text=True,
            )
            timed_out, timeout_error = _wait_for_worker(
                process,
                runtime_path,
                setup_timeout_seconds=options.setup_timeout_seconds,
                agent_timeout_seconds=options.agent_timeout_seconds,
            )
        except BaseException as exc:
            remember_controller_error(exc)
            if process is not None:
                try:
                    _terminate_worker(process)
                except BaseException as cleanup_exc:
                    timeout_error = (
                        "controller could not terminate worker: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
        finally:
            if process is not None:
                returncode = process.returncode

    result: dict[str, Any] = {}
    container_id = None
    container_restart_error = None
    log_redaction_errors: list[str] = []
    with defer_cleanup_signals() as deferred_signals:
        if process is not None and process.returncode is None:
            try:
                _terminate_worker(process)
                returncode = process.returncode
            except BaseException as exc:
                remember_controller_error(exc)
                timeout_error = (
                    "controller could not terminate worker during cleanup: "
                    f"{type(exc).__name__}: {exc}"
                )
        log_redaction_errors.extend(
            _redact_worker_logs((stdout_path, stderr_path), secret)
        )

        try:
            runtime = _load_json_if_present(runtime_path) or {}
            result = _load_json_if_present(result_path) or {}
            candidate = runtime.get("containerId")
            if isinstance(candidate, str):
                container_id = candidate
        except BaseException as exc:
            remember_controller_error(exc)
        if container_id is None:
            try:
                container_id = _find_benchmark_container(task_id)
            except BaseException as exc:
                remember_controller_error(exc)

        # Patch capture is always controller-owned. Even a malformed or missing
        # worker result may follow valid worktree edits, and restart is the
        # barrier that makes tracked and untracked background jobs harmless.
        if container_id:
            try:
                container_restart_error = _restart_container_for_capture(container_id)
                if container_restart_error:
                    patch, changed_paths, capture_error = (
                        "",
                        [],
                        container_restart_error,
                    )
                else:
                    patch, changed_paths, capture_error = _capture_patch_from_container(
                        container_id, row.base_commit
                    )
            except BaseException as exc:
                remember_controller_error(exc)
                patch, changed_paths, capture_error = (
                    "",
                    [],
                    f"controller patch capture failed: {type(exc).__name__}: {exc}",
                )
            result.update({
                "modelPatch": patch,
                "changedPaths": changed_paths,
                "patchCaptureError": capture_error,
            })
        else:
            result.update({
                "modelPatch": "",
                "changedPaths": [],
                "patchCaptureError": "container id unavailable",
            })

        try:
            if container_id:
                ownership_restart_error = container_restart_error
                if ownership_restart_error:
                    result["sandboxOwnershipError"] = ownership_restart_error
                else:
                    ownership_error = _restore_container_ownership(container_id)
                    if ownership_error:
                        result["sandboxOwnershipError"] = ownership_error
        except BaseException as exc:
            remember_controller_error(exc)
            result["sandboxOwnershipError"] = (
                f"controller ownership handoff failed: {type(exc).__name__}: {exc}"
            )
        container_cleanup_error = None
        for _attempt in range(2):
            try:
                container_cleanup_error = _remove_container(container_id)
                break
            except BaseException as exc:
                remember_controller_error(exc)
                container_cleanup_error = (
                    f"controller container cleanup failed: {type(exc).__name__}: {exc}"
                )
        if container_cleanup_error:
            result["containerCleanupError"] = container_cleanup_error
        else:
            try:
                sandbox_cleanup_error = _cleanup_sandbox_directory(
                    sandbox_path, image, options.docker_platform
                )
            except BaseException as exc:
                remember_controller_error(exc)
                sandbox_cleanup_error = (
                    f"controller sandbox cleanup failed: {type(exc).__name__}: {exc}"
                )
            if sandbox_cleanup_error:
                result["sandboxCleanupError"] = sandbox_cleanup_error
        log_redaction_errors.extend(
            _redact_worker_logs((stdout_path, stderr_path), secret)
        )
        result.update({
            "workerCommand": command,
            "workerReturnCode": returncode,
            "controllerStartedAt": started_at,
            "controllerCompletedAt": utc_now(),
            "timedOut": bool(result.get("timedOut")) or timed_out,
            "error": result.get("error") or timeout_error,
        })
        if log_redaction_errors:
            result["logRedactionErrors"] = list(dict.fromkeys(log_redaction_errors))
        redacted_result = _redact_secret(result, secret)
        try:
            atomic_write_json(instance_dir / "controller-result.json", redacted_result)
            atomic_write_text(
                instance_dir / "model.patch",
                redacted_result.get("modelPatch")
                if isinstance(redacted_result.get("modelPatch"), str)
                else "",
            )
        except OSError:
            pass

    if deferred_signals and controller_error is None:
        controller_error = (
            ControllerTerminated(deferred_signals[0])
            if deferred_signals[0] == signal.SIGTERM
            else KeyboardInterrupt()
        )
    if controller_error is not None:
        raise controller_error
    return redacted_result


def _prediction(row: SweBenchRow, model: str, result: dict[str, Any]) -> dict[str, str]:
    patch = result.get("modelPatch")
    return {
        "instance_id": row.instance_id,
        "model_name_or_path": prediction_model_name(model),
        "model_patch": patch if isinstance(patch, str) else "",
    }


def _instance_summary(
    row: SweBenchRow,
    image: str,
    result: dict[str, Any],
    prediction: dict[str, str],
) -> dict[str, Any]:
    return {
        "instanceId": row.instance_id,
        "repo": row.repo,
        "baseCommit": row.base_commit,
        "image": image,
        "attemptsUsed": DEFAULT_ATTEMPTS,
        "infrastructureRetriesUsed": DEFAULT_INFRASTRUCTURE_RETRIES,
        "semanticRetriesUsed": 0,
        "timedOut": bool(result.get("timedOut")),
        "workflowComplete": bool(result.get("workflowComplete")),
        "generationSucceeded": (
            not result.get("error")
            and not result.get("timedOut")
            and bool(result.get("workflowComplete"))
            and bool(prediction["model_patch"].strip())
        ),
        "changedPaths": result.get("changedPaths") or [],
        "patchCaptureError": result.get("patchCaptureError"),
        "sandboxOwnershipError": result.get("sandboxOwnershipError"),
        "sandboxCleanupError": result.get("sandboxCleanupError"),
        "containerCleanupError": result.get("containerCleanupError"),
        "error": result.get("error"),
        "workerReturnCode": result.get("workerReturnCode"),
        "phases": result.get("phases") or [],
        "startedAt": result.get("controllerStartedAt"),
        "completedAt": result.get("controllerCompletedAt"),
    }


def _manifest(
    options: InferenceOptions,
    rows: Sequence[SweBenchRow],
    predictions: Sequence[dict[str, str]],
    *,
    complete: bool,
) -> dict[str, Any]:
    content = encode_jsonl(predictions)
    return {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "dataset": DATASET_NAME,
        "datasetConfig": DATASET_CONFIG,
        "datasetSplit": DATASET_SPLIT,
        "runId": options.run_id,
        "framework": "hermes-agent",
        "hermesVersion": hermes_version(),
        "sourceIdentity": _source_identity_for_options(options),
        "model": canonical_model(options.model),
        "temperature": 0.1,
        "inferenceRuntime": "official-swebench-instance-image",
        "imageTemplate": options.image_template,
        "dockerPlatform": options.docker_platform,
        "includeHints": options.include_hints,
        "inferenceWorkers": DEFAULT_INFERENCE_WORKERS,
        "attemptsPerInstance": DEFAULT_ATTEMPTS,
        "maxInfrastructureRetries": DEFAULT_INFRASTRUCTURE_RETRIES,
        "apiMaxRetries": DEFAULT_API_MAX_RETRIES,
        "codingContext": DEFAULT_CODING_CONTEXT,
        "agentTimeoutSeconds": options.agent_timeout_seconds,
        "setupTimeoutSeconds": options.setup_timeout_seconds,
        "agentSequence": list(DEFAULT_AGENT_SEQUENCE),
        "agentBudgets": {
            "coordinator": DEFAULT_COORDINATOR_BUDGET,
            **DEFAULT_PHASE_BUDGETS,
        },
        "selectedInstances": [
            {
                "instanceId": row.instance_id,
                "repo": row.repo,
                "baseCommit": row.base_commit,
                "image": official_image(row.instance_id, options.image_template),
            }
            for row in rows
        ],
        "completedInstanceIds": [
            prediction["instance_id"] for prediction in predictions
        ],
        "complete": complete,
        "predictionCount": len(predictions),
        "nonEmptyPatchCount": sum(
            bool(prediction["model_patch"].strip()) for prediction in predictions
        ),
        "predictionsSha256": sha256_text(content),
        "generatedAt": utc_now(),
    }


def _write_progress(
    options: InferenceOptions,
    paths: RunPaths,
    rows: Sequence[SweBenchRow],
    summaries: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, str]],
    *,
    complete: bool,
) -> None:
    content = encode_jsonl(predictions)
    manifest = _manifest(options, rows, predictions, complete=complete)
    if manifest["predictionsSha256"] != sha256_text(content):
        raise AssertionError("prediction manifest digest mismatch")
    atomic_write_text(paths.predictions, content)
    atomic_write_json(paths.manifest, manifest)
    atomic_write_json(
        paths.summary,
        {
            "runId": options.run_id,
            "benchmark": BENCHMARK,
            "dataset": DATASET_NAME,
            "model": canonical_model(options.model),
            "selectedCount": len(rows),
            "completedCount": len(predictions),
            "generationSucceededCount": sum(
                summary.get("generationSucceeded") is True for summary in summaries
            ),
            "predictionCount": len(predictions),
            "nonEmptyPatchCount": sum(
                bool(prediction["model_patch"].strip()) for prediction in predictions
            ),
            "complete": complete,
            "predictionsPath": str(paths.predictions),
            "predictionManifestPath": str(paths.manifest),
            "predictionsSha256": manifest["predictionsSha256"],
            "summaries": list(summaries),
        },
    )


def _load_resume(
    options: InferenceOptions,
    paths: RunPaths,
    rows: Sequence[SweBenchRow],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not paths.manifest.exists():
        if any((paths.predictions.exists(), paths.summary.exists())):
            raise BenchmarkError(
                f"Run {options.run_id} has incomplete metadata; use --restart"
            )
        return [], []
    try:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise BenchmarkError("Could not read the incomplete run manifest") from exc
    if not isinstance(manifest, dict):
        raise BenchmarkError("Incomplete run manifest must be a JSON object")
    expected_instances = [
        {
            "instanceId": row.instance_id,
            "repo": row.repo,
            "baseCommit": row.base_commit,
            "image": official_image(row.instance_id, options.image_template),
        }
        for row in rows
    ]
    mismatches = []
    if manifest.get("schemaVersion") != MANIFEST_SCHEMA_VERSION:
        mismatches.append("manifest schema")
    if manifest.get("benchmark") != BENCHMARK:
        mismatches.append("benchmark")
    if manifest.get("dataset") != DATASET_NAME:
        mismatches.append("dataset")
    if manifest.get("datasetConfig") != DATASET_CONFIG:
        mismatches.append("dataset config")
    if manifest.get("datasetSplit") != DATASET_SPLIT:
        mismatches.append("dataset split")
    if manifest.get("framework") != "hermes-agent":
        mismatches.append("framework")
    if manifest.get("inferenceRuntime") != "official-swebench-instance-image":
        mismatches.append("inference runtime")
    if manifest.get("runId") != options.run_id:
        mismatches.append("run id")
    if manifest.get("model") != canonical_model(options.model):
        mismatches.append("model")
    if manifest.get("selectedInstances") != expected_instances:
        mismatches.append("selected instances")
    if manifest.get("imageTemplate") != options.image_template:
        mismatches.append("image template")
    if manifest.get("dockerPlatform") != options.docker_platform:
        mismatches.append("Docker platform")
    if manifest.get("includeHints") != options.include_hints:
        mismatches.append("hint policy")
    if manifest.get("agentTimeoutSeconds") != options.agent_timeout_seconds:
        mismatches.append("agent timeout")
    if manifest.get("setupTimeoutSeconds") != options.setup_timeout_seconds:
        mismatches.append("setup timeout")
    if manifest.get("hermesVersion") != hermes_version():
        mismatches.append("Hermes version")
    if manifest.get("sourceIdentity") != _source_identity_for_options(options):
        mismatches.append("Hermes source identity")
    if manifest.get("temperature") != 0.1:
        mismatches.append("temperature")
    if manifest.get("inferenceWorkers") != DEFAULT_INFERENCE_WORKERS:
        mismatches.append("inference workers")
    if manifest.get("attemptsPerInstance") != DEFAULT_ATTEMPTS:
        mismatches.append("attempt policy")
    if manifest.get("maxInfrastructureRetries") != DEFAULT_INFRASTRUCTURE_RETRIES:
        mismatches.append("retry policy")
    if manifest.get("apiMaxRetries") != DEFAULT_API_MAX_RETRIES:
        mismatches.append("API retry policy")
    if manifest.get("codingContext") != DEFAULT_CODING_CONTEXT:
        mismatches.append("coding context")
    if manifest.get("agentSequence") != list(DEFAULT_AGENT_SEQUENCE):
        mismatches.append("agent sequence")
    if manifest.get("agentBudgets") != {
        "coordinator": DEFAULT_COORDINATOR_BUDGET,
        **DEFAULT_PHASE_BUDGETS,
    }:
        mismatches.append("agent budgets")
    if mismatches:
        raise BenchmarkError(
            f"Existing run differs in {', '.join(mismatches)}; use --restart"
        )
    if manifest.get("complete"):
        raise BenchmarkError(
            f"Run {options.run_id} is already complete; use a new run id or --restart"
        )
    try:
        prediction_content = paths.predictions.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchmarkError(
            "Incomplete run is missing its predictions checkpoint"
        ) from exc
    if sha256_text(prediction_content) != manifest.get("predictionsSha256"):
        raise BenchmarkError(
            "Incomplete run predictions do not match their checkpoint manifest"
        )
    aggregate_predictions = []
    for line_number, line in enumerate(prediction_content.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(
                f"Incomplete run has invalid prediction JSONL line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise BenchmarkError("Incomplete run contains an invalid prediction")
        aggregate_predictions.append(value)
    if len(aggregate_predictions) != manifest.get("predictionCount"):
        raise BenchmarkError("Incomplete run prediction count does not match manifest")
    if [item.get("instance_id") for item in aggregate_predictions] != manifest.get(
        "completedInstanceIds"
    ):
        raise BenchmarkError("Incomplete run prediction IDs do not match its manifest")
    summaries = []
    predictions = []
    for row in rows:
        instance_dir = paths.run_dir / "instances" / row.instance_id
        prediction_path = instance_dir / "prediction.json"
        summary_path = instance_dir / "run.json"
        if not prediction_path.exists() or not summary_path.exists():
            break
        try:
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise BenchmarkError(
                f"Could not read resume checkpoint for {row.instance_id}"
            ) from exc
        if not isinstance(prediction, dict) or not isinstance(summary, dict):
            raise BenchmarkError(
                f"Resume checkpoint for {row.instance_id} must contain JSON objects"
            )
        if prediction.get("instance_id") != row.instance_id:
            raise BenchmarkError(f"Resume checkpoint does not match {row.instance_id}")
        if summary.get("instanceId") != row.instance_id:
            raise BenchmarkError(f"Resume summary does not match {row.instance_id}")
        if len(predictions) >= len(aggregate_predictions):
            raise BenchmarkError("Resume checkpoint exceeds aggregate predictions")
        if prediction != aggregate_predictions[len(predictions)]:
            raise BenchmarkError(
                f"Resume checkpoint prediction differs for {row.instance_id}"
            )
        predictions.append(prediction)
        summaries.append(summary)
    if len(predictions) != len(aggregate_predictions):
        raise BenchmarkError("Incomplete run is missing per-instance checkpoints")
    return summaries, predictions


def run_inference(
    options: InferenceOptions,
    *,
    dataset_client: DatasetRowsClient | None = None,
) -> RunPaths:
    if options.source_identity is None:
        options = replace(options, source_identity=hermes_source_identity())
    validate_run_id(options.run_id)
    validate_docker_platform(options.docker_platform)
    canonical_model(options.model)
    official_image(DEFAULT_SMOKE_INSTANCE_ID, options.image_template)
    owned_client = dataset_client is None
    client = dataset_client or DatasetRowsClient()
    try:
        rows = client.select(
            options.instance_ids,
            offset=options.offset,
            max_instances=options.max_instances,
        )
    finally:
        if owned_client:
            client.close()
    if not rows:
        raise BenchmarkError("Dataset selection produced no instances")

    paths = build_paths(options)
    if options.dry_run:
        print(
            json.dumps(
                {
                    "mode": "inference",
                    "runId": options.run_id,
                    "dataset": DATASET_NAME,
                    "instanceIds": [row.instance_id for row in rows],
                    "model": canonical_model(options.model),
                    "sourceIdentity": _source_identity_for_options(options),
                    "temperature": 0.1,
                    "inferenceWorkers": DEFAULT_INFERENCE_WORKERS,
                    "attemptsPerInstance": DEFAULT_ATTEMPTS,
                    "maxInfrastructureRetries": DEFAULT_INFRASTRUCTURE_RETRIES,
                    "apiMaxRetries": DEFAULT_API_MAX_RETRIES,
                    "codingContext": DEFAULT_CODING_CONTEXT,
                    "agentTimeoutSeconds": options.agent_timeout_seconds,
                    "setupTimeoutSeconds": options.setup_timeout_seconds,
                    "sequence": list(DEFAULT_AGENT_SEQUENCE),
                    "budgets": [
                        DEFAULT_COORDINATOR_BUDGET,
                        *DEFAULT_PHASE_BUDGETS.values(),
                    ],
                    "images": [
                        official_image(row.instance_id, options.image_template)
                        for row in rows
                    ],
                    "output": str(paths.run_dir),
                },
                indent=2,
            )
        )
        return paths

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise BenchmarkError("OPENROUTER_API_KEY is required for inference")
    docker_metadata = require_docker()
    if options.restart and paths.run_dir.exists():
        shutil.rmtree(paths.run_dir)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths.instances, encode_jsonl(row.public_dict() for row in rows))
    summaries, predictions = _load_resume(options, paths, rows)
    require_unchanged_source(options)
    _write_progress(options, paths, rows, summaries, predictions, complete=False)

    for row in rows[len(predictions) :]:
        instance_dir = paths.run_dir / "instances" / row.instance_id
        image = official_image(row.instance_id, options.image_template)
        image_metadata: dict[str, Any] = {"image": image}
        try:
            image_metadata = ensure_image(
                image, options.docker_platform, options.setup_timeout_seconds
            )
            image_metadata["docker"] = docker_metadata
            require_unchanged_source(options)
            result = _run_worker(options, row, instance_dir, image, image_metadata)
        except Exception as exc:
            result = {
                "modelPatch": "",
                "changedPaths": [],
                "workflowComplete": False,
                "timedOut": False,
                "error": f"{type(exc).__name__}: {exc}",
                "controllerStartedAt": utc_now(),
                "controllerCompletedAt": utc_now(),
            }
        require_unchanged_source(options)
        prediction = _prediction(row, options.model, result)
        summary = _instance_summary(row, image, result, prediction)
        instance_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(instance_dir / "model.patch", prediction["model_patch"])
        atomic_write_json(instance_dir / "prediction.json", prediction)
        atomic_write_json(instance_dir / "run.json", summary)
        predictions.append(prediction)
        summaries.append(summary)
        _write_progress(
            options,
            paths,
            rows,
            summaries,
            predictions,
            complete=len(predictions) == len(rows),
        )
        status = "ok" if summary["generationSucceeded"] else "failed"
        print(f"[{len(predictions)}/{len(rows)}] {row.instance_id}: {status}")
    require_unchanged_source(options)
    return paths


def verify_prediction_artifact(
    predictions_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        content = predictions_path.read_text(encoding="utf-8")
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise BenchmarkError(f"Could not read prediction artifact: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BenchmarkError("Prediction manifest must be a JSON object")
    if (
        manifest.get("schemaVersion") != MANIFEST_SCHEMA_VERSION
        or manifest.get("benchmark") != BENCHMARK
        or manifest.get("dataset") != DATASET_NAME
        or manifest.get("datasetConfig") != DATASET_CONFIG
        or manifest.get("datasetSplit") != DATASET_SPLIT
        or manifest.get("framework") != "hermes-agent"
        or not manifest.get("complete")
    ):
        raise BenchmarkError(
            "Prediction manifest is not a complete SWE-bench Verified run"
        )
    digest = sha256_text(content)
    if digest != manifest.get("predictionsSha256"):
        raise BenchmarkError("Predictions changed after inference; refusing evaluation")
    predictions = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(
                f"Invalid prediction JSONL line {line_number}"
            ) from exc
        if not isinstance(value, dict) or not all(
            isinstance(value.get(key), str)
            for key in ("instance_id", "model_name_or_path", "model_patch")
        ):
            raise BenchmarkError(f"Invalid prediction schema on line {line_number}")
        predictions.append(value)
    prediction_ids = [item["instance_id"] for item in predictions]
    selected_instances = manifest.get("selectedInstances")
    if not isinstance(selected_instances, list) or not all(
        isinstance(item, dict) for item in selected_instances
    ):
        raise BenchmarkError("Prediction manifest has invalid selected instances")
    selected_ids = [item.get("instanceId") for item in selected_instances]
    if prediction_ids != selected_ids or prediction_ids != manifest.get(
        "completedInstanceIds"
    ):
        raise BenchmarkError("Prediction IDs or ordering do not match the manifest")
    if len(predictions) != manifest.get("predictionCount"):
        raise BenchmarkError("Prediction count does not match the manifest")
    return manifest, predictions, digest


def build_evaluation_command(
    python: str,
    predictions_path: Path,
    run_id: str,
    instance_ids: Sequence[str],
    *,
    namespace_empty: bool,
) -> list[str]:
    command = [
        python,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        DATASET_NAME,
        "--split",
        DATASET_SPLIT,
        "--predictions_path",
        str(predictions_path.resolve()),
        "--max_workers",
        str(DEFAULT_EVALUATION_WORKERS),
        "--timeout",
        str(DEFAULT_EVALUATION_TIMEOUT_SECONDS),
        "--run_id",
        run_id,
        "--modal",
        "false",
    ]
    if namespace_empty:
        command.extend(["--namespace", "none"])
    command.extend(["--instance_ids", *instance_ids])
    return command


def parse_swebench_version(stdout: str) -> str | None:
    versions = [
        line.removeprefix(SWEBENCH_VERSION_SENTINEL).strip()
        for line in stdout.splitlines()
        if line.startswith(SWEBENCH_VERSION_SENTINEL)
    ]
    return versions[-1] if versions else None


def cleanup_evaluation_containers(
    run_id: str,
    instance_ids: Sequence[str],
    *,
    env: dict[str, str],
) -> list[str]:
    """Remove only official evaluator containers belonging to this run."""
    target_names = {
        f"sweb.eval.{instance_id.lower()}.{run_id}" for instance_id in instance_ids
    }
    try:
        listing = run_command(
            ["docker", "ps", "-a", "--format", "{{.ID}}\t{{.Names}}"],
            timeout=30,
            env=env,
        )
    except BenchmarkError as exc:
        return [str(exc)]
    if listing.returncode != 0:
        return [
            listing.stderr.strip()
            or listing.stdout.strip()
            or "could not list evaluator containers"
        ]
    container_ids = [
        container_id
        for line in listing.stdout.splitlines()
        for container_id, separator, name in (line.partition("\t"),)
        if separator
        and name in target_names
        and re.fullmatch(r"[0-9a-f]{12,64}", container_id)
    ]
    if not container_ids:
        return []
    try:
        removed = run_command(
            ["docker", "rm", "-f", *container_ids],
            timeout=60,
            env=env,
        )
    except BenchmarkError as exc:
        return [str(exc)]
    if removed.returncode == 0:
        return []
    return [
        removed.stderr.strip()
        or removed.stdout.strip()
        or "could not remove evaluator containers"
    ]


def run_evaluation(args: argparse.Namespace) -> None:
    validate_run_id(args.run_id)
    output_dir = Path(args.output_dir).resolve()
    run_dir = output_dir / "runs" / args.run_id
    predictions_path = (
        Path(args.predictions_path).resolve()
        if args.predictions_path
        else run_dir / "predictions.jsonl"
    )
    manifest_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path
        else run_dir / "prediction-manifest.json"
    )
    manifest, predictions, digest = verify_prediction_artifact(
        predictions_path, manifest_path
    )
    if manifest.get("runId") != args.run_id:
        raise BenchmarkError("Prediction manifest run id does not match --run-id")
    available_ids = [prediction["instance_id"] for prediction in predictions]
    requested_ids = args.instance_id or available_ids
    if len(set(requested_ids)) != len(requested_ids):
        raise BenchmarkError("Duplicate --instance-id values are not allowed")
    for instance_id in requested_ids:
        validate_instance_id(instance_id)
    missing = [item for item in requested_ids if item not in set(available_ids)]
    if missing:
        raise BenchmarkError(
            f"Prediction artifact does not contain: {', '.join(missing)}"
        )
    command = build_evaluation_command(
        args.python,
        predictions_path,
        args.run_id,
        requested_ids,
        namespace_empty=args.namespace_empty,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "evaluation",
                    "predictionsPath": str(predictions_path),
                    "manifestPath": str(manifest_path),
                    "predictionsSha256": digest,
                    "instanceIds": requested_ids,
                    "command": command,
                },
                indent=2,
            )
        )
        return
    evaluation_env = _evaluation_environment()
    require_docker(env=evaluation_env)
    version = run_command(
        [
            args.python,
            "-c",
            "import importlib.metadata; "
            f"print('{SWEBENCH_VERSION_SENTINEL}' + "
            "importlib.metadata.version('swebench'))",
        ],
        timeout=30,
        env=evaluation_env,
    )
    found_version = (
        parse_swebench_version(version.stdout) if version.returncode == 0 else None
    )
    if found_version != REQUIRED_SWEBENCH_VERSION:
        raise BenchmarkError(
            f"Expected swebench=={REQUIRED_SWEBENCH_VERSION}, "
            f"found {found_version or 'missing'}"
        )
    evaluation_manifest = run_dir / "evaluation-manifest.json"
    started_at = utc_now()
    evaluation_record = {
        "schemaVersion": 1,
        "benchmark": BENCHMARK,
        "dataset": DATASET_NAME,
        "runId": args.run_id,
        "predictionsPath": str(predictions_path),
        "predictionManifestPath": str(manifest_path),
        "predictionsSha256": digest,
        "swebenchVersion": found_version,
        "instanceIds": requested_ids,
        "maxWorkers": DEFAULT_EVALUATION_WORKERS,
        "testTimeoutSeconds": DEFAULT_EVALUATION_TIMEOUT_SECONDS,
        "command": command,
        "startedAt": started_at,
    }
    atomic_write_json(evaluation_manifest, {**evaluation_record, "status": "running"})
    result: CommandResult | None = None
    execution_error: BaseException | None = None
    try:
        result = run_command(command, cwd=run_dir, env=evaluation_env)
    except BaseException as exc:
        execution_error = exc
    with defer_cleanup_signals() as deferred_signals:
        cleanup_errors = cleanup_evaluation_containers(
            args.run_id, requested_ids, env=evaluation_env
        )
        atomic_write_text(
            run_dir / "evaluation.stdout.log", result.stdout if result else ""
        )
        atomic_write_text(
            run_dir / "evaluation.stderr.log", result.stderr if result else ""
        )
        interrupted = isinstance(execution_error, KeyboardInterrupt) or bool(
            deferred_signals
        )
        status = "interrupted" if interrupted else "failed"
        if (
            not interrupted
            and execution_error is None
            and result is not None
            and result.returncode == 0
            and not cleanup_errors
        ):
            status = "completed"
        final_record = {
            **evaluation_record,
            "status": status,
            "completedAt": utc_now(),
            "exitCode": result.returncode if result is not None else None,
            "containerCleanupErrors": cleanup_errors,
        }
        if execution_error is not None:
            final_record["error"] = (
                f"{type(execution_error).__name__}: {execution_error}"
            )
        atomic_write_json(evaluation_manifest, final_record)
    if deferred_signals and execution_error is None:
        execution_error = (
            ControllerTerminated(deferred_signals[0])
            if deferred_signals[0] == signal.SIGTERM
            else KeyboardInterrupt()
        )
        final_record["status"] = "interrupted"
        final_record["error"] = f"{type(execution_error).__name__}: {execution_error}"
        atomic_write_json(evaluation_manifest, final_record)
    if execution_error is not None:
        raise execution_error
    if result is None:
        raise BenchmarkError("Official SWE-bench evaluation produced no result")
    if result.returncode != 0:
        raise BenchmarkError(
            f"Official SWE-bench evaluation failed with exit code {result.returncode}"
        )
    if cleanup_errors:
        raise BenchmarkError(
            "Official SWE-bench evaluation left containers behind: "
            + "; ".join(cleanup_errors)
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-swebench-verified",
        description="Run Hermes Agent on SWE-bench Verified with local Docker evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Generate SWE-bench predictions")
    infer.add_argument("--run-id", default=default_run_id())
    infer.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    infer.add_argument("--instance-id", action="append", default=[])
    infer.add_argument("--max-instances", type=_positive_int)
    infer.add_argument("--offset", type=_nonnegative_int)
    infer.add_argument("--include-hints", action="store_true")
    infer.add_argument("--model", default=DEFAULT_MODEL)
    infer.add_argument("--image-template", default=DEFAULT_IMAGE_TEMPLATE)
    infer.add_argument("--docker-platform", default=DEFAULT_DOCKER_PLATFORM)
    infer.add_argument(
        "--agent-timeout-seconds",
        type=_positive_int,
        default=DEFAULT_AGENT_TIMEOUT_SECONDS,
    )
    infer.add_argument(
        "--setup-timeout-seconds",
        type=_positive_int,
        default=DEFAULT_SETUP_TIMEOUT_SECONDS,
    )
    infer.add_argument("--restart", action="store_true")
    infer.add_argument("--dry-run", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate", aliases=["eval"], help="Run the pinned official local evaluator"
    )
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    evaluate.add_argument("--predictions-path")
    evaluate.add_argument("--manifest-path")
    evaluate.add_argument("--instance-id", action="append", default=[])
    evaluate.add_argument("--python", default=sys.executable)
    evaluate.add_argument("--namespace-empty", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")

    listing = subparsers.add_parser("list", help="List selected public instances")
    listing.add_argument("--instance-id", action="append", default=[])
    listing.add_argument("--max-instances", type=_positive_int, default=1)
    listing.add_argument("--offset", type=_nonnegative_int, default=0)
    return parser


def options_from_args(args: argparse.Namespace) -> InferenceOptions:
    explicit_selection = (
        bool(args.instance_id)
        or args.max_instances is not None
        or args.offset is not None
    )
    instance_ids = tuple(args.instance_id)
    if not explicit_selection:
        instance_ids = (DEFAULT_SMOKE_INSTANCE_ID,)
    return InferenceOptions(
        run_id=args.run_id,
        output_dir=Path(args.output_dir),
        instance_ids=instance_ids,
        max_instances=args.max_instances if args.max_instances is not None else 1,
        offset=args.offset if args.offset is not None else 0,
        include_hints=args.include_hints,
        model=canonical_model(args.model),
        image_template=args.image_template,
        docker_platform=args.docker_platform,
        agent_timeout_seconds=args.agent_timeout_seconds,
        setup_timeout_seconds=args.setup_timeout_seconds,
        restart=args.restart,
        dry_run=args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    def terminate(signum: int, _frame: Any) -> None:
        raise ControllerTerminated(signum)

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        if args.command == "infer":
            paths = run_inference(options_from_args(args))
            if not args.dry_run:
                print(f"Predictions: {paths.predictions}")
            return 0
        if args.command in {"evaluate", "eval"}:
            run_evaluation(args)
            return 0
        with DatasetRowsClient() as client:
            rows = client.select(
                tuple(args.instance_id),
                offset=args.offset,
                max_instances=args.max_instances,
            )
        print(encode_jsonl(row.public_dict() for row in rows), end="")
        return 0
    except BenchmarkError as exc:
        parser.exit(2, f"error: {exc}\n")
    except ControllerTerminated as exc:
        print(
            "Benchmark interrupted; cleanup was attempted and partial artifacts "
            "were preserved.",
            file=sys.stderr,
        )
        return 128 + exc.signum
    except KeyboardInterrupt:
        print(
            "Benchmark interrupted; cleanup was attempted and partial artifacts "
            "were preserved.",
            file=sys.stderr,
        )
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
