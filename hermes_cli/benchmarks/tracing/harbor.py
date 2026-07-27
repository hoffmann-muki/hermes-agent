"""Harbor bridge utilities for Hermes benchmark traces."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tomllib
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from hermes_cli.benchmarks.tracing.agentsight import (
    AgentSightProfiler,
    AgentSightTarget,
    resolve_docker_compose_main_container,
)
from hermes_cli.benchmarks.tracing.coordination import (
    TraceRun,
    TraceSelection,
    TraceSelectionStrategy,
    attach_trace_run,
    finalize_trace_run,
)
from hermes_cli.benchmarks.tracing.runtime import attempt_directory


_ALLOCATION_FILENAME = ".harbor-attempts.json"
_LOCK_FILENAME = ".harbor-attempts.lock"
_AGENTSIGHT_STAGING_DIRECTORY = ".agentsight-profile"


@dataclass(frozen=True)
class HarborTraceAttempt:
    instance_id: str
    attempt: int
    agent_timeout_seconds: float
    container_image: str
    container_root: Path


@dataclass(frozen=True)
class HarborTraceHarness:
    """Reusable run-finalization adapter for any Harbor benchmark."""

    jobs_dir: Path | None
    job_name: str | None
    selected_instance_ids: tuple[str, ...] | None
    expected_instance_count: int
    expected_attempts_per_instance: int
    selection_strategy: TraceSelectionStrategy
    allow_observed_fallback: bool = False

    def prepare_finalization(self, run: TraceRun) -> None:
        _remove_allocator_files(run.root)

    def resolve_selection(
        self,
        run: TraceRun,
        observed_instance_ids: Sequence[str],
    ) -> TraceSelection:
        del run
        if self.selected_instance_ids is not None:
            instance_ids = self.selected_instance_ids
        elif self.allow_observed_fallback:
            instance_ids = tuple(sorted(observed_instance_ids))
        else:
            if self.jobs_dir is None or not self.job_name:
                raise ValueError(
                    "Harbor trace selection requires a job lock or explicit IDs"
                )
            instance_ids = tuple(
                trace_instance_ids_from_job(self.jobs_dir, self.job_name)
            )
        if len(instance_ids) != self.expected_instance_count:
            raise ValueError(
                "Trace run index omitted because the resolved instance count "
                f"{len(instance_ids)} does not match {self.expected_instance_count}"
            )
        return TraceSelection(
            instance_ids=instance_ids,
            strategy=self.selection_strategy,
            minimum_attempts_per_instance=self.expected_attempts_per_instance,
        )


def allocate_harbor_trace_attempt(
    *,
    logs_dir: Path,
    trace_root: Path,
) -> HarborTraceAttempt:
    """Assign one per-instance ordinal before Harbor starts the agent."""

    instance_id = trace_instance_id_from_trial_config(logs_dir)
    timeout = trace_agent_timeout_from_trial_config(logs_dir)
    image = trace_container_image_from_trial_config(logs_dir)
    root = trace_root.resolve()
    if not root.is_dir() or trace_root.is_symlink():
        raise ValueError(f"Harbor trace root must be a real directory: {trace_root}")

    with _allocation_lock(root):
        allocation_path = root / _ALLOCATION_FILENAME
        allocations = _read_allocations(allocation_path)
        attempt = allocations.get(instance_id, 0) + 1
        allocations[instance_id] = attempt
        _atomic_write_json(allocation_path, allocations)

    return HarborTraceAttempt(
        instance_id=instance_id,
        attempt=attempt,
        agent_timeout_seconds=timeout,
        container_image=image,
        container_root=Path("/logs/agent/benchmark-trace"),
    )


def promote_harbor_trace_attempt(
    *,
    logs_dir: Path,
    trace_root: Path,
    attempt: HarborTraceAttempt,
) -> Path:
    """Promote one sanitized in-container trace into the canonical run."""

    source = attempt_directory(
        logs_dir / attempt.container_root.name,
        attempt.instance_id,
        attempt.attempt,
    )
    destination = attempt_directory(
        trace_root.resolve(),
        attempt.instance_id,
        attempt.attempt,
    )
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"Harbor agent trace is missing: {source}")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Harbor agent trace contains a symbolic link")
    if destination.exists():
        raise FileExistsError(f"Harbor trace attempt already exists: {destination}")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    if os.name != "nt":
        destination.chmod(0o700)
    shutil.rmtree(logs_dir / attempt.container_root.name, ignore_errors=True)
    return destination


def start_harbor_agentsight_profile(
    *,
    logs_dir: Path,
    trace_run_id: str,
    benchmark: str,
    framework: str,
    attempt: HarborTraceAttempt,
    docker_session_id: str,
    tls_python_path: str | None,
    env: dict[str, str] | None = None,
) -> AgentSightProfiler:
    """Start one container-scoped AgentSight profile for a Harbor attempt."""

    effective_env = dict(os.environ if env is None else env)
    profile_id = harbor_agentsight_profile_id(
        run_id=trace_run_id,
        framework=framework,
        instance_id=attempt.instance_id,
        attempt=attempt.attempt,
    )
    staging = logs_dir / _AGENTSIGHT_STAGING_DIRECTORY
    correlation = {
        "runId": trace_run_id,
        "benchmark": benchmark,
        "framework": framework,
        "instanceId": attempt.instance_id,
        "attempt": attempt.attempt,
    }
    target = AgentSightTarget(
        attempt_dir=staging,
        profile_id=profile_id,
        container_id="disabled",
        capture_tls=True,
        tls_python_path=tls_python_path,
        correlation=correlation,
    )
    if _agentsight_disabled(effective_env):
        return AgentSightProfiler.start(target, env=effective_env)
    try:
        container_id = resolve_docker_compose_main_container(
            docker_session_id,
            effective_env,
        )
    except (OSError, RuntimeError, ValueError) as error:
        return AgentSightProfiler.unavailable(
            target,
            f"task-container resolution failed: {type(error).__name__}",
            expected_scopes=("task-container",),
            env=effective_env,
        )
    return AgentSightProfiler.start(
        AgentSightTarget(
            attempt_dir=staging,
            profile_id=profile_id,
            container_id=container_id,
            capture_tls=True,
            tls_python_path=tls_python_path,
            correlation=correlation,
        ),
        env=effective_env,
    )


def attach_harbor_agentsight_profile(
    *,
    logs_dir: Path,
    attempt: HarborTraceAttempt,
    profiler: AgentSightProfiler,
) -> Path:
    """Attach a staged profile to its finalized semantic trace attempt."""

    source = profiler.directory
    semantic_attempt = attempt_directory(
        logs_dir / attempt.container_root.name,
        attempt.instance_id,
        attempt.attempt,
    )
    destination = semantic_attempt / "profiles" / "agentsight"
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"Harbor AgentSight profile is missing: {source}")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Harbor AgentSight profile contains a symbolic link")
    if not semantic_attempt.is_dir() or semantic_attempt.is_symlink():
        raise ValueError(f"Harbor semantic trace is missing: {semantic_attempt}")
    if destination.exists():
        raise FileExistsError(
            f"Harbor semantic trace already has an AgentSight profile: {destination}"
        )
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    if os.name != "nt":
        destination.chmod(0o700)
    shutil.rmtree(profiler.target.attempt_dir, ignore_errors=True)
    return destination


def harbor_agentsight_profile_id(
    *,
    run_id: str,
    framework: str,
    instance_id: str,
    attempt: int,
) -> str:
    """Return a deterministic profiler identity for one semantic attempt."""

    if not run_id or not framework or not instance_id or attempt < 1:
        raise ValueError("Harbor AgentSight profile identity is invalid")
    digest = hashlib.sha256(
        "\0".join((run_id, framework, instance_id, str(attempt))).encode()
    ).hexdigest()
    return f"agentsight-{digest[:32]}"


def finalize_harbor_trace_run(
    *,
    trace_root: Path,
    run_id: str,
    benchmark: str,
    created_at: str,
    selected_instance_ids: Sequence[str] | None,
    expected_instance_count: int,
    expected_attempts_per_instance: int,
    selection_strategy: TraceSelectionStrategy,
) -> Path:
    """Compatibility wrapper around the generic trace-run coordinator."""

    return finalize_trace_run(
        attach_trace_run(
            root=trace_root,
            run_id=run_id,
            created_at=created_at,
            benchmark=benchmark,
            framework="hermes",
        ),
        HarborTraceHarness(
            jobs_dir=None,
            job_name=None,
            selected_instance_ids=(
                tuple(selected_instance_ids)
                if selected_instance_ids is not None
                else None
            ),
            expected_instance_count=expected_instance_count,
            expected_attempts_per_instance=expected_attempts_per_instance,
            selection_strategy=selection_strategy,
            allow_observed_fallback=selected_instance_ids is None,
        ),
    )


def trace_instance_id_from_trial_config(logs_dir: Path) -> str:
    config = _trial_config(logs_dir)
    task = config.get("task")
    if not isinstance(task, dict):
        raise ValueError("Harbor trial config has no task object")
    name = task.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Harbor trial config has no task name")
    return name.split("/", 1)[-1]


def trace_instance_ids_from_job(jobs_dir: Path, job_name: str) -> list[str]:
    """Return Harbor's resolved task order with repeated attempts collapsed."""

    value = json.loads((jobs_dir / job_name / "lock.json").read_text(encoding="utf-8"))
    trials = value.get("trials") if isinstance(value, dict) else None
    if not isinstance(trials, list):
        raise ValueError("Harbor job lock has no resolved trials")

    instance_ids: list[str] = []
    for trial in trials:
        task = trial.get("task") if isinstance(trial, dict) else None
        name = task.get("name") if isinstance(task, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError("Harbor job lock has a trial without a task name")
        instance_id = name.split("/", 1)[-1]
        if instance_id not in instance_ids:
            instance_ids.append(instance_id)
    if not instance_ids:
        raise ValueError("Harbor job lock selected no tasks")
    return instance_ids


def trace_agent_timeout_from_trial_config(logs_dir: Path) -> float:
    config = _trial_config(logs_dir)
    task = config.get("task")
    agent = config.get("agent")
    if not isinstance(task, dict):
        raise ValueError("Harbor trial config has no task object")
    override = agent.get("override_timeout_sec") if isinstance(agent, dict) else None
    if isinstance(override, int | float) and override > 0:
        timeout = float(override)
    else:
        with (_resolved_task_path(task) / "task.toml").open("rb") as file:
            task_document = tomllib.load(file)
        task_agent = task_document.get("agent")
        timeout_value = (
            task_agent.get("timeout_sec") if isinstance(task_agent, dict) else None
        )
        if not isinstance(timeout_value, int | float) or timeout_value <= 0:
            raise ValueError("Harbor task has no positive agent timeout")
        timeout = float(timeout_value)
    multiplier = config.get("agent_timeout_multiplier")
    if multiplier is None:
        multiplier = config.get("timeout_multiplier", 1)
    if not isinstance(multiplier, int | float) or multiplier <= 0:
        raise ValueError("Harbor trial timeout multiplier must be positive")
    return timeout * float(multiplier)


def trace_container_image_from_trial_config(logs_dir: Path) -> str:
    config = _trial_config(logs_dir)
    task = config.get("task")
    if not isinstance(task, dict):
        raise ValueError("Harbor trial config has no task object")
    with (_resolved_task_path(task) / "task.toml").open("rb") as file:
        task_document = tomllib.load(file)
    environment = task_document.get("environment")
    image = environment.get("docker_image") if isinstance(environment, dict) else None
    if not isinstance(image, str) or not image:
        raise ValueError("Harbor task has no Docker image")
    return image


def _trial_config(logs_dir: Path) -> dict[str, Any]:
    value = json.loads((logs_dir.parent / "config.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Harbor trial config must be an object")
    return value


def _resolved_task_path(task: dict[str, Any]) -> Path:
    name = task.get("name")
    reference = task.get("ref")
    if (
        not isinstance(name, str)
        or "/" not in name
        or not isinstance(reference, str)
        or not reference.startswith("sha256:")
    ):
        raise ValueError("Harbor package task is not pinned to a digest")
    # Harbor is installed by the benchmark environment, not Hermes core.
    from harbor.models.task.id import PackageTaskId

    organization, task_name = name.split("/", 1)
    return PackageTaskId(
        org=organization,
        name=task_name,
        ref=reference,
    ).get_local_path()


@contextmanager
def _allocation_lock(root: Path) -> Iterator[None]:
    import fcntl

    with (root / _LOCK_FILENAME).open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read_allocations(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, int) and item >= 0
        for key, item in value.items()
    ):
        raise ValueError("Harbor trace attempt allocation state is invalid")
    return value


def _atomic_write_json(path: Path, value: dict[str, int]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if os.name != "nt":
        temporary.chmod(0o600)
    temporary.replace(path)


def _remove_allocator_files(root: Path) -> None:
    for name in (_ALLOCATION_FILENAME, _LOCK_FILENAME):
        (root / name).unlink(missing_ok=True)


def _agentsight_disabled(env: dict[str, str]) -> bool:
    return env.get("BENCHMARK_AGENTSIGHT", "").strip().lower() in {
        "0",
        "false",
        "off",
        "disabled",
    }
