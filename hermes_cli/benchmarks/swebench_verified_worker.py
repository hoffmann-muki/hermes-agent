"""Per-instance worker for :mod:`hermes_cli.benchmarks.swebench_verified`.

The controller owns selection, deadlines, and checkpoints.  This worker owns
one Docker container and one coordinator run, keeping process-global Hermes
tool configuration isolated to a single benchmark instance.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import yaml

from hermes_cli.benchmarks.swebench_verified import (
    BENCHMARK,
    DEFAULT_API_MAX_RETRIES,
    DEFAULT_COORDINATOR_BUDGET,
    DEFAULT_CODING_CONTEXT,
    DEFAULT_PHASE_BUDGETS,
    BenchmarkError,
    SweBenchRow,
    atomic_write_json,
    canonical_model,
    hermes_source_identity,
    parse_swebench_row,
    provider_model,
    utc_now,
    validate_source_identity,
)


COORDINATOR_BUDGET = DEFAULT_COORDINATOR_BUDGET
PHASE_BUDGETS = DEFAULT_PHASE_BUDGETS
PHASE_ORDER = tuple(PHASE_BUDGETS)
TEMPERATURE = 0.1
WORKTREE = "/testbed"
CONDA_ACTIVATION = ". /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
MAX_PHASE_REPORT_CHARS = 20_000
PHASE_TOOL_TIMEOUT_GRACE_SECONDS = 60
READ_ONLY_TOOLSET = "swe-benchmark-readonly"


COORDINATOR_SYSTEM_PROMPT = """You are benchmark-coordinator for a SWE-bench Verified task.

You own the result, but three fresh specialists must work in the same foreground
workspace before you finish. Call swe_benchmark_phase exactly once for each role,
one call at a time, in this exact order:

1. navigator — investigate only; it is read-only and returns an evidence-backed plan.
2. patcher — implement the fix and run focused checks.
3. reviewer — independently inspect the diff/tests and make small corrective edits.

Wait for each call to return before making the next. Do not skip, repeat, parallelize,
or replace a phase with your own analysis. Do not invoke Hermes' general delegation.
After reviewer returns, reconcile all reports, inspect the final worktree, make only
necessary final corrections, run feasible verification, and provide a concise final
summary. The source edits in /testbed—not prose—are the benchmark answer.

Never seek a gold patch, hidden test patch, benchmark answer, or hidden grading data.
Do not modify tests or benchmark metadata unless the issue explicitly requires it.
"""


PHASE_SYSTEM_PROMPTS = {
    "navigator": """You are benchmark-navigator, a fresh read-only SWE specialist.
Investigate the issue and repository in /testbed. Trace the relevant implementation,
tests, and likely root cause. Do not modify files, install into the repository, or run
destructive commands. Return a concrete implementation and verification plan with
paths and symbols. You cannot delegate.""",
    "patcher": """You are benchmark-patcher, a fresh SWE implementation specialist.
Work directly in /testbed. Use the navigator evidence, inspect the code yourself,
implement the smallest correct fix, and run focused verification when feasible.
Do not seek benchmark answers or hidden tests. You cannot delegate. Leave all source
changes in the shared worktree and report changed paths, commands, and remaining risk.""",
    "reviewer": """You are benchmark-reviewer, a fresh independent code reviewer.
Inspect the issue, current /testbed worktree, and patcher report. Review the full diff
for correctness, regressions, style, and missing edge cases. Run focused checks and
make small corrective edits directly when justified. Do not rewrite a sound solution
for preference alone. You cannot delegate. Report findings, fixes, verification, and
residual risk.""",
}


PHASE_TOOL_SCHEMA: dict[str, Any] = {
    "description": (
        "Run the next required fresh SWE-bench specialist synchronously in the "
        "shared /testbed worktree. Call navigator, then patcher, then reviewer, "
        "exactly once each and wait for every result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "phase": {
                "type": "string",
                "enum": list(PHASE_ORDER),
                "description": "The next required benchmark phase.",
            }
        },
        "required": ["phase"],
        "additionalProperties": False,
    },
}

WORKER_BENCHMARK: str = BENCHMARK
ROW_PARSER = parse_swebench_row
PHASE_PROMPT_BUILDER: Callable[..., str]


@contextmanager
def worker_configuration(
    *,
    benchmark: str,
    worktree: str,
    conda_activation: str,
    row_parser: Callable[[Any], Any],
    coordinator_system_prompt: str,
    phase_system_prompts: dict[str, str],
    phase_tool_schema: dict[str, Any],
    phase_prompt_builder: Callable[..., str],
) -> Iterator[None]:
    """Temporarily specialize this process-isolated benchmark worker."""
    if worktree not in {"/app", "/testbed"}:
        raise BenchmarkError(f"Unsupported benchmark worktree: {worktree}")
    global WORKER_BENCHMARK, WORKTREE, CONDA_ACTIVATION, ROW_PARSER
    global COORDINATOR_SYSTEM_PROMPT, PHASE_SYSTEM_PROMPTS, PHASE_TOOL_SCHEMA
    global PHASE_PROMPT_BUILDER

    previous = (
        WORKER_BENCHMARK,
        WORKTREE,
        CONDA_ACTIVATION,
        ROW_PARSER,
        COORDINATOR_SYSTEM_PROMPT,
        PHASE_SYSTEM_PROMPTS,
        PHASE_TOOL_SCHEMA,
        PHASE_PROMPT_BUILDER,
    )
    WORKER_BENCHMARK = benchmark
    WORKTREE = worktree
    CONDA_ACTIVATION = conda_activation
    ROW_PARSER = row_parser
    COORDINATOR_SYSTEM_PROMPT = coordinator_system_prompt
    PHASE_SYSTEM_PROMPTS = phase_system_prompts
    PHASE_TOOL_SCHEMA = phase_tool_schema
    PHASE_PROMPT_BUILDER = phase_prompt_builder
    try:
        yield
    finally:
        (
            WORKER_BENCHMARK,
            WORKTREE,
            CONDA_ACTIVATION,
            ROW_PARSER,
            COORDINATOR_SYSTEM_PROMPT,
            PHASE_SYSTEM_PROMPTS,
            PHASE_TOOL_SCHEMA,
            PHASE_PROMPT_BUILDER,
        ) = previous


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    return value


def _terminal_config(request: dict[str, Any]) -> dict[str, Any]:
    hermes_home = Path(request["hermesHome"])
    return {
        "terminal": {
            "backend": "docker",
            "cwd": WORKTREE,
            "timeout": 180,
            "lifetime_seconds": 3600,
            "container_cpu": 0,
            "container_memory": 0,
            "container_disk": 0,
            # Persistent only within this worker so Hermes does not tear the
            # shared container down at each child turn. Cross-process reuse is
            # disabled and controller cleanup is unconditional.
            "container_persistent": True,
            "docker_image": request["image"],
            "docker_forward_env": [],
            "docker_env": {},
            "docker_volumes": [],
            "docker_mount_cwd_to_workspace": False,
            "docker_network": True,
            "docker_run_as_host_user": False,
            "docker_persist_across_processes": False,
            "docker_orphan_reaper": False,
            "sandbox_dir": str(hermes_home / "sandboxes"),
            "docker_extra_args": [
                "--platform",
                request["dockerPlatform"],
                "--user",
                "root",
                "--entrypoint",
                "",
            ],
        },
        "memory": {
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
        "checkpoints": {"enabled": False},
        "plugins": {"enabled": []},
        "approvals": {"mode": "off", "cron_mode": "deny"},
        "agent": {
            # One means one provider attempt: Hermes' application-level retry
            # loop performs no retry after a failed model request.
            "api_max_retries": DEFAULT_API_MAX_RETRIES,
            # The benchmark worktree exists only inside Docker. Without this override,
            # context discovery can fall back to the host Hermes checkout and
            # inject irrelevant repository state into the benchmark prompt.
            "coding_context": DEFAULT_CODING_CONTEXT,
        },
    }


def configure_worker(request: dict[str, Any]) -> None:
    hermes_home = Path(request["hermesHome"])
    hermes_home.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(hermes_home)
    # Hermes runs all model tools through a guarded worker pool whose normal
    # seven-minute ceiling is shorter than this benchmark's shared 30-minute
    # agent deadline. This existing internal runtime knob is scoped to the
    # disposable worker process; the controller remains the hard wall clock.
    os.environ["HERMES_CONCURRENT_TOOL_TIMEOUT_S"] = str(
        int(request["agentTimeoutSeconds"]) + PHASE_TOOL_TIMEOUT_GRACE_SECONDS
    )
    config = _terminal_config(request)
    config_path = hermes_home / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    os.chmod(config_path, 0o600)
    from hermes_cli.config import apply_terminal_config_to_env

    apply_terminal_config_to_env(config=config, override=True)


def _execute(env: Any, command: str, *, timeout: int = 120) -> dict[str, Any]:
    result = env.execute(command, cwd=WORKTREE, timeout=timeout)
    if not isinstance(result, dict):
        raise BenchmarkError("Docker environment returned an invalid command result")
    return result


def _require_success(result: dict[str, Any], description: str) -> None:
    if result.get("returncode") == 0:
        return
    output = str(result.get("output") or "").strip()
    raise BenchmarkError(f"{description} failed: {output[-2000:]}")


def setup_environment(request: dict[str, Any]) -> Any:
    from tools.terminal_tool import (
        get_active_env,
        register_task_env_overrides,
        terminal_tool,
    )

    task_id = request["taskId"]
    register_task_env_overrides(
        task_id, {"docker_image": request["image"], "cwd": WORKTREE}
    )
    base_commit = request["row"]["base_commit"]
    activation = f"{CONDA_ACTIVATION} && " if CONDA_ACTIVATION else ""
    command = (
        f"{activation}git config --global --add safe.directory {WORKTREE} && "
        f"git -C {WORKTREE} reset --hard {shlex.quote(base_commit)}"
    )
    raw = terminal_tool(
        command=command,
        task_id=task_id,
        workdir=WORKTREE,
        timeout=180,
        force=True,
    )
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BenchmarkError("Hermes terminal setup returned invalid JSON") from exc
    if result.get("exit_code") != 0:
        raise BenchmarkError(
            f"Could not initialize {WORKTREE}: "
            f"{result.get('error') or result.get('output')}"
        )
    env = get_active_env(task_id)
    if env is None:
        raise BenchmarkError("Hermes did not retain the benchmark Docker environment")
    return env


def _status(env: Any) -> str:
    result = _execute(
        env,
        f"git -C {WORKTREE} status --porcelain=v1 --untracked-files=all",
        timeout=30,
    )
    _require_success(result, "git status")
    return str(result.get("output") or "")


def _reset_worktree(env: Any, base_commit: str) -> None:
    result = _execute(
        env,
        f"git -C {WORKTREE} reset --hard {shlex.quote(base_commit)} && "
        f"git -C {WORKTREE} clean -fd",
        timeout=120,
    )
    _require_success(result, "navigator mutation rollback")


def capture_patch(env: Any, base_commit: str) -> tuple[str, list[str], str | None]:
    try:
        stage = _execute(env, f"git -C {WORKTREE} add -A -- .", timeout=60)
        _require_success(stage, "git add")
        patch_result = _execute(
            env,
            f"git -C {WORKTREE} diff --cached --binary --no-color "
            f"{shlex.quote(base_commit)} -- .",
            timeout=120,
        )
        _require_success(patch_result, "git diff")
        names_result = _execute(
            env,
            f"git -C {WORKTREE} diff --cached --name-only -z "
            f"{shlex.quote(base_commit)} -- .",
            timeout=30,
        )
        _require_success(names_result, "git diff --name-only")
        names = [
            item for item in str(names_result.get("output") or "").split("\0") if item
        ]
        return str(patch_result.get("output") or ""), names, None
    except Exception as exc:
        return "", [], f"{type(exc).__name__}: {exc}"


def restore_sandbox_ownership(env: Any) -> str | None:
    """Return run-scoped bind mounts to the host user before Docker teardown."""
    if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
        return None
    try:
        result = _execute(
            env,
            f"chown -R {os.getuid()}:{os.getgid()} /root /workspace",
            timeout=120,
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if result.get("returncode") == 0:
        return None
    output = str(result.get("output") or "")
    remaining = [
        line for line in output.splitlines() if "Read-only file system" not in line
    ]
    if output and not remaining:
        return None
    return ("\n".join(remaining) or "sandbox ownership handoff failed")[-2000:]


def _phase_user_prompt(
    phase: str,
    row: SweBenchRow,
    previous_records: Sequence[dict[str, Any]],
    *,
    include_hints: bool,
) -> str:
    handoffs = []
    for record in previous_records:
        report = str(record.get("report") or "").strip()
        if report:
            handoffs.extend([
                f"### {record['phase'].title()} handoff",
                report[:MAX_PHASE_REPORT_CHARS],
                "",
            ])
    issue = [
        f"Complete the {phase} phase for SWE-bench Verified instance {row.instance_id}.",
        f"Repository: {row.repo}",
        f"Base commit: {row.base_commit}",
        "Worktree: /testbed",
        "",
        "## Issue",
        row.problem_statement.strip(),
        "",
    ]
    if include_hints and row.hints_text:
        issue.extend(["## Public hints", row.hints_text.strip(), ""])
    return "\n".join([*issue, *handoffs])


PHASE_PROMPT_BUILDER = _phase_user_prompt


@dataclass
class PhaseState:
    request: dict[str, Any]
    row: Any
    env: Any
    api_key: str
    agent_factory: Callable[..., Any]
    coordinator: Any = None
    next_phase: int = 0
    in_progress: bool = False
    records: list[dict[str, Any]] = field(default_factory=list)
    protocol_errors: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def workflow_complete(self) -> bool:
        return (
            self.next_phase == len(PHASE_ORDER)
            and not self.in_progress
            and not self.protocol_errors
            and all(record.get("status") == "completed" for record in self.records)
        )

    def handler(self, args: dict[str, Any], **kwargs: Any) -> str:
        phase = args.get("phase")
        with self.lock:
            if not isinstance(phase, str):
                error = "Benchmark phase must be a string"
                self.protocol_errors.append(error)
                return json.dumps({"error": error})
            expected = (
                PHASE_ORDER[self.next_phase]
                if self.next_phase < len(PHASE_ORDER)
                else None
            )
            if self.in_progress:
                error = "A benchmark phase is already running; wait for it to finish"
                self.protocol_errors.append(error)
                return json.dumps({"error": error})
            if phase != expected:
                error = f"Expected phase {expected!r}, received {phase!r}"
                self.protocol_errors.append(error)
                return json.dumps({"error": error})
            self.in_progress = True
            previous_records = list(self.records)

        record = self._run_phase(phase, previous_records, kwargs.get("task_id"))
        with self.lock:
            self.records.append(record)
            self.next_phase += 1
            self.in_progress = False
        response = {
            "phase": phase,
            "status": record["status"],
            "report": str(record.get("report") or "")[:MAX_PHASE_REPORT_CHARS],
            "apiCalls": record.get("apiCalls", 0),
            "budget": record["budget"],
            "readOnlyViolation": record.get("readOnlyViolation", False),
            "error": record.get("error"),
            "nextRequiredPhase": (
                PHASE_ORDER[self.next_phase]
                if self.next_phase < len(PHASE_ORDER)
                else None
            ),
        }
        return json.dumps(response, ensure_ascii=False)

    def _run_phase(
        self,
        phase: str,
        previous_records: Sequence[dict[str, Any]],
        task_id: str | None,
    ) -> dict[str, Any]:
        started_at = utc_now()
        started = time.monotonic()
        child = None
        result: dict[str, Any] = {}
        error = None
        read_only_violation = False
        pre_status = ""
        try:
            if phase == "navigator":
                pre_status = _status(self.env)
                if pre_status:
                    raise BenchmarkError(
                        "Coordinator modified the worktree before the navigator phase"
                    )
            child = self.agent_factory(
                role=phase,
                budget=PHASE_BUDGETS[phase],
                api_key=self.api_key,
                request=self.request,
                parent_session_id=getattr(self.coordinator, "session_id", None),
            )
            if self.coordinator is not None:
                with self.coordinator._active_children_lock:
                    self.coordinator._active_children.append(child)
            result = child.run_conversation(
                PHASE_PROMPT_BUILDER(
                    phase,
                    self.row,
                    previous_records,
                    include_hints=bool(self.request.get("includeHints")),
                ),
                system_message=PHASE_SYSTEM_PROMPTS[phase],
                task_id=task_id or self.request["taskId"],
            )
            if phase == "navigator":
                post_status = _status(self.env)
                if post_status:
                    read_only_violation = True
                    _reset_worktree(self.env, self.row.base_commit)
                    error = "Navigator modified the read-only worktree; changes were discarded"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if child is not None and self.coordinator is not None:
                try:
                    with self.coordinator._active_children_lock:
                        self.coordinator._active_children.remove(child)
                except ValueError:
                    pass
            if child is not None:
                child.release_clients()

        report = result.get("final_response") if isinstance(result, dict) else None
        completed = bool(result.get("completed")) if isinstance(result, dict) else False
        status = "completed" if completed and not error else "failed"
        return {
            "phase": phase,
            "budget": PHASE_BUDGETS[phase],
            "status": status,
            "freshAgent": True,
            "report": report if isinstance(report, str) else "",
            "apiCalls": result.get("api_calls", 0) if isinstance(result, dict) else 0,
            "turnExitReason": result.get("turn_exit_reason")
            if isinstance(result, dict)
            else None,
            "completed": completed,
            "interrupted": bool(result.get("interrupted"))
            if isinstance(result, dict)
            else False,
            "readOnlyViolation": read_only_violation,
            "error": error,
            "messages": result.get("messages", []) if isinstance(result, dict) else [],
            "usage": {
                key: result.get(key, 0) if isinstance(result, dict) else 0
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                    "estimated_cost_usd",
                )
            },
            "startedAt": started_at,
            "completedAt": utc_now(),
            "durationSeconds": round(time.monotonic() - started, 3),
        }


def default_agent_factory(
    *,
    role: str,
    budget: int,
    api_key: str,
    request: dict[str, Any],
    parent_session_id: str | None = None,
) -> Any:
    from hermes_constants import OPENROUTER_BASE_URL
    from run_agent import AIAgent

    toolsets = [READ_ONLY_TOOLSET] if role == "navigator" else ["terminal", "file"]
    if role == "coordinator":
        toolsets.append("swe-benchmark")
    return AIAgent(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        provider="openrouter",
        model=provider_model(request["model"]),
        max_iterations=budget,
        tool_delay=0,
        enabled_toolsets=toolsets,
        save_trajectories=False,
        verbose_logging=False,
        quiet_mode=True,
        request_overrides={"temperature": TEMPERATURE},
        skip_context_files=True,
        load_soul_identity=False,
        skip_memory=True,
        session_id=f"{request['taskId']}-{role}",
        parent_session_id=parent_session_id or "",
        checkpoints_enabled=False,
    )


def register_phase_tool(state: PhaseState) -> None:
    from toolsets import create_custom_toolset
    from tools.registry import registry

    create_custom_toolset(
        READ_ONLY_TOOLSET,
        "Read-only file inspection for the SWE-bench navigator",
        tools=["read_file", "search_files"],
    )
    registry.register(
        name="swe_benchmark_phase",
        toolset="swe-benchmark",
        schema=PHASE_TOOL_SCHEMA,
        handler=state.handler,
        description=PHASE_TOOL_SCHEMA["description"],
        emoji="🧪",
    )


def deregister_phase_tool() -> None:
    from toolsets import TOOLSETS
    from tools.registry import registry

    registry.deregister("swe_benchmark_phase")
    TOOLSETS.pop(READ_ONLY_TOOLSET, None)


def reconciled_workflow(
    state: PhaseState | None,
    coordinator_result: dict[str, Any],
    error: str | None,
) -> bool:
    return bool(
        state is not None
        and state.workflow_complete
        and coordinator_result.get("completed")
        and not coordinator_result.get("interrupted")
        and not error
    )


def _runtime_metadata(request: dict[str, Any], env: Any) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "benchmark": WORKER_BENCHMARK,
        "taskId": request["taskId"],
        "containerId": getattr(env, "_container_id", None),
        "image": request["image"],
        "dockerPlatform": request["dockerPlatform"],
        "worktree": WORKTREE,
        "agentStartedAt": utc_now(),
        "credentialEnvironmentNames": ["OPENROUTER_API_KEY"],
        "dockerForwardEnvironment": [],
        "phaseToolTimeoutSeconds": (
            int(request["agentTimeoutSeconds"]) + PHASE_TOOL_TIMEOUT_GRACE_SECONDS
        ),
    }


def run_worker(
    request: dict[str, Any],
    *,
    agent_factory: Callable[..., Any] = default_agent_factory,
) -> dict[str, Any]:
    if request.get("benchmark") != WORKER_BENCHMARK:
        raise BenchmarkError("Worker request benchmark does not match")
    row = ROW_PARSER(request.get("row"))
    if canonical_model(str(request.get("model") or "")) != request.get("model"):
        raise BenchmarkError("Worker request model must be canonical")
    source_identity = validate_source_identity(request.get("sourceIdentity"))
    if hermes_source_identity() != source_identity:
        raise BenchmarkError("Worker source does not match the controller request")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise BenchmarkError("OPENROUTER_API_KEY is required in the worker environment")

    configure_worker(request)
    from tools.terminal_tool import clear_task_env_overrides

    env = None
    state = None
    coordinator = None
    coordinator_result: dict[str, Any] = {}
    error = None
    termination_requested = threading.Event()
    started_at = utc_now()
    runtime_metadata: dict[str, Any] | None = None

    def request_stop(_signum: int, _frame: Any) -> None:
        termination_requested.set()
        if coordinator is not None:
            coordinator.interrupt("benchmark controller deadline")

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        env = setup_environment(request)
        runtime_metadata = _runtime_metadata(request, env)
        atomic_write_json(Path(request["runtimePath"]), runtime_metadata)
        if termination_requested.is_set():
            raise BenchmarkError("Worker terminated during setup")

        state = PhaseState(
            request=request,
            row=row,
            env=env,
            api_key=api_key,
            agent_factory=agent_factory,
        )
        register_phase_tool(state)
        coordinator = agent_factory(
            role="coordinator",
            budget=COORDINATOR_BUDGET,
            api_key=api_key,
            request=request,
        )
        state.coordinator = coordinator
        coordinator_result = coordinator.run_conversation(
            request["prompt"],
            system_message=COORDINATOR_SYSTEM_PROMPT,
            task_id=request["taskId"],
        )
        if termination_requested.is_set():
            error = "Agent execution exceeded the controller deadline"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        timed_out = termination_requested.is_set()
        if runtime_metadata is not None and not timed_out:
            runtime_metadata["agentCompletedAt"] = utc_now()
            atomic_write_json(Path(request["runtimePath"]), runtime_metadata)
        if coordinator is not None:
            coordinator.release_clients()
        try:
            deregister_phase_tool()
        except Exception:
            pass
        try:
            clear_task_env_overrides(request["taskId"])
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)

    records = state.records if state is not None else []
    protocol_errors = state.protocol_errors if state is not None else []
    workflow_complete = reconciled_workflow(state, coordinator_result, error)
    result = {
        "schemaVersion": 1,
        "benchmark": WORKER_BENCHMARK,
        "instanceId": row.instance_id,
        "model": request["model"],
        "temperature": TEMPERATURE,
        "attempt": 1,
        "maxInfrastructureRetries": 0,
        "apiMaxRetries": DEFAULT_API_MAX_RETRIES,
        "codingContext": DEFAULT_CODING_CONTEXT,
        "sourceIdentity": source_identity,
        "agentTimeoutSeconds": request["agentTimeoutSeconds"],
        "coordinatorBudget": COORDINATOR_BUDGET,
        "phaseBudgets": PHASE_BUDGETS,
        "phaseOrder": list(PHASE_ORDER),
        "workflowComplete": workflow_complete,
        "protocolErrors": protocol_errors,
        "phases": records,
        "coordinator": {
            "completed": bool(coordinator_result.get("completed")),
            "interrupted": bool(coordinator_result.get("interrupted")),
            "apiCalls": coordinator_result.get("api_calls", 0),
            "turnExitReason": coordinator_result.get("turn_exit_reason"),
            "finalResponse": coordinator_result.get("final_response"),
            "messages": coordinator_result.get("messages", []),
            "usage": {
                key: coordinator_result.get(key, 0)
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                    "estimated_cost_usd",
                )
            },
        },
        # The controller always restarts the task container before capture.
        # This is the only reliable way to quiesce tracked and untracked
        # background commands before taking the final Git snapshot.
        "modelPatch": None,
        "changedPaths": [],
        "patchCaptureError": None,
        "sandboxOwnershipError": None,
        "timedOut": termination_requested.is_set(),
        "deferCaptureToController": True,
        "error": error,
        "startedAt": started_at,
        "completedAt": utc_now(),
    }
    return _redact(result, api_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Internal {WORKER_BENCHMARK} worker")
    parser.add_argument("--request", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request_path = Path(args.request).resolve()
    request: dict[str, Any] = {}
    result_path: Path | None = None
    try:
        value = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise BenchmarkError("Worker request must be a JSON object")
        request = value
        result_path = Path(request["resultPath"])
        result = run_worker(request)
        atomic_write_json(result_path, result)
        exit_code = 0 if result.get("error") is None else 1
        # Avoid atexit sandbox cleanup and terminate any detached Hermes tool
        # thread. The controller restarts the container to quiesce every exec
        # process, captures the patch, then removes it.
        os._exit(exit_code)
    except Exception as exc:
        if result_path is not None:
            secret = os.environ.get("OPENROUTER_API_KEY", "")
            atomic_write_json(
                result_path,
                _redact(
                    {
                        "schemaVersion": 1,
                        "benchmark": WORKER_BENCHMARK,
                        "modelPatch": None,
                        "changedPaths": [],
                        "workflowComplete": False,
                        "timedOut": False,
                        "deferCaptureToController": True,
                        "error": f"{type(exc).__name__}: {exc}",
                        "completedAt": utc_now(),
                    },
                    secret,
                ),
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
