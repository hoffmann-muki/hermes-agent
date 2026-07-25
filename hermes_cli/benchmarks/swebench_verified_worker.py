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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import yaml

from hermes_cli.benchmarks.swebench_verified import (
    BENCHMARK,
    DEFAULT_API_MAX_RETRIES,
    DEFAULT_COORDINATOR_BUDGET,
    DEFAULT_CODING_CONTEXT,
    DEFAULT_DELEGATION_MODE,
    DEFAULT_NATIVE_SUBAGENT_BUDGET,
    DEFAULT_NATIVE_SUBAGENT_COUNT,
    DEFAULT_PHASE_BUDGETS,
    BenchmarkError,
    atomic_write_json,
    canonical_model,
    hermes_source_identity,
    parse_swebench_row,
    provider_model,
    utc_now,
    validate_source_identity,
)
from hermes_cli.benchmarks.native_delegation import (
    audit_native_delegations as audit_delegations,
)


COORDINATOR_BUDGET = DEFAULT_COORDINATOR_BUDGET
PEER_PHASE_BUDGETS = DEFAULT_PHASE_BUDGETS
PHASE_ORDER = tuple(PEER_PHASE_BUDGETS)
NATIVE_SUBAGENT_BUDGET = DEFAULT_NATIVE_SUBAGENT_BUDGET
NATIVE_SUBAGENT_COUNT = DEFAULT_NATIVE_SUBAGENT_COUNT
TEMPERATURE = 0.1
WORKTREE = "/testbed"
CONDA_ACTIVATION = ". /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
MAX_PHASE_REPORT_CHARS = 20_000
DELEGATION_TOOL_TIMEOUT_GRACE_SECONDS = 60
DELEGATION_GOAL_MARKERS = {
    "navigator": "[benchmark-navigator]",
    "patcher": "[benchmark-patcher]",
    "reviewer": "[benchmark-reviewer]",
}


COORDINATOR_SYSTEM_PROMPT = """You are benchmark-coordinator for a SWE-bench Verified task.

You own the result, but three fresh specialists must work in the same shared
workspace before you finish. Use Hermes' native delegate_task tool for one fresh
leaf subagent at a time, in this order:

1. navigator — use a goal beginning [benchmark-navigator]. Ask it to investigate
   without changing state and return an evidence-backed plan.
2. patcher — after the navigator returns, use a goal beginning [benchmark-patcher].
   Pass the original issue and navigator handoff; require the smallest complete fix
   and focused verification.
3. reviewer — after the patcher returns, use a goal beginning [benchmark-reviewer].
   Pass the original issue and prior handoffs; require independent review, focused
   checks, and only small clearly necessary corrections.

For every call use role="leaf" and include the complete task, repository metadata,
worktree path, role constraints, and prior handoffs in context because native Hermes
subagents start with fresh context. Call sequentially, not as a tasks batch. Native
delegation returns synchronously in this benchmark runner. After the reviewer returns,
reconcile all reports, inspect the final worktree, make only necessary final
corrections, run feasible verification, and provide a concise final summary. The
source edits in /testbed—not prose—are the benchmark answer.

Never seek a gold patch, hidden test patch, benchmark answer, or hidden grading data.
Do not modify tests or benchmark metadata unless the issue explicitly requires it.
"""

WORKER_BENCHMARK: str = BENCHMARK
ROW_PARSER = parse_swebench_row


@contextmanager
def worker_configuration(
    *,
    benchmark: str,
    worktree: str,
    conda_activation: str,
    row_parser: Callable[[Any], Any],
    coordinator_system_prompt: str,
) -> Iterator[None]:
    """Temporarily specialize this process-isolated benchmark worker."""
    if worktree not in {"/app", "/testbed"}:
        raise BenchmarkError(f"Unsupported benchmark worktree: {worktree}")
    global WORKER_BENCHMARK, WORKTREE, CONDA_ACTIVATION, ROW_PARSER
    global COORDINATOR_SYSTEM_PROMPT

    previous = (
        WORKER_BENCHMARK,
        WORKTREE,
        CONDA_ACTIVATION,
        ROW_PARSER,
        COORDINATOR_SYSTEM_PROMPT,
    )
    WORKER_BENCHMARK = benchmark
    WORKTREE = worktree
    CONDA_ACTIVATION = conda_activation
    ROW_PARSER = row_parser
    COORDINATOR_SYSTEM_PROMPT = coordinator_system_prompt
    try:
        yield
    finally:
        (
            WORKER_BENCHMARK,
            WORKTREE,
            CONDA_ACTIVATION,
            ROW_PARSER,
            COORDINATOR_SYSTEM_PROMPT,
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
        "delegation": {
            "max_iterations": NATIVE_SUBAGENT_BUDGET,
            "max_concurrent_children": 1,
            "max_spawn_depth": 1,
            "orchestrator_enabled": False,
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
        int(request["agentTimeoutSeconds"]) + DELEGATION_TOOL_TIMEOUT_GRACE_SECONDS
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
    # A CWD-only override intentionally collapses to Hermes' shared "default"
    # container. Native delegate_task children use distinct task IDs but resolve
    # to that same container, so every fresh child sees the same worktree.
    register_task_env_overrides(task_id, {"cwd": WORKTREE})
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


def audit_native_delegations(
    messages: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    return audit_delegations(
        messages,
        phase_order=PHASE_ORDER,
        goal_markers=DELEGATION_GOAL_MARKERS,
        subagent_budget=NATIVE_SUBAGENT_BUDGET,
        max_report_chars=MAX_PHASE_REPORT_CHARS,
    )


def default_agent_factory(
    *,
    role: str,
    budget: int,
    api_key: str,
    request: dict[str, Any],
    parent_session_id: str | None = None,
    callbacks: Mapping[str, Callable[..., Any]] | None = None,
) -> Any:
    from hermes_constants import OPENROUTER_BASE_URL
    from run_agent import AIAgent

    if role != "coordinator":
        raise BenchmarkError(
            "Native benchmark delegation only constructs a coordinator"
        )
    return AIAgent(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        provider="openrouter",
        model=provider_model(request["model"]),
        max_iterations=budget,
        tool_delay=0,
        enabled_toolsets=["terminal", "file", "delegation"],
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
        **dict(callbacks or {}),
    )


def create_trace_adapter(request: dict[str, Any]) -> Any | None:
    value = request.get("trace")
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value.get("benchmark") != WORKER_BENCHMARK
        or not isinstance(value.get("runId"), str)
        or not isinstance(value.get("runRoot"), str)
        or not isinstance(value.get("createdAt"), str)
        or not isinstance(value.get("frameworkRevision"), str)
        or isinstance(value.get("evaluationTimeoutSeconds"), bool)
        or not isinstance(value.get("evaluationTimeoutSeconds"), int | float)
        or value["evaluationTimeoutSeconds"] <= 0
    ):
        raise BenchmarkError("Worker trace request has an invalid schema")
    from hermes_cli.benchmarks.tracing import (
        HermesTraceRun,
        create_hermes_attempt_trace,
    )

    return create_hermes_attempt_trace(
        run=HermesTraceRun(
            id=value["runId"],
            root=Path(value["runRoot"]).resolve(),
            created_at=value["createdAt"],
            benchmark=value["benchmark"],
        ),
        instance_id=request["row"]["instance_id"],
        attempt=int(request["attempt"]),
        framework_revision=value["frameworkRevision"],
        model=request["model"],
        agent_timeout_seconds=int(request["agentTimeoutSeconds"]),
        evaluation_workers=1,
        evaluation_timeout_seconds=value["evaluationTimeoutSeconds"],
    )


def reconciled_workflow(
    records: Sequence[dict[str, Any]],
    audit_errors: Sequence[str],
    coordinator_result: dict[str, Any],
    error: str | None,
) -> bool:
    return bool(
        [record.get("phase") for record in records] == list(PHASE_ORDER)
        and all(record.get("status") == "completed" for record in records)
        and not audit_errors
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
        "delegationToolTimeoutSeconds": (
            int(request["agentTimeoutSeconds"]) + DELEGATION_TOOL_TIMEOUT_GRACE_SECONDS
        ),
        "delegationMode": DEFAULT_DELEGATION_MODE,
        "nativeSubagentBudget": NATIVE_SUBAGENT_BUDGET,
        "nativeSubagentCount": NATIVE_SUBAGENT_COUNT,
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

    trace_adapter = create_trace_adapter(request)
    configure_worker(request)
    from tools.terminal_tool import clear_task_env_overrides
    from gateway.session_context import declare_stateless_channel

    env = None
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
        if trace_adapter is not None:
            trace_adapter.container_observed({
                "container_id": runtime_metadata["containerId"],
                "image": request["image"],
                "docker_platform": request["dockerPlatform"],
                "worktree": WORKTREE,
            })
        if termination_requested.is_set():
            raise BenchmarkError("Worker terminated during setup")

        # Benchmark workers have no completion-queue drain. This supported
        # one-shot mode makes native delegate_task calls return synchronously.
        declare_stateless_channel()
        agent_options: dict[str, Any] = {
            "role": "coordinator",
            "budget": COORDINATOR_BUDGET,
            "api_key": api_key,
            "request": request,
        }
        if trace_adapter is not None:
            agent_options["callbacks"] = trace_adapter.callbacks()
        coordinator = agent_factory(
            **agent_options,
        )
        if trace_adapter is not None:
            trace_adapter.start_session(f"{request['taskId']}-coordinator")
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
        if trace_adapter is not None:
            trace_adapter.end_execution(
                (
                    "timeout"
                    if termination_requested.is_set()
                    else "completed"
                    if coordinator_result.get("completed")
                    and not coordinator_result.get("interrupted")
                    and error is None
                    else "failed"
                ),
                messages=coordinator_result.get("messages"),
                error_message=error,
            )
        timed_out = termination_requested.is_set()
        if runtime_metadata is not None and not timed_out:
            runtime_metadata["agentCompletedAt"] = utc_now()
            atomic_write_json(Path(request["runtimePath"]), runtime_metadata)
        if coordinator is not None:
            coordinator.release_clients()
        try:
            clear_task_env_overrides(request["taskId"])
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)

    records, audit_errors = audit_native_delegations(coordinator_result.get("messages"))
    workflow_complete = reconciled_workflow(
        records, audit_errors, coordinator_result, error
    )
    trace_result = None
    if trace_adapter is not None:
        trace_status = (
            "timeout"
            if termination_requested.is_set()
            else "completed"
            if workflow_complete
            else "failed"
        )
        try:
            finalized_trace = trace_adapter.finish(
                trace_status,
                messages=coordinator_result.get("messages"),
                error_message=error,
            )
            trace_result = {
                "traceId": finalized_trace.trace_id,
                "traceDirectory": finalized_trace.attempt_dir,
                "traceHealth": finalized_trace.health,
                "traceComplete": finalized_trace.complete,
            }
        except Exception as trace_error:
            trace_result = {
                "traceId": trace_adapter.identity.trace_id,
                "traceDirectory": str(trace_adapter.attempt_dir),
                "traceHealth": "failed",
                "traceComplete": False,
                "error": type(trace_error).__name__,
            }
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
        "delegationMode": DEFAULT_DELEGATION_MODE,
        "nativeSubagentBudget": NATIVE_SUBAGENT_BUDGET,
        "nativeSubagentCount": NATIVE_SUBAGENT_COUNT,
        "peerPhaseBudgetReference": PEER_PHASE_BUDGETS,
        "phaseOrder": list(PHASE_ORDER),
        "workflowComplete": workflow_complete,
        "delegationAuditErrors": audit_errors,
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
    if trace_result is not None:
        result["trace"] = trace_result
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
