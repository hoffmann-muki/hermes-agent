"""Isolated multi-agent worker executed inside one Harbor task environment."""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

if __package__:
    from hermes_cli.benchmarks.native_delegation import (
        audit_native_delegations as audit_delegations,
    )
else:
    # The Harbor adapter uploads the invoking checkout's worker and helper so
    # dirty integration edits can run before their branch commit is installed.
    audit_delegations = importlib.import_module(
        "native_delegation"
    ).audit_native_delegations


BENCHMARK = "terminal-bench-2.1"
# This file is uploaded as a standalone Harbor entrypoint, so keep its parity
# constants local and cover their agreement with the wrapper in tests.
COORDINATOR_BUDGET = 24
PEER_PHASE_BUDGETS = {"navigator": 10, "patcher": 18, "reviewer": 12}
PHASE_ORDER = tuple(PEER_PHASE_BUDGETS)
DELEGATION_MODE = "native"
NATIVE_SUBAGENT_COUNT = len(PEER_PHASE_BUDGETS)
NATIVE_SUBAGENT_BUDGET = (
    sum(PEER_PHASE_BUDGETS.values()) // NATIVE_SUBAGENT_COUNT
)
TEMPERATURE = 0.1
API_MAX_RETRIES = 1
DELEGATION_TOOL_TIMEOUT_SECONDS = 24 * 60 * 60
MAX_PHASE_REPORT_CHARS = 20_000
DELEGATION_GOAL_MARKERS = {
    "navigator": "[benchmark-navigator]",
    "patcher": "[benchmark-patcher]",
    "reviewer": "[benchmark-reviewer]",
}
RESULT_PATH = Path("/logs/agent/hermes-result.json")
SESSION_PATH = Path("/logs/agent/hermes-session.jsonl")


COORDINATOR_SYSTEM_PROMPT = """You are the primary Terminal-Bench coordinator.

You own the final environment state. Use Hermes' native delegate_task tool for
one fresh leaf subagent at a time, in this order:

1. navigator — use a goal beginning [benchmark-navigator]. Ask it to investigate
   without changing state and return an evidence-backed plan.
2. patcher — after the navigator returns, use a goal beginning [benchmark-patcher].
   Pass the original task and navigator handoff; require the concrete work and
   focused verification.
3. reviewer — after the patcher returns, use a goal beginning [benchmark-reviewer].
   Pass the original task and prior handoffs; require independent verification and
   only small clearly necessary corrections.

For every call use role="leaf" and include the complete task, shared working
directory, role constraints, and prior handoffs in context because native Hermes
subagents start with fresh context. Call sequentially, not as a tasks batch. Native
delegation returns synchronously in this benchmark runner and all children share the
same task environment. After the reviewer returns, reconcile the reports, inspect any
remaining risk with your own tools, make only necessary final corrections, and give a
concise final answer. Changes to the environment—not prose—are the benchmark answer.
"""

def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _configure_runtime(workdir: Path) -> None:
    hermes_home = Path(os.environ.get("HERMES_HOME", "/tmp/hermes"))
    hermes_home.mkdir(parents=True, exist_ok=True)
    # Native delegate_task executes inline in this one-shot worker. Keep the
    # generic tool watchdog from becoming an undocumented benchmark deadline;
    # Harbor's task timeout remains the authoritative outer limit.
    os.environ["HERMES_CONCURRENT_TOOL_TIMEOUT_S"] = str(
        DELEGATION_TOOL_TIMEOUT_SECONDS
    )
    config = {
        "terminal": {
            "backend": "local",
            "cwd": str(workdir),
            "timeout": 180,
        },
        "memory": {
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
        "checkpoints": {"enabled": False},
        "plugins": {"enabled": []},
        "approvals": {"mode": "off", "cron_mode": "deny"},
        "agent": {
            "api_max_retries": API_MAX_RETRIES,
            "coding_context": "off",
        },
        "delegation": {
            "max_iterations": NATIVE_SUBAGENT_BUDGET,
            "max_concurrent_children": 1,
            "max_spawn_depth": 1,
            "orchestrator_enabled": False,
        },
    }
    config_path = hermes_home / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    os.chmod(config_path, 0o600)
    from hermes_cli.config import apply_terminal_config_to_env

    apply_terminal_config_to_env(config=config, override=True)


def _usage(result: dict[str, Any]) -> dict[str, int | float]:
    return {
        key: result.get(key, 0)
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "total_tokens",
            "estimated_cost_usd",
        )
    }


def _merge_usage(results: Sequence[dict[str, Any]]) -> dict[str, int | float]:
    merged: dict[str, int | float] = {}
    for result in results:
        for key, value in _usage(result).items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
    return merged


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
    model: str,
    task_id: str,
    parent_session_id: str | None = None,
    callbacks: Mapping[str, Callable[..., Any]] | None = None,
) -> Any:
    from hermes_constants import OPENROUTER_BASE_URL
    from run_agent import AIAgent

    if role != "coordinator":
        raise ValueError("Native benchmark delegation only constructs a coordinator")
    return AIAgent(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        provider="openrouter",
        model=model,
        max_iterations=budget,
        tool_delay=0,
        enabled_toolsets=["terminal", "file", "delegation"],
        save_trajectories=False,
        verbose_logging=False,
        quiet_mode=True,
        request_overrides={"temperature": TEMPERATURE},
        load_soul_identity=False,
        skip_memory=True,
        session_id=f"{task_id}-{role}-{uuid.uuid4().hex[:8]}",
        parent_session_id=parent_session_id or "",
        checkpoints_enabled=False,
        **dict(callbacks or {}),
    )


def create_trace_adapter() -> Any | None:
    raw = os.environ.get("HERMES_BENCHMARK_TRACE_CONFIG")
    if not raw:
        return None
    value = json.loads(raw)
    required = {
        "runId",
        "benchmark",
        "instanceId",
        "attempt",
        "runRoot",
        "createdAt",
        "frameworkRevision",
        "model",
        "evaluationWorkers",
        "agentTimeoutSeconds",
        "benchmarkRetries",
        "sessionId",
        "containerImage",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError("Hermes Harbor trace configuration is invalid")
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
        instance_id=value["instanceId"],
        attempt=int(value["attempt"]),
        framework_revision=value["frameworkRevision"],
        model=value["model"],
        agent_timeout_seconds=float(value["agentTimeoutSeconds"]),
        evaluation_workers=int(value["evaluationWorkers"]),
        benchmark_retries=int(value["benchmarkRetries"]),
        harness_revision=value.get("harborVersion"),
        agent_image=value["containerImage"],
    )


def _write_session(results: Sequence[dict[str, Any]], secret: str) -> None:
    messages = [
        message
        for result in results
        for message in result.get("messages", [])
        if isinstance(message, dict)
    ]
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.write_text(
        "".join(json.dumps(_redact(message, secret)) + "\n" for message in messages),
        encoding="utf-8",
    )
    os.chmod(SESSION_PATH, 0o600)


def run_worker(
    instruction: str,
    *,
    api_key: str,
    model: str,
    workdir: Path,
    agent_factory: Callable[..., Any] = default_agent_factory,
) -> dict[str, Any]:
    trace_adapter = create_trace_adapter()
    _configure_runtime(workdir)
    from gateway.session_context import declare_stateless_channel

    declare_stateless_channel()
    task_id = f"terminalbench-{uuid.uuid4().hex}"
    coordinator = None
    coordinator_result: dict[str, Any] = {}
    error = None
    started = time.monotonic()
    try:
        agent_options: dict[str, Any] = {
            "role": "coordinator",
            "budget": COORDINATOR_BUDGET,
            "api_key": api_key,
            "model": model,
            "task_id": task_id,
        }
        if trace_adapter is not None:
            agent_options["callbacks"] = trace_adapter.callbacks()
        coordinator = agent_factory(
            **agent_options,
        )
        if trace_adapter is not None:
            config = json.loads(os.environ["HERMES_BENCHMARK_TRACE_CONFIG"])
            trace_adapter.container_observed(
                {
                    "session_id": config["sessionId"],
                    "image": config["containerImage"],
                    "phase": "harbor.agent",
                }
            )
            trace_adapter.start_session(config["sessionId"])
        coordinator_result = coordinator.run_conversation(
            instruction,
            system_message=COORDINATOR_SYSTEM_PROMPT,
            task_id=task_id,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if coordinator is not None:
            coordinator.release_clients()

    records, audit_errors = audit_native_delegations(
        coordinator_result.get("messages")
    )
    workflow_complete = bool(
        [record.get("phase") for record in records] == list(PHASE_ORDER)
        and all(record.get("status") == "completed" for record in records)
        and not audit_errors
        and coordinator_result.get("completed")
        and not coordinator_result.get("interrupted")
        and not error
    )
    _write_session([coordinator_result], api_key)
    trace_result = None
    if trace_adapter is not None:
        # Trace lifecycle reports execution completion; delegation parity remains
        # a separate audit result and is not benchmark success.
        trace_status = (
            "completed"
            if coordinator_result.get("completed")
            and not coordinator_result.get("interrupted")
            and error is None
            else "failed"
        )
        try:
            finalized = trace_adapter.finish(
                trace_status,
                messages=coordinator_result.get("messages"),
                error_message=error,
            )
            trace_result = {
                "traceId": finalized.trace_id,
                "traceDirectory": finalized.attempt_dir,
                "traceHealth": finalized.health,
                "traceComplete": finalized.complete,
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
        "benchmark": BENCHMARK,
        "model": f"openrouter/{model}",
        "temperature": TEMPERATURE,
        "attempt": 1,
        "maxInfrastructureRetries": 0,
        "apiMaxRetries": API_MAX_RETRIES,
        "delegationToolTimeoutSeconds": DELEGATION_TOOL_TIMEOUT_SECONDS,
        "coordinatorBudget": COORDINATOR_BUDGET,
        "delegationMode": DELEGATION_MODE,
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
            "usage": _usage(coordinator_result),
        },
        "usage": _merge_usage([coordinator_result]),
        "error": error,
        "durationSeconds": round(time.monotonic() - started, 3),
        "trace": trace_result,
    }
    return _redact(result, api_key)


def main() -> int:
    instruction = os.environ.get("HARBOR_INSTRUCTION", "").strip()
    # Hold the credential only in Python memory. Local terminal commands inherit
    # this process environment, so removing it prevents agent shell tools from
    # reading or accidentally printing the provider secret.
    api_key = os.environ.pop("OPENROUTER_API_KEY", "").strip()
    model = os.environ.get("HERMES_BENCHMARK_MODEL", "").strip()
    if not instruction or not api_key or not model:
        missing = [
            name
            for name, value in (
                ("HARBOR_INSTRUCTION", instruction),
                ("OPENROUTER_API_KEY", api_key),
                ("HERMES_BENCHMARK_MODEL", model),
            )
            if not value
        ]
        print(f"Missing required environment: {', '.join(missing)}", file=sys.stderr)
        return 2
    result = run_worker(
        instruction,
        api_key=api_key,
        model=model,
        workdir=Path.cwd(),
    )
    _atomic_write_json(RESULT_PATH, result)
    response = str(result.get("coordinator", {}).get("finalResponse") or "").strip()
    if response:
        print(response)
    return 0 if result.get("error") is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
