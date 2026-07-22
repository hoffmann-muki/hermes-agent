"""Isolated multi-agent worker executed inside one Harbor task environment."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml


BENCHMARK = "terminal-bench-2.1"
COORDINATOR_BUDGET = 24
PHASE_BUDGETS = {"navigator": 10, "patcher": 18, "reviewer": 12}
PHASE_ORDER = tuple(PHASE_BUDGETS)
TEMPERATURE = 0.1
API_MAX_RETRIES = 1
PHASE_TOOL_TIMEOUT_SECONDS = 24 * 60 * 60
MAX_PHASE_REPORT_CHARS = 20_000
READ_ONLY_TOOLSET = "terminal-benchmark-readonly"
PHASE_TOOLSET = "terminal-benchmark"
RESULT_PATH = Path("/logs/agent/hermes-result.json")
SESSION_PATH = Path("/logs/agent/hermes-session.jsonl")


COORDINATOR_SYSTEM_PROMPT = """You are the primary Terminal-Bench coordinator.

You own the final environment state. Use terminal_benchmark_phase exactly once
for each fresh specialist, one call at a time, in this exact order:

1. navigator — investigate the environment and produce an evidence-backed plan.
2. patcher — perform the concrete work and run focused checks.
3. reviewer — independently verify the final state and make small clear corrections.

Wait for each phase result before calling the next. Do not skip, repeat, parallelize,
or replace a phase with general delegation. Passes are synchronous and share the same
task environment. After the reviewer returns, reconcile the reports, inspect any
remaining risk with your own tools, make only necessary final corrections, and give a
concise final answer. Changes to the environment—not prose—are the benchmark answer.
"""

PHASE_SYSTEM_PROMPTS = {
    "navigator": """You are a fresh, read-only Terminal-Bench navigator.
Investigate the task, environment, relevant files, constraints, likely root cause, and
a practical verification strategy. Do not modify state or delegate. Return a concise,
evidence-backed execution plan with concrete paths and commands for the patcher.""",
    "patcher": """You are a fresh Terminal-Bench implementation specialist.
Use the original task and navigator handoff, inspect the environment yourself, perform
the concrete work, and run focused checks. Do not delegate. Leave the required state in
the shared environment and report actions, verification, and remaining risk.""",
    "reviewer": """You are a fresh independent Terminal-Bench reviewer.
Inspect the original task, current environment, and prior handoffs. Verify the final
state, run feasible checks, and make small corrective changes when clearly necessary.
Do not delegate or redo sound work for preference alone. Report findings and risk.""",
}

PHASE_TOOL_SCHEMA: dict[str, Any] = {
    "description": (
        "Run the next required fresh Terminal-Bench specialist synchronously. "
        "Call navigator, then patcher, then reviewer, exactly once each."
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
    # The coordinator invokes each specialist through one synchronous custom
    # tool call. Keep Hermes' generic seven-minute tool watchdog from becoming
    # an undocumented benchmark deadline; Harbor's task timeout remains the
    # authoritative outer limit and terminates this process first.
    os.environ["HERMES_CONCURRENT_TOOL_TIMEOUT_S"] = str(PHASE_TOOL_TIMEOUT_SECONDS)
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
    }
    config_path = hermes_home / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    os.chmod(config_path, 0o600)
    from hermes_cli.config import apply_terminal_config_to_env

    apply_terminal_config_to_env(config=config, override=True)


def _phase_prompt(
    phase: str, instruction: str, records: Sequence[dict[str, Any]], workdir: Path
) -> str:
    handoffs = []
    for record in records:
        report = str(record.get("report") or "").strip()
        if report:
            handoffs.extend([
                f"### {str(record['phase']).title()} handoff",
                report[:MAX_PHASE_REPORT_CHARS],
                "",
            ])
    return "\n".join([
        f"Complete the {phase} phase for this Terminal-Bench 2.1 task.",
        f"Shared working directory: {workdir}",
        "",
        "## Original task",
        instruction.strip(),
        "",
        *handoffs,
    ])


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


@dataclass
class PhaseState:
    instruction: str
    workdir: Path
    api_key: str
    model: str
    task_id: str
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
            expected = (
                PHASE_ORDER[self.next_phase]
                if self.next_phase < len(PHASE_ORDER)
                else None
            )
            if not isinstance(phase, str):
                error = "Benchmark phase must be a string"
                self.protocol_errors.append(error)
                return json.dumps({"error": error})
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
        return json.dumps(
            {
                "phase": phase,
                "status": record["status"],
                "report": str(record.get("report") or "")[:MAX_PHASE_REPORT_CHARS],
                "apiCalls": record.get("apiCalls", 0),
                "budget": record["budget"],
                "error": record.get("error"),
                "nextRequiredPhase": (
                    PHASE_ORDER[self.next_phase]
                    if self.next_phase < len(PHASE_ORDER)
                    else None
                ),
            },
            ensure_ascii=False,
        )

    def _run_phase(
        self,
        phase: str,
        previous_records: Sequence[dict[str, Any]],
        task_id: str | None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        child = None
        result: dict[str, Any] = {}
        error = None
        try:
            child = self.agent_factory(
                role=phase,
                budget=PHASE_BUDGETS[phase],
                api_key=self.api_key,
                model=self.model,
                task_id=self.task_id,
                parent_session_id=getattr(self.coordinator, "session_id", None),
            )
            if self.coordinator is not None:
                with self.coordinator._active_children_lock:
                    self.coordinator._active_children.append(child)
            result = child.run_conversation(
                _phase_prompt(phase, self.instruction, previous_records, self.workdir),
                system_message=PHASE_SYSTEM_PROMPTS[phase],
                task_id=task_id or self.task_id,
            )
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

        completed = bool(result.get("completed"))
        return {
            "phase": phase,
            "budget": PHASE_BUDGETS[phase],
            "status": "completed" if completed and not error else "failed",
            "freshAgent": True,
            "report": result.get("final_response") or "",
            "apiCalls": result.get("api_calls", 0),
            "turnExitReason": result.get("turn_exit_reason"),
            "completed": completed,
            "interrupted": bool(result.get("interrupted")),
            "error": error,
            "messages": result.get("messages", []),
            "usage": _usage(result),
            "durationSeconds": round(time.monotonic() - started, 3),
        }


def default_agent_factory(
    *,
    role: str,
    budget: int,
    api_key: str,
    model: str,
    task_id: str,
    parent_session_id: str | None = None,
) -> Any:
    from hermes_constants import OPENROUTER_BASE_URL
    from run_agent import AIAgent

    if role == "navigator":
        toolsets = [READ_ONLY_TOOLSET]
    elif role == "coordinator":
        toolsets = ["terminal", "file", PHASE_TOOLSET]
    else:
        toolsets = ["terminal", "file"]
    return AIAgent(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        provider="openrouter",
        model=model,
        max_iterations=budget,
        tool_delay=0,
        enabled_toolsets=toolsets,
        save_trajectories=False,
        verbose_logging=False,
        quiet_mode=True,
        request_overrides={"temperature": TEMPERATURE},
        load_soul_identity=False,
        skip_memory=True,
        session_id=f"{task_id}-{role}-{uuid.uuid4().hex[:8]}",
        parent_session_id=parent_session_id or "",
        checkpoints_enabled=False,
    )


def register_phase_tool(state: PhaseState) -> None:
    from toolsets import create_custom_toolset
    from tools.registry import registry

    create_custom_toolset(
        READ_ONLY_TOOLSET,
        "Read-only investigation tools for the Terminal-Bench navigator",
        tools=["read_file", "search_files", "terminal", "process"],
    )
    registry.register(
        name="terminal_benchmark_phase",
        toolset=PHASE_TOOLSET,
        schema=PHASE_TOOL_SCHEMA,
        handler=state.handler,
        description=PHASE_TOOL_SCHEMA["description"],
        emoji="🧪",
    )


def deregister_phase_tool() -> None:
    from toolsets import TOOLSETS
    from tools.registry import registry

    registry.deregister("terminal_benchmark_phase")
    TOOLSETS.pop(READ_ONLY_TOOLSET, None)


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
    _configure_runtime(workdir)
    task_id = f"terminalbench-{uuid.uuid4().hex}"
    state = PhaseState(
        instruction=instruction,
        workdir=workdir,
        api_key=api_key,
        model=model,
        task_id=task_id,
        agent_factory=agent_factory,
    )
    coordinator = None
    coordinator_result: dict[str, Any] = {}
    error = None
    started = time.monotonic()
    try:
        register_phase_tool(state)
        coordinator = agent_factory(
            role="coordinator",
            budget=COORDINATOR_BUDGET,
            api_key=api_key,
            model=model,
            task_id=task_id,
        )
        state.coordinator = coordinator
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
        try:
            deregister_phase_tool()
        except Exception:
            pass

    phase_results = [
        {**record.get("usage", {}), "messages": record.get("messages", [])}
        for record in state.records
    ]
    all_results = [coordinator_result, *phase_results]
    _write_session(all_results, api_key)
    result = {
        "schemaVersion": 1,
        "benchmark": BENCHMARK,
        "model": f"openrouter/{model}",
        "temperature": TEMPERATURE,
        "attempt": 1,
        "maxInfrastructureRetries": 0,
        "apiMaxRetries": API_MAX_RETRIES,
        "phaseToolTimeoutSeconds": PHASE_TOOL_TIMEOUT_SECONDS,
        "coordinatorBudget": COORDINATOR_BUDGET,
        "phaseBudgets": PHASE_BUDGETS,
        "phaseOrder": list(PHASE_ORDER),
        "workflowComplete": state.workflow_complete,
        "protocolErrors": state.protocol_errors,
        "phases": state.records,
        "coordinator": {
            "completed": bool(coordinator_result.get("completed")),
            "interrupted": bool(coordinator_result.get("interrupted")),
            "apiCalls": coordinator_result.get("api_calls", 0),
            "turnExitReason": coordinator_result.get("turn_exit_reason"),
            "finalResponse": coordinator_result.get("final_response"),
            "messages": coordinator_result.get("messages", []),
            "usage": _usage(coordinator_result),
        },
        "usage": _merge_usage(all_results),
        "error": error,
        "durationSeconds": round(time.monotonic() - started, 3),
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
