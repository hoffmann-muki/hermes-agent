"""Hermes-native callback adapter for the benchmark trace contract."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from hermes_cli.benchmarks.tracing.runtime import (
    CAPABILITY_CATEGORIES,
    Capability,
    JsonObject,
    TraceFinalization,
    TraceRecorder,
    TraceStatus,
    utc_now,
)


@dataclass(frozen=True)
class _PendingSpan:
    span_id: str
    start_event_id: str
    start_type: str
    end_type: str
    family: str
    started_ns: int
    session_id: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    tool_name: str | None = None
    parent_span_id: str | None = None


@dataclass(frozen=True)
class _ToolCompletion:
    duration_seconds: float | None
    is_error: bool
    result: Any


_TOOL_GROUPS = {
    "shell": {"terminal", "shell", "bash"},
    "file_read": {"read_file", "read", "view_file"},
    "file_write": {"write_file", "write", "edit_file", "edit"},
    "file_patch": {"patch", "apply_patch"},
    "search": {"search_files", "search", "grep", "glob", "find"},
    "browser": {"browser", "web_search", "web_fetch"},
    "delegation": {"delegate_task"},
}


def hermes_capabilities(
    observed: Mapping[str, set[str]],
    *,
    delegation_enabled: bool,
) -> tuple[Capability, ...]:
    """Return an exhaustive, attempt-specific Hermes capability matrix."""

    disabled = {
        "browser",
        "memory",
        "evaluator.lifecycle",
        *(() if delegation_enabled else ("delegation",)),
    }
    unavailable = {"provider.exchange"}
    characteristics = {
        "agent.session": ("derived", "full", "derived"),
        "model.turn": ("derived", "partial", "derived"),
        "tool.invocation": (
            "captured",
            "partial" if delegation_enabled else "full",
            "not_available",
        ),
        "tool.result": ("captured", "full", "native_monotonic"),
        "tool.timing": ("captured", "full", "native_monotonic"),
        "shell": (
            "captured",
            "partial" if delegation_enabled else "full",
            "native_monotonic",
        ),
        "file": (
            "captured",
            "partial" if delegation_enabled else "full",
            "native_monotonic",
        ),
        "search": (
            "captured",
            "partial" if delegation_enabled else "full",
            "native_monotonic",
        ),
        "delegation": ("captured", "full", "native_monotonic"),
        "context.compaction": ("captured", "metadata_only", "not_available"),
        "harness.lifecycle": ("derived", "full", "derived"),
        "container.lifecycle": ("captured", "metadata_only", "not_available"),
        "patch": ("captured", "partial", "native_monotonic"),
        "native.evidence": ("captured", "partial", "native_wall"),
    }
    limitations = {
        "model.turn": (
            (
                "Root and delegated-child turn timing is derived from native step, "
                "tool, and conversation-return boundaries; exact provider bodies "
                "are not exposed."
                if delegation_enabled
                else "Model-turn timing is derived from native step, tool, and "
                "conversation-return boundaries; exact provider bodies are not exposed."
            ),
            "Token and cost accounting are intentionally excluded.",
        ),
        "provider.exchange": (
            "The benchmark does not enable provider-body logging; credentials, "
            "token usage, and cost fields are never retained.",
        ),
        "tool.invocation": (
            "Root tool arguments are complete; delegated child arguments are the "
            "display-safe values relayed by Hermes.",
        )
        if delegation_enabled
        else (),
        "tool.result": (
            (
                "Root and delegated child tool results are retained after mandatory "
                "credential and accounting-field sanitization."
                if delegation_enabled
                else "Tool results are retained after mandatory credential and "
                "accounting-field sanitization."
            ),
        ),
        "tool.timing": (
            (
                "Hermes exposes native durations for root and delegated child tools."
                if delegation_enabled
                else "Hermes exposes native durations for root tools."
            ),
        ),
        "shell": (
            "Child shell arguments are display-sanitized by Hermes before relay.",
        )
        if delegation_enabled
        else (),
        "file": (
            "Child file-tool arguments are display-sanitized by Hermes before relay.",
        )
        if delegation_enabled
        else (),
        "search": (
            "Child search arguments are display-sanitized by Hermes before relay.",
        )
        if delegation_enabled
        else (),
        "delegation": (
            "Hermes forwards native child lifecycle, text, tool inputs, tool "
            "results, and tool durations; child model activity is represented as "
            "atomic turns without provider request or response bodies.",
        )
        if delegation_enabled
        else (),
        "context.compaction": (
            "Hermes emits a completed compaction fact without a start boundary or "
            "duration.",
        ),
        "container.lifecycle": (
            "The worker exposes container identity after setup; controller-owned "
            "restart, patch capture, and teardown occur outside this adapter.",
        ),
        "patch": (
            "Agent patch-tool activity is visible, but the controller-owned final "
            "Git snapshot is outside the worker adapter.",
        ),
        "native.evidence": (
            "Native callback payloads are sanitized before persistence; accounting "
            "fields are removed by policy.",
        ),
    }
    capabilities = []
    for category in CAPABILITY_CATEGORIES:
        evidence = tuple(sorted(observed.get(category, set())))
        if category in disabled:
            capabilities.append(
                Capability(
                    category=category,
                    state="disabled",
                    coverage="none",
                    timing="not_applicable",
                    limitations=limitations.get(category, ()),
                )
            )
            continue
        if category in unavailable:
            capabilities.append(
                Capability(
                    category=category,
                    state="not_exposed",
                    coverage="none",
                    timing="not_available",
                    limitations=limitations.get(category, ()),
                )
            )
            continue
        if not evidence:
            capabilities.append(
                Capability(
                    category=category,
                    state="not_observed",
                    coverage="none",
                    timing="not_available",
                    limitations=limitations.get(category, ()),
                )
            )
            continue
        state, coverage, timing = characteristics[category]
        capabilities.append(
            Capability(
                category=category,
                state=state,
                coverage=coverage,
                timing=timing,
                evidence=evidence,
                limitations=limitations.get(category, ()),
            )
        )
    return tuple(capabilities)


class HermesTraceAdapter:
    """Translate Hermes' public agent callbacks into normalized trace events."""

    def __init__(
        self,
        recorder: TraceRecorder,
        *,
        delegation_enabled: bool,
    ) -> None:
        self._recorder = recorder
        self._delegation_enabled = delegation_enabled
        self._primary_agent_id = "coordinator" if delegation_enabled else "agent"
        self._lock = threading.RLock()
        self._observed: dict[str, set[str]] = defaultdict(set)
        self._tools: dict[str, _PendingSpan] = {}
        self._child_tools: dict[str, _PendingSpan] = {}
        self._child_models: dict[str, _PendingSpan] = {}
        self._tool_completions: dict[str, deque[_ToolCompletion]] = defaultdict(deque)
        self._subagents: dict[str, _PendingSpan] = {}
        self._delegation_parents: dict[str, deque[str]] = defaultdict(deque)
        self._model: _PendingSpan | None = None
        self._session: _PendingSpan | None = None
        self._transcript_recorded = False
        self._finished: TraceFinalization | None = None
        self._attempt_started_ns = time.monotonic_ns()
        self._startup_started_ns = self._attempt_started_ns
        self._execution_started_ns: int | None = None
        self._execution_ended = False
        self._shutdown_started_ns: int | None = None
        trace_id = recorder.identity.trace_id
        self._instance_span = f"instance-{trace_id}"
        self._attempt_span = f"attempt-{trace_id}"
        self._startup_span = f"hermes-startup-{trace_id}"
        self._execution_span = f"hermes-execution-{trace_id}"
        self._shutdown_span = f"hermes-shutdown-{trace_id}"
        self._record_lifecycle("instance.start", "instance", self._instance_span, None)
        self._record_lifecycle(
            "attempt.start",
            "attempt",
            self._attempt_span,
            self._instance_span,
            payload={
                "agent_configuration": {
                    "delegation_enabled": delegation_enabled,
                    "coordination_mode": "framework_native",
                    "delegation_sequence": (
                        ["navigator", "patcher", "reviewer"]
                        if delegation_enabled
                        else []
                    ),
                    "sequence_enforcement": "prompt_guided",
                }
            },
        )
        self._record_lifecycle(
            "harness.startup_start",
            "harness",
            self._startup_span,
            self._attempt_span,
        )
        self._observe("harness.lifecycle", "harness.startup_start")

    @property
    def identity(self):
        return self._recorder.identity

    @property
    def attempt_dir(self):
        return self._recorder.attempt_dir

    def callbacks(self) -> JsonObject:
        return {
            "tool_progress_callback": self.on_tool_progress,
            "tool_start_callback": self.on_tool_start,
            "tool_complete_callback": self.on_tool_complete,
            "step_callback": self.on_step,
            "event_callback": self.on_event,
        }

    def start_execution(self, *, entered: bool = True) -> None:
        """Transition from coarse worker setup into detailed agent work."""

        with self._lock:
            if self._execution_started_ns is not None:
                return
            boundary = time.monotonic_ns()
            occurred_at = utc_now()
            self._record(
                event_type="harness.startup_end",
                event_family="harness",
                phase="end",
                status="completed",
                span_id=self._startup_span,
                parent_span_id=self._attempt_span,
                origin=_harness_origin(),
                timing=_derived_duration(self._startup_started_ns, boundary),
                payload={},
                occurred_at=occurred_at,
            )
            self._observe("harness.lifecycle", "harness.startup_end")
            self._execution_started_ns = boundary
            self._record(
                event_type="agent.execution_start",
                event_family="agent",
                phase="start",
                status="started",
                span_id=self._execution_span,
                parent_span_id=self._attempt_span,
                origin=_harness_origin(),
                timing={"fidelity": "derived"},
                payload={"entered": entered},
                occurred_at=occurred_at,
            )

    def start_session(self, session_id: str) -> None:
        with self._lock:
            self.start_execution()
            if self._session is not None:
                return
            started_ns = time.monotonic_ns()
            span_id = f"hermes-session-{self.identity.trace_id}"
            event_id = self._record(
                event_type="agent.session_start",
                event_family="agent",
                phase="start",
                status="started",
                span_id=span_id,
                parent_span_id=self._execution_span,
                session_id=session_id,
                agent_id=self._primary_agent_id,
                origin=_derived_origin(),
                timing={"fidelity": "derived"},
                payload={"role": self._primary_agent_id},
            )
            if event_id is None:
                return
            self._session = _PendingSpan(
                span_id=span_id,
                start_event_id=event_id,
                start_type="agent.session_start",
                end_type="agent.session_end",
                family="agent",
                started_ns=started_ns,
                session_id=session_id,
                agent_id=self._primary_agent_id,
            )
            self._observe("agent.session", "agent.session_start")

    def container_observed(self, metadata: Mapping[str, Any]) -> None:
        with self._lock:
            event_id = self._record(
                event_type="container.observed",
                event_family="container",
                phase="instant",
                status="completed",
                span_id=f"hermes-container-{self.identity.trace_id}",
                parent_span_id=self._lifecycle_parent,
                origin=_harness_origin(),
                timing={"fidelity": "not_available"},
                payload=dict(metadata),
            )
            if event_id is not None:
                self._observe("container.lifecycle", "container.observed")

    def on_step(self, api_call_count: int, previous_tools: Any) -> None:
        with self._lock:
            if self._model is not None:
                self._end_model("completed", "next_model_turn")
            started_ns = time.monotonic_ns()
            span_id = f"hermes-model-{self.identity.trace_id}-{api_call_count}"
            event_id = self._record(
                event_type="model.turn_start",
                event_family="model",
                phase="start",
                status="started",
                span_id=span_id,
                parent_span_id=self._session_parent,
                session_id=self._session_id,
                agent_id=self._primary_agent_id,
                origin=_native_origin("agent.step_callback"),
                timing={"fidelity": "derived"},
                payload={"api_call_count": api_call_count},
            )
            if event_id is None:
                return
            self._model = _PendingSpan(
                span_id=span_id,
                start_event_id=event_id,
                start_type="model.turn_start",
                end_type="model.turn_end",
                family="model",
                started_ns=started_ns,
                session_id=self._session_id,
                agent_id=self._primary_agent_id,
            )
            self._observe("model.turn", "model.turn_start")
            self._native(
                "hermes.step_callback",
                {
                    "api_call_count": api_call_count,
                    "previous_tools": previous_tools,
                },
                ((event_id, "model.turn_start"),),
            )

    def on_tool_start(self, call_id: Any, name: Any, arguments: Any) -> None:
        with self._lock:
            tool_id = str(call_id)
            tool_name = str(name)
            if self._model is not None:
                self._end_model("completed", "tool_calls")
            classification = _classify_tool(tool_name)
            artifact = self._recorder.store_json_artifact(
                arguments, role="tool.arguments"
            )
            started_ns = time.monotonic_ns()
            span_id = f"hermes-tool-{tool_id}"
            event_id = self._record(
                event_type=classification[0],
                event_family=classification[2],
                phase="start",
                status="started",
                span_id=span_id,
                parent_span_id=self._session_parent,
                session_id=self._session_id,
                agent_id=self._primary_agent_id,
                origin=_native_origin("agent.tool_start_callback"),
                timing={
                    "fidelity": "native_monotonic",
                    "clock_id": "hermes.tool_duration",
                },
                payload={
                    "tool": {
                        "name": tool_name,
                        "call_id": tool_id,
                        "arguments": arguments,
                    }
                },
                artifacts=(artifact,) if artifact else (),
            )
            if event_id is None:
                return
            self._tools[tool_id] = _PendingSpan(
                span_id=span_id,
                start_event_id=event_id,
                start_type=classification[0],
                end_type=classification[1],
                family=classification[2],
                started_ns=started_ns,
                session_id=self._session_id,
                agent_id=self._primary_agent_id,
                tool_name=tool_name,
            )
            if classification[2] == "delegation" and isinstance(arguments, Mapping):
                goal = arguments.get("goal")
                if isinstance(goal, str):
                    self._delegation_parents[goal].append(span_id)
            self._observe("tool.invocation", classification[0])
            self._observe_tool_group(tool_name, classification[0])
            self._native(
                "hermes.tool_start_callback",
                {"call_id": tool_id, "name": tool_name, "arguments": arguments},
                ((event_id, classification[0]),),
            )

    def on_tool_complete(
        self, call_id: Any, name: Any, arguments: Any, result: Any
    ) -> None:
        with self._lock:
            tool_id = str(call_id)
            tool_name = str(name)
            pending = self._tools.pop(tool_id, None)
            if pending is None:
                self._recorder.report_issue(
                    "hermes.tool_completion_without_start",
                    "Hermes emitted a tool completion without a matching start",
                )
                return
            completion = (
                self._tool_completions[tool_name].popleft()
                if self._tool_completions[tool_name]
                else _ToolCompletion(None, False, result)
            )
            duration_ms = (
                max(0.0, completion.duration_seconds * 1000)
                if completion.duration_seconds is not None
                else (time.monotonic_ns() - pending.started_ns) / 1_000_000
            )
            artifact = (
                self._recorder.store_text_artifact(
                    result,
                    role="tool.error" if completion.is_error else "tool.output",
                )
                if isinstance(result, str)
                else self._recorder.store_json_artifact(
                    result,
                    role="tool.error" if completion.is_error else "tool.output",
                )
            )
            status = "failed" if completion.is_error else "completed"
            event_id = self._record(
                event_type=pending.end_type,
                event_family=pending.family,
                phase="end",
                status=status,
                span_id=pending.span_id,
                parent_span_id=self._session_parent,
                session_id=pending.session_id,
                agent_id=pending.agent_id,
                origin=_native_origin("agent.tool_complete_callback"),
                timing={
                    "fidelity": "native_monotonic",
                    "clock_id": "hermes.tool_duration",
                    "duration_ms": duration_ms,
                },
                payload={
                    "tool": {
                        "name": tool_name,
                        "call_id": tool_id,
                        "arguments": arguments,
                    }
                },
                artifacts=(artifact,) if artifact else (),
                error=(
                    {
                        "code": "hermes.tool_failed",
                        "message": "Hermes reported a failed tool invocation",
                    }
                    if completion.is_error
                    else None
                ),
                relations=({"type": "caused_by", "event_id": pending.start_event_id},),
            )
            if event_id is None:
                return
            self._observe("tool.result", pending.end_type)
            self._observe("tool.timing", pending.end_type)
            self._observe_tool_group(tool_name, pending.end_type)
            self._native(
                "hermes.tool_complete_callback",
                {
                    "call_id": tool_id,
                    "name": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "duration_seconds": completion.duration_seconds,
                    "is_error": completion.is_error,
                },
                ((event_id, pending.end_type),),
            )

    def on_tool_progress(
        self,
        event_type: Any,
        tool_name: Any = None,
        preview: Any = None,
        arguments: Any = None,
        **metadata: Any,
    ) -> None:
        with self._lock:
            native_type = str(event_type)
            name = str(tool_name or "")
            if native_type == "tool.completed":
                duration = metadata.get("duration")
                self._tool_completions[name].append(
                    _ToolCompletion(
                        float(duration) if isinstance(duration, (int, float)) else None,
                        bool(metadata.get("is_error")),
                        metadata.get("result"),
                    )
                )
                return
            if native_type in {"tool.started", "_thinking"}:
                return
            if native_type == "reasoning.available":
                self._instant_artifact_event(
                    "model.output",
                    "model",
                    preview or "",
                    "model.output",
                    native_type,
                )
                return
            if native_type == "subagent.start":
                self._subagent_start(preview, metadata)
                return
            if native_type == "subagent.complete":
                self._subagent_complete(preview, metadata)
                return
            if native_type in {"subagent.text", "subagent.thinking"}:
                self._instant_artifact_event(
                    (
                        "model.stream_delta"
                        if native_type == "subagent.text"
                        else "model.reasoning"
                    ),
                    "model",
                    preview or "",
                    native_type,
                    native_type,
                    metadata,
                )
                return
            if native_type == "subagent.tool":
                self._subagent_tool_start(
                    name,
                    preview,
                    arguments,
                    metadata,
                )
                return
            if native_type == "subagent.tool_complete":
                self._subagent_tool_complete(name, metadata)
                return
            if native_type == "subagent.model_turn":
                self._subagent_model_start(metadata)
                return
            if native_type == "subagent.model_turn_complete":
                self._subagent_model_complete(metadata)
                return
            self._instant_artifact_event(
                "agent.progress",
                "agent",
                _progress_payload(native_type, name, preview, arguments, metadata),
                "agent.progress",
                native_type,
                metadata,
            )

    def on_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            if event_type == "session:compress":
                event_id = self._record(
                    event_type="context.compaction",
                    event_family="context",
                    phase="instant",
                    status="completed",
                    span_id=f"hermes-compaction-{self.identity.trace_id}-{time.monotonic_ns()}",
                    parent_span_id=self._session_parent,
                    session_id=_optional_string(payload.get("session_id")),
                    origin=_native_origin(event_type),
                    timing={"fidelity": "not_available"},
                    payload=dict(payload),
                )
                if event_id is not None:
                    self._observe("context.compaction", "context.compaction")
                    self._native(
                        "hermes.event_callback.session_compress",
                        dict(payload),
                        ((event_id, "context.compaction"),),
                    )
                return
            self._instant_artifact_event(
                "agent.event",
                "agent",
                dict(payload),
                "agent.event",
                event_type,
            )

    def record_transcript(
        self,
        messages: Any,
        *,
        occurred_at: str | None = None,
    ) -> None:
        with self._lock:
            if self._transcript_recorded or not isinstance(messages, list):
                return
            self._transcript_recorded = True
            linked: list[tuple[str, str]] = []
            for index, message in enumerate(messages):
                if (
                    not isinstance(message, Mapping)
                    or message.get("role") != "assistant"
                ):
                    continue
                content = message.get("content")
                artifact = (
                    self._recorder.store_text_artifact(content, role="model.response")
                    if isinstance(content, str)
                    else self._recorder.store_json_artifact(
                        content, role="model.response"
                    )
                )
                event_id = self._record(
                    event_type="model.response",
                    event_family="model",
                    phase="instant",
                    status="completed",
                    span_id=f"hermes-response-{self.identity.trace_id}-{index}",
                    parent_span_id=self._session_parent,
                    session_id=self._session_id,
                    agent_id=self._primary_agent_id,
                    origin=_native_origin("run_conversation.messages"),
                    timing={"fidelity": "not_available"},
                    payload={
                        "message_index": index,
                        "has_tool_calls": bool(message.get("tool_calls")),
                    },
                    artifacts=(artifact,) if artifact else (),
                    occurred_at=occurred_at,
                )
                if event_id is not None:
                    linked.append((event_id, "model.response"))
                    self._observe("model.turn", "model.response")
            if linked:
                self._native(
                    "hermes.run_conversation.messages",
                    messages,
                    tuple(linked),
                )

    def end_execution(
        self,
        status: TraceStatus,
        *,
        messages: Any = None,
        error_message: str | None = None,
    ) -> None:
        """Close detailed agent activity and begin coarse worker shutdown."""

        with self._lock:
            if self._execution_ended:
                return
            boundary = time.monotonic_ns()
            occurred_at = utc_now()
            self.record_transcript(messages, occurred_at=occurred_at)
            if self._execution_started_ns is None:
                self._record(
                    event_type="harness.startup_end",
                    event_family="harness",
                    phase="end",
                    status=status,
                    span_id=self._startup_span,
                    parent_span_id=self._attempt_span,
                    origin=_harness_origin(),
                    timing=_derived_duration(self._startup_started_ns, boundary),
                    payload={},
                    error=_lifecycle_error(status, error_message),
                    occurred_at=occurred_at,
                )
                self._observe("harness.lifecycle", "harness.startup_end")
                self._execution_started_ns = boundary
                self._record(
                    event_type="agent.execution_start",
                    event_family="agent",
                    phase="start",
                    status="started",
                    span_id=self._execution_span,
                    parent_span_id=self._attempt_span,
                    origin=_harness_origin(),
                    timing={"fidelity": "derived"},
                    payload={"entered": False},
                    occurred_at=occurred_at,
                )
            if self._model is not None:
                self._end_model(
                    status,
                    "conversation_return",
                    ended_ns=boundary,
                    occurred_at=occurred_at,
                )
            for tool_id, pending in tuple(self._tools.items()):
                self._close_incomplete_tool(
                    tool_id,
                    pending,
                    status,
                    ended_ns=boundary,
                    occurred_at=occurred_at,
                )
            for tool_id, pending in tuple(self._child_tools.items()):
                self._close_incomplete_child_tool(
                    tool_id,
                    pending,
                    status,
                    ended_ns=boundary,
                    occurred_at=occurred_at,
                )
            for turn_id, pending in tuple(self._child_models.items()):
                self._close_incomplete_child_model(
                    turn_id,
                    pending,
                    status,
                    ended_ns=boundary,
                    occurred_at=occurred_at,
                )
            for subagent_id, pending in tuple(self._subagents.items()):
                self._close_incomplete_subagent(
                    subagent_id,
                    pending,
                    status,
                    ended_ns=boundary,
                    occurred_at=occurred_at,
                )
            if self._session is not None:
                self._end_session(
                    status,
                    ended_ns=boundary,
                    occurred_at=occurred_at,
                )
            assert self._execution_started_ns is not None
            self._record(
                event_type="agent.execution_end",
                event_family="agent",
                phase="end",
                status=status,
                span_id=self._execution_span,
                parent_span_id=self._attempt_span,
                origin=_harness_origin(),
                timing=_derived_duration(self._execution_started_ns, boundary),
                payload={},
                error=_lifecycle_error(status, error_message),
                occurred_at=occurred_at,
            )
            self._execution_ended = True
            self._shutdown_started_ns = boundary
            self._record(
                event_type="harness.shutdown_start",
                event_family="harness",
                phase="start",
                status="started",
                span_id=self._shutdown_span,
                parent_span_id=self._attempt_span,
                origin=_harness_origin(),
                timing={"fidelity": "derived"},
                payload={},
                occurred_at=occurred_at,
            )
            self._observe("harness.lifecycle", "harness.shutdown_start")

    def finish(
        self,
        status: TraceStatus,
        *,
        messages: Any = None,
        error_message: str | None = None,
    ) -> TraceFinalization:
        with self._lock:
            if self._finished is not None:
                return self._finished
            self.end_execution(
                status,
                messages=messages,
                error_message=error_message,
            )
            assert self._shutdown_started_ns is not None
            self._end_lifecycle(
                "harness.shutdown_end",
                "harness",
                self._shutdown_span,
                self._attempt_span,
                status,
                error_message,
                started_ns=self._shutdown_started_ns,
            )
            self._observe("harness.lifecycle", "harness.shutdown_end")
            self._end_lifecycle(
                "attempt.end",
                "attempt",
                self._attempt_span,
                self._instance_span,
                status,
                error_message,
            )
            self._end_lifecycle(
                "instance.end",
                "instance",
                self._instance_span,
                None,
                status,
                error_message,
            )
            self._recorder.update_capabilities(
                hermes_capabilities(
                    dict(self._observed),
                    delegation_enabled=self._delegation_enabled,
                )
            )
            self._finished = self._recorder.finalize()
            return self._finished

    @property
    def _session_id(self) -> str | None:
        return self._session.session_id if self._session else None

    @property
    def _session_parent(self) -> str:
        return self._session.span_id if self._session else self._execution_span

    @property
    def _lifecycle_parent(self) -> str:
        if self._execution_started_ns is None:
            return self._startup_span
        if not self._execution_ended:
            return self._execution_span
        return self._shutdown_span

    def _end_model(
        self,
        status: TraceStatus,
        reason: str,
        *,
        ended_ns: int | None = None,
        occurred_at: str | None = None,
    ) -> None:
        pending = self._model
        if pending is None:
            return
        event_id = self._record(
            event_type=pending.end_type,
            event_family=pending.family,
            phase="end",
            status=status,
            span_id=pending.span_id,
            parent_span_id=self._session_parent,
            session_id=pending.session_id,
            agent_id=pending.agent_id,
            origin=_derived_origin(),
            timing=_derived_duration(pending.started_ns, ended_ns),
            payload={"boundary": reason},
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
            occurred_at=occurred_at,
        )
        if event_id is not None:
            self._observe("model.turn", pending.end_type)
        self._model = None

    def _end_session(
        self,
        status: TraceStatus,
        *,
        ended_ns: int | None = None,
        occurred_at: str | None = None,
    ) -> None:
        pending = self._session
        if pending is None:
            return
        event_id = self._record(
            event_type=pending.end_type,
            event_family=pending.family,
            phase="end",
            status=status,
            span_id=pending.span_id,
            parent_span_id=self._execution_span,
            session_id=pending.session_id,
            agent_id=pending.agent_id,
            origin=_derived_origin(),
            timing=_derived_duration(pending.started_ns, ended_ns),
            payload={},
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
            occurred_at=occurred_at,
        )
        if event_id is not None:
            self._observe("agent.session", pending.end_type)
        self._session = None

    def _subagent_tool_start(
        self,
        tool_name: str,
        preview: Any,
        arguments: Any,
        metadata: Mapping[str, Any],
    ) -> None:
        subagent_id = str(
            metadata.get("subagent_id") or metadata.get("child_session_id") or ""
        )
        child_tool_id = str(
            metadata.get("child_tool_id") or f"unknown-{time.monotonic_ns()}"
        )
        key = f"{subagent_id}:{child_tool_id}"
        classification = _classify_tool(tool_name)
        parent_span_id = self._subagent_parent(metadata)
        artifact = self._recorder.store_json_artifact(
            arguments or {},
            role="tool.arguments",
        )
        started_ns = time.monotonic_ns()
        span_id = (
            f"hermes-child-tool-{self.identity.trace_id}-{subagent_id}-{child_tool_id}"
        )
        event_id = self._record(
            event_type=classification[0],
            event_family=classification[2],
            phase="start",
            status="started",
            span_id=span_id,
            parent_span_id=parent_span_id,
            session_id=_optional_string(metadata.get("child_session_id")),
            agent_id=_optional_string(metadata.get("subagent_id")),
            parent_agent_id=_optional_string(metadata.get("parent_id")),
            origin=_native_origin("subagent.tool"),
            timing={
                "fidelity": "native_monotonic",
                "clock_id": "hermes.subagent_tool_duration",
            },
            payload={
                "tool": {
                    "name": tool_name,
                    "call_id": child_tool_id,
                    "arguments": arguments or {},
                },
                "preview": str(preview or ""),
            },
            artifacts=(artifact,) if artifact else (),
        )
        if event_id is None:
            return
        self._child_tools[key] = _PendingSpan(
            span_id=span_id,
            start_event_id=event_id,
            start_type=classification[0],
            end_type=classification[1],
            family=classification[2],
            started_ns=started_ns,
            session_id=_optional_string(metadata.get("child_session_id")),
            agent_id=_optional_string(metadata.get("subagent_id")),
            parent_agent_id=_optional_string(metadata.get("parent_id")),
            tool_name=tool_name,
            parent_span_id=parent_span_id,
        )
        self._observe("tool.invocation", classification[0])
        self._observe_tool_group(tool_name, classification[0])
        self._native(
            "hermes.subagent.tool",
            _progress_payload(
                "subagent.tool",
                tool_name,
                preview,
                arguments,
                metadata,
            ),
            ((event_id, classification[0]),),
        )

    def _subagent_tool_complete(
        self,
        tool_name: str,
        metadata: Mapping[str, Any],
    ) -> None:
        subagent_id = str(
            metadata.get("subagent_id") or metadata.get("child_session_id") or ""
        )
        child_tool_id = str(metadata.get("child_tool_id") or "")
        pending = self._child_tools.pop(
            f"{subagent_id}:{child_tool_id}",
            None,
        )
        if pending is None:
            self._recorder.report_issue(
                "hermes.subagent_tool_completion_without_start",
                "Hermes emitted a child tool completion without a matching start",
            )
            return
        duration = metadata.get("duration_seconds")
        duration_ms = (
            max(0.0, float(duration) * 1000)
            if isinstance(duration, (int, float))
            else (time.monotonic_ns() - pending.started_ns) / 1_000_000
        )
        is_error = bool(metadata.get("is_error"))
        result = metadata.get("result")
        artifact = (
            self._recorder.store_text_artifact(
                result,
                role="tool.error" if is_error else "tool.output",
            )
            if isinstance(result, str)
            else self._recorder.store_json_artifact(
                result,
                role="tool.error" if is_error else "tool.output",
            )
        )
        event_id = self._record(
            event_type=pending.end_type,
            event_family=pending.family,
            phase="end",
            status="failed" if is_error else "completed",
            span_id=pending.span_id,
            parent_span_id=pending.parent_span_id or self._session_parent,
            session_id=pending.session_id,
            agent_id=pending.agent_id,
            parent_agent_id=pending.parent_agent_id,
            origin=_native_origin("subagent.tool_complete"),
            timing={
                "fidelity": "native_monotonic",
                "clock_id": "hermes.subagent_tool_duration",
                "duration_ms": duration_ms,
            },
            payload={
                "tool": {
                    "name": tool_name,
                    "call_id": child_tool_id,
                }
            },
            artifacts=(artifact,) if artifact else (),
            error=(
                {
                    "code": "hermes.subagent_tool_failed",
                    "message": "Hermes reported a failed child tool invocation",
                }
                if is_error
                else None
            ),
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
        )
        if event_id is None:
            return
        self._observe("tool.result", pending.end_type)
        self._observe("tool.timing", pending.end_type)
        self._observe_tool_group(tool_name, pending.end_type)
        self._native(
            "hermes.subagent.tool_complete",
            _progress_payload(
                "subagent.tool_complete",
                tool_name,
                None,
                None,
                metadata,
            ),
            ((event_id, pending.end_type),),
        )

    def _subagent_model_start(self, metadata: Mapping[str, Any]) -> None:
        subagent_id = str(
            metadata.get("subagent_id") or metadata.get("child_session_id") or ""
        )
        child_turn_id = str(
            metadata.get("child_turn_id") or f"unknown-{time.monotonic_ns()}"
        )
        key = f"{subagent_id}:{child_turn_id}"
        started_ns = time.monotonic_ns()
        parent_span_id = self._subagent_parent(metadata)
        span_id = (
            f"hermes-child-model-{self.identity.trace_id}-{subagent_id}-{child_turn_id}"
        )
        event_id = self._record(
            event_type="model.turn_start",
            event_family="model",
            phase="start",
            status="started",
            span_id=span_id,
            parent_span_id=parent_span_id,
            session_id=_optional_string(metadata.get("child_session_id")),
            agent_id=_optional_string(metadata.get("subagent_id")),
            parent_agent_id=_optional_string(metadata.get("parent_id")),
            turn_id=child_turn_id,
            origin=_native_origin("subagent.model_turn"),
            timing={
                "fidelity": "native_monotonic",
                "clock_id": "hermes.subagent_model_duration",
            },
            payload={
                "api_call_count": metadata.get("api_call_count"),
                "boundary": "child_step_callback",
            },
        )
        if event_id is None:
            return
        self._child_models[key] = _PendingSpan(
            span_id=span_id,
            start_event_id=event_id,
            start_type="model.turn_start",
            end_type="model.turn_end",
            family="model",
            started_ns=started_ns,
            session_id=_optional_string(metadata.get("child_session_id")),
            agent_id=_optional_string(metadata.get("subagent_id")),
            parent_agent_id=_optional_string(metadata.get("parent_id")),
            tool_name="model",
            parent_span_id=parent_span_id,
        )
        self._observe("model.turn", "model.turn_start")
        self._native(
            "hermes.subagent.model_turn",
            dict(metadata),
            ((event_id, "model.turn_start"),),
        )

    def _subagent_model_complete(self, metadata: Mapping[str, Any]) -> None:
        subagent_id = str(
            metadata.get("subagent_id") or metadata.get("child_session_id") or ""
        )
        child_turn_id = str(metadata.get("child_turn_id") or "")
        pending = self._child_models.pop(
            f"{subagent_id}:{child_turn_id}",
            None,
        )
        if pending is None:
            self._recorder.report_issue(
                "hermes.subagent_model_completion_without_start",
                "Hermes emitted a child model completion without a matching start",
            )
            return
        duration = metadata.get("duration_seconds")
        duration_ms = (
            max(0.0, float(duration) * 1000)
            if isinstance(duration, (int, float))
            else (time.monotonic_ns() - pending.started_ns) / 1_000_000
        )
        reported_status = str(metadata.get("status") or "completed")
        status = _normalized_native_status(reported_status)
        event_id = self._record(
            event_type="model.turn_end",
            event_family="model",
            phase="end",
            status=status,
            span_id=pending.span_id,
            parent_span_id=pending.parent_span_id or self._session_parent,
            session_id=pending.session_id,
            agent_id=pending.agent_id,
            parent_agent_id=pending.parent_agent_id,
            turn_id=child_turn_id,
            origin=_native_origin("subagent.model_turn_complete"),
            timing={
                "fidelity": "native_monotonic",
                "clock_id": "hermes.subagent_model_duration",
                "duration_ms": duration_ms,
            },
            payload={"boundary": str(metadata.get("boundary") or "unknown")},
            error=(
                {
                    "code": "hermes.subagent_model_failed",
                    "message": "Hermes reported an unsuccessful child model turn",
                }
                if status != "completed"
                else None
            ),
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
        )
        if event_id is None:
            return
        self._observe("model.turn", "model.turn_end")
        self._native(
            "hermes.subagent.model_turn_complete",
            dict(metadata),
            ((event_id, "model.turn_end"),),
        )

    def _subagent_start(self, preview: Any, metadata: Mapping[str, Any]) -> None:
        subagent_id = str(
            metadata.get("subagent_id")
            or metadata.get("child_session_id")
            or f"unknown-{time.monotonic_ns()}"
        )
        span_id = f"hermes-child-session-{subagent_id}"
        goal = str(metadata.get("goal") or preview or "")
        matching_parents = self._delegation_parents.get(goal)
        parent_span_id = (
            matching_parents.popleft()
            if matching_parents
            else next(
                (
                    pending.span_id
                    for pending in reversed(tuple(self._tools.values()))
                    if pending.family == "delegation"
                ),
                self._session_parent,
            )
        )
        if matching_parents is not None and not matching_parents:
            self._delegation_parents.pop(goal, None)
        artifact = self._recorder.store_json_artifact(
            _progress_payload("subagent.start", "", preview, None, metadata),
            role="agent.session.input",
        )
        event_id = self._record(
            event_type="agent.session_start",
            event_family="agent",
            phase="start",
            status="started",
            span_id=span_id,
            parent_span_id=parent_span_id,
            session_id=_optional_string(metadata.get("child_session_id")),
            agent_id=subagent_id,
            parent_agent_id=_optional_string(metadata.get("parent_id")),
            origin=_native_origin("subagent.start"),
            timing={
                "fidelity": "native_monotonic",
                "clock_id": "hermes.subagent_duration",
            },
            payload={
                "goal": str(metadata.get("goal") or preview or ""),
                "model": str(metadata.get("model") or ""),
                "toolsets": metadata.get("toolsets") or [],
                "depth": metadata.get("depth"),
            },
            artifacts=(artifact,) if artifact else (),
        )
        if event_id is None:
            return
        self._subagents[subagent_id] = _PendingSpan(
            span_id=span_id,
            start_event_id=event_id,
            start_type="agent.session_start",
            end_type="agent.session_end",
            family="agent",
            started_ns=time.monotonic_ns(),
            session_id=_optional_string(metadata.get("child_session_id")),
            agent_id=subagent_id,
            parent_agent_id=_optional_string(metadata.get("parent_id")),
            parent_span_id=parent_span_id,
        )
        self._observe("agent.session", "agent.session_start")
        self._native(
            "hermes.subagent.start",
            _progress_payload("subagent.start", "", preview, None, metadata),
            ((event_id, "agent.session_start"),),
        )

    def _subagent_complete(self, preview: Any, metadata: Mapping[str, Any]) -> None:
        subagent_id = str(
            metadata.get("subagent_id") or metadata.get("child_session_id") or ""
        )
        pending = self._subagents.pop(subagent_id, None)
        if pending is None:
            self._recorder.report_issue(
                "hermes.subagent_completion_without_start",
                "Hermes emitted a subagent completion without a matching start",
            )
            return
        native_duration = metadata.get("duration_seconds")
        duration_ms = (
            max(0.0, float(native_duration) * 1000)
            if isinstance(native_duration, (int, float))
            else (time.monotonic_ns() - pending.started_ns) / 1_000_000
        )
        artifact = self._recorder.store_text_artifact(
            str(preview or ""), role="agent.session.output"
        )
        native_status = str(metadata.get("status") or "completed")
        status = _normalized_native_status(native_status)
        event_id = self._record(
            event_type=pending.end_type,
            event_family=pending.family,
            phase="end",
            status=status,
            span_id=pending.span_id,
            parent_span_id=pending.parent_span_id or self._session_parent,
            session_id=pending.session_id,
            agent_id=pending.agent_id,
            parent_agent_id=pending.parent_agent_id,
            origin=_native_origin("subagent.complete"),
            timing={
                "fidelity": "native_monotonic",
                "clock_id": "hermes.subagent_duration",
                "duration_ms": duration_ms,
            },
            payload={
                "native_status": native_status,
                "tool_count": metadata.get("tool_count"),
            },
            artifacts=(artifact,) if artifact else (),
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
        )
        if event_id is None:
            return
        self._observe("agent.session", pending.end_type)
        self._native(
            "hermes.subagent.complete",
            _progress_payload("subagent.complete", "", preview, None, metadata),
            ((event_id, pending.end_type),),
        )

    def _instant_artifact_event(
        self,
        event_type: str,
        family: str,
        content: Any,
        role: str,
        native_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        artifact = (
            self._recorder.store_text_artifact(content, role=role)
            if isinstance(content, str)
            else self._recorder.store_json_artifact(content, role=role)
        )
        event_id = self._record(
            event_type=event_type,
            event_family=family,
            phase="instant",
            status="completed",
            span_id=f"hermes-{family}-{self.identity.trace_id}-{time.monotonic_ns()}",
            parent_span_id=(
                self._subagent_parent(metadata) if metadata else self._session_parent
            ),
            session_id=(
                _optional_string(metadata.get("child_session_id"))
                if metadata
                else self._session_id
            ),
            agent_id=(
                _optional_string(metadata.get("subagent_id"))
                if metadata
                else self._primary_agent_id
            ),
            parent_agent_id=(
                _optional_string(metadata.get("parent_id")) if metadata else None
            ),
            origin=_native_origin(native_type),
            timing={"fidelity": "not_available"},
            payload={"native_event_type": native_type},
            artifacts=(artifact,) if artifact else (),
        )
        if event_id is not None:
            if family == "model":
                self._observe("model.turn", event_type)
            self._native(
                f"hermes.{native_type}",
                (
                    _progress_payload(native_type, "", content, None, metadata)
                    if metadata
                    else {"event_type": native_type, "content": content}
                ),
                ((event_id, event_type),),
            )

    def _close_incomplete_tool(
        self,
        tool_id: str,
        pending: _PendingSpan,
        status: TraceStatus,
        *,
        ended_ns: int,
        occurred_at: str,
    ) -> None:
        self._record(
            event_type=pending.end_type,
            event_family=pending.family,
            phase="end",
            status=status,
            span_id=pending.span_id,
            parent_span_id=pending.parent_span_id or self._session_parent,
            session_id=pending.session_id,
            agent_id=pending.agent_id,
            origin=_derived_origin(),
            timing=_derived_duration(pending.started_ns, ended_ns),
            payload={
                "tool": {
                    "name": pending.tool_name or "unknown",
                    "call_id": tool_id,
                },
                "boundary": "adapter_finalization",
            },
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
            occurred_at=occurred_at,
        )
        self._tools.pop(tool_id, None)

    def _close_incomplete_subagent(
        self,
        subagent_id: str,
        pending: _PendingSpan,
        status: TraceStatus,
        *,
        ended_ns: int,
        occurred_at: str,
    ) -> None:
        self._record(
            event_type=pending.end_type,
            event_family=pending.family,
            phase="end",
            status=status,
            span_id=pending.span_id,
            parent_span_id=pending.parent_span_id or self._session_parent,
            session_id=pending.session_id,
            agent_id=pending.agent_id,
            parent_agent_id=pending.parent_agent_id,
            origin=_derived_origin(),
            timing=_derived_duration(pending.started_ns, ended_ns),
            payload={"boundary": "adapter_finalization"},
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
            occurred_at=occurred_at,
        )
        self._subagents.pop(subagent_id, None)

    def _close_incomplete_child_tool(
        self,
        tool_id: str,
        pending: _PendingSpan,
        status: TraceStatus,
        *,
        ended_ns: int,
        occurred_at: str,
    ) -> None:
        self._record(
            event_type=pending.end_type,
            event_family=pending.family,
            phase="end",
            status=status,
            span_id=pending.span_id,
            parent_span_id=pending.parent_span_id or self._session_parent,
            session_id=pending.session_id,
            agent_id=pending.agent_id,
            parent_agent_id=pending.parent_agent_id,
            origin=_derived_origin(),
            timing=_derived_duration(pending.started_ns, ended_ns),
            payload={
                "tool": {
                    "name": pending.tool_name or "unknown",
                    "call_id": tool_id,
                },
                "boundary": "execution_finalization",
            },
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
            occurred_at=occurred_at,
        )
        self._child_tools.pop(tool_id, None)

    def _close_incomplete_child_model(
        self,
        turn_id: str,
        pending: _PendingSpan,
        status: TraceStatus,
        *,
        ended_ns: int,
        occurred_at: str,
    ) -> None:
        self._record(
            event_type=pending.end_type,
            event_family=pending.family,
            phase="end",
            status=status,
            span_id=pending.span_id,
            parent_span_id=pending.parent_span_id or self._session_parent,
            session_id=pending.session_id,
            agent_id=pending.agent_id,
            parent_agent_id=pending.parent_agent_id,
            turn_id=turn_id.rsplit(":", 1)[-1],
            origin=_derived_origin(),
            timing=_derived_duration(pending.started_ns, ended_ns),
            payload={"boundary": "execution_finalization"},
            relations=({"type": "caused_by", "event_id": pending.start_event_id},),
            occurred_at=occurred_at,
        )
        self._child_models.pop(turn_id, None)

    def _record_lifecycle(
        self,
        event_type: str,
        family: str,
        span_id: str,
        parent_span_id: str | None,
        payload: JsonObject | None = None,
    ) -> None:
        self._record(
            event_type=event_type,
            event_family=family,
            phase="start",
            status="started",
            span_id=span_id,
            parent_span_id=parent_span_id,
            origin=_harness_origin(),
            timing={"fidelity": "derived"},
            payload=payload or {},
        )

    def _end_lifecycle(
        self,
        event_type: str,
        family: str,
        span_id: str,
        parent_span_id: str | None,
        status: TraceStatus,
        error_message: str | None = None,
        *,
        started_ns: int | None = None,
    ) -> None:
        self._record(
            event_type=event_type,
            event_family=family,
            phase="end",
            status=status,
            span_id=span_id,
            parent_span_id=parent_span_id,
            origin=_harness_origin(),
            timing=_derived_duration(started_ns or self._attempt_started_ns),
            payload={},
            error=_lifecycle_error(status, error_message),
        )

    def _native(
        self,
        source: str,
        content: Any,
        events: Sequence[tuple[str, str]],
    ) -> None:
        if self._recorder.record_native(
            source=source,
            content=content,
            event_ids=tuple(event_id for event_id, _ in events),
        ):
            for _, event_type in events:
                self._observe("native.evidence", event_type)

    def _observe(self, category: str, event_type: str) -> None:
        self._observed[category].add(event_type)

    def _observe_tool_group(self, tool_name: str, event_type: str) -> None:
        group = _tool_group(tool_name)
        if group in {"shell", "file", "search", "browser", "delegation"}:
            self._observe(group, event_type)
        if group == "file_patch":
            self._observe("file", event_type)
            self._observe("patch", event_type)

    def _subagent_parent(self, metadata: Mapping[str, Any] | None) -> str:
        if metadata:
            subagent_id = str(
                metadata.get("subagent_id") or metadata.get("child_session_id") or ""
            )
            pending = self._subagents.get(subagent_id)
            if pending:
                return pending.span_id
        return self._session_parent

    def _record(self, **kwargs: Any) -> str | None:
        try:
            return self._recorder.record_event(**kwargs)
        except Exception:
            self._recorder.report_issue(
                "hermes.adapter_callback_failed",
                "A Hermes callback could not be normalized",
            )
            return None


def _classify_tool(name: str) -> tuple[str, str, str]:
    group = _tool_group(name)
    if group == "shell":
        return ("shell.start", "shell.end", "shell")
    if group == "file_read":
        return ("file.read", "file.read", "file")
    if group == "file_write":
        return ("file.write", "file.write", "file")
    if group == "file_patch":
        return ("file.patch", "file.patch", "file")
    if group == "search":
        return ("search.start", "search.end", "search")
    if group == "browser":
        return ("browser.start", "browser.end", "browser")
    if group == "delegation":
        return ("delegation.start", "delegation.end", "delegation")
    return ("tool.start", "tool.end", "tool")


def _tool_group(name: str) -> str:
    normalized = name.strip().lower()
    return next(
        (group for group, names in _TOOL_GROUPS.items() if normalized in names),
        "tool",
    )


def _native_origin(event_type: str) -> JsonObject:
    return {
        "component": "hermes-agent",
        "capture_method": "native_hook",
        "native_event_type": event_type,
    }


def _derived_origin() -> JsonObject:
    return {
        "component": "hermes-benchmark-trace-adapter",
        "capture_method": "derived",
    }


def _harness_origin() -> JsonObject:
    return {
        "component": "hermes-benchmark-worker",
        "capture_method": "generic_harness",
    }


def _derived_duration(
    started_ns: int,
    ended_ns: int | None = None,
) -> JsonObject:
    return {
        "fidelity": "derived",
        "duration_ms": max(
            0.0,
            ((ended_ns or time.monotonic_ns()) - started_ns) / 1_000_000,
        ),
    }


def _lifecycle_error(
    status: TraceStatus,
    error_message: str | None,
) -> JsonObject | None:
    if status == "completed":
        return None
    return {
        "code": "hermes.attempt_failed",
        "message": error_message or "Hermes benchmark attempt did not complete",
    }


def _normalized_native_status(status: str) -> TraceStatus:
    if status in {"completed", "success"}:
        return "completed"
    if status == "timeout":
        return "timeout"
    if status in {"cancelled", "interrupted"}:
        return "cancelled"
    if status == "degraded":
        return "degraded"
    return "failed"


def _optional_string(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _progress_payload(
    event_type: str,
    tool_name: str,
    preview: Any,
    arguments: Any,
    metadata: Mapping[str, Any] | None,
) -> JsonObject:
    return {
        "event_type": event_type,
        "tool_name": tool_name,
        "preview": preview,
        "arguments": arguments,
        "metadata": dict(metadata or {}),
        "recorded_at": utc_now(),
    }
