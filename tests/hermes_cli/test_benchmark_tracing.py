from __future__ import annotations

import base64
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence, cast

import pytest

from hermes_cli.benchmarks import swebench_pro, swebench_verified
from hermes_cli.benchmarks import swebench_verified_worker as worker
from hermes_cli.benchmarks.tracing import (
    HermesTraceRun,
    TraceHarnessAdapter,
    TraceRun,
    TraceSelection,
    create_hermes_attempt_trace,
    create_hermes_trace_run,
    create_trace_run,
    finalize_hermes_trace_run,
    finalize_trace_run,
)
from hermes_cli.benchmarks import tracing as tracing_package
from hermes_cli.benchmarks.tracing.harbor import (
    HarborTraceHarness,
    allocate_harbor_trace_attempt,
    attach_harbor_agentsight_profile,
    finalize_harbor_trace_run,
    promote_harbor_trace_attempt,
    start_harbor_agentsight_profile,
    trace_instance_ids_from_job,
)
from hermes_cli.benchmarks.tracing.execution_tree import (
    build_execution_tree,
    execution_tree_event_ids,
)
from hermes_cli.benchmarks.tracing.runtime import (
    NATIVE_CHUNK_MEDIA_TYPE,
    SCHEMA_DIGEST,
    TraceIdentity,
    _Redactor,
)


INSTANCE = "owner__repo-1"
REVISION = "a" * 40
MODEL = "openrouter/qwen/qwen3-coder-next"


class _CustomHarness:
    prepared = False

    def prepare_finalization(self, run: TraceRun) -> None:
        assert run.benchmark == "custom-benchmark"
        self.prepared = True

    def resolve_selection(
        self,
        run: TraceRun,
        observed_instance_ids: Sequence[str],
    ) -> TraceSelection:
        assert run.framework == "hermes"
        assert tuple(observed_instance_ids) == ("custom-instance",)
        return TraceSelection(
            instance_ids=("custom-instance",),
            strategy="explicit_ids",
        )


def test_redactor_covers_private_keys_without_a_closing_boundary():
    result = _Redactor().sanitize_text(
        "-----BEGIN PRIVATE KEY-----\nsynthetic-truncated-payload"
    )

    assert result.value == "<redacted:private_key>"
    assert result.matches == 1
    assert result.rules == ("credential.private_key",)


def _projection_event(
    sequence,
    event_type,
    phase,
    status,
    span_id,
    timestamp_ms,
    parent_span_id=None,
    payload=None,
):
    timestamp = f"2026-01-01T00:00:00.{timestamp_ms:03d}Z"
    return {
        "event_id": f"event-{sequence:03d}",
        "sequence": sequence,
        "span_id": span_id,
        **({"parent_span_id": parent_span_id} if parent_span_id else {}),
        "occurred_at": timestamp,
        "recorded_at": timestamp,
        "event_type": event_type,
        "event_family": event_type.split(".", 1)[0],
        "phase": phase,
        "status": status,
        "origin": {"component": "test-agent", "capture_method": "derived"},
        "timing": {
            "fidelity": "derived",
            **({"duration_ms": timestamp_ms} if phase == "end" else {}),
        },
        "payload": payload or {},
        "artifacts": [],
    }


def test_generic_coordinator_supports_an_arbitrary_benchmark(tmp_path):
    run = create_trace_run(
        tmp_path / "traces",
        benchmark="custom-benchmark",
        framework="hermes",
    )
    adapter = create_hermes_attempt_trace(
        run=run,
        instance_id="custom-instance",
        attempt=1,
        framework_revision=REVISION,
        model=MODEL,
        agent_timeout_seconds=30,
        evaluation_workers=1,
    )
    adapter.start_session("custom-session")
    adapter._recorder.report_issue(
        "trace.synthetic_warning",
        "Synthetic observability warning",
        severity="warning",
    )
    result = adapter.finish("failed", messages=[])
    execution_tree = json.loads(
        (Path(result.attempt_dir) / "execution-tree.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (Path(result.attempt_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    events = (
        (Path(result.attempt_dir) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    custom_harness = _CustomHarness()
    harness: TraceHarnessAdapter = custom_harness

    path = finalize_trace_run(run, harness)

    assert custom_harness.prepared
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["benchmark"] == "custom-benchmark"
    assert document["framework"] == "hermes"
    assert document["selection"]["instance_ids"] == ["custom-instance"]
    assert document["attempts"][0]["status"] == "failed"
    assert execution_tree["complete"] is True
    assert execution_tree["source"]["event_count"] == len(events)
    assert execution_tree["source"]["represented_event_count"] == len(events)
    assert manifest["files"]["execution_tree"] == "execution-tree.json"


def test_error_level_trace_issue_marks_attempt_failed_and_partial(tmp_path):
    run = create_hermes_trace_run(tmp_path / "traces", "swe-bench-verified")
    adapter = create_hermes_attempt_trace(
        run=run,
        instance_id=INSTANCE,
        attempt=1,
        framework_revision=REVISION,
        model=MODEL,
        agent_timeout_seconds=1800,
        evaluation_workers=1,
    )
    adapter.start_session("task-coordinator")
    adapter.on_tool_complete("missing", "terminal", {}, "no result")

    result = adapter.finish("completed", messages=[])
    health = json.loads(
        (Path(result.attempt_dir) / "health.json").read_text(encoding="utf-8")
    )

    assert result.health == "failed"
    assert result.complete is False
    assert health["status"] == "failed"
    assert health["finalization"] == "partial"


def test_execution_tree_pairs_spans_and_marks_concurrent_siblings():
    events = [
        _projection_event(1, "attempt.start", "start", "started", "attempt", 0),
        _projection_event(
            2,
            "model.turn_start",
            "start",
            "started",
            "model-a",
            100,
            "attempt",
            {"request": "first"},
        ),
        _projection_event(
            3,
            "model.turn_start",
            "start",
            "started",
            "model-b",
            150,
            "attempt",
            {"request": "second"},
        ),
        _projection_event(
            4,
            "model.turn_end",
            "end",
            "completed",
            "model-b",
            250,
            "attempt",
            {"response": "second"},
        ),
        _projection_event(
            5,
            "model.turn_end",
            "end",
            "completed",
            "model-a",
            300,
            "attempt",
            {"response": "first"},
        ),
        _projection_event(
            6,
            "context.compaction",
            "instant",
            "completed",
            "compaction",
            400,
            "attempt",
        ),
        _projection_event(7, "attempt.end", "end", "completed", "attempt", 500),
    ]
    content = b"".join(
        json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events
    )

    tree = build_execution_tree(
        events,
        identity=TraceIdentity(
            trace_id="trace-tree",
            run_id="run-tree",
            benchmark="custom-benchmark",
            framework="hermes",
            instance_id="instance-1",
            attempt=1,
        ),
        schema_digest=SCHEMA_DIGEST,
        events_content=content,
    )
    attempt = tree["root"]["children"][0]
    first, second = attempt["children"][:2]

    assert tree["complete"] is True
    assert first["source_event_ids"] == ["event-002", "event-005"]
    assert first["input"] == {
        "payload": {"request": "first"},
        "artifacts": [],
    }
    assert first["output"] == {
        "payload": {"response": "first"},
        "artifacts": [],
    }
    assert first["concurrency_group"] == second["concurrency_group"]
    assert first["overlaps_with"] == [second["node_id"]]
    assert second["overlaps_with"] == [first["node_id"]]
    assert sorted(execution_tree_event_ids(tree)) == sorted(
        event["event_id"] for event in events
    )


def test_execution_tree_duration_is_stable_across_language_runtimes():
    events = [
        {
            **_projection_event(
                1,
                "attempt.start",
                "start",
                "started",
                "attempt",
                0,
            ),
            "occurred_at": "2026-01-01T00:00:00.000Z",
            "recorded_at": "2026-01-01T00:00:00.000Z",
        },
        {
            **_projection_event(
                2,
                "attempt.end",
                "end",
                "completed",
                "attempt",
                0,
            ),
            "occurred_at": "2026-01-01T00:02:08.824Z",
            "recorded_at": "2026-01-01T00:02:08.824Z",
            "timing": {"fidelity": "derived"},
        },
    ]
    content = b"".join(
        json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events
    )

    tree = build_execution_tree(
        events,
        identity=TraceIdentity(
            trace_id="trace-duration",
            run_id="run-duration",
            benchmark="custom-benchmark",
            framework="hermes",
            instance_id="instance-1",
            attempt=1,
        ),
        schema_digest=SCHEMA_DIGEST,
        events_content=content,
    )
    attempt = tree["root"]["children"][0]

    assert tree["root"]["duration_ms"] == 128_824.0
    assert attempt["duration_ms"] == 128_824.0


def test_execution_tree_represents_an_empty_recovered_journal():
    tree = build_execution_tree(
        [],
        identity=TraceIdentity(
            trace_id="trace-empty-tree",
            run_id="run-empty-tree",
            benchmark="custom-benchmark",
            framework="hermes",
            instance_id="instance-empty",
            attempt=1,
        ),
        schema_digest=SCHEMA_DIGEST,
        events_content=b"",
        generated_at="2026-01-01T00:00:00.000Z",
    )

    assert tree["complete"] is True
    assert tree["source"] == {
        "path": "events.jsonl",
        "sha256": ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        "event_count": 0,
        "represented_event_count": 0,
    }
    assert tree["root"] == {
        "node_id": "trace-root",
        "started_at": None,
        "ended_at": None,
        "duration_ms": 0.0,
        "children": [],
    }


def test_invalid_native_identifier_is_rejected_before_persistence(tmp_path):
    run = create_hermes_trace_run(tmp_path / "traces", "swe-bench-verified")
    adapter = create_hermes_attempt_trace(
        run=run,
        instance_id=INSTANCE,
        attempt=1,
        framework_revision=REVISION,
        model=MODEL,
        agent_timeout_seconds=1800,
        evaluation_workers=1,
    )
    adapter.start_session("x" * 600)

    result = adapter.finish(
        "failed",
        messages=[],
        error_message="Synthetic agent failure",
    )
    events = [
        json.loads(line)
        for line in (Path(result.attempt_dir) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    health = json.loads(
        (Path(result.attempt_dir) / "health.json").read_text(encoding="utf-8")
    )

    assert result.health == "failed"
    assert "event.invalid" in {issue["code"] for issue in health["issues"]}
    assert all(
        not isinstance(event.get("session_id"), str) or len(event["session_id"]) <= 512
        for event in events
    )


def test_harbor_job_lock_preserves_resolved_task_order(tmp_path):
    job = tmp_path / "jobs" / "run"
    job.mkdir(parents=True)
    (job / "lock.json").write_text(
        json.dumps({
            "trials": [
                {"task": {"name": "terminal-bench/task-b"}},
                {"task": {"name": "terminal-bench/task-a"}},
                {"task": {"name": "terminal-bench/task-b"}},
            ]
        }),
        encoding="utf-8",
    )

    assert trace_instance_ids_from_job(tmp_path / "jobs", "run") == [
        "task-b",
        "task-a",
    ]


def test_harbor_harness_resolves_any_benchmark_identity(tmp_path):
    job = tmp_path / "jobs" / "custom"
    job.mkdir(parents=True)
    (job / "lock.json").write_text(
        json.dumps({"trials": [{"task": {"name": "custom/task-a"}}]}),
        encoding="utf-8",
    )
    run = create_trace_run(
        tmp_path / "traces",
        benchmark="custom-harbor-benchmark",
        framework="hermes",
    )
    harness = HarborTraceHarness(
        jobs_dir=tmp_path / "jobs",
        job_name="custom",
        selected_instance_ids=None,
        expected_instance_count=1,
        expected_attempts_per_instance=1,
        selection_strategy="full_dataset",
    )

    selection = harness.resolve_selection(run, ())

    assert selection.instance_ids == ("task-a",)
    assert selection.strategy == "full_dataset"


def test_harbor_bridge_preserves_multiple_native_attempts(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[agent]\ntimeout_sec = 900\n\n"
        '[environment]\ndocker_image = "example/task:latest"\n',
        encoding="utf-8",
    )
    logs_dir = tmp_path / "job" / "trial" / "agent"
    logs_dir.mkdir(parents=True)
    (logs_dir.parent / "config.json").write_text(
        json.dumps({
            "task": {
                "name": "terminal-bench/task-a",
                "ref": "sha256:" + "a" * 64,
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_cli.benchmarks.tracing.harbor._resolved_task_path",
        lambda _task: task_dir,
    )
    run = create_hermes_trace_run(
        tmp_path / "traces",
        "terminal-bench-2.1",
    )

    for expected_attempt in (1, 2):
        allocation = allocate_harbor_trace_attempt(
            logs_dir=logs_dir,
            trace_root=run.root,
        )
        inner_run = HermesTraceRun(
            id=run.id,
            root=logs_dir / allocation.container_root.name,
            created_at=run.created_at,
            benchmark=run.benchmark,
        )
        adapter = create_hermes_attempt_trace(
            run=inner_run,
            instance_id=allocation.instance_id,
            attempt=allocation.attempt,
            framework_revision=REVISION,
            model=MODEL,
            agent_timeout_seconds=allocation.agent_timeout_seconds,
            evaluation_workers=1,
            agent_image=allocation.container_image,
        )
        adapter.container_observed({"image": allocation.container_image})
        adapter.start_session(f"task-a-attempt-{expected_attempt}")
        adapter.finish("completed", messages=[])
        profiler = start_harbor_agentsight_profile(
            logs_dir=logs_dir,
            trace_run_id=run.id,
            benchmark=run.benchmark,
            framework="hermes",
            attempt=allocation,
            docker_session_id=f"task-a__trial-{expected_attempt}",
            tls_python_path="/home/agent/hermes/venv/bin/python",
            env={"BENCHMARK_AGENTSIGHT": "off"},
        )
        profiler.finish()
        assert profiler.target.capture_tls is True
        attach_harbor_agentsight_profile(
            logs_dir=logs_dir,
            attempt=allocation,
            profiler=profiler,
        )
        promoted = promote_harbor_trace_attempt(
            logs_dir=logs_dir,
            trace_root=run.root,
            attempt=allocation,
        )

        assert allocation.attempt == expected_attempt
        assert promoted.name == f"attempt-{expected_attempt}"
        profile = json.loads(
            (promoted / "profiles" / "agentsight" / "profile.json").read_text()
        )
        assert profile["correlation"]["attempt"] == expected_attempt
        assert profile["correlation"]["framework"] == "hermes"

    run_index = finalize_harbor_trace_run(
        trace_root=run.root,
        run_id=run.id,
        benchmark=run.benchmark,
        created_at=run.created_at,
        selected_instance_ids=["task-a"],
        expected_instance_count=1,
        expected_attempts_per_instance=2,
        selection_strategy="explicit_ids",
    )
    attempts = json.loads(run_index.read_text(encoding="utf-8"))["attempts"]

    assert [attempt["attempt"] for attempt in attempts] == [1, 2]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _artifact(attempt_dir: Path, event: dict) -> str:
    reference = event["artifacts"][0]
    return (attempt_dir / reference["path"]).read_text(encoding="utf-8")


def test_hermes_adapter_records_native_tools_delegation_and_compaction(tmp_path):
    run = create_hermes_trace_run(tmp_path / "traces", "swe-bench-verified")
    adapter = create_hermes_attempt_trace(
        run=run,
        instance_id=INSTANCE,
        attempt=1,
        framework_revision=REVISION,
        model=MODEL,
        agent_timeout_seconds=1800,
        evaluation_workers=1,
    )
    adapter.start_session("task-coordinator")
    adapter.container_observed({
        "container_id": "a" * 64,
        "image": "example/task:latest",
        "worktree": "/testbed",
    })
    adapter.on_step(
        1,
        [
            {
                "name": "terminal",
                "arguments": {"command": "pwd"},
                "result": "/testbed",
                "usage": {"total_tokens": 99},
                "metrics": {
                    "tokens": {"input": 10, "output": 5},
                    "duration_ms": 25,
                },
            }
        ],
    )
    adapter.on_tool_start(
        "call-1",
        "terminal",
        {
            "command": (
                "OPENROUTER_API_KEY=synthetic-secret-value "
                "--custom-access-token synthetic-token-value"
            ),
            "OPENROUTER_API_KEY": "must-not-persist",
        },
    )
    adapter.on_tool_progress(
        "tool.completed",
        "terminal",
        duration=0.125,
        is_error=False,
        result="./a.py\n./b.py\n",
    )
    adapter.on_tool_complete(
        "call-1",
        "terminal",
        {
            "command": (
                "OPENROUTER_API_KEY=synthetic-secret-value "
                "--custom-access-token synthetic-token-value"
            ),
            "OPENROUTER_API_KEY": "must-not-persist",
            "provider_usage": {"input_tokens": 10},
        },
        "./a.py\n./b.py\n",
    )
    child = {
        "subagent_id": "child-1",
        "parent_id": "coordinator",
        "child_session_id": "child-session-1",
        "goal": "[benchmark-navigator] inspect the repository",
        "model": "qwen/qwen3-coder-next",
        "toolsets": ["terminal", "file"],
        "depth": 1,
    }
    adapter.on_tool_start(
        "delegate-1",
        "delegate_task",
        {
            "goal": child["goal"],
            "agent": "benchmark-navigator",
        },
    )
    adapter.on_tool_progress(
        "subagent.start",
        preview=child["goal"],
        **child,
    )
    adapter.on_tool_progress(
        "subagent.model_turn",
        child_turn_id="child-turn-1",
        api_call_count=1,
        previous_tools=[],
        **child,
    )
    adapter.on_tool_progress(
        "subagent.model_turn_complete",
        child_turn_id="child-turn-1",
        duration_seconds=0.1,
        boundary="tool_calls",
        status="completed",
        **child,
    )
    adapter.on_tool_progress(
        "subagent.tool",
        "search_files",
        "find symbol",
        {"query": "Widget", "path": "."},
        child_tool_id="child-tool-1",
        **child,
    )
    adapter.on_tool_progress(
        "subagent.tool_complete",
        "search_files",
        child_tool_id="child-tool-1",
        duration_seconds=0.05,
        is_error=False,
        result="src/widget.py:10:class Widget",
        **child,
    )
    adapter.on_tool_progress(
        "subagent.text",
        preview="I found the implementation.",
        **child,
    )
    adapter.on_tool_progress(
        "subagent.complete",
        preview="Navigator report",
        duration_seconds=0.25,
        status="completed",
        tool_count=1,
        cost_usd=1.25,
        **child,
    )
    adapter.on_tool_progress(
        "tool.completed",
        "delegate_task",
        duration=0.25,
        is_error=False,
        result="Navigator report",
    )
    adapter.on_tool_complete(
        "delegate-1",
        "delegate_task",
        {
            "goal": child["goal"],
            "agent": "benchmark-navigator",
        },
        "Navigator report",
    )
    adapter.on_event(
        "session:compress",
        {
            "session_id": "task-coordinator",
            "old_session_id": "old-task-coordinator",
            "in_place": False,
            "compression_count": 1,
        },
    )
    result = adapter.finish(
        "completed",
        messages=[
            {
                "role": "assistant",
                "content": "Finished.",
                "usage": {"total_tokens": 123},
            }
        ],
    )
    finalize_hermes_trace_run(
        run=run,
        instance_ids=[INSTANCE],
        selection_strategy="explicit_ids",
    )

    attempt_dir = Path(result.attempt_dir)
    events = _read_jsonl(attempt_dir / "events.jsonl")
    event_types = [event["event_type"] for event in events]
    assert event_types[:3] == [
        "instance.start",
        "attempt.start",
        "harness.startup_start",
    ]
    assert events[1]["payload"]["agent_configuration"] == {
        "delegation_enabled": True,
        "coordination_mode": "framework_native",
        "delegation_sequence": ["navigator", "patcher", "reviewer"],
        "sequence_enforcement": "prompt_guided",
    }
    assert {
        "agent.session_start",
        "container.observed",
        "model.turn_start",
        "model.turn_end",
        "shell.start",
        "shell.end",
        "delegation.start",
        "delegation.end",
        "search.start",
        "search.end",
        "model.stream_delta",
        "context.compaction",
        "model.response",
        "agent.session_end",
        "agent.execution_end",
        "harness.shutdown_start",
        "harness.shutdown_end",
        "attempt.end",
        "instance.end",
    } <= set(event_types)
    assert event_types.count("delegation.start") == 1
    assert event_types.count("delegation.end") == 1
    child_session_start = next(
        event
        for event in events
        if event["event_type"] == "agent.session_start"
        and event.get("agent_id") == "child-1"
    )
    assert (
        child_session_start["parent_span_id"]
        == next(event for event in events if event["event_type"] == "delegation.start")[
            "span_id"
        ]
    )
    shell_end = next(event for event in events if event["event_type"] == "shell.end")
    assert shell_end["timing"]["duration_ms"] == 125
    assert _artifact(attempt_dir, shell_end) == "./a.py\n./b.py\n"
    child_model_end = next(
        event
        for event in events
        if event["event_type"] == "model.turn_end"
        and event.get("agent_id") == "child-1"
    )
    assert child_model_end["timing"]["duration_ms"] == 100
    assert child_model_end["payload"]["boundary"] == "tool_calls"
    execution_end = next(
        event for event in events if event["event_type"] == "agent.execution_end"
    )
    transcript_response = next(
        event for event in events if event["event_type"] == "model.response"
    )
    session_end = next(
        event
        for event in events
        if event["event_type"] == "agent.session_end"
        and event.get("agent_id") == "coordinator"
    )
    assert transcript_response["occurred_at"] == execution_end["occurred_at"]
    assert session_end["occurred_at"] == execution_end["occurred_at"]
    assert result.health == "healthy"
    assert result.complete is True

    capabilities = json.loads(
        (attempt_dir / "capabilities.json").read_text(encoding="utf-8")
    )
    states = {item["category"]: item["state"] for item in capabilities["capabilities"]}
    coverage = {
        item["category"]: item["coverage"] for item in capabilities["capabilities"]
    }
    assert states["tool.result"] == "captured"
    assert states["tool.timing"] == "captured"
    assert states["delegation"] == "captured"
    assert coverage["delegation"] == "full"
    assert states["context.compaction"] == "captured"
    assert states["provider.exchange"] == "not_exposed"
    assert states["memory"] == "disabled"
    assert states["browser"] == "disabled"

    retained = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in attempt_dir.rglob("*")
        if path.is_file()
    )
    assert "must-not-persist" not in retained
    assert "synthetic-secret-value" not in retained
    assert "synthetic-token-value" not in retained
    assert '"provider_usage"' not in retained
    assert '"usage"' not in retained
    assert '"total_tokens"' not in retained
    assert '"tokens"' not in retained
    assert '"duration_ms":25' in retained
    assert '"cost_usd"' not in retained
    native = _read_jsonl(attempt_dir / "native" / "index.jsonl")
    native_paths = {record["artifact"]["path"] for record in native}
    assert len(native_paths) == 1
    assert native[0]["artifact"]["media_type"] == NATIVE_CHUNK_MEDIA_TYPE
    members = [
        json.loads(line)
        for line in gzip.decompress(
            (attempt_dir / next(iter(native_paths))).read_bytes()
        ).splitlines()
    ]
    assert len(members) == len(native)
    packed = b"\n".join(
        base64.b64decode(member["content_base64"], validate=True) for member in members
    ).decode("utf-8")
    assert "must-not-persist" not in packed
    assert "synthetic-secret-value" not in packed
    assert '"usage"' not in packed
    if os.name != "nt":
        assert (attempt_dir / "events.jsonl").stat().st_ino == (
            attempt_dir / "journal.jsonl"
        ).stat().st_ino
    assert (run.root / "run.json").stat().st_mode & 0o777 == 0o600


def test_single_agent_trace_uses_native_role_and_disables_delegation(tmp_path):
    run = create_hermes_trace_run(tmp_path / "traces", "swe-bench-lite")
    adapter = create_hermes_attempt_trace(
        run=run,
        instance_id=INSTANCE,
        attempt=1,
        framework_revision=REVISION,
        model="openrouter/poolside/laguna-s-2.1:free",
        agent_timeout_seconds=900,
        evaluation_workers=1,
        delegation_enabled=False,
    )
    adapter.start_session("task-agent")
    adapter.on_step(1, [])
    adapter.on_tool_start("call-1", "terminal", {"command": "pwd"})
    adapter.on_tool_progress(
        "tool.completed",
        "terminal",
        duration=0.125,
        is_error=False,
        result="/testbed\n",
    )
    adapter.on_tool_complete(
        "call-1",
        "terminal",
        {"command": "pwd"},
        "/testbed\n",
    )

    result = adapter.finish(
        "completed",
        messages=[{"role": "assistant", "content": "Finished."}],
    )
    attempt_dir = Path(result.attempt_dir)
    events = _read_jsonl(attempt_dir / "events.jsonl")
    attempt_start = next(
        event for event in events if event["event_type"] == "attempt.start"
    )
    root_events = [
        event
        for event in events
        if event["event_type"]
        in {
            "agent.session_start",
            "agent.session_end",
            "model.turn_start",
            "model.turn_end",
            "model.response",
            "shell.start",
            "shell.end",
        }
    ]
    capabilities = json.loads(
        (attempt_dir / "capabilities.json").read_text(encoding="utf-8")
    )["capabilities"]
    by_category = {item["category"]: item for item in capabilities}

    assert result.health == "healthy"
    assert attempt_start["payload"]["agent_configuration"] == {
        "delegation_enabled": False,
        "coordination_mode": "framework_native",
        "delegation_sequence": [],
        "sequence_enforcement": "prompt_guided",
    }
    assert root_events
    assert {event.get("agent_id") for event in root_events} == {"agent"}
    assert (
        next(event for event in events if event["event_type"] == "agent.session_start")[
            "payload"
        ]["role"]
        == "agent"
    )
    assert by_category["delegation"] == {
        "category": "delegation",
        "state": "disabled",
        "coverage": "none",
        "timing": "not_applicable",
        "evidence": [],
        "limitations": [],
    }
    assert by_category["tool.invocation"]["coverage"] == "full"
    assert by_category["shell"]["coverage"] == "full"
    assert by_category["tool.invocation"]["limitations"] == []
    assert by_category["shell"]["limitations"] == []


def test_worker_wires_trace_callbacks_without_a_model_call(tmp_path, monkeypatch):
    run = create_hermes_trace_run(tmp_path / "traces", "swe-bench-verified")
    runtime_path = tmp_path / "runtime.json"
    source_identity = {
        "commit": REVISION,
        "dirty": False,
        "fingerprint": "test",
    }
    request = {
        "benchmark": swebench_verified.BENCHMARK,
        "row": {
            "repo": "owner/repo",
            "instance_id": INSTANCE,
            "base_commit": REVISION,
            "problem_statement": "Fix it.",
        },
        "model": MODEL,
        "taskId": "trace-test",
        "prompt": "Fix it.",
        "runtimePath": str(runtime_path),
        "hermesHome": str(tmp_path / "home"),
        "image": "example/task:latest",
        "dockerPlatform": "linux/amd64",
        "agentTimeoutSeconds": 1800,
        "attempt": 1,
        "sourceIdentity": source_identity,
        "trace": {
            "runId": run.id,
            "runRoot": str(run.root),
            "createdAt": run.created_at,
            "benchmark": run.benchmark,
            "frameworkRevision": REVISION,
            "evaluationTimeoutSeconds": 3600,
        },
    }
    fake_environment = SimpleNamespace(_container_id="a" * 64)
    profile_targets = []

    class OfflineProfiler:
        strict = False

        def finish(self):
            return None

    class OfflineAgent:
        def __init__(self, callbacks):
            self.callbacks = callbacks

        def run_conversation(self, *_args, **_kwargs):
            self.callbacks["step_callback"](1, [])
            self.callbacks["tool_start_callback"](
                "call-1", "read_file", {"path": "a.py"}
            )
            self.callbacks["tool_progress_callback"](
                "tool.completed",
                "read_file",
                duration=0.01,
                is_error=False,
                result="print('ok')\n",
            )
            self.callbacks["tool_complete_callback"](
                "call-1",
                "read_file",
                {"path": "a.py"},
                "print('ok')\n",
            )
            return {
                "completed": True,
                "interrupted": False,
                "messages": [{"role": "assistant", "content": "done"}],
            }

        def release_clients(self):
            return None

        def interrupt(self, _reason):
            return None

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setattr(worker, "hermes_source_identity", lambda: source_identity)
    monkeypatch.setattr(worker, "configure_worker", lambda _request: None)
    monkeypatch.setattr(worker, "setup_environment", lambda _request: fake_environment)
    monkeypatch.setattr(
        worker.AgentSightProfiler,
        "start",
        lambda target: profile_targets.append(target) or OfflineProfiler(),
    )
    monkeypatch.setattr(
        "tools.terminal_tool.clear_task_env_overrides", lambda _task_id: None
    )

    def agent_factory(**kwargs):
        assert set(kwargs["callbacks"]) == {
            "tool_progress_callback",
            "tool_start_callback",
            "tool_complete_callback",
            "step_callback",
            "event_callback",
        }
        return OfflineAgent(kwargs["callbacks"])

    result = worker.run_worker(request, agent_factory=agent_factory)

    assert result["trace"]["traceHealth"] == "healthy"
    assert result["trace"]["traceComplete"] is True
    attempt_dir = Path(result["trace"]["traceDirectory"])
    manifest = json.loads((attempt_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution"]["inference_timeout_seconds"] == 1800
    assert manifest["execution"]["evaluation_timeout_seconds"] == 3600
    assert len(profile_targets) == 1
    assert profile_targets[0].host_pid == os.getpid()
    assert profile_targets[0].capture_host_tls is True
    assert profile_targets[0].capture_tls is False
    assert (attempt_dir / "manifest.json").is_file()
    assert "file.read" in {
        event["event_type"] for event in _read_jsonl(attempt_dir / "events.jsonl")
    }


@pytest.mark.parametrize(
    "parser",
    (swebench_verified.build_parser, swebench_pro.build_parser),
)
def test_swe_cli_trace_directory_defaults_to_repo_and_is_inference_only(
    parser, tmp_path
):
    defaults = parser().parse_args(["infer"])
    assert defaults.trace_dir == swebench_verified.DEFAULT_TRACE_DIR
    args = parser().parse_args(["infer", "--trace-dir", str(tmp_path)])
    assert args.trace_dir == tmp_path
    assert parser().parse_args(["infer", "--no-trace"]).trace_dir is None
    with pytest.raises(SystemExit):
        parser().parse_args(["infer", "--trace-dir", str(tmp_path), "--no-trace"])
    with pytest.raises(SystemExit):
        parser().parse_args([
            "evaluate",
            "--run-id",
            "test",
            "--trace-dir",
            str(tmp_path),
        ])


def test_trace_revision_must_be_exact(tmp_path):
    run = create_hermes_trace_run(tmp_path / "traces", "swe-bench-pro")
    with pytest.raises(ValueError, match="exact 40-character"):
        create_hermes_attempt_trace(
            run=run,
            instance_id=INSTANCE,
            attempt=1,
            framework_revision="dirty",
            model=MODEL,
            agent_timeout_seconds=1800,
            evaluation_workers=1,
        )


def test_verified_controller_passes_one_private_trace_run(tmp_path, monkeypatch):
    row = swebench_verified.parse_swebench_row({
        "repo": "owner/repo",
        "instance_id": INSTANCE,
        "base_commit": REVISION,
        "problem_statement": "Fix it.",
    })

    class Dataset:
        def select(self, *_args, **_kwargs):
            return [row]

    source_identity = {
        "commit": REVISION,
        "dirty": False,
        "fingerprint": "test",
    }
    observed = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setattr(
        swebench_verified, "require_docker", lambda: {"serverVersion": "test"}
    )
    monkeypatch.setattr(
        swebench_verified,
        "ensure_image",
        lambda image, *_args: {"image": image},
    )
    monkeypatch.setattr(
        swebench_verified, "require_unchanged_source", lambda _options: None
    )

    def run_worker(*_args, **kwargs):
        observed["run"] = kwargs["trace_run"]
        return {
            "modelPatch": "diff --git a/a.py b/a.py\n",
            "changedPaths": ["a.py"],
            "workflowComplete": True,
            "timedOut": False,
        }

    def finalize(run, harness):
        observed["finalize"] = (run, harness)
        return run.root / "run.json"

    monkeypatch.setattr(swebench_verified, "_run_worker", run_worker)
    monkeypatch.setattr(tracing_package, "finalize_trace_run", finalize)
    options = swebench_verified.InferenceOptions(
        run_id="trace-verified",
        output_dir=tmp_path / "runs",
        instance_ids=(INSTANCE,),
        max_instances=1,
        offset=0,
        include_hints=False,
        model=MODEL,
        image_template=swebench_verified.DEFAULT_IMAGE_TEMPLATE,
        docker_platform="linux/amd64",
        agent_timeout_seconds=1800,
        setup_timeout_seconds=600,
        restart=False,
        dry_run=False,
        trace_dir=tmp_path / "traces",
        source_identity=source_identity,
    )

    swebench_verified.run_inference(
        options,
        dataset_client=cast(swebench_verified.DatasetRowsClient, Dataset()),
    )

    assert observed["run"].root.parent == (tmp_path / "traces").resolve()
    assert observed["finalize"][0] == observed["run"]
    assert observed["finalize"][1].selection.instance_ids == (INSTANCE,)
    assert observed["finalize"][1].selection.strategy == "explicit_ids"


def test_pro_controller_uses_the_same_trace_lifecycle(tmp_path, monkeypatch):
    row = swebench_pro.parse_swebench_pro_row({
        "repo": "owner/repo",
        "instance_id": INSTANCE,
        "base_commit": REVISION,
        "problem_statement": "Fix it.",
        "requirements": "",
        "interface": "",
        "repo_language": "Python",
        "dockerhub_tag": "owner__repo-1",
    })

    class Dataset:
        def select(self, *_args, **_kwargs):
            return [row]

    source_identity = {
        "commit": REVISION,
        "dirty": False,
        "fingerprint": "test",
    }
    observed = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setattr(
        swebench_verified, "require_docker", lambda: {"serverVersion": "test"}
    )
    monkeypatch.setattr(
        swebench_verified,
        "ensure_image",
        lambda image, *_args: {"image": image},
    )
    monkeypatch.setattr(
        swebench_verified, "require_unchanged_source", lambda _options: None
    )

    def run_worker(*_args, **kwargs):
        observed["run"] = kwargs["trace_run"]
        return {
            "modelPatch": "diff --git a/a.py b/a.py\n",
            "changedPaths": ["a.py"],
            "workflowComplete": True,
            "timedOut": False,
        }

    def finalize(run, harness):
        observed["finalize"] = (run, harness)
        return run.root / "run.json"

    monkeypatch.setattr(swebench_verified, "_run_worker", run_worker)
    monkeypatch.setattr(tracing_package, "finalize_trace_run", finalize)
    options = swebench_pro.InferenceOptions(
        run_id="trace-pro",
        output_dir=tmp_path / "runs",
        instance_ids=(INSTANCE,),
        max_instances=1,
        offset=0,
        model=MODEL,
        image_prefix=swebench_pro.DEFAULT_IMAGE_PREFIX,
        docker_platform="linux/amd64",
        agent_timeout_seconds=1800,
        setup_timeout_seconds=600,
        restart=False,
        dry_run=False,
        trace_dir=tmp_path / "traces",
        source_identity=source_identity,
    )

    swebench_pro.run_inference(
        options,
        client=cast(swebench_pro.DatasetRowsClient, Dataset()),
    )

    assert observed["run"].root.parent == (tmp_path / "traces").resolve()
    assert observed["finalize"][0] == observed["run"]
    assert observed["finalize"][1].selection.instance_ids == (INSTANCE,)
    assert observed["finalize"][1].selection.strategy == "explicit_ids"
