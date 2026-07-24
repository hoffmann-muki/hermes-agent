from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from hermes_cli.benchmarks import swebench_pro, swebench_verified
from hermes_cli.benchmarks import swebench_verified_worker as worker
from hermes_cli.benchmarks.tracing import (
    create_hermes_attempt_trace,
    create_hermes_trace_run,
    finalize_hermes_trace_run,
)
from hermes_cli.benchmarks import tracing as tracing_package


INSTANCE = "owner__repo-1"
REVISION = "a" * 40
MODEL = "openrouter/qwen/qwen3-coder-next"


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
    adapter.container_observed(
        {
            "container_id": "a" * 64,
            "image": "example/task:latest",
            "worktree": "/testbed",
        }
    )
    adapter.on_step(
        1,
        [
            {
                "name": "terminal",
                "arguments": {"command": "pwd"},
                "result": "/testbed",
                "usage": {"total_tokens": 99},
            }
        ],
    )
    adapter.on_tool_start(
        "call-1",
        "terminal",
        {"command": "find . -maxdepth 1", "api_key": "must-not-persist"},
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
        {"command": "find . -maxdepth 1", "api_key": "must-not-persist"},
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
    adapter.on_tool_progress(
        "subagent.start",
        preview=child["goal"],
        **child,
    )
    adapter.on_tool_progress(
        "subagent.tool",
        "search_files",
        "find symbol",
        {"query": "Widget", "path": "."},
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
    assert event_types[:3] == ["instance.start", "attempt.start", "harness.start"]
    assert {
        "agent.session_start",
        "container.observed",
        "model.turn_start",
        "model.turn_end",
        "shell.start",
        "shell.end",
        "delegation.start",
        "delegation.end",
        "search.observed",
        "model.stream_delta",
        "context.compaction",
        "model.response",
        "agent.session_end",
        "harness.end",
        "attempt.end",
        "instance.end",
    } <= set(event_types)
    shell_end = next(event for event in events if event["event_type"] == "shell.end")
    assert shell_end["timing"]["duration_ms"] == 125
    assert _artifact(attempt_dir, shell_end) == "./a.py\n./b.py\n"
    assert result.health == "healthy"
    assert result.complete is True

    capabilities = json.loads(
        (attempt_dir / "capabilities.json").read_text(encoding="utf-8")
    )
    states = {
        item["category"]: item["state"] for item in capabilities["capabilities"]
    }
    assert states["tool.result"] == "captured"
    assert states["tool.timing"] == "captured"
    assert states["delegation"] == "captured"
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
    assert '"usage"' not in retained
    assert '"total_tokens"' not in retained
    assert '"cost_usd"' not in retained
    assert (run.root / "run.json").stat().st_mode & 0o777 == 0o600


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
        },
    }
    fake_environment = SimpleNamespace(_container_id="a" * 64)

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
    assert (attempt_dir / "manifest.json").is_file()
    assert "file.read" in {
        event["event_type"] for event in _read_jsonl(attempt_dir / "events.jsonl")
    }


@pytest.mark.parametrize(
    "parser",
    (swebench_verified.build_parser, swebench_pro.build_parser),
)
def test_swe_cli_trace_directory_is_explicit_and_inference_only(parser, tmp_path):
    args = parser().parse_args(["infer", "--trace-dir", str(tmp_path)])
    assert args.trace_dir == tmp_path
    with pytest.raises(SystemExit):
        parser().parse_args(["evaluate", "--run-id", "test", "--trace-dir", str(tmp_path)])


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
    row = swebench_verified.parse_swebench_row(
        {
            "repo": "owner/repo",
            "instance_id": INSTANCE,
            "base_commit": REVISION,
            "problem_statement": "Fix it.",
        }
    )

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

    def finalize(**kwargs):
        observed["finalize"] = kwargs
        return kwargs["run"].root / "run.json"

    monkeypatch.setattr(swebench_verified, "_run_worker", run_worker)
    monkeypatch.setattr(tracing_package, "finalize_hermes_trace_run", finalize)
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
    assert observed["finalize"]["run"] == observed["run"]
    assert observed["finalize"]["instance_ids"] == [INSTANCE]
    assert observed["finalize"]["selection_strategy"] == "explicit_ids"


def test_pro_controller_uses_the_same_trace_lifecycle(tmp_path, monkeypatch):
    row = swebench_pro.parse_swebench_pro_row(
        {
            "repo": "owner/repo",
            "instance_id": INSTANCE,
            "base_commit": REVISION,
            "problem_statement": "Fix it.",
            "requirements": "",
            "interface": "",
            "repo_language": "Python",
            "dockerhub_tag": "owner__repo-1",
        }
    )

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

    def finalize(**kwargs):
        observed["finalize"] = kwargs
        return kwargs["run"].root / "run.json"

    monkeypatch.setattr(swebench_verified, "_run_worker", run_worker)
    monkeypatch.setattr(tracing_package, "finalize_hermes_trace_run", finalize)
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
    assert observed["finalize"]["run"] == observed["run"]
    assert observed["finalize"]["instance_ids"] == [INSTANCE]
    assert observed["finalize"]["selection_strategy"] == "explicit_ids"
