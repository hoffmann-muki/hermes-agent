from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.benchmarks import terminalbench as benchmark
from hermes_cli.benchmarks import terminalbench_worker as worker
from hermes_cli.benchmarks.tracing import HermesTraceRun


def test_safe_defaults_match_peer_terminal_bench_runners():
    options = benchmark.parse_args(["--run-id", "smoke"])

    assert options.task_names == ()
    assert options.max_tasks == 1
    assert options.attempts == 1
    assert options.concurrency == 1
    assert options.max_retries == 0
    assert options.model == "openrouter/qwen/qwen3-coder-next"
    assert options.hermes_version == "play"
    assert options.hermes_repository.startswith("https://")
    assert len(options.hermes_commit) == 40
    assert options.environment == "docker"
    assert options.upload is False
    assert options.public is False
    assert options.leaderboard is False
    assert options.trace_dir == benchmark.DEFAULT_TRACE_DIR
    assert worker.COORDINATOR_BUDGET == 24
    assert worker.PEER_PHASE_BUDGETS == {
        "navigator": 10,
        "patcher": 18,
        "reviewer": 12,
    }
    assert worker.NATIVE_SUBAGENT_BUDGET == 13
    assert worker.NATIVE_SUBAGENT_COUNT == 3
    assert benchmark.DEFAULT_DELEGATION_MODE == "native"
    assert worker.DELEGATION_MODE == benchmark.DEFAULT_DELEGATION_MODE
    assert worker.COORDINATOR_BUDGET == benchmark.DEFAULT_COORDINATOR_BUDGET
    assert worker.PEER_PHASE_BUDGETS == benchmark.DEFAULT_PHASE_BUDGETS
    assert worker.NATIVE_SUBAGENT_BUDGET == benchmark.DEFAULT_NATIVE_SUBAGENT_BUDGET
    assert worker.NATIVE_SUBAGENT_COUNT == benchmark.DEFAULT_NATIVE_SUBAGENT_COUNT
    assert worker.TEMPERATURE == 0.1
    assert worker.API_MAX_RETRIES == 1


def test_explicit_task_ids_preserve_order_and_cli_aliases():
    options = benchmark.parse_args([
        "--run-id",
        "selected",
        "--task-id",
        "terminal-bench/task-a",
        "--task-name",
        "task-b",
    ])

    assert options.task_names == ("task-a", "task-b")
    with pytest.raises(SystemExit):
        benchmark.parse_args([
            "--task-id",
            "task-a",
            "--task-id",
            "terminal-bench/task-a",
        ])
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--task-id", "another-package/task-a"])


def test_full_and_leaderboard_modes_have_peer_semantics():
    full = benchmark.parse_args(["--run-id", "full", "--all-tasks"])
    leaderboard = benchmark.parse_args([
        "--run-id",
        "leaderboard",
        "--leaderboard",
        "--concurrency",
        "4",
    ])

    assert full.max_tasks is None
    assert full.attempts == 1
    assert full.upload is False
    assert leaderboard.max_tasks is None
    assert leaderboard.attempts == 5
    assert leaderboard.concurrency == 4
    assert leaderboard.upload is True
    assert leaderboard.public is True

    with pytest.raises(SystemExit):
        benchmark.parse_args(["--all-tasks", "--max-tasks", "2"])
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--leaderboard", "--task-id", "task-a"])
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--leaderboard", "--attempts", "4"])


def test_harbor_command_is_reproducible_and_credential_free(tmp_path: Path):
    options = benchmark.parse_args([
        "--run-id",
        "selected",
        "--task-id",
        "task-a",
        "--attempts",
        "2",
        "--concurrency",
        "3",
    ])
    command = benchmark.build_harbor_command(options, tmp_path / "jobs")
    rendered = json.dumps(command)

    assert command[:8] == [
        "harbor",
        "run",
        "--dataset",
        "terminal-bench/terminal-bench-2-1",
        "--agent",
        benchmark.AGENT_IMPORT_PATH,
        "--model",
        "openrouter/qwen/qwen3-coder-next",
    ]
    assert command[command.index("--agent-kwarg") + 1] == "version=play"
    agent_kwargs = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--agent-kwarg"
    ]
    assert agent_kwargs == [
        "version=play",
        f"repository={options.hermes_repository}",
        f"commit={options.hermes_commit}",
    ]
    assert command[command.index("--env") + 1] == "docker"
    assert command[command.index("--n-attempts") + 1] == "2"
    assert command[command.index("--n-concurrent") + 1] == "3"
    assert command[command.index("--max-retries") + 1] == "0"
    assert command[command.index("--include-task-name") + 1] == "terminal-bench/task-a"
    assert command[command.index("--n-tasks") + 1] == "1"
    assert "OPENROUTER_API_KEY" not in rendered


def test_trace_command_passes_nonsecret_trial_metadata(tmp_path: Path):
    options = benchmark.parse_args([
        "--run-id",
        "traced",
        "--trace-dir",
        str(tmp_path / "traces"),
    ])
    trace_run = HermesTraceRun(
        id="trace-run-test",
        root=tmp_path / "traces" / "trace-run-test",
        created_at="2026-07-20T12:34:56+00:00",
        benchmark="terminal-bench-2.1",
    )

    command = benchmark.build_harbor_command(
        options,
        tmp_path / "jobs",
        trace_run=trace_run,
        harbor_version="0.20.0",
    )
    agent_kwargs = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--agent-kwarg"
    ]

    assert options.trace_dir == tmp_path / "traces"
    assert f"trace_root={trace_run.root}" in agent_kwargs
    assert "trace_run_id=trace-run-test" in agent_kwargs
    assert "trace_benchmark=terminal-bench-2.1" in agent_kwargs
    assert "evaluation_workers=1" in agent_kwargs
    assert "benchmark_retries=0" in agent_kwargs
    assert "harbor_version=0.20.0" in agent_kwargs
    assert "API_KEY" not in " ".join(agent_kwargs)


def test_tracing_can_be_disabled_explicitly():
    assert (
        benchmark.parse_args(["--run-id", "untraced", "--no-trace"]).trace_dir is None
    )


def test_git_remote_is_normalized_for_credential_free_container_install(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_git_output",
        lambda *args: (
            "git@github.com:example/hermes-agent.git"
            if args == ("config", "--get", "remote.origin.url")
            else ""
        ),
    )

    assert benchmark.default_hermes_repository() == (
        "https://github.com/example/hermes-agent.git"
    )
    assert benchmark.valid_hermes_repository(
        "https://github.com/example/hermes-agent.git"
    )
    assert not benchmark.valid_hermes_repository(
        "https://credential@github.com/example/hermes-agent.git"
    )


def test_benchmark_environment_exposes_only_adapter_path_changes(monkeypatch):
    monkeypatch.setattr(benchmark, "REPO_ROOT", Path("/repo/hermes"))
    env = benchmark.benchmark_process_env({
        "PYTHONPATH": "/existing",
        "OPENROUTER_API_KEY": "secret",
    })

    assert env["PYTHONPATH"] == f"/repo/hermes{benchmark.os.pathsep}/existing"
    assert env["HARBOR_TELEMETRY"] == "off"
    assert env["OPENROUTER_API_KEY"] == "secret"


def native_delegation_messages(phases=("navigator", "patcher", "reviewer")):
    messages = []
    for index, phase in enumerate(phases):
        call_id = f"delegate-{index}"
        messages.extend([
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "function": {
                            "name": "delegate_task",
                            "arguments": json.dumps({
                                "goal": f"[benchmark-{phase}] perform {phase}",
                                "context": "Full task and prior handoffs.",
                                "role": "leaf",
                            }),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps({
                    "results": [
                        {
                            "status": "completed",
                            "summary": f"{phase} report",
                            "api_calls": index + 1,
                            "duration_seconds": 0.5,
                        }
                    ]
                }),
            },
        ])
    return messages


def test_native_delegation_audit_records_sequential_leaf_handoffs():
    records, errors = worker.audit_native_delegations(native_delegation_messages())

    assert errors == []
    assert [record["phase"] for record in records] == list(worker.PHASE_ORDER)
    assert [record["budget"] for record in records] == [13, 13, 13]
    assert [record["report"] for record in records] == [
        "navigator report",
        "patcher report",
        "reviewer report",
    ]
    assert all(record["nativeDelegation"] for record in records)
    assert all(record["freshAgent"] for record in records)


def test_native_delegation_audit_reports_wrong_order():
    _records, errors = worker.audit_native_delegations(
        native_delegation_messages(("patcher", "navigator", "reviewer"))
    )

    assert any("Expected native delegation order" in error for error in errors)


def test_worker_config_is_local_isolated_and_retry_free(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    captured = {}
    monkeypatch.setattr(
        "hermes_cli.config.apply_terminal_config_to_env",
        lambda *, config, override: captured.update(config=config, override=override),
    )

    worker._configure_runtime(tmp_path)
    config = captured["config"]

    assert config["terminal"] == {
        "backend": "local",
        "cwd": str(tmp_path),
        "timeout": 180,
    }
    assert config["agent"] == {"api_max_retries": 1, "coding_context": "off"}
    assert config["delegation"] == {
        "max_iterations": 13,
        "max_concurrent_children": 1,
        "max_spawn_depth": 1,
        "orchestrator_enabled": False,
    }
    assert config["memory"]["memory_enabled"] is False
    assert config["checkpoints"]["enabled"] is False
    assert config["plugins"]["enabled"] == []
    assert captured["override"] is True
    assert worker.os.environ["HERMES_CONCURRENT_TOOL_TIMEOUT_S"] == "86400"


def test_worker_session_artifact_redacts_provider_credentials(
    tmp_path: Path, monkeypatch
):
    session_path = tmp_path / "hermes-session.jsonl"
    monkeypatch.setattr(worker, "SESSION_PATH", session_path)

    worker._write_session(
        [{"messages": [{"role": "tool", "content": "key=private-value"}]}],
        "private-value",
    )

    content = session_path.read_text(encoding="utf-8")
    assert "private-value" not in content
    assert "[REDACTED]" in content
    assert session_path.stat().st_mode & 0o777 == 0o600


def test_trace_completion_is_independent_of_delegation_audit(
    tmp_path: Path,
    monkeypatch,
):
    statuses = []

    class Trace:
        def callbacks(self):
            return {}

        def container_observed(self, _metadata):
            return None

        def start_session(self, _session_id):
            return None

        def end_execution(self, _status, **_kwargs):
            return None

        def finish(self, status, **_kwargs):
            statuses.append(status)
            return SimpleNamespace(
                trace_id="trace-test",
                attempt_dir=str(tmp_path / "attempt"),
                health="healthy",
                complete=True,
            )

    class Agent:
        def run_conversation(self, *_args, **_kwargs):
            return {
                "completed": True,
                "interrupted": False,
                "messages": [],
            }

        def release_clients(self):
            return None

    monkeypatch.setattr(worker, "create_trace_adapter", Trace)
    monkeypatch.setattr(worker, "_configure_runtime", lambda _workdir: None)
    monkeypatch.setattr(worker, "SESSION_PATH", tmp_path / "session.jsonl")
    monkeypatch.setenv(
        "HERMES_BENCHMARK_TRACE_CONFIG",
        json.dumps({
            "sessionId": "session-test",
            "containerImage": "example/task:latest",
        }),
    )

    result = worker.run_worker(
        "Complete the task.",
        api_key="test-only-key",
        model="qwen/qwen3-coder-next",
        workdir=tmp_path,
        agent_factory=lambda **_kwargs: Agent(),
    )

    assert result["workflowComplete"] is False
    assert statuses == ["completed"]
