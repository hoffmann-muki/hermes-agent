from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from hermes_cli.benchmarks import terminalbench as benchmark
from hermes_cli.benchmarks import terminalbench_worker as worker


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
    assert worker.COORDINATOR_BUDGET == 24
    assert worker.PHASE_BUDGETS == {
        "navigator": 10,
        "patcher": 18,
        "reviewer": 12,
    }
    assert worker.TEMPERATURE == 0.1
    assert worker.API_MAX_RETRIES == 1


def test_explicit_task_ids_preserve_order_and_cli_aliases():
    options = benchmark.parse_args([
        "--run-id",
        "selected",
        "--task-id",
        "task-a",
        "--task-name",
        "task-b",
    ])

    assert options.task_names == ("task-a", "task-b")


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
    assert command[command.index("--include-task-name") + 1] == "task-a"
    assert command[command.index("--n-tasks") + 1] == "1"
    assert "OPENROUTER_API_KEY" not in rendered


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


class FakeAgent:
    def __init__(self, role: str, budget: int, calls: list[dict[str, object]]):
        self.role = role
        self.budget = budget
        self.calls = calls
        self.session_id = f"session-{role}-{len(calls)}"
        self._active_children = []
        self._active_children_lock = threading.Lock()
        self.released = False

    def run_conversation(self, prompt: str, *, system_message: str, task_id: str):
        self.calls.append({
            "role": self.role,
            "budget": self.budget,
            "prompt": prompt,
            "system": system_message,
            "task_id": task_id,
            "agent": self,
        })
        return {
            "completed": True,
            "interrupted": False,
            "api_calls": self.budget,
            "turn_exit_reason": "text_response(stop)",
            "final_response": f"{self.role} report",
            "messages": [{"role": "assistant", "content": f"{self.role} report"}],
        }

    def release_clients(self):
        self.released = True


def make_phase_state(tmp_path: Path):
    calls: list[dict[str, object]] = []

    def factory(*, role, budget, **_kwargs):
        return FakeAgent(role, budget, calls)

    state = worker.PhaseState(
        instruction="Create the requested artifact.",
        workdir=tmp_path,
        api_key="not-a-real-key",
        model="qwen/qwen3-coder-next",
        task_id="shared-task",
        agent_factory=factory,
    )
    state.coordinator = FakeAgent("coordinator", 24, calls)
    return state, calls


def test_phase_tool_enforces_fresh_blocking_order_and_handoffs(tmp_path: Path):
    state, calls = make_phase_state(tmp_path)

    responses = [
        json.loads(state.handler({"phase": phase}, task_id="shared-task"))
        for phase in worker.PHASE_ORDER
    ]

    assert [call["role"] for call in calls] == list(worker.PHASE_ORDER)
    assert [call["budget"] for call in calls] == [10, 18, 12]
    assert len({id(call["agent"]) for call in calls}) == 3
    assert all(call["task_id"] == "shared-task" for call in calls)
    assert "navigator report" in str(calls[1]["prompt"])
    assert "patcher report" in str(calls[2]["prompt"])
    assert [response["status"] for response in responses] == [
        "completed",
        "completed",
        "completed",
    ]
    assert state.workflow_complete is True
    assert all(call["agent"].released for call in calls)
    assert state.coordinator._active_children == []


def test_phase_tool_rejects_skips_repeats_and_parallel_calls(tmp_path: Path):
    state, calls = make_phase_state(tmp_path)
    skipped = json.loads(state.handler({"phase": "patcher"}))

    assert "error" in skipped
    assert calls == []

    state, calls = make_phase_state(tmp_path)
    state.handler({"phase": "navigator"})
    repeated = json.loads(state.handler({"phase": "navigator"}))

    assert "error" in repeated
    assert [call["role"] for call in calls] == ["navigator"]
    assert state.workflow_complete is False


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
