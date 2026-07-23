from __future__ import annotations

import json
import os
import re
import signal
import shlex
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from hermes_cli.benchmarks import swebench_verified as benchmark
from hermes_cli.benchmarks import swebench_verified_worker as worker


INSTANCE = "scikit-learn__scikit-learn-13439"
BASE_COMMIT = "a" * 40


def make_row(**overrides):
    value = {
        "repo": "scikit-learn/scikit-learn",
        "instance_id": INSTANCE,
        "base_commit": BASE_COMMIT,
        "problem_statement": "Fix the estimator regression.",
        "hints_text": "Inspect validation.",
        "difficulty": "medium",
        "patch": "SECRET GOLD PATCH",
        "test_patch": "SECRET HIDDEN TEST",
        "FAIL_TO_PASS": "hidden test name",
        "PASS_TO_PASS": "hidden test name",
    }
    value.update(overrides)
    return value


def make_options(tmp_path: Path, **overrides):
    value = {
        "run_id": "test-run",
        "output_dir": tmp_path,
        "instance_ids": (INSTANCE,),
        "max_instances": 1,
        "offset": 0,
        "include_hints": False,
        "model": benchmark.DEFAULT_MODEL,
        "image_template": benchmark.DEFAULT_IMAGE_TEMPLATE,
        "docker_platform": benchmark.DEFAULT_DOCKER_PLATFORM,
        "agent_timeout_seconds": benchmark.DEFAULT_AGENT_TIMEOUT_SECONDS,
        "setup_timeout_seconds": benchmark.DEFAULT_SETUP_TIMEOUT_SECONDS,
        "restart": False,
        "dry_run": False,
        "source_identity": {
            "commit": "test",
            "dirty": False,
            "fingerprint": "test",
        },
    }
    value.update(overrides)
    return benchmark.InferenceOptions(**value)


def test_cli_defaults_match_other_frameworks(tmp_path):
    args = benchmark.build_parser().parse_args(["infer", "--output-dir", str(tmp_path)])
    options = benchmark.options_from_args(args)

    assert options.instance_ids == (benchmark.DEFAULT_SMOKE_INSTANCE_ID,)
    assert options.model == "openrouter/qwen/qwen3-coder-next"
    assert options.agent_timeout_seconds == 1800
    assert options.setup_timeout_seconds == 600
    assert benchmark.DEFAULT_INFERENCE_WORKERS == 1
    assert benchmark.DEFAULT_EVALUATION_WORKERS == 1
    assert benchmark.DEFAULT_ATTEMPTS == 1
    assert benchmark.DEFAULT_INFRASTRUCTURE_RETRIES == 0
    assert benchmark.DEFAULT_API_MAX_RETRIES == 1
    assert benchmark.DEFAULT_CODING_CONTEXT == "off"
    assert worker.COORDINATOR_BUDGET == 24
    assert worker.PEER_PHASE_BUDGETS == {
        "navigator": 10,
        "patcher": 18,
        "reviewer": 12,
    }
    assert worker.NATIVE_SUBAGENT_BUDGET == 13
    assert worker.NATIVE_SUBAGENT_COUNT == 3
    assert benchmark.DEFAULT_DELEGATION_MODE == "native"
    assert worker.TEMPERATURE == 0.1


def test_explicit_instance_ids_preserve_order_and_disable_smoke_default(tmp_path):
    ids = ["django__django-10000", INSTANCE]
    args = benchmark.build_parser().parse_args([
        "infer",
        "--output-dir",
        str(tmp_path),
        "--instance-id",
        ids[0],
        "--instance-id",
        ids[1],
    ])

    assert benchmark.options_from_args(args).instance_ids == tuple(ids)

    window_args = benchmark.build_parser().parse_args([
        "infer",
        "--output-dir",
        str(tmp_path),
        "--max-instances",
        "2",
    ])
    assert benchmark.options_from_args(window_args).instance_ids == ()

    one_args = benchmark.build_parser().parse_args(["infer", "--max-instances", "1"])
    zero_args = benchmark.build_parser().parse_args(["infer", "--offset", "0"])
    assert benchmark.options_from_args(one_args).instance_ids == ()
    assert benchmark.options_from_args(zero_args).instance_ids == ()


def test_dataset_projection_never_retains_gold_or_hidden_fields():
    row = benchmark.parse_swebench_row(make_row())

    assert set(row.public_dict()) == benchmark.SAFE_DATASET_FIELDS
    serialized = json.dumps(row.public_dict())
    assert "SECRET GOLD PATCH" not in serialized
    assert "SECRET HIDDEN TEST" not in serialized
    assert "FAIL_TO_PASS" not in serialized
    assert "PASS_TO_PASS" not in serialized


def test_dataset_client_returns_explicit_ids_in_requested_order():
    first = make_row(instance_id="django__django-10000", repo="django/django")
    second = make_row()

    def respond(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        rows = [first, second] if offset == 0 else []
        return httpx.Response(
            200,
            json={"rows": [{"row": row} for row in rows]},
        )

    client = benchmark.DatasetRowsClient(
        client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    rows = client.select([INSTANCE, "django__django-10000"], offset=0, max_instances=1)

    assert [row.instance_id for row in rows] == [INSTANCE, "django__django-10000"]
    with pytest.raises(benchmark.BenchmarkError, match="Duplicate"):
        client.select([INSTANCE, INSTANCE], offset=0, max_instances=1)


def test_dataset_window_selection_paginates_without_truncation():
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        length = int(request.url.params["length"])
        requests.append((offset, length))
        rows = [
            make_row(
                instance_id=f"repo__issue-{index}",
                repo="repo/repo",
            )
            for index in range(offset, offset + length)
        ]
        return httpx.Response(200, json={"rows": [{"row": row} for row in rows]})

    client = benchmark.DatasetRowsClient(
        client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    rows = client.select([], offset=25, max_instances=225)

    assert len(rows) == 225
    assert rows[0].instance_id == "repo__issue-25"
    assert rows[-1].instance_id == "repo__issue-249"
    assert requests == [(25, 100), (125, 100), (225, 25)]


def test_official_image_derivation_and_validation():
    assert benchmark.official_image(INSTANCE) == (
        "docker.io/swebench/sweb.eval.x86_64."
        "scikit-learn_1776_scikit-learn-13439:latest"
    )
    with pytest.raises(benchmark.BenchmarkError, match="Invalid"):
        benchmark.official_image("../unsafe")
    with pytest.raises(benchmark.BenchmarkError, match="Unsupported placeholder"):
        benchmark.official_image(INSTANCE, "example/{unknown}:latest")

    task_id = benchmark.docker_task_id("a-very-long-run-id-" * 8, INSTANCE)
    assert len(task_id) <= 63
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", task_id)


def test_hints_are_opt_in_for_benchmark_prompt():
    row = benchmark.parse_swebench_row(make_row())
    without_hints = benchmark.build_prompt(row, include_hints=False)
    with_hints = benchmark.build_prompt(row, include_hints=True)

    assert "Inspect validation." not in without_hints
    assert "Inspect validation." in with_hints


def native_delegation_messages(
    phases=("navigator", "patcher", "reviewer"),
    *,
    role="leaf",
    batch=False,
):
    messages = []
    for index, phase in enumerate(phases):
        call_id = f"delegate-{index}"
        arguments = {
            "goal": f"[benchmark-{phase}] complete the {phase} role",
            "context": "Full task and prior handoffs.",
            "role": role,
        }
        if batch:
            arguments["tasks"] = [
                {
                    "goal": arguments.pop("goal"),
                    "context": arguments.pop("context"),
                    "role": role,
                }
            ]
        messages.extend([
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "function": {
                            "name": "delegate_task",
                            "arguments": json.dumps(arguments),
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
    records, errors = worker.audit_native_delegations(
        native_delegation_messages()
    )

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


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (
            native_delegation_messages(("patcher", "navigator", "reviewer")),
            "Expected native delegation order",
        ),
        (native_delegation_messages(role="orchestrator"), "was not a leaf"),
        (native_delegation_messages(batch=True), "used batch mode"),
    ],
)
def test_native_delegation_audit_reports_parity_violations(messages, expected):
    _records, errors = worker.audit_native_delegations(messages)

    assert any(expected in error for error in errors)


def test_initial_setup_preserves_image_provided_ignored_artifacts(monkeypatch):
    commands = []
    overrides = []
    fake_environment = object()

    def terminal_tool(**kwargs):
        commands.append(kwargs["command"])
        return json.dumps({"exit_code": 0, "output": ""})

    monkeypatch.setattr(
        "tools.terminal_tool.register_task_env_overrides",
        lambda *args: overrides.append(args),
    )
    monkeypatch.setattr("tools.terminal_tool.terminal_tool", terminal_tool)
    monkeypatch.setattr(
        "tools.terminal_tool.get_active_env", lambda _task_id: fake_environment
    )
    request = {
        "taskId": "setup-test",
        "image": benchmark.official_image(INSTANCE),
        "row": {"base_commit": BASE_COMMIT},
    }

    assert worker.setup_environment(request) is fake_environment
    assert len(commands) == 1
    assert worker.CONDA_ACTIVATION in commands[0]
    assert f"reset --hard {BASE_COMMIT}" in commands[0]
    assert "git clean" not in commands[0]
    assert overrides == [("setup-test", {"cwd": "/testbed"})]


def test_worker_terminal_config_is_local_docker_without_resource_caps_or_forwarded_secrets(
    tmp_path,
):
    request = {
        "hermesHome": str(tmp_path / "home"),
        "image": benchmark.official_image(INSTANCE),
        "dockerPlatform": "linux/amd64",
    }

    config = worker._terminal_config(request)
    terminal = config["terminal"]

    assert terminal["backend"] == "docker"
    assert terminal["cwd"] == "/testbed"
    assert terminal["container_cpu"] == 0
    assert terminal["container_memory"] == 0
    assert terminal["container_disk"] == 0
    assert terminal["docker_forward_env"] == []
    assert terminal["docker_env"] == {}
    assert terminal["docker_volumes"] == []
    assert terminal["docker_persist_across_processes"] is False
    assert terminal["docker_orphan_reaper"] is False
    assert terminal["docker_extra_args"] == [
        "--platform",
        "linux/amd64",
        "--user",
        "root",
        "--entrypoint",
        "",
    ]
    assert config["agent"] == {
        "api_max_retries": 1,
        "coding_context": "off",
    }
    assert config["delegation"] == {
        "max_iterations": 13,
        "max_concurrent_children": 1,
        "max_spawn_depth": 1,
        "orchestrator_enabled": False,
    }
    assert type(config["agent"]["api_max_retries"]) is int


def test_worker_config_reaches_terminal_runtime_consumption_point(
    tmp_path, monkeypatch
):
    request = {
        "hermesHome": str(tmp_path / "home"),
        "image": benchmark.official_image(INSTANCE),
        "dockerPlatform": "linux/amd64",
        "agentTimeoutSeconds": 1800,
        "sourceIdentity": {
            "commit": "test",
            "dirty": False,
            "fingerprint": "test",
        },
    }
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP

    terminal_config = worker._terminal_config(request)["terminal"]
    for key in terminal_config:
        env_name = TERMINAL_CONFIG_ENV_MAP.get(key)
        if env_name:
            monkeypatch.setenv(env_name, "before-test")
    monkeypatch.setenv("HERMES_HOME", "before-test")
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "420")

    worker.configure_worker(request)

    from tools.terminal_tool import _get_env_config

    consumed = _get_env_config()
    assert consumed["env_type"] == "docker"
    assert consumed["cwd"] == "/testbed"
    assert consumed["docker_image"] == request["image"]
    assert consumed["container_cpu"] == 0
    assert consumed["container_memory"] == 0
    assert consumed["container_disk"] == 0
    assert consumed["container_persistent"] is True
    assert consumed["docker_forward_env"] == []
    assert consumed["docker_env"] == {}
    assert consumed["docker_volumes"] == []
    assert consumed["docker_persist_across_processes"] is False
    assert consumed["docker_orphan_reaper"] is False
    assert consumed["docker_extra_args"] == terminal_config["docker_extra_args"]


def test_actual_coordinator_exposes_native_delegation(tmp_path, monkeypatch):
    request = {
        "taskId": "scope-test",
        "model": benchmark.DEFAULT_MODEL,
        "hermesHome": str(tmp_path / "home"),
        "image": benchmark.official_image(INSTANCE),
        "dockerPlatform": "linux/amd64",
        "agentTimeoutSeconds": 1800,
        "sourceIdentity": {
            "commit": "test",
            "dirty": False,
            "fingerprint": "test",
        },
    }
    worker.configure_worker(request)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-benchmark-key")
    coordinator = None
    try:
        coordinator = worker.default_agent_factory(
            role="coordinator",
            budget=24,
            api_key="dummy-benchmark-key",
            request=request,
        )

        assert coordinator.valid_tool_names == {
            "delegate_task",
            "patch",
            "process",
            "read_file",
            "search_files",
            "terminal",
            "write_file",
        }
        assert coordinator.model == "qwen/qwen3-coder-next"
        assert coordinator.max_iterations == 24
        assert coordinator.request_overrides["temperature"] == 0.1
        assert coordinator._api_max_retries == 1
        assert coordinator.client.max_retries == 0
        from agent.coding_context import resolve_runtime_mode

        workspace = tmp_path / "host-workspace"
        workspace.mkdir()
        (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        mode = resolve_runtime_mode(
            platform="cli",
            cwd=workspace,
            model=coordinator.model,
        )
        assert mode.config_mode == "off"
        assert mode.is_coding is False
        assert mode.system_blocks() == []
    finally:
        if coordinator is not None:
            coordinator.release_clients()


def test_worker_environment_passes_only_openrouter_credential(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-openrouter")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-pass")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-pass")
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/benchmark-docker.sock")
    monkeypatch.setenv("DOCKER_CONTEXT", "benchmark-context")

    env = benchmark._worker_environment(tmp_path)

    assert env["OPENROUTER_API_KEY"] == "secret-openrouter"
    assert "GITHUB_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env["DOCKER_HOST"] == "unix:///tmp/benchmark-docker.sock"
    assert env["DOCKER_CONTEXT"] == "benchmark-context"
    assert env["HERMES_HOME"] == str(tmp_path)


def test_delegation_tool_pool_timeout_exceeds_shared_agent_deadline(
    tmp_path, monkeypatch
):
    request = {
        "hermesHome": str(tmp_path / "home"),
        "image": benchmark.official_image(INSTANCE),
        "dockerPlatform": "linux/amd64",
        "agentTimeoutSeconds": 1800,
        "sourceIdentity": {
            "commit": "test",
            "dirty": False,
            "fingerprint": "test",
        },
    }
    monkeypatch.setattr(
        "hermes_cli.config.apply_terminal_config_to_env",
        lambda **_kwargs: os.environ,
    )
    monkeypatch.setenv("HERMES_HOME", "before-test")
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "420")

    worker.configure_worker(request)

    assert os.environ["HERMES_CONCURRENT_TOOL_TIMEOUT_S"] == "1860"


def test_controller_distinguishes_setup_and_agent_timeouts(tmp_path, monkeypatch):
    process = SimpleNamespace(poll=lambda: None)
    terminations = []
    monkeypatch.setattr(benchmark, "_terminate_worker", terminations.append)
    monotonic_values = iter([0.0, 2.0])
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(benchmark.time, "sleep", lambda _seconds: None)

    assert benchmark._wait_for_worker(
        process,
        tmp_path / "missing-runtime.json",
        setup_timeout_seconds=1,
        agent_timeout_seconds=10,
    ) == (False, "setup timeout")
    assert terminations == [process]

    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps({"agentStartedAt": "now"}), encoding="utf-8")
    terminations.clear()
    monotonic_values = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(monotonic_values))
    assert benchmark._wait_for_worker(
        process,
        runtime_path,
        setup_timeout_seconds=10,
        agent_timeout_seconds=1,
    ) == (True, "agent timeout")
    assert terminations == [process]

    runtime_path.write_text(
        json.dumps({"agentStartedAt": "now", "agentCompletedAt": "now"}),
        encoding="utf-8",
    )
    terminations.clear()
    monotonic_values = iter([0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(monotonic_values))
    assert benchmark._wait_for_worker(
        process,
        runtime_path,
        setup_timeout_seconds=1,
        agent_timeout_seconds=10,
    ) == (False, "teardown timeout")
    assert terminations == [process]


def test_timed_out_worker_defers_capture_and_container_cleanup(tmp_path, monkeypatch):
    runtime_path = tmp_path / "runtime.json"
    request = {
        "benchmark": benchmark.BENCHMARK,
        "row": make_row(),
        "model": benchmark.DEFAULT_MODEL,
        "taskId": "timeout-task",
        "prompt": "Fix the issue.",
        "runtimePath": str(runtime_path),
        "hermesHome": str(tmp_path / "home"),
        "image": benchmark.official_image(INSTANCE),
        "dockerPlatform": "linux/amd64",
        "agentTimeoutSeconds": 1800,
        "sourceIdentity": benchmark.hermes_source_identity(),
    }
    fake_environment = SimpleNamespace(_container_id="a" * 64)
    installed_handlers = {}

    class InterruptingAgent:
        session_id = "coordinator"

        def __init__(self):
            self._active_children = []
            self._active_children_lock = threading.Lock()

        def run_conversation(self, *_args, **_kwargs):
            installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
            return {"completed": False, "interrupted": True}

        def interrupt(self, _reason):
            return None

        def release_clients(self):
            return None

    def fake_signal(signum, handler):
        previous = installed_handlers.get(signum, signal.SIG_DFL)
        installed_handlers[signum] = handler
        return previous

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-benchmark-key")
    monkeypatch.setattr(worker, "configure_worker", lambda _request: None)
    monkeypatch.setattr(worker, "setup_environment", lambda _request: fake_environment)
    monkeypatch.setattr(worker.signal, "signal", fake_signal)
    monkeypatch.setattr(
        worker,
        "capture_patch",
        lambda *_args: pytest.fail("worker must not capture a timed-out patch"),
    )
    monkeypatch.setattr(
        worker,
        "restore_sandbox_ownership",
        lambda *_args: pytest.fail("worker must leave the timed-out container alone"),
    )
    monkeypatch.setattr(
        "tools.terminal_tool.clear_task_env_overrides", lambda task_id: None
    )

    result = worker.run_worker(
        request,
        agent_factory=lambda **_kwargs: InterruptingAgent(),
    )

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert "agentCompletedAt" not in runtime
    assert result["timedOut"] is True
    assert result["deferCaptureToController"] is True
    assert result["modelPatch"] is None


def test_completed_worker_also_defers_capture_and_container_cleanup(
    tmp_path, monkeypatch
):
    runtime_path = tmp_path / "runtime.json"
    request = {
        "benchmark": benchmark.BENCHMARK,
        "row": make_row(),
        "model": benchmark.DEFAULT_MODEL,
        "taskId": "completed-task",
        "prompt": "Fix the issue.",
        "runtimePath": str(runtime_path),
        "hermesHome": str(tmp_path / "home"),
        "image": benchmark.official_image(INSTANCE),
        "dockerPlatform": "linux/amd64",
        "agentTimeoutSeconds": 1800,
        "sourceIdentity": benchmark.hermes_source_identity(),
    }
    fake_environment = SimpleNamespace(_container_id="a" * 64)

    class CompletingAgent:
        session_id = "coordinator"

        def __init__(self):
            self._active_children = []
            self._active_children_lock = threading.Lock()

        def run_conversation(self, *_args, **_kwargs):
            return {"completed": True, "interrupted": False}

        def release_clients(self):
            return None

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-benchmark-key")
    monkeypatch.setattr(worker, "configure_worker", lambda _request: None)
    monkeypatch.setattr(worker, "setup_environment", lambda _request: fake_environment)
    monkeypatch.setattr(
        worker,
        "capture_patch",
        lambda *_args: pytest.fail("worker must defer every patch capture"),
    )
    monkeypatch.setattr(
        worker,
        "restore_sandbox_ownership",
        lambda *_args: pytest.fail("worker must leave every container to controller"),
    )
    monkeypatch.setattr(
        "tools.terminal_tool.clear_task_env_overrides", lambda _task_id: None
    )

    result = worker.run_worker(
        request,
        agent_factory=lambda **_kwargs: CompletingAgent(),
    )

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime["agentCompletedAt"]
    assert result["timedOut"] is False
    assert result["deferCaptureToController"] is True
    assert result["modelPatch"] is None


@pytest.mark.parametrize(
    ("worker_result", "wait_result", "expected_timed_out"),
    [
        (
            {
                "modelPatch": None,
                "timedOut": False,
                "deferCaptureToController": True,
                "error": None,
            },
            (False, None),
            False,
        ),
        (
            {
                "modelPatch": None,
                "timedOut": True,
                "deferCaptureToController": True,
                "error": "agent timeout",
            },
            (True, "agent timeout"),
            True,
        ),
        (
            {
                "modelPatch": "stale worker patch",
                "timedOut": False,
                "error": "late worker failure",
            },
            (False, None),
            False,
        ),
    ],
    ids=["completed", "timed-out", "fallback-result"],
)
def test_controller_quiesces_and_captures_after_worker_exit(
    tmp_path, monkeypatch, worker_result, wait_result, expected_timed_out
):
    options = make_options(tmp_path)
    row = benchmark.parse_swebench_row(make_row())
    instance_dir = tmp_path / "instance"
    exited = False
    calls = []

    class FakeProcess:
        returncode = -9

    monkeypatch.setattr(
        benchmark.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess()
    )

    def fake_wait(_process, runtime_path, **_kwargs):
        nonlocal exited
        benchmark.atomic_write_json(
            runtime_path,
            {"containerId": "a" * 64, "agentStartedAt": "now"},
        )
        benchmark.atomic_write_json(
            instance_dir / "worker-result.json",
            worker_result,
        )
        exited = True
        return wait_result

    def fake_restart(_container_id):
        assert exited is True
        calls.append("restart")
        return None

    def fake_capture(_container_id, _base_commit):
        assert calls == ["restart"]
        calls.append("capture")
        return "diff --git a/a b/a\n", ["a"], None

    monkeypatch.setattr(benchmark, "_wait_for_worker", fake_wait)
    monkeypatch.setattr(benchmark, "_restart_container_for_capture", fake_restart)
    monkeypatch.setattr(benchmark, "_capture_patch_from_container", fake_capture)
    monkeypatch.setattr(
        benchmark,
        "_restore_container_ownership",
        lambda _container_id: calls.append("ownership"),
    )
    monkeypatch.setattr(
        benchmark, "_remove_container", lambda _container_id: calls.append("remove")
    )

    result = benchmark._run_worker(
        options,
        row,
        instance_dir,
        benchmark.official_image(INSTANCE),
        {"id": "image"},
    )

    assert result["modelPatch"] == "diff --git a/a b/a\n"
    assert result["timedOut"] is expected_timed_out
    assert calls == ["restart", "capture", "ownership", "remove"]
    assert (instance_dir / "worker.stdout.log").stat().st_mode & 0o777 == 0o600
    assert (instance_dir / "worker.stderr.log").stat().st_mode & 0o777 == 0o600


def test_worker_logs_are_private_and_redacted_when_controller_wait_fails(
    tmp_path, monkeypatch
):
    options = make_options(tmp_path)
    row = benchmark.parse_swebench_row(make_row())
    instance_dir = tmp_path / "instance"
    secret = "not-a-real-benchmark-secret"

    class FakeProcess:
        returncode = None

    def fake_popen(*_args, **kwargs):
        kwargs["stdout"].write(f"stdout {secret}\n")
        kwargs["stdout"].flush()
        kwargs["stderr"].write(f"stderr {secret}\n")
        kwargs["stderr"].flush()
        return FakeProcess()

    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setattr(benchmark.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        benchmark,
        "_terminate_worker",
        lambda process: setattr(process, "returncode", -15),
    )
    monkeypatch.setattr(benchmark, "_find_benchmark_container", lambda _task_id: None)
    monkeypatch.setattr(
        benchmark,
        "_wait_for_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("wait failed")),
    )

    with pytest.raises(RuntimeError, match="wait failed"):
        benchmark._run_worker(
            options,
            row,
            instance_dir,
            benchmark.official_image(INSTANCE),
            {"id": "image"},
        )

    for path in (
        instance_dir / "worker.stdout.log",
        instance_dir / "worker.stderr.log",
    ):
        assert path.stat().st_mode & 0o777 == 0o600
        content = path.read_text(encoding="utf-8")
        assert secret not in content
        assert "[REDACTED]" in content


def test_controller_interrupt_reaps_worker_before_patch_capture_and_cleanup(
    tmp_path, monkeypatch
):
    options = make_options(tmp_path)
    row = benchmark.parse_swebench_row(make_row())
    instance_dir = tmp_path / "instance"
    calls = []

    class FakeProcess:
        returncode = None

    process = FakeProcess()
    monkeypatch.setattr(
        benchmark.subprocess, "Popen", lambda *_args, **_kwargs: process
    )

    def interrupted_wait(_process, runtime_path, **_kwargs):
        benchmark.atomic_write_json(
            runtime_path,
            {"containerId": "a" * 64, "agentStartedAt": "now"},
        )
        raise KeyboardInterrupt

    def terminate(_process):
        calls.append("terminate")
        process.returncode = -15

    def restart(_container_id):
        assert process.returncode == -15
        calls.append("restart")
        return None

    monkeypatch.setattr(benchmark, "_wait_for_worker", interrupted_wait)
    monkeypatch.setattr(benchmark, "_terminate_worker", terminate)
    monkeypatch.setattr(benchmark, "_restart_container_for_capture", restart)
    monkeypatch.setattr(
        benchmark,
        "_capture_patch_from_container",
        lambda *_args: (
            calls.append("capture")
            or ("diff --git a/recovered b/recovered\n", ["recovered"], None)
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_restore_container_ownership",
        lambda _container_id: calls.append("ownership"),
    )
    monkeypatch.setattr(
        benchmark, "_remove_container", lambda _container_id: calls.append("remove")
    )

    with pytest.raises(KeyboardInterrupt):
        benchmark._run_worker(
            options,
            row,
            instance_dir,
            benchmark.official_image(INSTANCE),
            {"id": "image"},
        )

    assert calls == ["terminate", "restart", "capture", "ownership", "remove"]
    assert (instance_dir / "model.patch").read_text(encoding="utf-8") == (
        "diff --git a/recovered b/recovered\n"
    )
    persisted = json.loads(
        (instance_dir / "controller-result.json").read_text(encoding="utf-8")
    )
    assert persisted["changedPaths"] == ["recovered"]


def test_interrupt_during_ownership_still_removes_container_and_persists_patch(
    tmp_path, monkeypatch
):
    options = make_options(tmp_path)
    row = benchmark.parse_swebench_row(make_row())
    instance_dir = tmp_path / "instance"
    calls = []

    class FakeProcess:
        returncode = 0

    monkeypatch.setattr(
        benchmark.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess()
    )

    def completed_wait(_process, runtime_path, **_kwargs):
        benchmark.atomic_write_json(
            runtime_path,
            {"containerId": "a" * 64, "agentStartedAt": "now"},
        )
        benchmark.atomic_write_json(
            instance_dir / "worker-result.json",
            {"deferCaptureToController": True, "timedOut": False},
        )
        return False, None

    monkeypatch.setattr(benchmark, "_wait_for_worker", completed_wait)
    monkeypatch.setattr(
        benchmark,
        "_restart_container_for_capture",
        lambda _container_id: calls.append("restart"),
    )
    monkeypatch.setattr(
        benchmark,
        "_capture_patch_from_container",
        lambda *_args: (
            calls.append("capture") or ("diff --git a/saved b/saved\n", ["saved"], None)
        ),
    )

    def interrupt_ownership(_container_id):
        calls.append("ownership")
        raise KeyboardInterrupt

    monkeypatch.setattr(benchmark, "_restore_container_ownership", interrupt_ownership)
    monkeypatch.setattr(
        benchmark, "_remove_container", lambda _container_id: calls.append("remove")
    )

    with pytest.raises(KeyboardInterrupt):
        benchmark._run_worker(
            options,
            row,
            instance_dir,
            benchmark.official_image(INSTANCE),
            {"id": "image"},
        )

    assert calls == ["restart", "capture", "ownership", "remove"]
    assert (instance_dir / "model.patch").read_text(encoding="utf-8") == (
        "diff --git a/saved b/saved\n"
    )


def test_workflow_requires_coordinator_reconciliation():
    records, errors = worker.audit_native_delegations(
        native_delegation_messages()
    )

    assert worker.reconciled_workflow(
        records, errors, {"completed": False}, None
    ) is False
    assert (
        worker.reconciled_workflow(
            records,
            errors,
            {"completed": True, "interrupted": False},
            None,
        )
        is True
    )


class LocalGitEnvironment:
    def __init__(self, repository: Path):
        self.repository = repository

    def execute(self, command, cwd, timeout):
        mapped = command.replace("/testbed", shlex.quote(str(self.repository)))
        completed = subprocess.run(
            ["bash", "-lc", mapped],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "output": completed.stdout + completed.stderr,
            "returncode": completed.returncode,
        }


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_controller_patch_capture_never_starts_a_login_shell(monkeypatch):
    commands = []

    def fake_run_command(args, **_kwargs):
        commands.append(args)
        return benchmark.CommandResult(tuple(args), 0, "", "")

    monkeypatch.setattr(benchmark, "run_command", fake_run_command)

    patch, changed_paths, error = benchmark._capture_patch_from_container(
        "a" * 64, BASE_COMMIT
    )

    assert (patch, changed_paths, error) == ("", [], None)
    assert len(commands) == 2
    assert all(command[-3] == "bash" and command[-2] == "-c" for command in commands)


def test_patch_capture_includes_modified_deleted_untracked_and_binary_files(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    git(repository, "init")
    git(repository, "config", "user.email", "benchmark@example.com")
    git(repository, "config", "user.name", "Benchmark")
    (repository / "modified.txt").write_text("before\n", encoding="utf-8")
    (repository / "deleted.txt").write_text("delete me\n", encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "base")
    base_commit = git(repository, "rev-parse", "HEAD")
    (repository / "modified.txt").write_text("after\n", encoding="utf-8")
    (repository / "deleted.txt").unlink()
    (repository / "untracked.txt").write_text("new\n", encoding="utf-8")
    (repository / "binary.bin").write_bytes(b"\x00\x01\xff\x10")

    patch, changed_paths, error = worker.capture_patch(
        LocalGitEnvironment(repository), base_commit
    )

    assert error is None
    assert set(changed_paths) == {
        "binary.bin",
        "deleted.txt",
        "modified.txt",
        "untracked.txt",
    }
    assert "modified.txt" in patch
    assert "deleted.txt" in patch
    assert "untracked.txt" in patch
    assert "GIT binary patch" in patch or "Binary files" in patch


def test_prediction_manifest_is_hash_bound_and_tampering_is_rejected(tmp_path):
    options = make_options(tmp_path)
    paths = benchmark.build_paths(options)
    row = benchmark.parse_swebench_row(make_row())
    prediction = {
        "instance_id": INSTANCE,
        "model_name_or_path": "hermes-agent@source:" + benchmark.DEFAULT_MODEL,
        "model_patch": "diff --git a/a b/a\n",
    }
    benchmark._write_progress(
        options,
        paths,
        [row],
        [{"generationSucceeded": True}],
        [prediction],
        complete=True,
    )

    manifest, predictions, digest = benchmark.verify_prediction_artifact(
        paths.predictions, paths.manifest
    )
    assert manifest["complete"] is True
    assert manifest["apiMaxRetries"] == 1
    assert manifest["codingContext"] == "off"
    assert set(manifest["sourceIdentity"]) == {"commit", "dirty", "fingerprint"}
    assert predictions == [prediction]
    assert digest == manifest["predictionsSha256"]

    paths.predictions.write_text(
        paths.predictions.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(benchmark.BenchmarkError, match="changed after inference"):
        benchmark.verify_prediction_artifact(paths.predictions, paths.manifest)


def test_summary_counts_empty_predictions_separately_from_nonempty_patches(tmp_path):
    options = make_options(tmp_path)
    paths = benchmark.build_paths(options)
    row = benchmark.parse_swebench_row(make_row())
    prediction = {
        "instance_id": INSTANCE,
        "model_name_or_path": "hermes-agent@source:" + benchmark.DEFAULT_MODEL,
        "model_patch": "",
    }
    benchmark._write_progress(
        options,
        paths,
        [row],
        [{"generationSucceeded": False}],
        [prediction],
        complete=True,
    )

    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    assert summary["predictionCount"] == 1
    assert summary["nonEmptyPatchCount"] == 0


@pytest.mark.parametrize(
    ("option_updates", "row_updates", "mismatch"),
    [
        ({"image_template": "example/{repo}/{name}:latest"}, {}, "image"),
        ({"docker_platform": "linux/arm64"}, {}, "Docker platform"),
        ({"setup_timeout_seconds": 601}, {}, "setup timeout"),
        (
            {
                "source_identity": {
                    "commit": "different",
                    "dirty": False,
                    "fingerprint": "different",
                }
            },
            {},
            "source identity",
        ),
        ({}, {"base_commit": "b" * 40}, "selected instances"),
    ],
)
def test_resume_refuses_mixed_runtime_configuration(
    tmp_path, option_updates, row_updates, mismatch
):
    options = make_options(tmp_path)
    paths = benchmark.build_paths(options)
    row = benchmark.parse_swebench_row(make_row())
    benchmark._write_progress(options, paths, [row], [], [], complete=False)

    resumed_options = make_options(tmp_path, **option_updates)
    resumed_row = benchmark.parse_swebench_row(make_row(**row_updates))
    with pytest.raises(benchmark.BenchmarkError, match=mismatch):
        benchmark._load_resume(resumed_options, paths, [resumed_row])


def test_running_inference_refuses_changed_hermes_source(tmp_path, monkeypatch):
    options = make_options(tmp_path)
    monkeypatch.setattr(
        benchmark,
        "hermes_source_identity",
        lambda: {"commit": "changed", "dirty": True, "fingerprint": "changed"},
    )

    with pytest.raises(benchmark.BenchmarkError, match="source changed"):
        benchmark.require_unchanged_source(options)


def test_official_evaluator_command_is_local_single_worker_with_one_hour_test_timeout(
    tmp_path,
):
    predictions = tmp_path / "predictions.jsonl"
    command = benchmark.build_evaluation_command(
        "/venv/bin/python",
        predictions,
        "run-1",
        [INSTANCE],
        namespace_empty=False,
    )

    assert command[:3] == [
        "/venv/bin/python",
        "-m",
        "swebench.harness.run_evaluation",
    ]
    assert command[command.index("--max_workers") + 1] == "1"
    assert command[command.index("--timeout") + 1] == "3600"
    assert command[command.index("--split") + 1] == "test"
    assert command[command.index("--modal") + 1] == "false"
    assert command[command.index("--instance_ids") + 1 :] == [INSTANCE]


def test_noisy_evaluator_version_output_is_parsed_by_sentinel():
    output = (
        "OpenHands benchmark patches active\n"
        "another startup banner\n"
        f"{benchmark.SWEBENCH_VERSION_SENTINEL}4.1.0\n"
    )
    assert benchmark.parse_swebench_version(output) == "4.1.0"


def test_evaluation_subprocesses_receive_no_provider_or_cloud_secrets(
    tmp_path, monkeypatch
):
    options = make_options(tmp_path)
    paths = benchmark.build_paths(options)
    row = benchmark.parse_swebench_row(make_row())
    prediction = {
        "instance_id": INSTANCE,
        "model_name_or_path": "hermes-agent@source:" + benchmark.DEFAULT_MODEL,
        "model_patch": "diff --git a/a b/a\n",
    }
    benchmark._write_progress(
        options,
        paths,
        [row],
        [{"generationSucceeded": True}],
        [prediction],
        complete=True,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-pass")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-pass")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-pass")
    observed_environments = []
    docker_environments = []

    def fake_require_docker(*, env=None):
        docker_environments.append(env)
        return {"serverVersion": "test"}

    def fake_run_command(args, *, cwd=None, timeout=None, env=None):
        observed_environments.append(env)
        if "-c" in args:
            stdout = (
                "OpenHands startup banner\n"
                f"{benchmark.SWEBENCH_VERSION_SENTINEL}4.1.0\n"
            )
            return benchmark.CommandResult(tuple(args), 0, stdout, "")
        return benchmark.CommandResult(tuple(args), 0, "evaluation complete", "")

    monkeypatch.setattr(benchmark, "require_docker", fake_require_docker)
    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    args = SimpleNamespace(
        run_id=options.run_id,
        output_dir=str(options.output_dir),
        predictions_path=None,
        manifest_path=None,
        instance_id=[],
        python="/evaluator/bin/python",
        namespace_empty=False,
        dry_run=False,
    )

    benchmark.run_evaluation(args)

    assert len(observed_environments) == 3
    for env in [*observed_environments, *docker_environments]:
        assert "OPENROUTER_API_KEY" not in env
        assert "GITHUB_TOKEN" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env


def test_evaluation_refuses_a_manifest_from_a_different_run(tmp_path):
    options = make_options(tmp_path)
    paths = benchmark.build_paths(options)
    row = benchmark.parse_swebench_row(make_row())
    benchmark._write_progress(
        options,
        paths,
        [row],
        [{"generationSucceeded": True}],
        [
            {
                "instance_id": INSTANCE,
                "model_name_or_path": "hermes-agent@source:" + benchmark.DEFAULT_MODEL,
                "model_patch": "diff --git a/a b/a\n",
            }
        ],
        complete=True,
    )
    args = SimpleNamespace(
        run_id="different-run",
        output_dir=str(options.output_dir),
        predictions_path=str(paths.predictions),
        manifest_path=str(paths.manifest),
        instance_id=[],
        python="/evaluator/bin/python",
        namespace_empty=False,
        dry_run=True,
    )

    with pytest.raises(benchmark.BenchmarkError, match="run id"):
        benchmark.run_evaluation(args)


def test_interrupted_evaluation_records_state_and_removes_run_containers(
    tmp_path, monkeypatch
):
    options = make_options(tmp_path)
    paths = benchmark.build_paths(options)
    row = benchmark.parse_swebench_row(make_row())
    benchmark._write_progress(
        options,
        paths,
        [row],
        [{"generationSucceeded": True}],
        [
            {
                "instance_id": INSTANCE,
                "model_name_or_path": "hermes-agent@source:" + benchmark.DEFAULT_MODEL,
                "model_patch": "diff --git a/a b/a\n",
            }
        ],
        complete=True,
    )
    container_id = "a" * 12
    container_name = f"sweb.eval.{INSTANCE.lower()}.{options.run_id}"
    removed = []

    def fake_run_command(args, *, cwd=None, timeout=None, env=None):
        if "-c" in args:
            return benchmark.CommandResult(
                tuple(args),
                0,
                f"{benchmark.SWEBENCH_VERSION_SENTINEL}4.1.0\n",
                "",
            )
        if args[:3] == ["docker", "ps", "-a"]:
            return benchmark.CommandResult(
                tuple(args), 0, f"{container_id}\t{container_name}\n", ""
            )
        if args[:3] == ["docker", "rm", "-f"]:
            removed.extend(args[3:])
            return benchmark.CommandResult(tuple(args), 0, "", "")
        raise KeyboardInterrupt

    monkeypatch.setattr(benchmark, "require_docker", lambda **_kwargs: {})
    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    args = SimpleNamespace(
        run_id=options.run_id,
        output_dir=str(options.output_dir),
        predictions_path=None,
        manifest_path=None,
        instance_id=[],
        python="/evaluator/bin/python",
        namespace_empty=False,
        dry_run=False,
    )

    with pytest.raises(KeyboardInterrupt):
        benchmark.run_evaluation(args)

    evaluation = json.loads(paths.evaluation_manifest.read_text(encoding="utf-8"))
    assert evaluation["status"] == "interrupted"
    assert evaluation["containerCleanupErrors"] == []
    assert removed == [container_id]


def test_dry_run_reports_parity_contract_without_requiring_api_key(tmp_path, capsys):
    class StaticDataset:
        def select(self, *_args, **_kwargs):
            return [benchmark.parse_swebench_row(make_row())]

    options = make_options(tmp_path, dry_run=True)
    benchmark.run_inference(options, dataset_client=StaticDataset())
    payload = json.loads(capsys.readouterr().out)

    assert payload["model"] == benchmark.DEFAULT_MODEL
    assert payload["inferenceWorkers"] == 1
    assert payload["attemptsPerInstance"] == 1
    assert payload["maxInfrastructureRetries"] == 0
    assert payload["apiMaxRetries"] == 1
    assert payload["codingContext"] == "off"
    assert payload["agentTimeoutSeconds"] == 1800
    assert payload["setupTimeoutSeconds"] == 600
    assert payload["sequence"] == ["coordinator", "navigator", "patcher", "reviewer"]
    assert payload["delegationMode"] == "native"
    assert payload["coordinatorBudget"] == 24
    assert payload["nativeSubagentBudget"] == 13
    assert payload["nativeSubagentCount"] == 3
    assert payload["nativeSubagentTotalBudget"] == 39
    assert payload["peerPhaseBudgetReference"] == {
        "navigator": 10,
        "patcher": 18,
        "reviewer": 12,
    }
