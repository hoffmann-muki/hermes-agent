from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli.benchmarks.tracing import privileges
from hermes_cli.benchmarks.tracing.agentsight import (
    AgentSightProfiler,
    AgentSightTarget,
    _wait_for_ready,
    build_docker_sidecar_args,
    build_host_collector_args,
    docker_compose_project_name,
    resolve_container_tls_binary,
    resolve_docker_compose_main_container,
    resolve_host_tls_binary,
)
from hermes_cli.benchmarks.tracing.privileges import (
    SudoAuthorizationError,
    authorize_sudo,
    build_sudo_supervised_command,
)


def test_host_scope_uses_research_capture_and_readiness(tmp_path: Path) -> None:
    args = build_host_collector_args(
        binary="/usr/local/bin/agentsight",
        source_dir=tmp_path / "host",
        profile_id="trace-1",
        pid=42,
    )

    assert args[:4] == ["/usr/local/bin/agentsight", "record", "--pid", "42"]
    assert "research" in args
    assert "--profile-dir" in args
    assert "--ready-file" in args
    assert "--no-server" in args


def test_host_collector_can_pin_tls_to_one_binary(tmp_path: Path) -> None:
    args = build_host_collector_args(
        binary="/usr/local/bin/agentsight",
        source_dir=tmp_path / "host",
        profile_id="trace-1",
        pid=42,
        tls_binary_path="/usr/lib/libssl.so.3",
    )

    assert args[args.index("--binary-path") + 1] == "/usr/lib/libssl.so.3"
    assert "--tls-binary-only" in args


def test_host_tls_binary_resolves_current_python_runtime() -> None:
    binary = Path(resolve_host_tls_binary(os.getpid()))

    assert binary.is_absolute()
    assert binary.is_file()


def test_host_collector_can_run_under_validated_sudo(tmp_path: Path) -> None:
    collector = build_host_collector_args(
        binary="/usr/local/bin/agentsight",
        source_dir=tmp_path / "host",
        profile_id="trace-1",
        pid=42,
    )
    args = build_sudo_supervised_command(
        privileges.SudoAuthorization(
            command_prefix=("/usr/bin/sudo", "-n", "--"),
            method="sudo-cache",
        ),
        collector,
        owner_pid=os.getpid(),
        stop_file=tmp_path / "stop",
        stop_timeout=15,
    )

    assert args[:3] == [
        "/usr/bin/sudo",
        "-n",
        "--",
    ]
    collector_start = args.index("/usr/local/bin/agentsight")
    assert args[collector_start : collector_start + 7] == [
        "/usr/local/bin/agentsight",
        "record",
        "--pid",
        "42",
        "--capture-level",
        "research",
        "--profile-dir",
    ]
    assert "_supervise" in args
    assert str(tmp_path / "stop") in args


def test_password_file_authorizes_noninteractive_sudo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password_file = tmp_path / "sudo-password"
    password_file.write_text("test-only-password\n")
    password_file.chmod(0o600)
    calls: list[tuple[list[str], object]] = []

    def run(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((args, kwargs["stdin"]))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(privileges.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        privileges.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/sudo"
    )
    monkeypatch.setattr(privileges.subprocess, "run", run)

    authorization = authorize_sudo({
        "PATH": "/usr/bin",
        "BENCHMARK_SUDO_PASSWORD_FILE": str(password_file),
    })

    assert authorization.command_prefix == (
        "/usr/bin/sudo",
        "-S",
        "-p",
        "",
        "--",
    )
    assert authorization.method == "sudo-password-file"
    assert calls[0][0] == ["/usr/bin/sudo", "-S", "-p", "", "-v"]
    assert hasattr(calls[0][1], "read")


def test_supervisor_command_contains_no_password(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password_file = tmp_path / "sudo-password"
    password_file.write_text("test-only-password\n")
    password_file.chmod(0o600)

    def run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(privileges.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        privileges.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/sudo"
    )
    monkeypatch.setattr(privileges.subprocess, "run", run)
    authorization = authorize_sudo({
        "PATH": "/usr/bin",
        "BENCHMARK_SUDO_PASSWORD_FILE": str(password_file),
    })
    command = build_sudo_supervised_command(
        authorization,
        ["/usr/local/bin/agentsight", "record"],
        owner_pid=os.getpid(),
        stop_file=tmp_path / "stop",
        stop_timeout=15,
    )

    assert command[:5] == ["/usr/bin/sudo", "-S", "-p", "", "--"]
    assert "test-only-password" not in " ".join(command)
    assert "_supervise" in command


def test_supervisor_stops_child_via_private_marker(tmp_path: Path) -> None:
    stop_file = tmp_path / "stop"
    process = subprocess.Popen([
        privileges.sys.executable,
        "-S",
        str(Path(privileges.__file__).resolve()),
        "_supervise",
        str(stop_file),
        "1",
        str(os.getpid()),
        privileges.sys.executable,
        "-c",
        "import time; time.sleep(30)",
    ])
    stop_file.touch(mode=0o600)

    assert process.wait(timeout=3) == 0
    assert not stop_file.exists()


def test_supervisor_stops_child_when_owner_exits(tmp_path: Path) -> None:
    owner = subprocess.Popen(["/bin/sleep", "0.2"])
    process = subprocess.Popen([
        privileges.sys.executable,
        "-S",
        str(Path(privileges.__file__).resolve()),
        "_supervise",
        str(tmp_path / "stop"),
        "1",
        str(owner.pid),
        "/bin/sleep",
        "30",
    ])

    assert owner.wait(timeout=3) == 0
    assert process.wait(timeout=3) == 0


def test_sudo_password_file_rejects_broad_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password_file = tmp_path / "sudo-password"
    password_file.write_text("test-only-password\n")
    password_file.chmod(0o644)

    monkeypatch.setattr(privileges.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        privileges.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/sudo"
    )

    with pytest.raises(SudoAuthorizationError, match="permissions"):
        authorize_sudo({
            "PATH": "/usr/bin",
            "BENCHMARK_SUDO_PASSWORD_FILE": str(password_file),
        })


def test_missing_password_file_reuses_an_existing_sudo_ticket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(privileges.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        privileges.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/sudo"
    )
    monkeypatch.setattr(privileges.subprocess, "run", run)

    authorization = authorize_sudo({
        "PATH": "/usr/bin",
        "BENCHMARK_SUDO_PASSWORD_FILE": str(tmp_path / "missing"),
    })

    assert authorization.method == "sudo-cache"
    assert calls == [["/usr/bin/sudo", "-n", "-v"]]


def test_task_container_scope_uses_privileged_pid_host_sidecar(
    tmp_path: Path,
) -> None:
    args = build_docker_sidecar_args(
        image="agentsight:play",
        sidecar="agentsight-trace",
        source_dir=tmp_path / "task-container",
        profile_id="trace-1",
        init_pid=4242,
        stop_timeout=15,
    )

    assert args[:3] == ["docker", "run", "--detach"]
    assert "--privileged" in args
    assert args[args.index("--pid") + 1] == "host"
    assert "research" in args
    assert "--no-ssl" in args
    assert "--no-stdio" in args
    assert args[args.index("--pidns-filter") + 1] == "/proc/4242/ns/pid"

    tls_args = build_docker_sidecar_args(
        image="agentsight:play",
        sidecar="agentsight-terminal-trace",
        source_dir=tmp_path / "terminal-task-container",
        profile_id="trace-2",
        init_pid=4343,
        stop_timeout=15,
        tls_binary_path="/proc/4343/root/usr/lib/libssl.so.3",
    )
    assert "--no-ssl" not in tls_args
    assert "--no-stdio" in tls_args
    assert tls_args[tls_args.index("--binary-path") + 1] == (
        "/proc/4343/root/usr/lib/libssl.so.3"
    )
    assert "--tls-binary-only" in tls_args


def test_container_tls_binary_uses_the_requested_python_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = "/container-only/libssl.so.3"
    calls: list[list[str]] = []

    def run(
        command: list[str],
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, f"{library}\n", "")

    monkeypatch.setattr(
        "hermes_cli.benchmarks.tracing.agentsight._run",
        run,
    )

    resolved = resolve_container_tls_binary(
        "task-container",
        "/opt/agent-venv/bin/python",
        os.getpid(),
        {"PATH": "/usr/bin"},
    )

    assert resolved == f"/proc/{os.getpid()}/root/{library.lstrip('/')}"
    assert calls[0][:4] == [
        "docker",
        "exec",
        "task-container",
        "/opt/agent-venv/bin/python",
    ]


def test_harbor_container_resolution_matches_compose_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run(
        command: list[str],
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert env == {"PATH": "/usr/bin"}
        return subprocess.CompletedProcess(command, 0, "container-id\n", "")

    monkeypatch.setattr("hermes_cli.benchmarks.tracing.agentsight._run", run)

    assert docker_compose_project_name("_Run.ID") == "0_run-id"
    assert (
        resolve_docker_compose_main_container(
            "_Run.ID",
            {"PATH": "/usr/bin"},
        )
        == "container-id"
    )
    assert "label=com.docker.compose.project=0_run-id" in calls[0]
    assert "label=com.docker.compose.service=main" in calls[0]


def test_collector_exit_without_terminal_health_is_not_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = AgentSightProfiler(
        AgentSightTarget(
            attempt_dir=tmp_path,
            profile_id="trace-missing-health",
            container_id="container",
        ),
        env={},
    )
    profiler.sources_dir.mkdir(parents=True)

    def stop_docker() -> None:
        profiler._source_status["task-container"] = {
            "status": "stopped",
            "complete": True,
            "exitCode": 0,
        }

    monkeypatch.setattr(profiler, "_stop_docker", stop_docker)
    profiler.finish()

    health = json.loads((profiler.directory / "health.json").read_text())
    assert health["status"] == "unavailable"
    assert health["complete"] is False
    assert (
        health["sources"]["task-container"]["reason"]
        == "collector health record is missing"
    )


def test_disabled_profiler_records_a_nonfatal_unavailable_profile(
    tmp_path: Path,
) -> None:
    profiler = AgentSightProfiler.start(
        AgentSightTarget(
            attempt_dir=tmp_path,
            profile_id="trace-disabled",
            host_pid=42,
            container_id="container",
        ),
        env={
            "BENCHMARK_AGENTSIGHT": "disabled",
            "BENCHMARK_AGENTSIGHT_STRICT": "1",
        },
    )
    profiler.finish()

    profile = json.loads(
        (tmp_path / "profiles" / "agentsight" / "profile.json").read_text()
    )
    health = json.loads(
        (tmp_path / "profiles" / "agentsight" / "health.json").read_text()
    )
    assert profile["schema"] == "benchmark-agentsight-profile/v1"
    assert profile["status"] == "unavailable"
    assert health["complete"] is False
    assert health["sources"]["host"]["reason"] == "disabled"
    assert health["sources"]["task-container"]["reason"] == "disabled"


def test_readiness_must_match_profile_and_scope(tmp_path: Path) -> None:
    ready = tmp_path / "ready.json"
    ready.write_text(
        json.dumps({
            "schema": "agentsight-capture-ready/v1",
            "profile_id": "old-profile",
            "scope_id": "host",
        })
    )

    assert not _wait_for_ready(
        ready,
        0.01,
        profile_id="new-profile",
        scope_id="host",
    )

    ready.write_text(
        json.dumps({
            "schema": "agentsight-capture-ready/v1",
            "profile_id": "new-profile",
            "scope_id": "host",
        })
    )
    assert _wait_for_ready(
        ready,
        0.01,
        profile_id="new-profile",
        scope_id="host",
    )


def test_strict_mode_requires_every_expected_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def start_host(profiler: AgentSightProfiler) -> None:
        profiler._source_status["host"] = {
            "status": "capturing",
            "complete": False,
        }

    def start_docker(profiler: AgentSightProfiler) -> None:
        profiler._source_status["task-container"] = {
            "status": "unavailable",
            "complete": False,
            "reason": "probe unavailable",
        }

    monkeypatch.setattr(AgentSightProfiler, "_start_host", start_host)
    monkeypatch.setattr(AgentSightProfiler, "_start_docker", start_docker)

    with pytest.raises(RuntimeError, match="task-container"):
        AgentSightProfiler.start(
            AgentSightTarget(
                attempt_dir=tmp_path,
                profile_id="trace-strict",
                host_pid=42,
                container_id="container",
            ),
            env={"BENCHMARK_AGENTSIGHT_STRICT": "1"},
        )


def test_strict_mode_requires_requested_container_tls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def start_docker(profiler: AgentSightProfiler) -> None:
        profiler._source_status["task-container"] = {
            "status": "capturing",
            "complete": False,
        }

    monkeypatch.setattr(AgentSightProfiler, "_start_docker", start_docker)

    with pytest.raises(RuntimeError, match="task-container-tls"):
        AgentSightProfiler.start(
            AgentSightTarget(
                attempt_dir=tmp_path,
                profile_id="trace-strict-tls",
                container_id="container",
                capture_tls=True,
                tls_python_path="/opt/agent-venv/bin/python",
            ),
            env={"BENCHMARK_AGENTSIGHT_STRICT": "1"},
        )


def test_strict_mode_requires_requested_host_tls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def start_host(profiler: AgentSightProfiler) -> None:
        profiler._source_status["host"] = {
            "status": "capturing",
            "complete": False,
        }

    monkeypatch.setattr(AgentSightProfiler, "_start_host", start_host)

    with pytest.raises(RuntimeError, match="host-tls"):
        AgentSightProfiler.start(
            AgentSightTarget(
                attempt_dir=tmp_path,
                profile_id="trace-strict-host-tls",
                host_pid=42,
                capture_host_tls=True,
            ),
            env={"BENCHMARK_AGENTSIGHT_STRICT": "1"},
        )


def test_unexpected_start_failure_is_nonfatal_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_start(_profiler: AgentSightProfiler) -> None:
        raise OSError("synthetic failure")

    monkeypatch.setattr(AgentSightProfiler, "_start", fail_start)
    profiler = AgentSightProfiler.start(
        AgentSightTarget(
            attempt_dir=tmp_path,
            profile_id="trace-best-effort",
            host_pid=42,
        ),
        env={},
    )
    profiler.finish()

    health = json.loads(
        (tmp_path / "profiles" / "agentsight" / "health.json").read_text()
    )
    assert health["status"] == "unavailable"
    assert health["reason"] == "profiler initialization failed: OSError"
