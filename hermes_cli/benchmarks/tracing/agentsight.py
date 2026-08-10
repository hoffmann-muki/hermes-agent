"""Benchmark-independent AgentSight profiling companion.

This adapter maps a benchmark worker's host process and task container into
AgentSight collector scopes. It does not instrument Hermes core behavior.
"""

from __future__ import annotations

import json
import os
import shutil
import ssl
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from hermes_cli.benchmarks.tracing.privileges import (
    SudoAuthorizationError,
    authorized_sudo_stdin,
    authorize_sudo,
    build_sudo_supervised_command,
)


PROFILE_SCHEMA = "benchmark-agentsight-profile/v1"
HEALTH_SCHEMA = "benchmark-agentsight-health/v1"
DEFAULT_IMAGE = "agentsight:play"
DEFAULT_READY_TIMEOUT_SECONDS = 30.0
DEFAULT_STOP_TIMEOUT_SECONDS = 15


@dataclass(frozen=True, slots=True)
class AgentSightTarget:
    attempt_dir: Path
    profile_id: str
    host_pid: int | None = None
    container_id: str | None = None
    capture_host_tls: bool = False
    capture_tls: bool = False
    tls_python_path: str | None = None
    correlation: Mapping[str, Any] | None = None


@dataclass(slots=True)
class _HostCollector:
    process: subprocess.Popen[bytes]
    stdout: BinaryIO
    stderr: BinaryIO
    stop_file: Path


@dataclass(frozen=True, slots=True)
class _DockerCollector:
    sidecar: str
    source_dir: Path


class AgentSightProfiler:
    """Manage all collector planes for one benchmark attempt."""

    def __init__(
        self,
        target: AgentSightTarget,
        *,
        env: dict[str, str] | None = None,
    ) -> None:
        self.target = target
        self.env = dict(os.environ if env is None else env)
        self.directory = target.attempt_dir / "profiles" / "agentsight"
        self.sources_dir = self.directory / "sources"
        self.image = self.env.get("AGENTSIGHT_IMAGE", "").strip() or DEFAULT_IMAGE
        self.image_id: str | None = None
        self.strict = self.env.get("BENCHMARK_AGENTSIGHT_STRICT") == "1"
        self.ready_timeout = _positive_float(
            self.env.get("AGENTSIGHT_READY_TIMEOUT_SECONDS"),
            DEFAULT_READY_TIMEOUT_SECONDS,
        )
        self.stop_timeout = int(
            _positive_float(
                self.env.get("AGENTSIGHT_STOP_TIMEOUT_SECONDS"),
                DEFAULT_STOP_TIMEOUT_SECONDS,
            )
        )
        self.started_at = time.time()
        self._host: _HostCollector | None = None
        self._docker: _DockerCollector | None = None
        self._source_status: dict[str, dict[str, Any]] = {}
        self._host_privilege: str | None = None
        self._host_tls_binary_path: str | None = None
        self._host_tls_error: str | None = None
        self._container_tls_binary_path: str | None = None
        self._container_tls_error: str | None = None
        self._finished = False

    @classmethod
    def start(
        cls,
        target: AgentSightTarget,
        *,
        env: dict[str, str] | None = None,
    ) -> AgentSightProfiler:
        profiler = cls(target, env=env)
        try:
            profiler._start()
        except Exception as error:
            try:
                profiler._stop_host()
            except Exception:
                pass
            try:
                profiler._stop_docker()
            except Exception:
                pass
            if profiler.strict:
                profiler._finished = True
                raise
            reason = f"profiler initialization failed: {type(error).__name__}"
            try:
                profiler._mark_unavailable(reason)
            except OSError:
                pass
            profiler._finished = True
        return profiler

    @classmethod
    def unavailable(
        cls,
        target: AgentSightTarget,
        reason: str,
        *,
        expected_scopes: tuple[str, ...],
        env: dict[str, str] | None = None,
    ) -> AgentSightProfiler:
        """Persist a non-capturing profile when a runtime scope cannot resolve."""

        profiler = cls(target, env=env)
        profiler._mark_unavailable(reason, expected_scopes=expected_scopes)
        profiler._finished = True
        if profiler.strict and not _disabled(profiler.env):
            raise RuntimeError(f"AgentSight {reason}")
        return profiler

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        for scope, stop in (
            ("host", self._stop_host),
            ("task-container", self._stop_docker),
        ):
            try:
                stop()
            except Exception as error:
                self._source_status[scope] = {
                    "status": "unavailable",
                    "complete": False,
                    "reason": (
                        f"collector finalization failed: {type(error).__name__}"
                    ),
                }
        source_health = {
            scope: health
            if (health := _read_object(self.sources_dir / scope / "health.json"))
            is not None
            else (
                {
                    **source_status,
                    "complete": False,
                    "reason": "collector health record is missing",
                }
                if source_status.get("complete") is True
                else source_status
            )
            for scope, source_status in self._source_status.items()
        }
        expected = self._expected_scopes()
        completed = [
            scope
            for scope, health in source_health.items()
            if health.get("complete") is True
        ]
        tls_degraded = (
            self.target.host_pid is not None
            and self.target.capture_host_tls
            and self._host_tls_binary_path is None
        ) or (
            self.target.container_id is not None
            and self.target.capture_tls
            and self._container_tls_binary_path is None
        )
        status = (
            "completed"
            if expected and set(completed) == set(expected) and not tls_degraded
            else "degraded"
            if completed
            else "unavailable"
        )
        self._write_root_profile(status=status, source_scopes=completed)
        _write_json_atomic(
            self.directory / "health.json",
            {
                "schema": HEALTH_SCHEMA,
                "profileId": self.target.profile_id,
                "status": status,
                "complete": status == "completed",
                "sources": source_health,
                "hostTls": self._host_tls_health(),
                "containerTls": self._container_tls_health(),
            },
        )
        _write_json_atomic(
            self.directory / "summary.json",
            {
                "status": status,
                "complete": status == "completed",
                "sources": {
                    scope: _summarize_health(health)
                    for scope, health in source_health.items()
                },
                "hostTls": self._host_tls_health(),
                "containerTls": self._container_tls_health(),
            },
        )
        if self.strict and not _disabled(self.env) and status != "completed":
            raise RuntimeError(
                f"AgentSight profile {self.target.profile_id} finished as {status}"
            )

    def _start(self) -> None:
        self.sources_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if _disabled(self.env):
            for scope in self._expected_scopes():
                self._source_status[scope] = {
                    "status": "unavailable",
                    "complete": False,
                    "reason": "disabled",
                }
            self._write_root_profile(
                status="unavailable",
                source_scopes=[],
                reason="disabled",
            )
            return
        if self.target.host_pid is not None:
            self._start_host()
        if self.target.container_id is not None:
            self._start_docker()
        ready = [
            scope
            for scope, status in self._source_status.items()
            if status.get("status") == "capturing"
        ]
        self._write_root_profile(
            status="capturing" if ready else "unavailable",
            source_scopes=ready,
        )
        expected = self._expected_scopes()
        host_tls_missing = (
            self.target.host_pid is not None
            and self.target.capture_host_tls
            and self._host_tls_binary_path is None
        )
        container_tls_missing = (
            self.target.container_id is not None
            and self.target.capture_tls
            and self._container_tls_binary_path is None
        )
        if self.strict and (
            set(ready) != set(expected) or host_tls_missing or container_tls_missing
        ):
            missing = sorted(set(expected) - set(ready))
            if host_tls_missing:
                missing.append("host-tls")
            if container_tls_missing:
                missing.append("task-container-tls")
            try:
                self.finish()
            except RuntimeError:
                pass
            raise RuntimeError(
                "AgentSight could not start required collector scopes: "
                + ", ".join(missing)
            )

    def _expected_scopes(self) -> list[str]:
        return [
            scope
            for scope, present in (
                ("host", self.target.host_pid is not None),
                ("task-container", self.target.container_id is not None),
            )
            if present
        ]

    def _container_tls_health(self) -> dict[str, Any]:
        if self.target.container_id is None or not self.target.capture_tls:
            return {"requested": False, "active": False}
        return {
            "requested": True,
            "active": self._container_tls_binary_path is not None,
            **(
                {"binaryPath": self._container_tls_binary_path}
                if self._container_tls_binary_path is not None
                else {}
            ),
            **(
                {"reason": self._container_tls_error}
                if self._container_tls_error is not None
                else {}
            ),
        }

    def _host_tls_health(self) -> dict[str, Any]:
        if self.target.host_pid is None or not self.target.capture_host_tls:
            return {"requested": False, "active": False}
        return {
            "requested": True,
            "active": self._host_tls_binary_path is not None,
            **(
                {"binaryPath": self._host_tls_binary_path}
                if self._host_tls_binary_path is not None
                else {}
            ),
            **(
                {"reason": self._host_tls_error}
                if self._host_tls_error is not None
                else {}
            ),
        }

    def _mark_unavailable(
        self,
        reason: str,
        *,
        expected_scopes: tuple[str, ...] | None = None,
    ) -> None:
        if self.target.capture_host_tls and self._host_tls_error is None:
            self._host_tls_error = reason
        if self.target.capture_tls and self._container_tls_error is None:
            self._container_tls_error = reason
        scopes = expected_scopes or tuple(self._expected_scopes())
        sources = {
            scope: {
                "status": "unavailable",
                "complete": False,
                "reason": reason,
            }
            for scope in scopes
        }
        self._source_status = sources
        self._write_root_profile(
            status="unavailable",
            source_scopes=[],
            reason=reason,
        )
        _write_json_atomic(
            self.directory / "health.json",
            {
                "schema": HEALTH_SCHEMA,
                "profileId": self.target.profile_id,
                "status": "unavailable",
                "complete": False,
                "reason": reason,
                "sources": sources,
                "hostTls": self._host_tls_health(),
                "containerTls": self._container_tls_health(),
            },
        )
        _write_json_atomic(
            self.directory / "summary.json",
            {
                "status": "unavailable",
                "complete": False,
                "reason": reason,
                "sources": {
                    scope: _summarize_health(status)
                    for scope, status in sources.items()
                },
                "hostTls": self._host_tls_health(),
                "containerTls": self._container_tls_health(),
            },
        )

    def _start_host(self) -> None:
        source_dir = self.sources_dir / "host"
        source_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        binary = self.env.get("AGENTSIGHT_BIN", "").strip() or shutil.which(
            "agentsight"
        )
        if not binary:
            self._source_status["host"] = {
                "status": "unavailable",
                "complete": False,
                "reason": "agentsight executable is not on PATH; set AGENTSIGHT_BIN",
            }
            return
        try:
            authorization = authorize_sudo(self.env)
        except SudoAuthorizationError as error:
            self._source_status["host"] = {
                "status": "unavailable",
                "complete": False,
                "reason": f"host collector privilege authorization failed: {error}",
            }
            return
        self._host_privilege = authorization.method
        if self.target.capture_host_tls:
            try:
                self._host_tls_binary_path = resolve_host_tls_binary(
                    self.target.host_pid
                )
            except (OSError, RuntimeError, ValueError) as error:
                self._host_tls_error = (
                    f"TLS binary resolution failed: {type(error).__name__}"
                )
        stdout = None
        try:
            stdout = (source_dir / "collector.stdout.log").open("wb")
            stderr = (source_dir / "collector.stderr.log").open("wb")
        except OSError as error:
            if stdout is not None:
                stdout.close()
            self._source_status["host"] = {
                "status": "unavailable",
                "complete": False,
                "reason": f"host collector log setup failed: {type(error).__name__}",
            }
            return
        try:
            with authorized_sudo_stdin(authorization) as stdin:
                collector_args = build_host_collector_args(
                    binary=binary,
                    source_dir=source_dir,
                    profile_id=self.target.profile_id,
                    pid=self.target.host_pid,
                    tls_binary_path=self._host_tls_binary_path,
                )
                stop_file = source_dir / "stop"
                stop_file.unlink(missing_ok=True)
                process = subprocess.Popen(
                    build_sudo_supervised_command(
                        authorization,
                        collector_args,
                        owner_pid=self.target.host_pid,
                        stop_file=stop_file,
                        stop_timeout=self.stop_timeout,
                    ),
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    env=self.env,
                    start_new_session=True,
                )
        except (OSError, SudoAuthorizationError) as error:
            stdout.close()
            stderr.close()
            self._source_status["host"] = {
                "status": "unavailable",
                "complete": False,
                "reason": f"host collector launch failed: {error}",
            }
            return
        if not _wait_for_ready(
            source_dir / "ready.json",
            self.ready_timeout,
            process,
            profile_id=self.target.profile_id,
            scope_id="host",
        ):
            _terminate_process(process, self.stop_timeout, stop_file)
            stdout.close()
            stderr.close()
            self._source_status["host"] = {
                "status": "unavailable",
                "complete": False,
                "reason": (
                    "host collector did not become ready after privilege authorization"
                ),
            }
            return
        self._host = _HostCollector(
            process=process,
            stdout=stdout,
            stderr=stderr,
            stop_file=stop_file,
        )
        self._source_status["host"] = {
            "status": "capturing",
            "complete": False,
        }

    def _start_docker(self) -> None:
        source_dir = self.sources_dir / "task-container"
        source_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        image = _run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", self.image],
            self.env,
        )
        if image.returncode != 0:
            self._source_status["task-container"] = {
                "status": "unavailable",
                "complete": False,
                "reason": (
                    f"image unavailable: {self.image}; build the AgentSight play "
                    "image or set AGENTSIGHT_IMAGE"
                ),
            }
            return
        self.image_id = image.stdout.strip()
        inspected = _run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Pid}}",
                str(self.target.container_id),
            ],
            self.env,
        )
        try:
            init_pid = int(inspected.stdout.strip())
        except ValueError:
            init_pid = 0
        if inspected.returncode != 0 or init_pid <= 0:
            self._source_status["task-container"] = {
                "status": "unavailable",
                "complete": False,
                "reason": (
                    "could not resolve task-container init PID: "
                    f"{inspected.stderr.strip() or inspected.stdout.strip()}"
                ),
            }
            return
        if self.target.capture_tls:
            try:
                self._container_tls_binary_path = resolve_container_tls_binary(
                    str(self.target.container_id),
                    self.target.tls_python_path,
                    init_pid,
                    self.env,
                )
            except (OSError, RuntimeError, ValueError) as error:
                self._container_tls_error = (
                    f"TLS binary resolution failed: {type(error).__name__}"
                )
        sidecar = _sidecar_name(self.target.profile_id, "task-container")
        _run(["docker", "rm", "--force", sidecar], self.env)
        launched = _run(
            build_docker_sidecar_args(
                image=self.image,
                sidecar=sidecar,
                source_dir=source_dir,
                profile_id=self.target.profile_id,
                init_pid=init_pid,
                stop_timeout=self.stop_timeout,
                tls_binary_path=self._container_tls_binary_path,
            ),
            self.env,
        )
        if launched.returncode != 0:
            self._source_status["task-container"] = {
                "status": "unavailable",
                "complete": False,
                "reason": (
                    "task-container collector launch failed: "
                    f"{launched.stderr.strip() or launched.stdout.strip()}"
                ),
            }
            return
        if not _wait_for_ready(
            source_dir / "ready.json",
            self.ready_timeout,
            profile_id=self.target.profile_id,
            scope_id="task-container",
        ):
            _run(
                ["docker", "stop", "--time", str(self.stop_timeout), sidecar],
                self.env,
            )
            logs = _run(["docker", "logs", sidecar], self.env)
            _write_text_atomic(
                source_dir / "collector.log",
                f"{logs.stdout}{logs.stderr}",
            )
            _run(["docker", "rm", "--force", sidecar], self.env)
            self._source_status["task-container"] = {
                "status": "unavailable",
                "complete": False,
                "reason": (
                    "task-container collector did not become ready within "
                    f"{self.ready_timeout:g} seconds"
                ),
            }
            return
        self._docker = _DockerCollector(sidecar=sidecar, source_dir=source_dir)
        self._source_status["task-container"] = {
            "status": "capturing",
            "complete": False,
        }

    def _stop_host(self) -> None:
        if self._host is None:
            return
        collector = self._host
        self._host = None
        try:
            _terminate_process(
                collector.process,
                self.stop_timeout,
                collector.stop_file,
            )
        finally:
            collector.stdout.close()
            collector.stderr.close()
        self._source_status["host"] = {
            "status": "stopped",
            "complete": collector.process.returncode == 0,
            "exitCode": collector.process.returncode,
        }

    def _stop_docker(self) -> None:
        if self._docker is None:
            return
        collector = self._docker
        self._docker = None
        stopped = _run(
            [
                "docker",
                "stop",
                "--time",
                str(self.stop_timeout),
                collector.sidecar,
            ],
            self.env,
        )
        logs = _run(["docker", "logs", collector.sidecar], self.env)
        try:
            _write_text_atomic(
                collector.source_dir / "collector.log",
                f"{logs.stdout}{logs.stderr}",
            )
        finally:
            _run(["docker", "rm", "--force", collector.sidecar], self.env)
        self._source_status["task-container"] = {
            "status": "stopped",
            "complete": stopped.returncode == 0,
            "exitCode": stopped.returncode,
        }

    def _write_root_profile(
        self,
        *,
        status: str,
        source_scopes: list[str],
        reason: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema": PROFILE_SCHEMA,
            "profileId": self.target.profile_id,
            "status": status,
            "topology": _topology(self.target),
            "sourceScopes": source_scopes,
            "startedAt": _iso(self.started_at),
            "updatedAt": _iso(time.time()),
            "hostPid": self.target.host_pid,
            "targetContainer": self.target.container_id,
            "image": self.image,
            "imageId": self.image_id,
            "hostPrivilege": self._host_privilege,
            "captureTls": (
                (
                    self.target.capture_host_tls
                    and self._host_tls_binary_path is not None
                )
                or (
                    self.target.capture_tls
                    and self._container_tls_binary_path is not None
                )
            ),
            "hostTls": self._host_tls_health(),
            "containerTls": self._container_tls_health(),
        }
        if self.target.correlation is not None:
            payload["correlation"] = dict(self.target.correlation)
        if reason:
            payload["reason"] = reason
        _write_json_atomic(self.directory / "profile.json", payload)


def build_host_collector_args(
    *,
    binary: str,
    source_dir: Path,
    profile_id: str,
    pid: int | None,
    tls_binary_path: str | None = None,
) -> list[str]:
    if pid is None or pid <= 0:
        raise ValueError("host AgentSight capture requires a positive PID")
    args = [
        binary,
        "record",
        "--pid",
        str(pid),
        "--capture-level",
        "research",
        "--profile-dir",
        str(source_dir),
        "--profile-id",
        profile_id,
        "--scope-id",
        "host",
        "--ready-file",
        str(source_dir / "ready.json"),
        "--no-server",
    ]
    if tls_binary_path is not None:
        args.extend(["--binary-path", tls_binary_path, "--tls-binary-only"])
    return args


def build_docker_sidecar_args(
    *,
    image: str,
    sidecar: str,
    source_dir: Path,
    profile_id: str,
    init_pid: int,
    stop_timeout: int,
    tls_binary_path: str | None = None,
) -> list[str]:
    record = [
        "record",
        "--pidns-filter",
        f"/proc/{init_pid}/ns/pid",
        "--capture-level",
        "research",
        "--profile-dir",
        "/output",
        "--profile-id",
        profile_id,
        "--scope-id",
        "task-container",
        "--ready-file",
        "/output/ready.json",
        "--no-server",
        "--no-stdio",
    ]
    if tls_binary_path is None:
        record.append("--no-ssl")
    else:
        record.extend([
            "--binary-path",
            tls_binary_path,
            "--tls-binary-only",
        ])
    return [
        "docker",
        "run",
        "--detach",
        "--name",
        sidecar,
        "--privileged",
        "--pid",
        "host",
        "--network",
        "none",
        "--stop-timeout",
        str(stop_timeout),
        "--volume",
        "/sys:/sys:ro",
        "--volume",
        f"{source_dir.resolve()}:/output",
        image,
        *record,
    ]


def resolve_container_tls_binary(
    container_id: str,
    python_path: str | None,
    init_pid: int,
    env: dict[str, str],
) -> str:
    """Resolve the container TLS-bearing library or Python executable."""

    if not container_id or not python_path or not python_path.startswith("/"):
        raise ValueError("container TLS discovery requires an absolute Python path")
    if init_pid <= 0:
        raise ValueError("container TLS discovery requires a positive init PID")
    result = _run(
        [
            "docker",
            "exec",
            container_id,
            python_path,
            "-c",
            (
                "import ssl,sys;"
                "from pathlib import Path;"
                "paths=sorted(set(line.split(None,5)[-1].removesuffix(' (deleted)') "
                "for line in Path('/proc/self/maps').read_text().splitlines() "
                "if '/libssl.so' in line));"
                "assert len(paths)<=1;"
                "binary=Path(paths[0] if paths else sys.executable).resolve();"
                "assert binary.is_file();"
                "print(binary)"
            ),
        ],
        env,
    )
    library = result.stdout.strip()
    if result.returncode != 0 or "\n" in library or not library.startswith("/"):
        raise RuntimeError("Python runtime did not expose one TLS-bearing binary")
    # Docker Desktop and WSL place dockerd in another VM/PID namespace. The
    # path is validated inside the target above and becomes visible only inside
    # the --pid=host AgentSight sidecar.
    return str(Path(f"/proc/{init_pid}/root") / library.removeprefix("/"))


def resolve_host_tls_binary(pid: int | None) -> str:
    """Resolve the TLS-bearing library or executable of a host Python worker."""

    if pid is None or pid <= 0:
        raise ValueError("host TLS discovery requires a positive PID")
    if pid == os.getpid():
        _ = ssl.OPENSSL_VERSION
    libraries = {
        line.split(None, 5)[-1].removesuffix(" (deleted)")
        for line in Path(f"/proc/{pid}/maps").read_text(encoding="utf-8").splitlines()
        if "/libssl.so" in line
    }
    if len(libraries) > 1:
        raise RuntimeError("Python runtime exposed multiple OpenSSL libraries")
    binary = (
        Path(libraries.pop())
        if libraries
        else Path(f"/proc/{pid}/exe").resolve(strict=True)
    )
    if not binary.is_absolute() or not binary.is_file():
        raise RuntimeError("host TLS-bearing binary is not accessible")
    return str(binary)


def resolve_docker_compose_main_container(
    session_id: str,
    env: dict[str, str] | None = None,
) -> str:
    """Resolve Harbor's running main service without Harbor-private APIs."""

    if not session_id:
        raise ValueError("Docker Compose session ID cannot be empty")
    effective_env = dict(os.environ if env is None else env)
    project = docker_compose_project_name(session_id)
    result = _run(
        [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--filter",
            "label=com.docker.compose.service=main",
            "--format",
            "{{.ID}}",
        ],
        effective_env,
    )
    containers = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or len(containers) != 1:
        detail = (
            result.stderr.strip() or result.stdout.strip() or "no matching container"
        )
        raise RuntimeError(
            f"could not resolve Harbor main container for {project}: {detail}"
        )
    return containers[0]


def docker_compose_project_name(value: str) -> str:
    """Match Docker Compose's project-name normalization used by Harbor."""

    normalized = value.lower()
    allowed_initial = "abcdefghijklmnopqrstuvwxyz0123456789"
    allowed = f"{allowed_initial}_-"
    if not normalized or normalized[0] not in allowed_initial:
        normalized = f"0{normalized}"
    return "".join(char if char in allowed else "-" for char in normalized)


def _wait_for_ready(
    path: Path,
    timeout: float,
    process: subprocess.Popen[bytes] | None = None,
    *,
    profile_id: str,
    scope_id: str,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready = _read_object(path)
        if (
            ready
            and ready.get("schema") == "agentsight-capture-ready/v1"
            and ready.get("profile_id") == profile_id
            and ready.get("scope_id") == scope_id
        ):
            return True
        if process is not None and process.poll() is not None:
            return False
        time.sleep(0.1)
    return False


def _terminate_process(
    process: subprocess.Popen[bytes],
    timeout: int,
    stop_file: Path,
) -> None:
    if process.poll() is not None:
        stop_file.unlink(missing_ok=True)
        return
    try:
        stop_file.touch(mode=0o600, exist_ok=True)
        process.wait(timeout=timeout + 7)
    finally:
        stop_file.unlink(missing_ok=True)


def _run(
    command: list[str],
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess(command, 127, "", str(error))


def _disabled(env: dict[str, str]) -> bool:
    return env.get("BENCHMARK_AGENTSIGHT", "").strip().lower() in {
        "0",
        "false",
        "off",
        "disabled",
    }


def _positive_float(value: str | None, fallback: float) -> float:
    try:
        parsed = float(value or "")
    except ValueError:
        return fallback
    return parsed if parsed > 0 else fallback


def _sidecar_name(profile_id: str, scope_id: str) -> str:
    value = "".join(
        char if char.isalnum() or char in "_.-" else "-"
        for char in f"agentsight-{profile_id}-{scope_id}".lower()
    )
    return value[:120].rstrip("-_.") or "agentsight-profile"


def _topology(target: AgentSightTarget) -> str:
    if target.host_pid is not None and target.container_id is not None:
        return "host-process-plus-docker-pid-host-sidecar"
    if target.container_id is not None:
        return "docker-pid-host-sidecar"
    if target.host_pid is not None:
        return "host-process"
    return "unresolved"


def _summarize_health(health: dict[str, Any]) -> dict[str, Any]:
    evidence = health.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return {
        "status": health.get("status", "missing"),
        "complete": health.get("complete") is True,
        "events": evidence.get("events_written", 0),
        "writeErrors": evidence.get("write_errors", 0),
        "eventsBySource": evidence.get("events_by_source", {}),
        "diagnosticsByType": evidence.get("diagnostics_by_type", {}),
    }


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _iso(timestamp: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
