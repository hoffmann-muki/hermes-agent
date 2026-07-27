"""Privilege acquisition shared by tracing and profiling companions."""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping, Sequence


DEFAULT_SUDO_PASSWORD_FILE = Path("~/.config/benchmark-tools/sudo-password")
DEFAULT_SUDO_TIMEOUT_SECONDS = 15.0


class SudoAuthorizationError(RuntimeError):
    """Raised when a privileged tracing companion cannot be authorized."""


@dataclass(frozen=True, slots=True)
class SudoAuthorization:
    """A validated prefix for one privileged child process."""

    command_prefix: tuple[str, ...]
    method: str
    password_file: Path | None = None


def authorize_sudo(
    env: Mapping[str, str] | None = None,
) -> SudoAuthorization:
    """Validate sudo once without exposing credentials to argv or the environment."""

    effective_env = dict(os.environ if env is None else env)
    if os.geteuid() == 0:
        return SudoAuthorization(
            command_prefix=(),
            method="root",
        )

    sudo = shutil.which("sudo", path=effective_env.get("PATH"))
    if sudo is None:
        raise SudoAuthorizationError("sudo executable is not available")

    configured = effective_env.get("BENCHMARK_SUDO_PASSWORD_FILE", "").strip()
    password_file = Path(configured).expanduser() if configured else (
        DEFAULT_SUDO_PASSWORD_FILE.expanduser()
    )
    timeout = _positive_float(
        effective_env.get("BENCHMARK_SUDO_TIMEOUT_SECONDS"),
        DEFAULT_SUDO_TIMEOUT_SECONDS,
    )

    try:
        secret = _open_password_file(password_file)
    except FileNotFoundError:
        result = _run_sudo_validation(
            [sudo, "-n", "-v"],
            env=effective_env,
            timeout=timeout,
        )
        if result.returncode == 0:
            return SudoAuthorization(
                command_prefix=(sudo, "-n", "--"),
                method="sudo-cache",
            )
        raise SudoAuthorizationError(
            "sudo authentication is unavailable and the configured password "
            "file does not exist"
        ) from None
    except OSError as error:
        raise SudoAuthorizationError(
            f"sudo password file could not be opened: {type(error).__name__}"
        ) from None

    try:
        _validate_password_file(secret, password_file)
        result = _run_sudo_validation(
            [sudo, "-S", "-p", "", "-v"],
            env=effective_env,
            timeout=timeout,
            stdin=secret,
        )
    finally:
        secret.close()

    if result.returncode != 0:
        raise SudoAuthorizationError("sudo password validation failed")
    return SudoAuthorization(
        command_prefix=(sudo, "-S", "-p", "", "--"),
        method="sudo-password-file",
        password_file=password_file,
    )


def build_sudo_supervised_command(
    authorization: SudoAuthorization,
    command: Sequence[str],
    *,
    stop_file: Path,
    stop_timeout: float,
) -> list[str]:
    """Wrap a command in a root supervisor controlled by a credential-free file."""

    if not command:
        raise ValueError("sudo supervisor requires a command")
    return [
        *authorization.command_prefix,
        sys.executable,
        "-S",
        str(Path(__file__).resolve()),
        "_supervise",
        str(stop_file),
        str(stop_timeout),
        *command,
    ]


@contextmanager
def authorized_sudo_stdin(
    authorization: SudoAuthorization,
) -> Iterator[BinaryIO | int]:
    """Provide stdin for an authorized command without materializing the password."""

    if authorization.password_file is None:
        yield subprocess.DEVNULL
        return

    try:
        secret = _open_password_file(authorization.password_file)
    except OSError as error:
        raise SudoAuthorizationError(
            f"sudo password file could not be reopened: {type(error).__name__}"
        ) from None
    try:
        _validate_password_file(secret, authorization.password_file)
        yield secret
    finally:
        secret.close()


def _supervise(stop_file: Path, stop_timeout: float, command: Sequence[str]) -> int:
    """Run one privileged child until its unprivileged owner requests shutdown."""

    try:
        os.close(0)
    except OSError:
        pass
    os.open(os.devnull, os.O_RDONLY)
    interrupted = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        while process.poll() is None and not interrupted and not stop_file.exists():
            time.sleep(0.1)
        requested_stop = interrupted or stop_file.exists()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=stop_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        return 0 if requested_stop else process.returncode or 0
    finally:
        stop_file.unlink(missing_ok=True)


def _main() -> int:
    if len(sys.argv) < 5 or sys.argv[1] != "_supervise":
        return 64
    try:
        timeout = float(sys.argv[3])
    except ValueError:
        return 64
    if timeout <= 0:
        return 64
    return _supervise(Path(sys.argv[2]), timeout, sys.argv[4:])


def _open_password_file(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    return os.fdopen(os.open(path, flags), "rb")


def _validate_password_file(secret: BinaryIO, path: Path) -> None:
    metadata = os.fstat(secret.fileno())
    if not stat.S_ISREG(metadata.st_mode):
        raise SudoAuthorizationError("sudo password file is not a regular file")
    if metadata.st_uid != os.getuid():
        raise SudoAuthorizationError(
            "sudo password file must be owned by the benchmark user"
        )
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise SudoAuthorizationError(
            "sudo password file permissions must not grant group or other access"
        )
    if metadata.st_size <= 0 or metadata.st_size > 4096:
        raise SudoAuthorizationError(
            f"sudo password file has an invalid size: {path}"
        )


def _run_sudo_validation(
    args: list[str],
    *,
    env: Mapping[str, str],
    timeout: float,
    stdin: BinaryIO | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            args,
            stdin=stdin if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise SudoAuthorizationError("sudo password validation timed out") from None
    except OSError as error:
        raise SudoAuthorizationError(
            f"sudo password validation could not run: {type(error).__name__}"
        ) from None


def _positive_float(value: str | None, fallback: float) -> float:
    try:
        parsed = float(value) if value else fallback
    except ValueError:
        return fallback
    return parsed if parsed > 0 else fallback


if __name__ == "__main__":
    raise SystemExit(_main())
