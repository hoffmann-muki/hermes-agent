"""Credential-file handoff for Harbor-managed benchmark containers."""

from __future__ import annotations

import os
import re
import shlex
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment


ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


async def stage_secret_environment(
    environment: BaseEnvironment,
    staging_dir: Path,
    values: Mapping[str, str],
) -> str:
    """Upload a mode-0400 shell environment file without putting values in argv."""

    if not values:
        raise ValueError("secret environment cannot be empty")
    if any(not ENVIRONMENT_NAME.fullmatch(name) for name in values):
        raise ValueError("secret environment contains an invalid variable name")
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise ValueError("secret environment values must be non-empty strings")

    identity = await environment.exec(command="id -u; id -g")
    parts = (identity.stdout or "").splitlines() if identity.return_code == 0 else []
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise RuntimeError("could not resolve the Harbor agent identity")

    staging_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = f".benchmark-secret-env-{uuid.uuid4().hex}"
    local_path = staging_dir / name
    target_path = f"/installed-agent/{name}"
    descriptor = os.open(
        local_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as secret:
            secret.write(
                "".join(
                    f"export {key}={shlex.quote(value)}\n"
                    for key, value in sorted(values.items())
                )
            )
        try:
            await environment.upload_file(
                source_path=local_path,
                target_path=target_path,
            )
        except BaseException:
            await remove_secret_environment(environment, target_path)
            raise
    finally:
        local_path.unlink(missing_ok=True)

    try:
        secured = await environment.exec(
            command=(
                f"chown {parts[0]}:{parts[1]} {shlex.quote(target_path)} && "
                f"chmod 0400 {shlex.quote(target_path)}"
            ),
            user="root",
        )
    except BaseException:
        await remove_secret_environment(environment, target_path)
        raise
    if secured.return_code == 0:
        return target_path
    await remove_secret_environment(environment, target_path)
    raise RuntimeError("could not secure the Harbor credential handoff")


async def remove_secret_environment(
    environment: BaseEnvironment,
    target_path: str,
) -> None:
    """Best-effort removal for a staged credential file."""

    try:
        await environment.exec(
            command=f"rm -f -- {shlex.quote(target_path)}",
            user="root",
        )
    except Exception:
        pass


def source_secret_environment(target_path: str) -> str:
    """Return shell setup that imports and immediately unlinks staged secrets."""

    quoted = shlex.quote(target_path)
    return f". {quoted}\nrm -f -- {quoted}\n"
