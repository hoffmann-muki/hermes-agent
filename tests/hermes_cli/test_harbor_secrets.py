from __future__ import annotations

import stat
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.benchmarks.harbor_secrets import (
    remove_secret_environment,
    source_secret_environment,
    stage_secret_environment,
)


class FakeEnvironment:
    def __init__(self) -> None:
        self.commands: list[tuple[str, str | None]] = []
        self.uploads: list[tuple[str, str, str, int]] = []

    async def exec(self, command: str, user: str | None = None):
        self.commands.append((command, user))
        if command == "id -u; id -g":
            return SimpleNamespace(return_code=0, stdout="1000\n1001\n")
        return SimpleNamespace(return_code=0, stdout="")

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        self.uploads.append((
            str(source_path),
            target_path,
            source_path.read_text(encoding="utf-8"),
            stat.S_IMODE(source_path.stat().st_mode),
        ))


@pytest.mark.asyncio
async def test_secret_environment_uses_a_transient_file_not_command_arguments(
    tmp_path: Path,
) -> None:
    environment = FakeEnvironment()
    value = "synthetic secret with ' quoting"

    target = await stage_secret_environment(
        environment,
        tmp_path,
        {"OPENROUTER_API_KEY": value},
    )

    assert len(environment.uploads) == 1
    source, uploaded_target, content, mode = environment.uploads[0]
    assert uploaded_target == target
    assert content == f"export OPENROUTER_API_KEY={shlex.quote(value)}\n"
    assert mode == 0o600
    assert not Path(source).exists()
    assert all(value not in command for command, _user in environment.commands)
    assert value not in target
    assert value not in source_secret_environment(target)
    assert "chown 1000:1001" in environment.commands[1][0]
    assert "chmod 0400" in environment.commands[1][0]

    await remove_secret_environment(environment, target)
    assert environment.commands[-1][1] == "root"


@pytest.mark.asyncio
async def test_secret_environment_rejects_invalid_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid variable"):
        await stage_secret_environment(
            FakeEnvironment(),
            tmp_path,
            {"BAD-NAME": "value"},
        )
