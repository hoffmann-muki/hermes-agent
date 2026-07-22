"""Harbor adapter for the parity-configured Hermes Terminal-Bench worker."""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import override
from urllib.parse import urlsplit

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.hermes import Hermes
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class BenchmarkHermes(Hermes):
    """Install Hermes, then run its isolated benchmark coordinator."""

    def __init__(
        self,
        logs_dir: Path,
        prompt_template_path: Path | str | None = None,
        repository: str = "https://github.com/NousResearch/hermes-agent.git",
        commit: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        parsed_repository = urlsplit(repository)
        if not (
            parsed_repository.scheme == "https"
            and parsed_repository.hostname
            and parsed_repository.path not in {"", "/"}
            and parsed_repository.username is None
            and parsed_repository.password is None
            and not parsed_repository.query
            and not parsed_repository.fragment
        ):
            raise ValueError("repository must be a credential-free HTTPS Git URL")
        if commit is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            raise ValueError("commit must be a full 40-character Git commit")
        self._repository = repository
        self._commit = commit
        if prompt_template_path is None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            prompt_template_path = logs_dir / "multiagent-prompt.j2"
            prompt_template_path.write_text("{{ instruction }}\n", encoding="utf-8")
        super().__init__(
            logs_dir=logs_dir,
            prompt_template_path=prompt_template_path,
            *args,
            **kwargs,
        )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if not self._version:
            raise ValueError("A Hermes branch or tag is required")
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl git ripgrep xz-utils",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        commit_flag = f" --commit {shlex.quote(self._commit)}" if self._commit else ""
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                'INSTALL_DIR="$HOME/.hermes/hermes-agent"; '
                'mkdir -p "$(dirname "$INSTALL_DIR")"; '
                f"git clone --filter=blob:none --branch {shlex.quote(self._version)} "
                f'{shlex.quote(self._repository)} "$INSTALL_DIR"; '
                'bash "$INSTALL_DIR/scripts/install.sh" --skip-setup '
                f"--branch {shlex.quote(self._version)}{commit_flag} "
                '--dir "$INSTALL_DIR"; '
                'export PATH="$HOME/.local/bin:$PATH"; '
                "hermes version"
            ),
        )
        worker_source = Path(__file__).with_name("terminalbench_worker.py")
        worker_copy = self.logs_dir / "terminalbench_worker.py"
        worker_copy.write_text(
            worker_source.read_text(encoding="utf-8"), encoding="utf-8"
        )
        await environment.upload_file(
            source_path=worker_copy,
            target_path="/installed-agent/terminalbench_worker.py",
        )
        result = await environment.exec(
            command="chmod 0555 /installed-agent/terminalbench_worker.py",
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to install the Hermes Terminal-Bench worker: {result.stderr}"
            )

    @with_prompt_template
    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        if not self.model_name or not self.model_name.startswith("openrouter/"):
            raise ValueError("Benchmark Hermes requires an openrouter/<model> model")
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

        env = {
            "HARBOR_INSTRUCTION": instruction,
            "HERMES_BENCHMARK_MODEL": self.model_name.removeprefix("openrouter/"),
            "HERMES_HOME": "/tmp/hermes",
            "OPENROUTER_API_KEY": api_key,
        }
        command = """set -e
export PATH="$HOME/.local/bin:$PATH"
HERMES_LAUNCHER="$(command -v hermes)"
HERMES_ENTRY="$(grep '^exec "' "$HERMES_LAUNCHER" | head -n 1 | cut -d'"' -f2 || true)"
if [ -z "$HERMES_ENTRY" ]; then
  HERMES_ENTRY="$(readlink -f "$HERMES_LAUNCHER")"
fi
"$(dirname "$HERMES_ENTRY")/python" /installed-agent/terminalbench_worker.py \
  2>&1 | stdbuf -oL tee /logs/agent/hermes.txt"""
        # BaseInstalledAgent._exec includes per-command environment values in
        # debug metadata. Invoke the environment directly so credentials never
        # enter Harbor's agent log record.
        result = await environment.exec(
            command=f"set -o pipefail; {command}",
            env=env,
        )
        if result.return_code != 0:
            raise self._classify_exec_error(command, result)

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)
        result_path = self.logs_dir / "hermes-result.json"
        if not result_path.exists():
            return
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        usage = result.get("usage")
        if not isinstance(usage, dict):
            return
        context.n_input_tokens = int(usage.get("input_tokens") or 0)
        context.n_cache_tokens = int(usage.get("cache_read_tokens") or 0)
        context.n_output_tokens = int(usage.get("output_tokens") or 0)
        context.cost_usd = float(usage.get("estimated_cost_usd") or 0)
        context.metadata = {
            **(context.metadata or {}),
            "workflow_complete": bool(result.get("workflowComplete")),
            "phase_order": result.get("phaseOrder"),
            "phase_budgets": result.get("phaseBudgets"),
        }
