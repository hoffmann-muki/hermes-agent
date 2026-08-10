"""Harbor adapter for the parity-configured Hermes Terminal-Bench worker."""

from __future__ import annotations

import asyncio
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

from hermes_cli.benchmarks.tracing.agentsight import AgentSightProfiler
from hermes_cli.benchmarks.tracing.harbor import (
    HarborTraceAttempt,
    allocate_harbor_trace_attempt,
    attach_harbor_agentsight_profile,
    promote_harbor_trace_attempt,
    start_harbor_agentsight_profile,
)


AGENT_TOPOLOGIES = {"supervisor-delegation", "single-agent"}


class BenchmarkHermes(Hermes):
    """Install Hermes, then run its isolated benchmark coordinator."""

    def __init__(
        self,
        logs_dir: Path,
        prompt_template_path: Path | str | None = None,
        repository: str = "https://github.com/NousResearch/hermes-agent.git",
        commit: str | None = None,
        agent_topology: str = "supervisor-delegation",
        trace_root: str | None = None,
        trace_run_id: str | None = None,
        trace_created_at: str | None = None,
        trace_benchmark: str | None = None,
        evaluation_workers: int = 1,
        benchmark_retries: int = 0,
        harbor_version: str = "unknown",
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
        if agent_topology not in AGENT_TOPOLOGIES:
            raise ValueError(
                "agent_topology must be supervisor-delegation or single-agent"
            )
        self._repository = repository
        self._commit = commit
        self._agent_topology = agent_topology
        trace_values = (
            trace_root,
            trace_run_id,
            trace_created_at,
            trace_benchmark,
        )
        if any(value is not None for value in trace_values) and not all(
            value is not None for value in trace_values
        ):
            raise ValueError("Hermes Harbor tracing requires complete metadata")
        if trace_root is not None and commit is None:
            raise ValueError("Hermes Harbor tracing requires an exact commit")
        if trace_benchmark is not None and not trace_benchmark.strip():
            raise ValueError("trace_benchmark cannot be empty")
        if evaluation_workers < 1 or benchmark_retries < 0:
            raise ValueError("Hermes Harbor trace execution metadata is invalid")
        self._trace_root = Path(trace_root).resolve() if trace_root else None
        self._trace_run_id = trace_run_id
        self._trace_created_at = trace_created_at
        self._trace_benchmark = trace_benchmark
        self._evaluation_workers = evaluation_workers
        self._benchmark_retries = benchmark_retries
        self._harbor_version = harbor_version
        self._trace_attempt: HarborTraceAttempt | None = None
        self._agentsight_profiler: AgentSightProfiler | None = None
        self._agentsight_python_path: str | None = None
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
        if self._trace_root is not None:
            self._trace_attempt = allocate_harbor_trace_attempt(
                logs_dir=self.logs_dir,
                trace_root=self._trace_root,
            )
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
                'bash "$INSTALL_DIR/scripts/install.sh" --skip-setup --skip-browser '
                f"--branch {shlex.quote(self._version)}{commit_flag} "
                '--dir "$INSTALL_DIR"; '
                'export PATH="$HOME/.local/bin:$PATH"; '
                "hermes version"
            ),
        )
        if self._trace_attempt is not None:
            python_result = await self.exec_as_agent(
                environment,
                command=('readlink -f "$HOME/.hermes/hermes-agent/venv/bin/python"'),
            )
            python_path = python_result.stdout.strip()
            if (
                python_result.return_code == 0
                and "\n" not in python_path
                and python_path.startswith("/")
            ):
                self._agentsight_python_path = python_path
        for name in ("terminalbench_worker.py", "native_delegation.py"):
            source = Path(__file__).with_name(name)
            uploaded = self.logs_dir / name
            uploaded.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            await environment.upload_file(
                source_path=uploaded,
                target_path=f"/installed-agent/{name}",
            )
        result = await environment.exec(
            command=(
                "chmod 0555 /installed-agent/terminalbench_worker.py "
                "/installed-agent/native_delegation.py"
            ),
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
            "HERMES_BENCHMARK_AGENT_TOPOLOGY": self._agent_topology,
            "HERMES_BENCHMARK_MODEL": self.model_name.removeprefix("openrouter/"),
            "HERMES_HOME": "/tmp/hermes",
            "OPENROUTER_API_KEY": api_key,
        }
        if self._trace_attempt is not None:
            if (
                self.session_id is None
                or self._commit is None
                or self._trace_run_id is None
                or self._trace_benchmark is None
            ):
                raise ValueError("Harbor did not initialize Hermes trace identity")
            env["HERMES_BENCHMARK_TRACE_CONFIG"] = json.dumps(
                {
                    "runId": self._trace_run_id,
                    "benchmark": self._trace_benchmark,
                    "instanceId": self._trace_attempt.instance_id,
                    "attempt": self._trace_attempt.attempt,
                    "runRoot": str(self._trace_attempt.container_root),
                    "createdAt": self._trace_created_at,
                    "frameworkRevision": self._commit.lower(),
                    "model": self.model_name,
                    "evaluationWorkers": self._evaluation_workers,
                    "agentTimeoutSeconds": (self._trace_attempt.agent_timeout_seconds),
                    "benchmarkRetries": self._benchmark_retries,
                    "harborVersion": self._harbor_version,
                    "sessionId": self.session_id,
                    "containerImage": self._trace_attempt.container_image,
                    "agentTopology": self._agent_topology,
                },
                separators=(",", ":"),
            )
            self._agentsight_profiler = await asyncio.to_thread(
                start_harbor_agentsight_profile,
                logs_dir=self.logs_dir,
                trace_run_id=self._trace_run_id,
                benchmark=self._trace_benchmark,
                framework="hermes",
                attempt=self._trace_attempt,
                docker_session_id=environment.session_id,
                tls_python_path=self._agentsight_python_path,
            )
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
        try:
            result = await environment.exec(
                command=f"set -o pipefail; {command}",
                env=env,
            )
        finally:
            if self._agentsight_profiler is not None:
                await asyncio.to_thread(self._agentsight_profiler.finish)
        if result.return_code != 0:
            raise self._classify_exec_error(command, result)

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)
        if self._trace_attempt is not None and self._trace_root is not None:
            metadata = {**(context.metadata or {})}
            try:
                if self._agentsight_profiler is None:
                    raise ValueError("Hermes AgentSight profiler was not initialized")
                attach_harbor_agentsight_profile(
                    logs_dir=self.logs_dir,
                    attempt=self._trace_attempt,
                    profiler=self._agentsight_profiler,
                )
            except Exception as error:
                metadata["agentsight_profile"] = {
                    "health": "failed",
                    "error": type(error).__name__,
                }
                if (
                    self._agentsight_profiler is not None
                    and self._agentsight_profiler.strict
                ):
                    raise
            try:
                destination = promote_harbor_trace_attempt(
                    logs_dir=self.logs_dir,
                    trace_root=self._trace_root,
                    attempt=self._trace_attempt,
                )
                metadata["benchmark_trace"] = {
                    "path": str(destination),
                    "instance_id": self._trace_attempt.instance_id,
                    "attempt": self._trace_attempt.attempt,
                }
                profile_path = destination / "profiles" / "agentsight"
                if profile_path.is_dir() and self._agentsight_profiler is not None:
                    metadata["agentsight_profile"] = {
                        "path": str(profile_path),
                        "profile_id": self._agentsight_profiler.target.profile_id,
                    }
            except Exception as error:
                metadata["benchmark_trace"] = {
                    "health": "failed",
                    "error": type(error).__name__,
                }
            context.metadata = metadata
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
            "agent_topology": result.get("agentTopology"),
            "agent_sequence": result.get("agentSequence"),
            "primary_agent_role": result.get("primaryAgentRole"),
            "delegation_enabled": result.get("delegationEnabled"),
            "workflow_complete": bool(result.get("workflowComplete")),
            "phase_order": result.get("phaseOrder"),
            "delegation_mode": result.get("delegationMode"),
            "native_subagent_budget": result.get("nativeSubagentBudget"),
            "native_subagent_count": result.get("nativeSubagentCount"),
            "peer_phase_budget_reference": result.get("peerPhaseBudgetReference"),
        }
