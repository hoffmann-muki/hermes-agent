"""Lifecycle integration between Hermes benchmark runners and trace adapters."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from hermes_cli.benchmarks.tracing.hermes import (
    HermesTraceAdapter,
    hermes_capabilities,
)
from hermes_cli.benchmarks.tracing.runtime import (
    CONTRACT_VERSION,
    TraceConfig,
    TraceIdentity,
    TraceRecorder,
    attempt_directory,
    utc_now,
    write_run_index,
)


@dataclass(frozen=True)
class HermesTraceRun:
    id: str
    root: Path
    created_at: str
    benchmark: str


def create_hermes_trace_run(base_directory: Path, benchmark: str) -> HermesTraceRun:
    """Create one private trace root before any benchmark agent work."""

    expanded = base_directory.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"Trace base cannot be a symbolic link: {expanded}")
    base = expanded.resolve()
    existed = base.exists()
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not base.is_dir() or base.is_symlink():
        raise ValueError(f"Trace base must be a real directory: {base}")
    if not existed and os.name != "nt":
        base.chmod(0o700)
    run_id = f"trace-run-{uuid4().hex}"
    root = base / run_id
    root.mkdir(mode=0o700)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Trace root must be a real directory: {root}")
    if os.name != "nt":
        root.chmod(0o700)
    return HermesTraceRun(
        id=run_id,
        root=root,
        created_at=utc_now(),
        benchmark=benchmark,
    )


def create_hermes_attempt_trace(
    *,
    run: HermesTraceRun,
    instance_id: str,
    attempt: int,
    framework_revision: str,
    model: str,
    agent_timeout_seconds: int,
    evaluation_workers: int,
) -> HermesTraceAdapter:
    if not re.fullmatch(r"[0-9a-f]{40}", framework_revision):
        raise ValueError("Tracing requires an exact 40-character Hermes revision")
    identity = TraceIdentity.create(
        run_id=run.id,
        benchmark=run.benchmark,
        framework="hermes",
        instance_id=instance_id,
        attempt=attempt,
    )
    recorder = TraceRecorder(
        TraceConfig(
            attempt_dir=attempt_directory(run.root, instance_id, attempt),
            identity=identity,
            producer={
                "name": "hermes_cli.benchmarks.tracing.hermes",
                "version": CONTRACT_VERSION,
            },
            provenance={
                "benchmark": {
                    "name": "hermes-agent-benchmarks",
                    "revision": framework_revision,
                },
                "framework": {
                    "name": "Hermes Agent",
                    "revision": framework_revision,
                },
                "adapter": {
                    "name": "hermes_cli.benchmarks.tracing.hermes",
                    "revision": framework_revision,
                },
            },
            execution={
                "model": model,
                "evaluation_workers": evaluation_workers,
                "inference_timeout_seconds": agent_timeout_seconds,
                "benchmark_retries": 0,
                "provider_attempts": 1,
            },
            capabilities=hermes_capabilities({}),
        )
    )
    return HermesTraceAdapter(recorder)


def finalize_hermes_trace_run(
    *,
    run: HermesTraceRun,
    instance_ids: Sequence[str],
    selection_strategy: str,
) -> Path:
    return write_run_index(
        root=run.root,
        run_id=run.id,
        benchmark=run.benchmark,
        created_at=run.created_at,
        instance_ids=instance_ids,
        selection_strategy=selection_strategy,
    )
