"""Hermes-native trace adapter construction for benchmark runners."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from hermes_cli.benchmarks.tracing.coordination import (
    DirectTraceHarness,
    TraceRun,
    TraceSelection,
    TraceSelectionStrategy,
    create_trace_run,
    finalize_trace_run,
)
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
)


@dataclass(frozen=True)
class HermesTraceRun(TraceRun):
    """Backward-compatible Hermes-bound trace-run identity."""

    framework: str = "hermes"


def create_hermes_trace_run(base_directory: Path, benchmark: str) -> TraceRun:
    """Compatibility wrapper around generic trace-run creation."""

    return create_trace_run(
        base_directory,
        benchmark=benchmark,
        framework="hermes",
    )


def create_hermes_attempt_trace(
    *,
    run: TraceRun,
    instance_id: str,
    attempt: int,
    framework_revision: str,
    model: str,
    agent_timeout_seconds: int | float,
    evaluation_workers: int,
    benchmark_retries: int = 0,
    harness_revision: str | None = None,
    agent_image: str | None = None,
) -> HermesTraceAdapter:
    if run.framework != "hermes":
        raise ValueError("Hermes trace adapter requires framework='hermes'")
    if not re.fullmatch(r"[0-9a-f]{40}", framework_revision):
        raise ValueError("Tracing requires an exact 40-character Hermes revision")
    if (
        not instance_id
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
        or not model
        or evaluation_workers < 1
        or agent_timeout_seconds <= 0
        or benchmark_retries < 0
    ):
        raise ValueError("Hermes trace attempt metadata is invalid")
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
                **(
                    {
                        "harness": {
                            "name": "Harbor",
                            "revision": harness_revision,
                        }
                    }
                    if harness_revision
                    else {}
                ),
                **({"agent_image": agent_image} if agent_image else {}),
            },
            execution={
                "model": model,
                "evaluation_workers": evaluation_workers,
                "inference_timeout_seconds": agent_timeout_seconds,
                "benchmark_retries": benchmark_retries,
                "provider_attempts": 1,
            },
            capabilities=hermes_capabilities({}),
        )
    )
    return HermesTraceAdapter(recorder)


def finalize_hermes_trace_run(
    *,
    run: TraceRun,
    instance_ids: Sequence[str],
    selection_strategy: TraceSelectionStrategy,
) -> Path:
    """Compatibility wrapper around generic direct-harness finalization."""

    return finalize_trace_run(
        run,
        DirectTraceHarness(
            TraceSelection(
                instance_ids=tuple(instance_ids),
                strategy=selection_strategy,
            )
        ),
    )
