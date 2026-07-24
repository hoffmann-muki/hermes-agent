"""Framework-neutral trace-run coordination for benchmark integrations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence
from uuid import uuid4

from hermes_cli.benchmarks.tracing.runtime import utc_now, write_run_index


@dataclass(frozen=True)
class TraceRun:
    id: str
    root: Path
    created_at: str
    benchmark: str
    framework: str


TraceSelectionStrategy = Literal[
    "explicit_ids",
    "full_dataset",
    "ordered_window",
]


@dataclass(frozen=True)
class TraceSelection:
    instance_ids: tuple[str, ...]
    strategy: TraceSelectionStrategy
    minimum_attempts_per_instance: int = 1

    def __post_init__(self) -> None:
        if not self.instance_ids or any(
            not value.strip() for value in self.instance_ids
        ):
            raise ValueError("Trace selection requires non-empty instance IDs")
        if len(set(self.instance_ids)) != len(self.instance_ids):
            raise ValueError("Trace selection instance IDs must be unique")
        if (
            isinstance(self.minimum_attempts_per_instance, bool)
            or not isinstance(self.minimum_attempts_per_instance, int)
            or self.minimum_attempts_per_instance < 1
        ):
            raise ValueError("Trace selection requires at least one attempt")


class TraceHarnessAdapter(Protocol):
    def prepare_finalization(self, run: TraceRun) -> None: ...

    def resolve_selection(
        self,
        run: TraceRun,
        observed_instance_ids: Sequence[str],
    ) -> TraceSelection: ...


@dataclass(frozen=True)
class DirectTraceHarness:
    selection: TraceSelection

    def prepare_finalization(self, run: TraceRun) -> None:
        del run

    def resolve_selection(
        self,
        run: TraceRun,
        observed_instance_ids: Sequence[str],
    ) -> TraceSelection:
        del run, observed_instance_ids
        return self.selection


def create_trace_run(
    base_directory: Path,
    *,
    benchmark: str,
    framework: str,
) -> TraceRun:
    """Create one private trace root before any benchmark agent work."""

    if not benchmark.strip() or not framework.strip():
        raise ValueError("Trace benchmark and framework cannot be empty")
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
    return TraceRun(
        id=run_id,
        root=root,
        created_at=utc_now(),
        benchmark=benchmark,
        framework=framework,
    )


def attach_trace_run(
    *,
    root: Path,
    run_id: str,
    created_at: str,
    benchmark: str,
    framework: str,
) -> TraceRun:
    resolved = root.resolve()
    if (
        not run_id
        or not created_at
        or not benchmark.strip()
        or not framework.strip()
        or not resolved.is_dir()
        or root.is_symlink()
    ):
        raise ValueError("Existing trace run identity or root is invalid")
    return TraceRun(
        id=run_id,
        root=resolved,
        created_at=created_at,
        benchmark=benchmark,
        framework=framework,
    )


def finalize_trace_run(run: TraceRun, harness: TraceHarnessAdapter) -> Path:
    """Validate discovered coverage before writing a generic run index."""

    if run.root.is_symlink() or not run.root.resolve().is_dir():
        raise ValueError("Trace run root must be a real directory")
    harness.prepare_finalization(run)
    observed: dict[str, set[int]] = {}
    for manifest_path in sorted(run.root.glob("instances/*/attempt-*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        instance_id = (
            manifest.get("instance_id") if isinstance(manifest, dict) else None
        )
        attempt = manifest.get("attempt") if isinstance(manifest, dict) else None
        if (
            not isinstance(manifest, dict)
            or manifest.get("run_id") != run.id
            or manifest.get("benchmark") != run.benchmark
            or manifest.get("framework") != run.framework
            or not isinstance(instance_id, str)
            or not instance_id
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 1
            or not (manifest_path.parent / "events.jsonl").is_file()
            or not (manifest_path.parent / "health.json").is_file()
        ):
            raise ValueError(
                f"Trace attempt does not belong to its run: {manifest_path.parent}"
            )
        if attempt in observed.setdefault(instance_id, set()):
            raise ValueError(f"Duplicate trace attempt {attempt} for {instance_id}")
        observed[instance_id].add(attempt)

    selection = harness.resolve_selection(run, tuple(observed))
    if set(selection.instance_ids) != set(observed):
        raise ValueError(
            "Trace run index omitted because selected instances lack finalized traces"
        )
    if any(
        len(observed[instance_id]) < selection.minimum_attempts_per_instance
        for instance_id in selection.instance_ids
    ):
        raise ValueError(
            "Trace run index omitted because an instance lacks a requested attempt"
        )
    return write_run_index(
        root=run.root,
        run_id=run.id,
        benchmark=run.benchmark,
        framework=run.framework,
        created_at=run.created_at,
        instance_ids=selection.instance_ids,
        selection_strategy=selection.strategy,
    )
