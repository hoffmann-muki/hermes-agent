"""Per-instance Hermes worker specialization for SWE-bench Lite."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

from hermes_cli.benchmarks import swebench_verified_worker as shared
from hermes_cli.benchmarks.swebench_verified import (
    SWE_BENCH_LITE,
    parse_swebench_row,
)


@contextmanager
def configuration() -> Iterator[None]:
    with shared.worker_configuration(
        benchmark=SWE_BENCH_LITE.benchmark,
        worktree="/testbed",
        conda_activation=shared.CONDA_ACTIVATION,
        row_parser=parse_swebench_row,
        coordinator_system_prompt=shared.COORDINATOR_SYSTEM_PROMPT,
    ):
        yield


def run_worker(
    request: dict[str, Any],
    *,
    agent_factory: Callable[..., Any] = shared.default_agent_factory,
) -> dict[str, Any]:
    with configuration():
        return shared.run_worker(request, agent_factory=agent_factory)


def main(argv: Sequence[str] | None = None) -> int:
    with configuration():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
