"""Per-instance Hermes worker specialization for SWE-bench Pro."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

from hermes_cli.benchmarks import swebench_verified_worker as shared
from hermes_cli.benchmarks.swebench_pro import (
    BENCHMARK,
    parse_swebench_pro_row,
)


WORKTREE = "/app"

COORDINATOR_SYSTEM_PROMPT = """You are benchmark-coordinator for a SWE-bench Pro task.

You own the result, but three fresh specialists must work in the same shared
workspace before you finish. Use Hermes' native delegate_task tool for one fresh
leaf subagent at a time, in this order:

1. navigator — use a goal beginning [benchmark-navigator]. Ask it to investigate
   without changing state and return an evidence-backed plan.
2. patcher — after the navigator returns, use a goal beginning [benchmark-patcher].
   Pass the complete Pro issue fields and navigator handoff; require the smallest
   complete fix and focused verification.
3. reviewer — after the patcher returns, use a goal beginning [benchmark-reviewer].
   Pass the complete issue and prior handoffs; require independent review, focused
   checks, and only small clearly necessary corrections.

For every call use role="leaf" and include the complete task, repository metadata,
worktree path, role constraints, and prior handoffs in context because native Hermes
subagents start with fresh context. Call sequentially, not as a tasks batch. Native
delegation returns synchronously in this benchmark runner. After the reviewer returns,
reconcile all reports, inspect the final worktree, make only necessary final
corrections, run feasible verification, and provide a concise final summary. The
source edits in /app—not prose—are the benchmark answer.

Never seek a gold patch, hidden test patch, benchmark answer, or hidden grading data.
Do not modify tests or benchmark metadata unless the issue explicitly requires it.
"""

@contextmanager
def configuration() -> Iterator[None]:
    with shared.worker_configuration(
        benchmark=BENCHMARK,
        worktree=WORKTREE,
        conda_activation="",
        row_parser=parse_swebench_pro_row,
        coordinator_system_prompt=COORDINATOR_SYSTEM_PROMPT,
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
