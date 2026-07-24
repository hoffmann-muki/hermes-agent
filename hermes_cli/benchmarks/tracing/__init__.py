"""Framework-neutral trace contract support for Hermes benchmark adapters."""

from hermes_cli.benchmarks.tracing.hermes import (
    HermesTraceAdapter,
    hermes_capabilities,
)
from hermes_cli.benchmarks.tracing.coordination import (
    DirectTraceHarness,
    TraceHarnessAdapter,
    TraceRun,
    TraceSelection,
    TraceSelectionStrategy,
    attach_trace_run,
    create_trace_run,
    finalize_trace_run,
)
from hermes_cli.benchmarks.tracing.integration import (
    HermesTraceRun,
    create_hermes_attempt_trace,
    create_hermes_trace_run,
    finalize_hermes_trace_run,
)

__all__ = [
    "DirectTraceHarness",
    "HermesTraceAdapter",
    "HermesTraceRun",
    "TraceHarnessAdapter",
    "TraceRun",
    "TraceSelection",
    "TraceSelectionStrategy",
    "attach_trace_run",
    "create_hermes_attempt_trace",
    "create_hermes_trace_run",
    "create_trace_run",
    "finalize_hermes_trace_run",
    "finalize_trace_run",
    "hermes_capabilities",
]
