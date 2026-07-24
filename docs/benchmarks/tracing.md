# Benchmark tracing architecture

Hermes benchmark tracing emits the shared `benchmark-trace/v1` contract without
coupling native agent instrumentation to a particular benchmark.

The integration has three independent layers:

- The generic trace-run coordinator creates private runs, resolves canonical
  selection, discovers finalized attempts, verifies coverage, and writes the run
  index.
- `HermesTraceAdapter` consumes Hermes-native callbacks for sessions, model
  turns, tools, delegation, compaction, timing, outputs, and errors. It receives
  benchmark identity as data and contains no benchmark-specific extraction.
- A `TraceHarnessAdapter` reports execution topology owned by the harness.
  Direct SWE runners use `DirectTraceHarness`. Harbor-backed runners use
  `HarborTraceHarness`, which resolves Harbor task order and infrastructure
  attempts without changing Hermes delegation.

To trace another direct benchmark, create a run with `create_trace_run`, create
one `create_hermes_attempt_trace` adapter per agent attempt, and finalize with a
`TraceSelection` supplied to `DirectTraceHarness`. To trace another Harbor
dataset, reuse `HarborTraceHarness`; the installed agent receives the benchmark
identity through non-secret trace metadata.

A new tracing implementation is required only for a genuinely new agent
framework or execution harness. Missing native hooks are reported through the
capability matrix rather than inferred.

## Runtime collection

Hermes SWE-bench and Terminal-Bench inference commands enable capture by default
beneath the repository-local `.benchmark-traces/` base. The runner creates a
private `trace-run-<uuid>` before provider work and the native callbacks record
activity as the agent performs it. Use `--trace-dir <base-directory>` to
override the base or `--no-trace` for an intentional untraced run. This is not
post-hoc log extraction, and no collector command needs to run before, during,
or after the benchmark.

SWE runners print the exact trace-run path. Terminal-Bench persists it in the
benchmark manifest and prints it at completion. The selected base remains a
stable discovery location containing finalized trace-run children.

Framework-native records are sanitized into one durable internal journal during
execution and finalized into size-bounded gzip chunks. Every record keeps its
sequence, timestamp, source, identity, and canonical-event links; chunking
changes only physical storage and never samples or coalesces deltas. Many
`native/index.jsonl` rows can therefore share one content-addressed artifact.
On filesystems with hard-link support, finalized `journal.jsonl` and
`events.jsonl` also share one inode, with an atomic-copy fallback.

## Researcher tooling

The canonical OpenHands-benchmarks environment provides one read-only
`benchmark-trace` CLI for output from Hermes, OpenCode, and OpenHands:

```bash
uv run benchmark-trace validate /path/to/traces
uv run benchmark-trace inspect /path/to/traces/trace-run-<uuid>
uv run benchmark-trace summarize /path/to/traces/trace-run-<uuid>
uv run benchmark-trace render /path/to/traces/trace-run-<uuid>
uv run benchmark-trace compare <trace-run-a> <trace-run-b>
```

The tool operates on the normalized contract rather than framework or benchmark
names. It validates, inspects provenance and capability boundaries, aggregates
activity and timing, checks comparison parity, and renders deterministic nested
timelines. `--format json` provides machine-readable output. It never launches
Hermes or collects a trace, and it does not inspect artifact contents or
calculate token usage or cost.
