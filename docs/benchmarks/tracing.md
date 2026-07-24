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
