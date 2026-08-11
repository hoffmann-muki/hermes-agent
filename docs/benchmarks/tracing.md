# Benchmark tracing architecture

Hermes benchmark tracing emits the shared `benchmark-trace/v1` contract without
coupling native agent instrumentation to a particular benchmark.

The integration has three independent layers:

- The generic trace-run coordinator creates private runs, resolves canonical
  selection, discovers finalized attempts, verifies coverage, and writes the run
  index.
- `HermesTraceAdapter` consumes Hermes-native callbacks for sessions, model
  turns, tools, delegation, compaction, timing, outputs, and errors. Native
  child step and progress hooks expose atomic child model turns plus correlated
  child tool inputs, complete sanitized results, and durations. The adapter
  represents each parent `delegate_task` call as one logical delegation and
  nests the native child session beneath it. It receives benchmark identity as
  data and contains no benchmark-specific extraction.
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

Every complete attempt has one coarse startup span, one detailed agent-execution
span, and one coarse shutdown span. These generic envelopes account for the
whole attempt without encoding benchmark-specific setup or teardown. Detailed
execution preserves overlapping root and child lanes; it is not forced into a
serial history. Native occurrence and recorder capture timestamps remain
distinct.

SWE runners print the exact trace-run path. Terminal-Bench persists it in the
benchmark manifest and prints it at completion. The selected base remains a
stable discovery location containing finalized trace-run children.

Framework-native records are sanitized into one durable internal journal during
execution and finalized into size-bounded gzip chunks. Every record keeps its
sequence, timestamp, source, identity, and canonical-event links; chunking
changes only physical storage and never samples or coalesces deltas. Many
`native/index.jsonl` rows can therefore share one content-addressed artifact.
Chunk artifacts are the only valid native-evidence representation; loose
per-record artifacts are outside the shared trace contract.
On filesystems with hard-link support, finalized `journal.jsonl` and
`events.jsonl` also share one inode, with an atomic-copy fallback.

If a timeout or process termination kills an attempt after its durable
checkpoint is written but before finalization, run coordination recovers the
complete journal records before checking coverage. A torn final JSONL record is
discarded, the attempt is explicitly marked `degraded` with `recovered`
finalization, and `run.json` can still account for the attempt. Recovery never
invents missing agent actions or closing events.

## AgentSight companion profiles

Every traced SWE-bench Verified, SWE-bench Pro, and Terminal-Bench 2.1 attempt
starts AgentSight automatically. The profiling adapter remains
benchmark-independent and maps only Hermes's runtime topology:

- a host scope follows the benchmark worker and descendants, including
  host-side Hermes, delegation, and model activity; and
- a task-container scope runs a privileged, network-isolated sidecar filtered
  by the container's PID namespace, including later `docker exec` processes.

SWE-bench uses both scopes because the Hermes coordinator and provider activity
are host-side while repository tools run in the task container. Harbor installs
the complete Terminal-Bench coordinator inside its `main` task container, so
Terminal-Bench uses only one task-container sidecar. It does not start a
redundant host collector or require sudo.

AgentSight supplies independent process, filesystem, network, signal, memory,
system, stdio, and TLS/HTTP evidence. It does not replace Hermes-native
callbacks and does not change prompts, native delegation, provider attempts,
timeouts, or retries. For SWE-bench, the adapter attaches TLS probes to the
TLS-bearing binary used by the exact host Python worker that makes model
requests: its loaded `libssl`, or the Python executable when OpenSSL is
statically embedded. For Terminal-Bench, it resolves the equivalent binary in
the exact container Python runtime and attaches it through
`/proc/<container-init>/root`. Container probes are filtered to the task PID
namespace, so capture needs neither a proxy nor a global TLS probe nor another
collector. If discovery fails, best-effort mode retains
process/filesystem/network evidence and reports the requested TLS scope as
inactive; strict mode aborts before inference. The sidecar disables stdio
because namespace-wide stdio capture is unavailable.

Output is colocated under each trace attempt at `profiles/agentsight/`.
Aggregate `profile.json`, `health.json`, and `summary.json` files report
cross-scope status. Every `sources/<scope>/` directory contains AgentSight
provenance and health, `capture.db`, and the compressed
`system-events.jsonl.zst` evidence journal. Readiness must match the current
profile and scope, and collectors stop before their observed runtime is
destroyed.
Terminal-Bench stages the stopped profile in the Harbor trial log and
co-locates it with the semantic attempt during promotion. The aggregate
correlation record names the run, benchmark, framework, instance, and attempt.

AgentSight can visualize the host and task-container databases as one
scope-aware timeline without copying or rewriting them:

```bash
agentsight report --profile-dir <attempt>/profiles/agentsight serve
```

The loader validates source profile identities, merges by normalized wall-clock
time, and preserves `scope_id` on every row. Equal numeric PIDs and row IDs in
different scopes remain distinct, concurrent activity remains concurrent, and
the UI provides scope lanes and filtering. The same command works for
single-scope Terminal-Bench profiles.

Profiling is best-effort by default and has no effect on benchmark success or
retry decisions. Set `BENCHMARK_AGENTSIGHT_STRICT=1` to require every expected
scope before provider work, or `BENCHMARK_AGENTSIGHT=off` to disable it.
`AGENTSIGHT_BIN`, `AGENTSIGHT_IMAGE`, `AGENTSIGHT_READY_TIMEOUT_SECONDS`, and
`AGENTSIGHT_STOP_TIMEOUT_SECONDS` configure the executable, image, and
supervision. Before an unprivileged host collector starts, the generic tracing
privilege helper now performs the equivalent of `sudo -v` automatically. It
reads the password from `~/.config/benchmark-tools/sudo-password` by default;
use `BENCHMARK_SUDO_PASSWORD_FILE` to select another private regular file and
`BENCHMARK_SUDO_TIMEOUT_SECONDS` to configure validation timeout. The file must
belong to the benchmark user and grant no group or other permissions. Its
contents are supplied only on sudo's standard input and never enter command
arguments, environment variables, logs, profiles, or Git. Password-file
authorization occurs before evidence capture becomes ready. A small privileged
supervisor then owns the collector and observes a private stop marker, so
shutdown needs neither another sudo call nor the password and remains reliable
even after the host's sudo ticket expires. If the file is absent, an
already-valid sudo ticket is accepted non-interactively.

Parsed authentication headers are removed before persistence, but profiles can
still contain prompts, responses, commands, output, paths, and network targets.
Treat them as sensitive research artifacts. The canonical `events.jsonl`
retains agent semantics and concurrency; AgentSight contributes correlated
OS-level facts.

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
timelines. Summaries report detailed execution coverage, explicit unattributed
gaps, concurrency, lanes, ordering inversions, and capture delay. Rendering
defaults to source-time order; `--order capture` and `--order sequence` expose
arrival and durable-journal order. `--format json` provides machine-readable
output. It never launches Hermes or collects a trace, and it does not inspect
artifact contents or calculate token usage or cost.
