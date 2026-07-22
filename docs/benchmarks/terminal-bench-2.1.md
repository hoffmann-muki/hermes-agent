# Terminal-Bench 2.1

Hermes runs Terminal-Bench 2.1 through the official Harbor harness and its
local Docker environment. Harbor owns dataset retrieval, task-container setup,
verification, and result persistence; the Hermes wrapper supplies reproducible
agent defaults, prerequisite checks, and run metadata.

## Parity defaults

The runner matches the OpenCode and OpenHands Terminal-Bench configuration in
this workspace:

| Setting | Default |
| --- | --- |
| Dataset | `terminal-bench/terminal-bench-2-1` (89 tasks) |
| Smoke selection | First Harbor task (`--n-tasks 1`) |
| Model | `openrouter/qwen/qwen3-coder-next` |
| Temperature | `0.1` |
| Harbor attempts / infrastructure retries | `1` / `0` |
| Concurrent Harbor trials | `1` |
| Environment | Local Docker |
| Agent sequence | coordinator → navigator → patcher → reviewer |
| Iteration budgets | `24` / `10` / `18` / `12` |
| Hermes API attempts/retries per model call | `1` / `0` (`api_max_retries: 1`) |

The coordinator invokes a benchmark-only synchronous phase tool. Each phase
uses a fresh Hermes agent in the same Harbor task container. The navigator is
instructed to use file and terminal inspection tools without changing state;
the patcher performs the task; the reviewer independently
checks the resulting state and may make small corrections. The runtime rejects
skipped, repeated, out-of-order, or overlapping phase calls.

Harbor's official task definitions remain authoritative for setup, agent, and
verifier timeouts. The wrapper does not impose the SWE-bench 1,800-second agent
deadline on Terminal-Bench tasks.

## Prerequisites

- Harbor on `PATH` (the integration is tested with Harbor 0.20.0).
- A running local Docker daemon. Configure Docker Desktop with at least 16 GiB
  of memory for the broader benchmark suite.
- `OPENROUTER_API_KEY` exported in the shell that starts the run.
- The Hermes `play` branch and current commit available on the checkout's
  `origin` remote, because Harbor installs that exact revision into each task
  container by default.

The API key is inherited by Harbor and forwarded only to the agent process. It
is never accepted as a command-line argument or written to the wrapper's
command or manifest. Worker result and trajectory artifacts are redacted before
they are persisted.

## Run

From the Hermes repository:

```bash
export OPENROUTER_API_KEY='...'

# Credential-free command preview; creates no run directory.
hermes-terminalbench --dry-run

# Safe one-task smoke evaluation.
hermes-terminalbench --run-id qwen-next-terminal-smoke

# Explicit task IDs, retained in the supplied order.
hermes-terminalbench \
  --run-id qwen-next-terminal-selected \
  --task-id task-a \
  --task-id task-b

# Complete local dataset, one attempt per task, without upload.
hermes-terminalbench --all-tasks --run-id qwen-next-terminal-full
```

`--task-name` is an alias for `--task-id`, `--n-limit` aliases `--max-tasks`,
and `--num-workers` aliases `--concurrency`. The default remains one worker;
raise it only for an intentional higher-throughput run.

Use `--leaderboard` for the official protocol. It rejects task filters and
explicit limits, runs the full dataset, raises attempts to at least five, and
enables a public Harbor upload.

Harbor installs the Hermes agent core from `--hermes-version play`, the HTTPS
form of the checkout's `origin`, and its current 40-character commit. Override
those with `--hermes-repository` and `--hermes-commit` when intentionally
testing another remote revision. The repo-local Harbor adapter also uploads the
exact benchmark worker from the invoking checkout into the task container, so
local integration edits are exercised without building a separate agent image.
The manifest records both the installed remote revision and the invoking
checkout's full source identity, including dirty-state fingerprinting.

Artifacts are written below:

```text
.benchmark-runs/terminal-bench-2.1/runs/<run-id>/
```

They include a manifest, streamed Harbor stdout/stderr, native Harbor jobs,
verifier results, Hermes phase records, usage totals, and an ATIF trajectory.
