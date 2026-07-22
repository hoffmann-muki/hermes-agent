# SWE-bench Verified

Hermes runs SWE-bench Verified against the official per-instance images and
scores predictions with the official local Docker harness. Inference and
evaluation are separate commands: finishing inference never starts scoring or
spends evaluator resources automatically.

## Parity defaults

The runner intentionally matches the OpenCode and OpenHands benchmark setup in
this workspace:

| Setting | Default |
| --- | --- |
| Dataset | `princeton-nlp/SWE-bench_Verified` (`test`) |
| Smoke instance | `scikit-learn__scikit-learn-13439` |
| Model | `openrouter/qwen/qwen3-coder-next` |
| Temperature | `0.1` |
| Inference/evaluation workers | `1` / `1` |
| Attempts/retries | `1` / `0` |
| Hermes API attempts/retries per model call | `1` / `0` (`api_max_retries: 1`) |
| Agent deadline | 1,800 seconds |
| Agent sequence | coordinator → navigator → patcher → reviewer |
| Iteration budgets | `24` / `10` / `18` / `12` |
| Evaluator | `swebench==4.1.0`, local Docker |

The coordinator invokes three fresh, blocking specialist agents through a
benchmark-only runtime tool. All four agents share one `/testbed` container.
The navigator is checked for worktree mutations; any changes it makes are
discarded and mark the workflow unsuccessful. The patcher and reviewer edit the
shared checkout, and the coordinator performs final reconciliation.

## Prerequisites

- A working local Docker daemon. On Docker Desktop, configure the VM with at
  least 16 GiB of memory. Hermes does not apply a smaller per-container memory,
  CPU, or disk cap for this benchmark.
- `OPENROUTER_API_KEY` exported in the shell that starts inference.
- For evaluation only, a Python environment containing exactly
  `swebench==4.1.0`.

The existing OpenHands benchmark environment can supply that version, but it
also loads OpenHands' `sitecustomize.py` patches at interpreter startup. A small,
dedicated evaluator environment containing only the pinned SWE-bench harness is
preferred for reproducible Hermes scoring.

The API key is passed only to the isolated worker process. It is not accepted as
a CLI argument, written to artifacts, forwarded into the task container, or
included in the worker request. The runner uses a fresh Hermes home, with user
memory, context files, checkpoints, plugins, user-configured volumes, and user
credential mounts absent for the attempt. Hermes uses run-scoped host directories
for `/root` and `/workspace` only so one container survives all four agent turns;
those directories contain no user profile and the controller removes the
container after the attempt.

## Run inference

From the Hermes repository:

```bash
export OPENROUTER_API_KEY='...'
hermes-swebench-verified infer --run-id qwen-next-verified-smoke
```

Use repeatable explicit IDs to run a chosen set in that order:

```bash
hermes-swebench-verified infer \
  --run-id qwen-next-verified \
  --instance-id scikit-learn__scikit-learn-13439 \
  --instance-id django__django-10097
```

Alternatively, select a dataset window with `--offset` and
`--max-instances`. Either option disables the implicit smoke instance. Public
hints are excluded unless `--include-hints` is supplied. Use `--dry-run` to
inspect resolved IDs, images, budgets, and output paths without Docker or an API
call.

The runner inspects or pulls the official image
`docker.io/swebench/sweb.eval.x86_64.{repo}_1776_{name}:latest`; it does not
build a Hermes agent image. A fresh worker subprocess and container are used for
each instance, with one outer attempt and no benchmark-level retry. Partial and
empty patches are preserved after failures and timeouts. Initial checkout setup
activates the image's `testbed` Conda environment and resets tracked files to the
instance commit while preserving ignored artifacts provided by the official
image, such as prebuilt extension modules.

The 1,800-second deadline covers agent inference. Image/setup and worker teardown
use separate 600-second guards, while controller-side patch capture and cleanup
use their own bounded operations, so artifact collection does not consume the
model's execution budget. The isolated config also disables Hermes coding-context
discovery, preventing the host Hermes checkout from being added to task prompts
when `/testbed` exists only inside Docker.

After every inference attempt, the worker exits without capturing or removing the
task container. The controller restarts that container to stop tracked or
untracked background commands, then captures a stable patch, restores bind-mount
ownership, and removes the container. The same path handles normal completion,
failure, timeout, and interruption. Worker logs are created with mode `0600` and
redacted on controller exit paths before artifacts are returned.

Artifacts are written below:

```text
.benchmark-runs/swe-bench-verified/runs/<run-id>/
```

They include the selected public rows, prompt, worker logs and messages,
per-phase records, patch, official prediction JSONL, summary, and a completion
manifest bound to the predictions with SHA-256. The manifest also records the
exact Hermes Git commit and a fingerprint of local source changes; resume refuses
to mix predictions produced by different runner code, and an active run aborts if
that source changes between instances. A matching incomplete run resumes at the
first unfinished instance. Use `--restart` to replace it. If the controller is
interrupted, the latest captured patch and redacted controller result remain in
the instance directory even though the interrupted instance is not checkpointed
as complete.

## Run local evaluation

Point `--python` at the environment containing the pinned harness:

```bash
hermes-swebench-verified evaluate \
  --run-id qwen-next-verified-smoke \
  --python /path/to/swebench-4.1.0/bin/python
```

Before starting Docker evaluation, Hermes verifies the completed manifest,
run ID, prediction order/count, and SHA-256 digest, then confirms the exact
harness version. Evaluation uses one worker and a 3,600-second per-test timeout.
Supply repeatable `--instance-id` flags to score a subset already present in the
prediction artifact. `--dry-run` prints the exact evaluator command after
artifact verification. On interruption, the evaluation manifest is finalized as
`interrupted` and only containers whose exact names belong to that run are
removed.
