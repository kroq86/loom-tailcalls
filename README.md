# Loom

Stack-safe async state machines for Python.

Loom lets you write long-running async processes as explicit state
transitions:

```python
from loom import tailrec


@tailrec
async def agent_loop(state):
    if state.done:
        return state

    event = await run_next_step(state)
    return await agent_loop(state.apply(event))
```

`@tailrec` rewrites the tail-position `return await agent_loop(...)` into an
async `while` loop. You keep the recursive state-machine shape, but runtime
uses constant stack.

## Install

```bash
pip install loom-tailcalls
```

## Loom stack

Small composable pieces for long-running agent loops:

| Package | Role |
| --- | --- |
| **[loom-tailcalls](https://github.com/kroq86/loom-tailcalls)** (this repo) | Stack-safe async transitions |
| **[flow-xray](https://github.com/kroq86/flow-xray)** | Local HTML traces |
| **[loom-runner](https://github.com/kroq86/loom-runner)** | SQLite checkpoint/resume and CLI inspection |

## What's new in 0.2

- **`**kwargs` in tail-call** — `return await agent_loop(state, **bindings)` without hand-written `while`
- **Structured tail positions** — tail-call inside `try`/`except`, `with`/`async with`, and loops
- **Integration lab** — [`demo-loom-flow/`](demo-loom-flow/) with cases 01–08 (Ollama agent, 100k stress, `explain_tailcalls` smoke)

## Why Loom

Python already has `while`, but long-running async systems often want a
different shape:

```text
state -> await step -> event -> next state -> ... -> result
```

That is the natural shape of agent runners, workflow engines, retry/backoff
systems, protocol sessions, streaming parsers, and resumable state machines.
Without Loom, Python forces these systems into manual mutable loops:

```python
async def run_job(state):
    if state.done:
        return state.result

    while True:
        event = await execute_next_step(state)
        state = state.apply(event)
        if state.done:
            return state.result
```

That works, but as workflows grow, the loop often accumulates flags, mutable
locals, nested `continue`/`break`, retry counters, checkpoints, and stream
state. Loom lets the transition stay explicit:

```python
@tailrec
async def run_job(state):
    if state.done:
        return state.result

    event = await execute_next_step(state)
    return await run_job(state.apply(event))
```

This is not recursion for recursion's sake. It is a way to model a long-running
process as a pure-ish transition:

```text
delta : State -> Result + State
```

Concrete jobs this unlocks in Python:

- **Agent/tool runtimes**: conversation state -> tool/model call -> updated
  state -> final answer.
- **Retry/backoff workflows**: attempt state -> async call -> delay/update ->
  next attempt.
- **Polling monitors**: snapshot -> await next check -> compare/update ->
  continue.
- **Protocol/session handlers**: session state -> receive message -> transition
  -> respond/continue.
- **Workflow/saga engines**: workflow state -> run step -> persist checkpoint ->
  continue/compensate/finish.
- **Streaming parsers and agents**: parser state -> read chunk -> emit events ->
  next parser state.
- **Small interpreters/evaluators**: machine state -> execute instruction ->
  next machine state.

Plain async recursion can express the same transition style, but it grows the
Python call stack. Loom keeps the transition style and removes the stack growth.

Run the comparison:

```bash
python3 examples/deep_agent_loop_comparison.py
```

Expected output:

```text
plain recursion failed: RecursionError
loom recursion survived: AgentState(remaining=0, total=100000)
```

## Who It Is For

Loom is for long-running async workflows that are naturally written as
tail-recursive state transitions:

- AI agent and tool loops: think, call a tool, update state, continue.
- Async orchestration and workflow engines with many sequential steps.
- Retry/backoff jobs, polling monitors, protocol sessions, and resumable
  workflows.
- Streaming agents that yield events while moving to the next state.
- Explicit state machines shaped as `state -> next_state -> result`.
- Libraries that need a small, testable, semantics-preserving transform rather
  than an ad hoc recursion trick.

The goal is:

```text
write async tail recursion
run it as a loop
keep O(1) stack
preserve observable behavior
```

Loom is not a general Python speed optimizer. If the code is already a simple
CPU-bound `for` or `while` loop, write the loop directly. For numeric or
maximum-throughput CPU work, use the usual tools: vectorization, native
extensions, Cython, Rust, or similar.

Loom is also intentionally narrow: non-tail recursion such as
`return 1 + await fn(...)` is rejected rather than transformed unsafely.

In one sentence:

```text
Loom is a stack-safety compiler for async tail-recursive state machines.
```

For streaming agents, `@tailstream` optimizes this terminal async-generator pattern:

```python
from loom import tailstream


@tailstream
async def stream_agent(state):
    if state.done:
        yield {"type": "final", "state": state}
        return

    async for event in run_step(state):
        yield event

    async for event in stream_agent(state.next()):
        yield event
    return
```

## Comparison

Loom is a stack-safety transform, not a full agent or workflow framework.

| Approach | Async | Stack-safe | Streaming | Role |
| --- | --- | --- | --- | --- |
| **Loom** | yes | yes | `@tailstream` | Write tail-recursive async state machines |
| **Hand-written `while`** | yes | yes | manual | Same runtime shape, more mutable loop state |
| **[tacopy](https://github.com/raaidrt/tacopy)** | no | yes | no | Sync-only AST tail-call optimization |
| **LangGraph / Temporal** | yes | n/a | yes | Orchestration, persistence, tools — different layer |

Loom is primarily about stack safety and explicit transitions, not beating a
hand-written loop on speed. On CPython 3.11+ with direct rebinding, a local
benchmark at `n=100000` typically shows roughly **1.1–1.2x** overhead versus an
equivalent hand-written `while` loop. See
[docs/performance-tracing.md](docs/performance-tracing.md) for the measurement
helper and expected overhead model.

## Hooks And Guardrails

Hooks, iteration budgets, and checkpoints belong **outside** `@tailrec`. Loom
transforms the loop body; cross-cutting concerns compose around the step
function or live in immutable state:

```python
async def run_step_with_hooks(state, hooks):
    for hook in hooks:
        await hook.before_step(state)
    event = await run_step(state)
    for hook in hooks:
        event = await hook.after_step(state, event)
    return event
```

Runaway-loop protection works the same way: keep a `remaining_steps` (or similar)
field in state and terminate when the budget is exhausted. See
[examples/agent_loop_with_budget.py](examples/agent_loop_with_budget.py) and
[examples/agent_loop_with_hooks.py](examples/agent_loop_with_hooks.py).

## Debugging Transforms

If `@tailrec` or `@tailstream` rejects a function, call `explain_tailcalls(fn)`
to inspect optimized sites, binding mode, and rejected recursive calls:

```python
from loom import explain_tailcalls, tailrec

@tailrec
async def agent_loop(state):
    ...

print(explain_tailcalls(agent_loop))
```

Rejected shapes include non-tail returns such as `return 1 + await fn(...)`,
recursive calls in `try` finally / `with` context expressions / loop tests,
and recursive calls that are not returned. Error messages include line/column
and a fix hint.

## Integration lab

[`demo-loom-flow/`](demo-loom-flow/) runs Loom + [flow-xray](https://github.com/kroq86/flow-xray) + optional Ollama:

```bash
cd demo-loom-flow && python run_all_cases.py
```

See [`demo-loom-flow/README.md`](demo-loom-flow/README.md) and [`demo-loom-flow/ROADMAP.md`](demo-loom-flow/ROADMAP.md).

## Run

```bash
python3 -m unittest tests.test_loom_tailcalls
python3 -m unittest discover -s tests
python3 examples/agent_tool_loop.py
python3 examples/agent_loop_with_budget.py
python3 examples/agent_loop_with_hooks.py
python3 examples/deep_agent_loop_comparison.py
python3 examples/streaming_agent.py
python3 scripts/bench_tailcalls.py --n 100000 --samples 5
```

Optional local Ollama fuzzing chooses trusted test templates and bounded
parameters; it does not execute model-written Python. In `demo-loom-flow`,
case 07 runs automatically when Ollama is reachable. To run the unittest directly:

```bash
LOOM_OLLAMA_FUZZ=1 python3 -m unittest tests.test_ollama_contract
```

Skip Ollama in the demo runner: `LOOM_SKIP_OLLAMA=1 python demo-loom-flow/run_all_cases.py`

## API

- `tailrec`: optimizes async self-tail recursion via `return await fn(...)`.
- `tailstream`: optimizes terminal recursive async-generator streaming loops.
- `explain_tailcalls`: returns readable or JSON-serializable transform metadata.

## Formal Core

Loom's correctness contract is specified in [docs/formal-core.md](docs/formal-core.md).
Future transforms should preserve that model or extend it explicitly before
accepting new source shapes.

Mathematically, Loom is grounded in operational semantics, transition systems,
partial functions, induction, and semantics-preserving program transformation.
Set theory supplies the domains; operational semantics supplies the execution
model.

Opcode tracing and performance expectations are described in
[docs/performance-tracing.md](docs/performance-tracing.md).

Concurrency expectations are described in
[docs/concurrency-contract.md](docs/concurrency-contract.md).
