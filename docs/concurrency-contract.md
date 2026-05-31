# Concurrency Contract

Loom transforms one async function body into an explicit loop. It does not
create shared scheduler state, global mutable execution state, worker threads,
or process-local runtime registries. Each call keeps its own Python frame and
locals.

## Supported Execution Contexts

The core contract is expected to hold in these contexts:

```text
asyncio tasks
threads running independent event loops
processes importing source-backed modules
greenlet execution when greenlet is installed
```

The semantic theorem is unchanged:

```text
sem(P)(sigma) = sem(T(P))(sigma)
```

Concurrency only changes how multiple calls are interleaved. It does not
change the single-call transition relation `delta`.

## What The Tests Prove

The concurrency suite checks:

1. `asyncio.gather` can run several optimized coroutines independently.
2. Async interleaving preserves per-task local state and trace order.
3. Multiple OS threads can run optimized functions inside independent event
   loops.
4. Multiprocessing with `spawn` can import a source-backed module and run the
   transformed function in child processes.
5. Greenlet execution is tested when the optional `greenlet` package is
   installed.

## Boundaries

Loom does not make user data thread-safe or process-safe. If the function body
mutates shared objects, normal Python synchronization rules still apply.

Loom also does not guarantee compatibility with monkey-patched schedulers as a
separate theorem. Gevent/eventlet-style behavior should be validated in a
project that depends on those packages. The core transform is scheduler-neutral:
it preserves `await`, `yield`, exceptions, and parameter rebinding, but it does
not control the scheduler.

## Source-Backed Requirement

Child processes and greenlets must execute functions whose source is available
to `inspect.getsource`. Dynamically created stdin functions are rejected by
design.
