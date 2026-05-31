# Performance And Opcode Tracing

Loom does not treat Python opcode numbers or exact bytecode offsets as a stable
public contract. CPython may change or specialize bytecode between versions.
The stable contract is the semantic and structural shape of the transformed
function.

## Bytecode Shape Contract

For accepted `@tailrec` functions, the transformed bytecode must satisfy:

```text
self recursive function name is absent from bytecode globals/names
JUMP_BACKWARD is present
RETURN_VALUE is present on terminal paths
```

For accepted `@tailstream` functions:

```text
self recursive function name is absent from bytecode globals/names
JUMP_BACKWARD is present
YIELD_VALUE is present
```

For direct-rebinding sites, `__loom_bind` is absent and temporary locals named
`__loom_next_*` are present. For fallback binding sites, `__loom_bind` remains
present.

The bytecode tests check these shape properties rather than exact instruction
offsets.

## Runtime Opcode Tracing

Runtime opcode tracing uses `sys.settrace` with `frame.f_trace_opcodes = True`
and filters events to the transformed function's code object. The trace should
show execution through the loop instruction and terminal return path for a
small input.

This proves the optimized function is executing as a loop at runtime. It does
not replace semantic tests; it complements them.

## Performance Model

Let:

```text
K = number of tail transitions
n = number of parameters
kwargs = number of keyword arguments
```

The transformed function keeps runtime linear in `K`:

```text
time = O(K * step_cost)
stack = O(1)
```

Each tail transition pays rebinding overhead:

```text
extra = O(n + kwargs)
```

Loom uses three rebinding paths:

```text
direct rebinding  exact full-arity positional self-calls
fast binding      simple positional-or-keyword and keyword-only signatures
signature binding conservative fallback through inspect.Signature.bind
```

The direct path avoids `inspect.Signature.bind`, tuple construction, kwargs dict
construction, bound dict construction, and bound dict lookups. It evaluates all
next argument values into temporary locals before assigning parameters, so
aliasing such as `return await f(acc, n)` is preserved.

The fast path avoids `inspect.Signature.bind` on every tail transition when the
call still needs simple Python-style binding. Complex signatures fall back to
Python's binding machinery. This keeps semantic preservation ahead of speed.

This means Loom is primarily a stack-safety transform. It is not guaranteed to
beat a hand-written loop for tiny bodies. The target is to make Loom
substantially cheaper than the old per-step `inspect.Signature.bind` path while
preserving Python call semantics.

Use the benchmark helper for local measurement:

```bash
python3 scripts/bench_tailcalls.py --n 100000 --samples 5
```

Example output on CPython 3.11+ with direct rebinding:

```text
n=100000
samples=5
binding=direct
hand_loop best=0.002670s ... per_iter_best=0.027us
loom_loop best=0.003049s ... per_iter_best=0.030us
loom_to_hand_best_ratio=1.14x
```

Treat `loom_to_hand_best_ratio` as **expected overhead**, not a speed promise.
Numbers vary by Python version, CPU, signature shape, and binding path. Loom
targets stack safety first; a hand-written `while` loop remains the baseline for
minimum per-iteration cost on tiny bodies.

Rough expectations:

```text
direct rebinding   ~1.1–1.3x versus an equivalent hand-written while loop
fast binding       somewhat higher; still avoids per-step Signature.bind
signature binding  highest; used for complex signatures only
```

## Known Pressure Points

1. Exact opcodes and offsets may differ across CPython versions.
2. CPython adaptive specialization may change displayed instruction forms.
3. Fallback `__loom_bind` still has measurable cost and may dominate tiny loop
   bodies that cannot use direct rebinding.
4. Source-backed transformation requires `inspect.getsource`, so stdin-created
   functions are rejected.
5. Unsupported source shapes are rejected rather than optimized unsafely.
