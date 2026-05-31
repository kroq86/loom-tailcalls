# Loom Formal Core

This document fixes the mathematical core that future Loom features must
preserve. The core is intentionally smaller than Python: unsupported programs
must be rejected before transformation.

## Mathematical Classification

Loom's core belongs to **program semantics**, more specifically:

```text
small-step operational semantics
+ transition systems
+ partial functions
+ structural induction over finite ASTs
+ induction over execution length
+ semantics-preserving program transformation
```

It is **not** based on red-black tree theory, graph algorithms, or
combinatorics. Set theory is used as the foundation for the domains and
functions, but it is not enough by itself: Loom must preserve ordered
execution, effects, exceptions, and divergence.

The mathematical stack is:

| Layer | Theory | What it gives Loom |
| --- | --- | --- |
| Objects | Set theory | Domains such as `Sigma`, `R`, `X`, `E*`, products, sums, and functions. |
| Execution | Operational semantics | A precise step relation for what the program does next. |
| State change | Transition systems | The `delta` transition from one machine state to a result, exception, or next state. |
| Nontermination | Partial functions | A program may return, raise, or diverge. |
| Recursion | Fixed-point semantics | Recursive behavior is the least solution of the recursive equation. |
| Correctness proof | Induction | Equivalence is proved by induction over the number of tail transitions. |
| AST analysis | Structural induction | The checker terminates because AST traversal descends through finite subtrees. |
| Complexity | Algorithm analysis | Compile-time and runtime costs are bounded by AST size and transition count. |

The central claim is semantic preservation:

```text
T is correct iff forall P in L_supported, forall sigma in Sigma:
  sem(P)(sigma) = sem(T(P))(sigma)
```

The contract is checked from several sides:

1. **Domain checks**: tests cover the supported input forms and unsupported
   forms from `L_supported`.
2. **Trace checks**: tests verify argument order, exactly-once evaluation,
   stream event order, and exception preservation.
3. **State checks**: tests compare undecorated recursive baselines against
   transformed functions.
4. **Rejection checks**: tests require `TailCallError` for forms without a
   proof under this core.
5. **Fuzz checks**: optional Ollama tests select only trusted templates and
   parameter ranges, then reuse the same semantic-equivalence oracle.
6. **Binding-path checks**: tests assert `explain_tailcalls` reports
   `direct` / `fast` / `signature` binding for reference fixtures so
   implementation refinements do not silently regress.
7. **Structured-block checks**: tests compare recursive baselines against
   `@tailrec` for tail calls inside `try`, `with`, `async with`, and loops.
8. **Kwargs-spread checks**: tests verify `**dict` tail calls, merge order,
   explicit override, and preserved binding `TypeError`s.

So the foundation is:

```text
set-theoretic domains
  -> operational transition delta
  -> recursive semantics F
  -> loop semantics G
  -> induction proof F = G
  -> executable contract tests that sample the theorem's cases
```

## Semantic Domain

Let a supported function have parameters:

```text
x1 : A1, x2 : A2, ..., xn : An
```

The parameter configuration space is:

```text
C = A1 x A2 x ... x An
```

For the full execution model, use machine states rather than bare parameter
tuples:

```text
Sigma = Env x Store x C
```

Let:

```text
R = result values
X = exceptions
E = observable effects
E* = finite traces of observable effects
```

One supported execution step is a partial transition:

```text
delta : Sigma partial-> E* x (R + Sigma + X)
```

`delta(sigma) = (tau, in_R(r))` means the function returns `r` after trace
`tau`.

`delta(sigma) = (tau, in_Sigma(sigma'))` means the function performs trace
`tau` and continues via a tail self-call represented by `sigma'`.

`delta(sigma) = (tau, in_X(x))` means the function raises `x` after trace
`tau`.

## Recursive And Loop Semantics

The source recursive semantics `F` is:

```text
F(sigma) =
  (tau, r)          if delta(sigma) = (tau, in_R(r))
  (tau, raise x)    if delta(sigma) = (tau, in_X(x))
  tau . F(sigma')   if delta(sigma) = (tau, in_Sigma(sigma'))
```

The transformed loop semantics `G` is:

```text
G(sigma0):
  sigma := sigma0
  trace := epsilon

  while true:
    delta(sigma) = (tau, y)
    trace := trace . tau

    if y = in_R(r):
      return (trace, r)

    if y = in_X(x):
      raise x with trace

    if y = in_Sigma(sigma'):
      sigma := sigma'
      continue
```

## Correctness Theorem

For every accepted program `P` and initial state `sigma`:

```text
sem(P)(sigma) = sem(T(P))(sigma)
```

Expanded:

```text
F(sigma) returns (tau, r)  iff  G(sigma) returns (tau, r)
F(sigma) raises  (tau, x)  iff  G(sigma) raises  (tau, x)
F(sigma) diverges          iff  G(sigma) diverges
```

The proof is by induction on the number of tail transitions before
termination.

Base case:

```text
delta(sigma) = (tau, in_R(r)) or (tau, in_X(x))
```

Both `F` and `G` stop immediately with the same trace and same terminal
outcome.

Inductive step:

```text
delta(sigma) = (tau, in_Sigma(sigma'))
```

Assume correctness for `sigma'`. `F(sigma)` performs `tau` and recurses to
`F(sigma')`. `G(sigma)` performs `tau`, assigns `sigma := sigma'`, and
continues as `G(sigma')`. By the induction hypothesis, both have the same
remaining behavior.

## Rewriting Rule

The only coroutine tail-call rewrite is:

```text
return await f(a1, ..., am, **s1, ..., **sk, kw1=b1, ..., kwp=bp)
```

where each `**si` is a keyword-dict spread and explicit keywords may follow
spreads. Explicit keywords override earlier spread keys left-to-right.

to:

```text
v_args   := eval_LTR(a1, ..., am, sigma)
v_kw_i   := eval_LTR(si, sigma)           for each spread left-to-right
v_kwargs := merge(v_kw_1, ..., v_kw_k, {kw1: b1, ..., kwp: bp})
beta     := bind(signature_f, v_args, v_kwargs)
sigma'   := update_params(sigma, beta)
continue with sigma'
```

`merge` is left-to-right dict union: later keys win. Each spread expression is
evaluated exactly once.

The required invariant is:

```text
Env(new recursive call f(...)) = Env(after transformed assignment + continue)
```

## Binding Refinement

The mathematical rule uses one abstract operation:

```text
beta = bind(signature_f, v_args, v_kwargs)
```

The implementation may choose one of three concrete rebinding implementations:

```text
bind_impl =
  direct_assign(plan)   for exact full-arity positional self-calls
  fast_bind(plan)       for simple proven signatures
  signature.bind        for conservative fallback
```

The direct path is valid only under this refinement law:

```text
forall full_arity_positional args:
  eval_LTR(args, sigma) = values
  direct_assign(params, values) = bind(signature_f, values, {})
```

with the additional temp-variable condition:

```text
all values are evaluated before any parameter is reassigned
```

The fast binding path is valid only under this refinement law:

```text
forall args, kwargs:
  fast_bind(plan, args, kwargs) = signature.bind(args, kwargs).apply_defaults()
```

or both sides raise a binding `TypeError` before any parameter rebinding occurs.

The direct path is allowed only when the function signature contains only
`POSITIONAL_OR_KEYWORD` parameters and the recursive call supplies exactly one
plain positional argument per parameter, with no keywords, `*args`, or
`**kwargs` spreads.

Tail calls that use `**kwargs` spreads always take the bind path (`bind` site),
never direct assignment. The semantic operation is unchanged.

The fast path is allowed only for signatures whose parameters are:

```text
POSITIONAL_OR_KEYWORD
KEYWORD_ONLY
```

with no `POSITIONAL_ONLY`, `VAR_POSITIONAL`, or `VAR_KEYWORD` parameters. All
other signatures use the fallback binder. Therefore the proof of the transform
does not change: the `bind` node in the rewriting rule is still the semantic
operation. Direct assignment and the fast binder are implementation refinements
of that operation, not new semantic rules.

The strengthened invariant is:

```text
Env(new recursive call f(...))
= Env(after bind_impl/direct_assign + transformed assignment + continue)
```

where `bind_impl` must satisfy the refinement law above.

For async generators, the only streaming tail-call rewrite is the terminal
pattern:

```text
async for item in f(...):
    yield item
return
```

to the same parameter rebinding plus `continue`.

## Structured Tail Positions

A tail-call rewrite is allowed when the terminal statement of a block is:

```text
return await f(...)
```

inside any of:

```text
try body / except handler body / try else
with body / async with body
for body / async for body / while body
if branch body / match case body
```

The outer transformed loop restarts the enclosing function body on `continue`,
so structured blocks execute again under `sigma'` exactly as they would under a
new recursive stack frame.

### Explicit Rejects

These shapes remain outside `L_supported`:

```text
recursive call in try finally
recursive call in except type expression
recursive call in with / async with context expression
recursive call in for iter / while test
recursive call in assignment, expression statement, or non-tail expression
*args spread in tail-call argument list
```

Rejection is preferred to accepting a program whose block ordering or binding
semantics are not proved under this core.

## Soundness And Completeness

Let `L_supported` be the accepted source fragment.

Soundness:

```text
accept(P) => P in L_supported and sem(P) = sem(T(P))
```

This property is mandatory.

Completeness is relative, not absolute:

```text
P in L_supported => accept(P)
```

This is desirable only for the explicitly supported fragment. Loom may reject
valid but unimplemented tail-recursive shapes. Rejection is preferred to
accepting a program whose semantics are not proved.

## Computability

The analysis and transformation are computable because Python ASTs are finite
trees and the checker is structural recursion over strictly smaller subtrees.

For:

```text
N = number of AST nodes
q = number of optimized tail-call sites
n = number of parameters
K = number of runtime tail transitions
```

compile-time analysis and rewriting:

```text
O(N)
```

transformed AST size:

```text
O(N + qn)
```

runtime stack:

```text
source recursion: O(K)
transformed loop: O(1)
```

runtime time remains linear in the number of semantic transitions:

```text
O(K * step_cost)
```

with per-tail-step rebinding overhead:

```text
O(n + kwargs)
```

## Non-Negotiable Invariants

1. Argument evaluation order is preserved.
2. Each argument expression is evaluated exactly once.
3. Python signature binding is preserved, including defaults and keywords.
4. Exceptions raised during argument evaluation or binding are preserved.
5. Fallthrough termination is preserved as `return None` for coroutines and
   generator termination for async generators.
6. Nested functions and classes are not part of the outer function's tail-call
   analysis.
7. The transformed async generator remains an async generator even when all
   observable yields are removed by tail-call rewriting.
8. Unsupported shapes are rejected rather than guessed.
