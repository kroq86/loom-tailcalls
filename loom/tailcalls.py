"""AST-backed tail-call optimization for async Python functions.

The public decorators in this module intentionally support a narrow but useful
shape: self-recursive tail calls in async functions and async generators. That
keeps the transform predictable enough for agent loops while still avoiding
Python call-stack growth.
"""

from __future__ import annotations

import ast
import functools
import inspect
import json
import textwrap
from dataclasses import dataclass, field
from types import FunctionType
from typing import Any, Callable


class TailCallError(SyntaxError):
    """Raised when a function cannot be safely tail-call optimized."""


@dataclass(frozen=True)
class TailCallSite:
    line: int
    column: int
    kind: str


@dataclass
class TailCallReport:
    function: str
    mode: str
    binding: str = "signature"
    binding_reasons: list[str] = field(default_factory=list)
    binding_sites: list[str] = field(default_factory=list)
    optimized: list[TailCallSite] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "mode": self.mode,
            "binding": self.binding,
            "binding_reasons": list(self.binding_reasons),
            "binding_sites": list(self.binding_sites),
            "optimized": [
                {"line": site.line, "column": site.column, "kind": site.kind}
                for site in self.optimized
            ],
            "rejected": list(self.rejected),
        }

    def format(self) -> str:
        lines = [f"{self.function}: {self.mode}"]
        lines.append(f"binding: {self.binding}")
        if self.binding_reasons:
            lines.extend(f"  {item}" for item in self.binding_reasons)
        if self.binding_sites:
            lines.append("binding sites:")
            lines.extend(f"  {item}" for item in self.binding_sites)
        if self.optimized:
            lines.append("optimized tail calls:")
            for site in self.optimized:
                lines.append(f"  line {site.line}:{site.column} {site.kind}")
        else:
            lines.append("optimized tail calls: none")
        if self.rejected:
            lines.append("rejected recursive calls:")
            lines.extend(f"  {item}" for item in self.rejected)
        return "\n".join(lines)


def tailrec(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Optimize ``return await fn(...)`` self-recursion in an async function."""

    if not inspect.iscoroutinefunction(fn):
        raise TailCallError("@tailrec currently supports async def functions only")
    return _compile_transformed(fn, mode="tailrec")


def tailstream(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Optimize terminal recursive async-generator streaming loops."""

    if not inspect.isasyncgenfunction(fn):
        raise TailCallError("@tailstream supports async generator functions only")
    return _compile_transformed(fn, mode="tailstream")


def explain_tailcalls(fn: Callable[..., Any], *, as_json: bool = False) -> str | dict[str, Any]:
    """Return a readable or JSON-serializable summary of Loom's transform."""

    report = getattr(fn, "__loom_tailcalls__", None)
    if report is None:
        report = _analyze_untransformed(fn)
    data = report.to_dict()
    if as_json:
        return data
    return report.format()


def _compile_transformed(fn: Callable[..., Any], *, mode: str) -> Callable[..., Any]:
    source = _get_source(fn)
    module = ast.parse(source)
    func = _find_target_function(module, fn.__name__)
    func.decorator_list = []

    signature = inspect.signature(fn)
    param_names = list(signature.parameters)
    report = TailCallReport(function=fn.__qualname__, mode=mode)
    binder, binding, binding_reasons = _make_binder(signature)
    report.binding = binding
    report.binding_reasons = binding_reasons
    direct_param_names = _direct_param_names(signature)

    if mode == "tailrec":
        transformer: ast.NodeTransformer = _TailRecTransformer(
            fn.__name__, param_names, direct_param_names, report
        )
    else:
        transformer = _TailStreamTransformer(fn.__name__, param_names, direct_param_names, report)

    transformed_func = transformer.visit(func)
    assert isinstance(transformed_func, (ast.AsyncFunctionDef, ast.FunctionDef))
    loop_body = [*transformed_func.body, ast.Return(value=None)]
    loop = ast.While(test=ast.Constant(value=True), body=loop_body, orelse=[])
    transformed_func.body = [_async_generator_marker(), loop] if mode == "tailstream" else [loop]

    module.body = [transformed_func]
    ast.fix_missing_locations(module)

    if report.binding_sites and all(site == "direct" for site in report.binding_sites):
        report.binding = "direct"

    env = _execution_env(fn)
    env["__loom_bind"] = binder
    code = compile(module, filename=inspect.getsourcefile(fn) or "<loom-tailcalls>", mode="exec")
    exec(code, env)
    optimized = env[fn.__name__]
    functools.update_wrapper(optimized, fn)
    optimized.__signature__ = signature  # type: ignore[attr-defined]
    optimized.__loom_tailcalls__ = report  # type: ignore[attr-defined]
    return optimized


def _get_source(fn: Callable[..., Any]) -> str:
    try:
        return textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:
        raise TailCallError(
            f"{fn.__qualname__} needs source-backed code for AST tail-call optimization"
        ) from exc


def _find_target_function(module: ast.Module, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    for item in module.body:
        if isinstance(item, (ast.AsyncFunctionDef, ast.FunctionDef)) and item.name == name:
            return item
    raise TailCallError(f"could not locate function {name!r} in source")


def _execution_env(fn: Callable[..., Any]) -> dict[str, Any]:
    env = dict(fn.__globals__)
    if fn.__closure__:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__, strict=False):
            try:
                env[name] = cell.cell_contents
            except ValueError:
                # A recursive local function can have an empty self-reference
                # cell while its decorator is still running.
                continue
    return env


@dataclass(frozen=True)
class _FastBindingPlan:
    positional_names: tuple[str, ...]
    keyword_only_names: tuple[str, ...]
    param_names: tuple[str, ...]
    param_set: frozenset[str]
    defaults: dict[str, Any]
    required: frozenset[str]


def _make_binder(signature: inspect.Signature) -> tuple[Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]], str, list[str]]:
    plan, reasons = _fast_binding_plan(signature)
    if plan is None:
        return functools.partial(_bind_next_arguments, signature), "signature", reasons
    return functools.partial(_fast_bind_next_arguments, plan), "fast", []


def _fast_binding_plan(signature: inspect.Signature) -> tuple[_FastBindingPlan | None, list[str]]:
    positional_names: list[str] = []
    keyword_only_names: list[str] = []
    defaults: dict[str, Any] = {}
    required: set[str] = set()
    reasons: list[str] = []

    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            positional_names.append(param.name)
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            keyword_only_names.append(param.name)
        else:
            reasons.append(f"fallback: unsupported parameter kind {param.kind.description} for {param.name}")
            continue

        if param.default is inspect.Parameter.empty:
            required.add(param.name)
        else:
            defaults[param.name] = param.default

    if reasons:
        return None, reasons
    return (
        _FastBindingPlan(
            positional_names=tuple(positional_names),
            keyword_only_names=tuple(keyword_only_names),
            param_names=tuple(positional_names + keyword_only_names),
            param_set=frozenset(positional_names + keyword_only_names),
            defaults=defaults,
            required=frozenset(required),
        ),
        [],
    )


def _direct_param_names(signature: inspect.Signature) -> list[str] | None:
    names: list[str] = []
    for param in signature.parameters.values():
        if param.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            return None
        names.append(param.name)
    return names


def _fast_bind_next_arguments(plan: _FastBindingPlan, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) > len(plan.positional_names):
        raise TypeError(f"too many positional arguments: expected at most {len(plan.positional_names)}")

    bound: dict[str, Any] = {}
    for index, value in enumerate(args):
        bound[plan.positional_names[index]] = value

    for key, value in kwargs.items():
        if key not in plan.param_set:
            raise TypeError(f"got an unexpected keyword argument {key!r}")
        if key in bound:
            raise TypeError(f"multiple values for argument {key!r}")
        bound[key] = value

    for name in plan.param_names:
        if name not in bound:
            if name in plan.defaults:
                bound[name] = plan.defaults[name]
            elif name in plan.required:
                raise TypeError(f"missing a required argument: {name!r}")

    return bound


def _bind_next_arguments(signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _async_generator_marker() -> ast.If:
    return ast.If(
        test=ast.Constant(value=False),
        body=[ast.Expr(value=ast.Yield(value=ast.Constant(value=None)))],
        orelse=[],
    )


class _RecursiveCallFinder(ast.NodeVisitor):
    def __init__(self, function_name: str) -> None:
        self.function_name = function_name
        self.calls: list[ast.Call] = []
        self._function_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._function_depth > 0:
            return None
        self._function_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self._function_depth -= 1
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._function_depth > 0:
            return None
        self._function_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self._function_depth -= 1
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None

    def visit_Call(self, node: ast.Call) -> None:
        if _is_self_call(node, self.function_name):
            self.calls.append(node)
        self.generic_visit(node)


class _TailRecTransformer(ast.NodeTransformer):
    def __init__(
        self,
        function_name: str,
        param_names: list[str],
        direct_param_names: list[str] | None,
        report: TailCallReport,
    ) -> None:
        self.function_name = function_name
        self.param_names = param_names
        self.direct_param_names = direct_param_names
        self.report = report
        self._inside_target = False

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        if self._inside_target:
            return node
        self._inside_target = True
        node.body = _flatten_statements([self.visit(stmt) for stmt in node.body])
        self._inside_target = False
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return node

    def visit_Return(self, node: ast.Return) -> list[ast.stmt] | ast.Return:
        self.generic_visit(node)
        if node.value is not None and _is_tail_await_call(node.value, self.function_name):
            call = node.value.value  # Await.value
            assert isinstance(call, ast.Call)
            statements, binding_site = _continuation_statements(
                call, self.param_names, self.direct_param_names
            )
            self.report.binding_sites.append(binding_site)
            self.report.optimized.append(
                TailCallSite(node.lineno, node.col_offset, f"return await self(...) [{binding_site}]")
            )
            return statements

        calls = _find_recursive_calls(node.value, self.function_name)
        if calls:
            raise _non_tail_error(self.function_name, calls[0], "recursive call is not in tail position")
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        calls = _find_recursive_calls(node.test, self.function_name)
        if calls:
            raise _non_tail_error(self.function_name, calls[0], "recursive call in if condition is not tail position")
        node.body = [self.visit(stmt) for stmt in node.body]  # type: ignore[list-item]
        node.orelse = [self.visit(stmt) for stmt in node.orelse]  # type: ignore[list-item]
        node.body = _flatten_statements(node.body)
        node.orelse = _flatten_statements(node.orelse)
        return node

    def visit_Match(self, node: ast.Match) -> ast.Match:
        calls = _find_recursive_calls(node.subject, self.function_name)
        if calls:
            raise _non_tail_error(self.function_name, calls[0], "recursive call in match subject is not tail position")
        for case in node.cases:
            if case.guard is not None:
                guard_calls = _find_recursive_calls(case.guard, self.function_name)
                if guard_calls:
                    raise _non_tail_error(
                        self.function_name,
                        guard_calls[0],
                        "recursive call in match guard is not tail position",
                    )
            case.body = _flatten_statements([self.visit(stmt) for stmt in case.body])
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        calls = _find_recursive_calls(node.value, self.function_name)
        if calls:
            raise _non_tail_error(self.function_name, calls[0], "recursive call must be returned")
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        return self._rejecting_generic_visit(node, "recursive call in assignment is not tail position")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        return self._rejecting_generic_visit(node, "recursive call in assignment is not tail position")

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign:
        return self._rejecting_generic_visit(node, "recursive call in assignment is not tail position")

    def visit_For(self, node: ast.For) -> ast.For:
        return self._rejecting_generic_visit(node, "recursive call in for loop is not tail position")

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AsyncFor:
        return self._rejecting_generic_visit(node, "recursive call in async for loop is not tail position")

    def visit_While(self, node: ast.While) -> ast.While:
        return self._rejecting_generic_visit(node, "recursive call in while loop is not tail position")

    def visit_With(self, node: ast.With) -> ast.With:
        return self._rejecting_generic_visit(node, "recursive call in with block is not tail position")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AsyncWith:
        return self._rejecting_generic_visit(node, "recursive call in async with block is not tail position")

    def visit_Try(self, node: ast.Try) -> ast.Try:
        return self._rejecting_generic_visit(node, "recursive call in try block is not supported in MVP")

    def _rejecting_generic_visit(self, node: ast.AST, reason: str) -> Any:
        calls = _find_recursive_calls(node, self.function_name)
        if calls:
            raise _non_tail_error(self.function_name, calls[0], reason)
        return self.generic_visit(node)


class _TailStreamTransformer(ast.NodeTransformer):
    def __init__(
        self,
        function_name: str,
        param_names: list[str],
        direct_param_names: list[str] | None,
        report: TailCallReport,
    ) -> None:
        self.function_name = function_name
        self.param_names = param_names
        self.direct_param_names = direct_param_names
        self.report = report

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.body = self._rewrite_block(node.body)
        return node

    def _rewrite_block(self, body: list[ast.stmt]) -> list[ast.stmt]:
        rewritten: list[ast.stmt] = []
        i = 0
        while i < len(body):
            stmt = body[i]
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                rewritten.append(stmt)
                i += 1
                continue

            if isinstance(stmt, ast.If):
                stmt.body = self._rewrite_block(stmt.body)
                stmt.orelse = self._rewrite_block(stmt.orelse)
                rewritten.append(stmt)
                i += 1
                continue

            if (
                isinstance(stmt, ast.AsyncFor)
                and i + 1 < len(body)
                and isinstance(body[i + 1], ast.Return)
                and _is_stream_tail_loop(stmt, self.function_name)
            ):
                call = stmt.iter
                assert isinstance(call, ast.Call)
                statements, binding_site = _continuation_statements(
                    call, self.param_names, self.direct_param_names
                )
                self.report.binding_sites.append(binding_site)
                self.report.optimized.append(
                    TailCallSite(
                        stmt.lineno,
                        stmt.col_offset,
                        f"async for yield self(...); return [{binding_site}]",
                    )
                )
                rewritten.extend(statements)
                i += 2
                continue

            calls = _find_recursive_calls(stmt, self.function_name)
            if calls:
                raise _non_tail_error(
                    self.function_name,
                    calls[0],
                    "recursive async generator call must be terminal async-for/yield/return",
                )
            rewritten.append(stmt)
            i += 1
        return rewritten


def _continuation_statements(
    call: ast.Call, param_names: list[str], direct_param_names: list[str] | None
) -> tuple[list[ast.stmt], str]:
    if _is_direct_rebind_call(call, direct_param_names):
        assert direct_param_names is not None
        return _direct_rebind_statements(call, direct_param_names), "direct"
    return _bind_rebind_statements(call, param_names), "bind"


def _is_direct_rebind_call(call: ast.Call, direct_param_names: list[str] | None) -> bool:
    return (
        direct_param_names is not None
        and not call.keywords
        and len(call.args) == len(direct_param_names)
        and not any(isinstance(arg, ast.Starred) for arg in call.args)
    )


def _direct_rebind_statements(call: ast.Call, param_names: list[str]) -> list[ast.stmt]:
    statements: list[ast.stmt] = []
    temp_names = [f"__loom_next_{index}" for index in range(len(param_names))]
    for temp_name, value in zip(temp_names, call.args, strict=True):
        statements.append(
            ast.Assign(
                targets=[ast.Name(id=temp_name, ctx=ast.Store())],
                value=value,
            )
        )
    for param_name, temp_name in zip(param_names, temp_names, strict=True):
        statements.append(
            ast.Assign(
                targets=[ast.Name(id=param_name, ctx=ast.Store())],
                value=ast.Name(id=temp_name, ctx=ast.Load()),
            )
        )
    statements.append(ast.Continue())
    return statements


def _bind_rebind_statements(call: ast.Call, param_names: list[str]) -> list[ast.stmt]:
    assign_args = ast.Assign(
        targets=[ast.Name(id="__loom_args", ctx=ast.Store())],
        value=ast.Tuple(elts=list(call.args), ctx=ast.Load()),
    )
    assign_kwargs = ast.Assign(
        targets=[ast.Name(id="__loom_kwargs", ctx=ast.Store())],
        value=ast.Dict(
            keys=[ast.Constant(value=kw.arg) for kw in call.keywords if kw.arg is not None],
            values=[kw.value for kw in call.keywords if kw.arg is not None],
        ),
    )
    if any(kw.arg is None for kw in call.keywords):
        raise TailCallError("tail recursive calls with **kwargs expansion are not supported in MVP")
    bind = ast.Assign(
        targets=[ast.Name(id="__loom_bound", ctx=ast.Store())],
        value=ast.Call(
            func=ast.Name(id="__loom_bind", ctx=ast.Load()),
            args=[
                ast.Name(id="__loom_args", ctx=ast.Load()),
                ast.Name(id="__loom_kwargs", ctx=ast.Load()),
            ],
            keywords=[],
        ),
    )
    assignments: list[ast.stmt] = [assign_args, assign_kwargs, bind]
    for name in param_names:
        assignments.append(
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id="__loom_bound", ctx=ast.Load()),
                    slice=ast.Constant(value=name),
                    ctx=ast.Load(),
                ),
            )
        )
    assignments.append(ast.Continue())
    return assignments


def _flatten_statements(items: list[ast.stmt | list[ast.stmt]]) -> list[ast.stmt]:
    flattened: list[ast.stmt] = []
    for item in items:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _is_tail_await_call(node: ast.AST, function_name: str) -> bool:
    return isinstance(node, ast.Await) and isinstance(node.value, ast.Call) and _is_self_call(
        node.value, function_name
    )


def _is_self_call(node: ast.Call, function_name: str) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == function_name


def _find_recursive_calls(node: ast.AST | None, function_name: str) -> list[ast.Call]:
    if node is None:
        return []
    finder = _RecursiveCallFinder(function_name)
    finder.visit(node)
    return finder.calls


def _is_stream_tail_loop(node: ast.AsyncFor, function_name: str) -> bool:
    if not (isinstance(node.iter, ast.Call) and _is_self_call(node.iter, function_name)):
        return False
    if len(node.body) != 1 or node.orelse:
        return False
    only = node.body[0]
    return (
        isinstance(only, ast.Expr)
        and isinstance(only.value, ast.Yield)
        and isinstance(only.value.value, ast.Name)
        and isinstance(node.target, ast.Name)
        and only.value.value.id == node.target.id
    )


def _non_tail_error(function_name: str, call: ast.Call, reason: str) -> TailCallError:
    return TailCallError(
        f"{function_name}: {reason} at line {getattr(call, 'lineno', '?')}, "
        f"column {getattr(call, 'col_offset', '?')}"
    )


def _analyze_untransformed(fn: Callable[..., Any]) -> TailCallReport:
    source = _get_source(fn)
    module = ast.parse(source)
    func = _find_target_function(module, fn.__name__)
    mode = "tailstream" if inspect.isasyncgenfunction(fn) else "tailrec"
    report = TailCallReport(function=fn.__qualname__, mode=mode)
    calls = _find_recursive_calls(func, fn.__name__)
    report.rejected = [
        f"line {call.lineno}:{call.col_offset} recursive call has not been optimized"
        for call in calls
    ]
    return report


def _json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)
