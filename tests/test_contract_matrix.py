import asyncio
import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from loom import TailCallError, explain_tailcalls, tailrec, tailstream


_TEMP_DIRS: list[object] = []


def load_module(source: str):
    tempdir = tempfile.TemporaryDirectory()
    _TEMP_DIRS.append(tempdir)
    path = Path(tempdir.name) / "case_module.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"case_module_{len(_TEMP_DIRS)}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    module.__tempdir = tempdir
    return module


async def collect(async_iterable):
    return [item async for item in async_iterable]


class TestTailRecContractMatrix(unittest.IsolatedAsyncioTestCase):
    async def test_positional_varargs_keywords_globals_and_match_are_equivalent(self) -> None:
        module = load_module(
            """
            from loom import tailrec

            GLOBAL_BONUS = 3

            async def baseline(n, acc, /, *items, step=1, scale=2):
                match n % 3:
                    case _ if n <= 0:
                        return acc + len(items) + GLOBAL_BONUS
                    case 0:
                        return await baseline(n - step, acc + scale, *items, step=step, scale=scale)
                    case 1:
                        return await baseline(n - step, acc + scale + len(items), *items, step=step, scale=scale)
                    case _:
                        return await baseline(n - step, acc + scale + GLOBAL_BONUS, *items, step=step, scale=scale)

            @tailrec
            async def optimized(n, acc, /, *items, step=1, scale=2):
                match n % 3:
                    case _ if n <= 0:
                        return acc + len(items) + GLOBAL_BONUS
                    case 0:
                        return await optimized(n - step, acc + scale, *items, step=step, scale=scale)
                    case 1:
                        return await optimized(n - step, acc + scale + len(items), *items, step=step, scale=scale)
                    case _:
                        return await optimized(n - step, acc + scale + GLOBAL_BONUS, *items, step=step, scale=scale)
            """
        )

        expected = await module.baseline(8, 0, "a", "b", step=1, scale=5)
        actual = await module.optimized(8, 0, "a", "b", step=1, scale=5)

        self.assertEqual(actual, expected)

    async def test_argument_exception_trace_is_preserved(self) -> None:
        seen: list[str] = []

        class MarkerError(RuntimeError):
            pass

        def first(value: int) -> int:
            seen.append("first")
            return value - 1

        def explode(_: int) -> int:
            seen.append("explode")
            raise MarkerError("boom")

        @tailrec
        async def loop(n: int, acc: int) -> int:
            if n <= 0:
                return acc
            return await loop(first(n), explode(acc))

        with self.assertRaisesRegex(MarkerError, "boom"):
            await loop(3, 0)
        self.assertEqual(seen, ["first", "explode"])

    async def test_closure_reads_are_preserved(self) -> None:
        offset = 7

        @tailrec
        async def loop(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc + offset
            return await loop(n - 1, acc + offset)

        self.assertEqual(await loop(3), 28)

    async def test_explain_untransformed_reports_outer_calls_only(self) -> None:
        async def analyzed(n: int) -> int:
            async def helper() -> object:
                return await analyzed(n - 1)

            if n <= 0:
                return 0
            _ = helper
            return await analyzed(n - 1)

        report = explain_tailcalls(analyzed, as_json=True)

        self.assertEqual(report["mode"], "tailrec")
        self.assertEqual(len(report["rejected"]), 1)


class TestKwargsSpreadContractMatrix(unittest.IsolatedAsyncioTestCase):
    async def test_kwargs_merge_order_and_explicit_override(self) -> None:
        module = load_module(
            """
            from loom import tailrec

            async def baseline(n, *, step=1, acc=0):
                if n <= 0:
                    return acc
                first = {"acc": acc + step}
                second = {"step": step}
                return await baseline(n - step, **first, **second)

            @tailrec
            async def optimized(n, *, step=1, acc=0):
                if n <= 0:
                    return acc
                first = {"acc": acc + step}
                second = {"step": step}
                return await optimized(n - step, **first, **second)
            """
        )

        self.assertEqual(await module.optimized(4, step=1), await module.baseline(4, step=1))

    async def test_kwargs_bind_raises_type_error(self) -> None:
        @tailrec
        async def loop(n: int, *, step: int = 1) -> int:
            if n <= 0:
                return 0
            kwargs = {"nope": step}
            return await loop(n - step, **kwargs)

        with self.assertRaises(TypeError):
            await loop(2)

    async def test_kwargs_explicit_overrides_spread(self) -> None:
        @tailrec
        async def baseline(n: int, *, step: int = 1, acc: int = 0) -> int:
            if n <= 0:
                return acc
            extra = {"step": step + 9, "acc": acc + step}
            return await baseline(n - step, **extra, step=step)

        @tailrec
        async def optimized(n: int, *, step: int = 1, acc: int = 0) -> int:
            if n <= 0:
                return acc
            extra = {"step": step + 9, "acc": acc + step}
            return await optimized(n - step, **extra, step=step)

        self.assertEqual(await optimized(3, step=1), await baseline(3, step=1))


class TestStructuredTailPositions(unittest.IsolatedAsyncioTestCase):
    async def test_try_except_retry_is_equivalent(self) -> None:
        class TransientError(RuntimeError):
            pass

        async def baseline(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            try:
                if n % 3 == 0:
                    raise TransientError("retry")
                value = 1
            except TransientError:
                return await baseline(n - 1, acc)
            return await baseline(n - 1, acc + value)

        @tailrec
        async def optimized(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            try:
                if n % 3 == 0:
                    raise TransientError("retry")
                value = 1
            except TransientError:
                return await optimized(n - 1, acc)
            return await optimized(n - 1, acc + value)

        self.assertEqual(await optimized(8), await baseline(8))

    async def test_async_with_scope_is_equivalent(self) -> None:
        class Lock:
            def __init__(self) -> None:
                self.active = 0
                self.peak = 0

            async def __aenter__(self) -> None:
                self.active += 1
                self.peak = max(self.peak, self.active)

            async def __aexit__(self, *_: object) -> None:
                self.active -= 1

        lock = Lock()

        async def baseline(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            async with lock:
                return await baseline(n - 1, acc + 1)

        @tailrec
        async def optimized(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            async with lock:
                return await optimized(n - 1, acc + 1)

        self.assertEqual(await optimized(5), await baseline(5))

    async def test_for_loop_tail_call_is_equivalent(self) -> None:
        async def baseline(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            for _ in range(1):
                return await baseline(n - 1, acc + 1)
            return acc

        @tailrec
        async def optimized(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            for _ in range(1):
                return await optimized(n - 1, acc + 1)
            return acc

        self.assertEqual(await optimized(4), await baseline(4))

    async def test_while_loop_tail_call_is_equivalent(self) -> None:
        async def baseline(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            while n > 0:
                return await baseline(n - 1, acc + 1)
            return acc

        @tailrec
        async def optimized(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            while n > 0:
                return await optimized(n - 1, acc + 1)
            return acc

        self.assertEqual(await optimized(4), await baseline(4))

    async def test_agent_shaped_try_and_async_with_fixture(self) -> None:
        class TransientError(RuntimeError):
            pass

        class Session:
            def __init__(self) -> None:
                self.steps = 0

            async def step(self, n: int) -> int:
                self.steps += 1
                if n % 4 == 0:
                    raise TransientError("retry")
                return 1

            def lock(self) -> "Lock":
                return Lock()

        class Lock:
            def __init__(self) -> None:
                self.entries = 0

            async def __aenter__(self) -> None:
                self.entries += 1

            async def __aexit__(self, *_: object) -> None:
                pass

        session = Session()

        async def baseline(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            try:
                value = await session.step(n)
            except TransientError:
                return await baseline(n - 1, acc)
            async with session.lock():
                return await baseline(n - 1, acc + value)

        @tailrec
        async def optimized(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            try:
                value = await session.step(n)
            except TransientError:
                return await optimized(n - 1, acc)
            async with session.lock():
                return await optimized(n - 1, acc + value)

        self.assertEqual(await optimized(6), await baseline(6))


class TestTailStreamContractMatrix(unittest.IsolatedAsyncioTestCase):
    async def test_stream_equivalence_with_branches(self) -> None:
        module = load_module(
            """
            from loom import tailstream

            async def baseline(n):
                if n <= 0:
                    yield ("final", n)
                    return
                if n % 2 == 0:
                    yield ("even", n)
                else:
                    yield ("odd", n)
                async for item in baseline(n - 1):
                    yield item
                return

            @tailstream
            async def optimized(n):
                if n <= 0:
                    yield ("final", n)
                    return
                if n % 2 == 0:
                    yield ("even", n)
                else:
                    yield ("odd", n)
                async for item in optimized(n - 1):
                    yield item
                return
            """
        )

        self.assertEqual(await collect(module.optimized(6)), await collect(module.baseline(6)))


class TestRejectionContractMatrix(unittest.TestCase):
    def test_rejects_recursive_call_in_if_condition(self) -> None:
        with self.assertRaises(TailCallError):

            @tailrec
            async def bad(n: int) -> int:
                if await bad(n - 1):
                    return 1
                return 0

    def test_rejects_recursive_call_in_try_finally(self) -> None:
        with self.assertRaises(TailCallError):

            @tailrec
            async def bad(n: int) -> int:
                try:
                    pass
                finally:
                    return await bad(n - 1)

    def test_rejects_recursive_call_in_with_context(self) -> None:
        with self.assertRaises(TailCallError):

            @tailrec
            async def bad(n: int) -> int:
                with await bad(n - 1):
                    return 0

    def test_rejects_recursive_call_in_for_iter(self) -> None:
        with self.assertRaises(TailCallError):

            @tailrec
            async def bad(n: int) -> int:
                for _ in await bad(n - 1):
                    return 0
                return 0

    def test_rejects_stream_extra_body_statement(self) -> None:
        with self.assertRaises(TailCallError):

            @tailstream
            async def bad(n: int):
                async for item in bad(n - 1):
                    item = item
                    yield item
                return

    def test_rejects_stream_async_for_else(self) -> None:
        with self.assertRaises(TailCallError):

            @tailstream
            async def bad(n: int):
                async for item in bad(n - 1):
                    yield item
                else:
                    yield "else"
                return

    def test_rejects_stream_mismatched_yield_name(self) -> None:
        with self.assertRaises(TailCallError):

            @tailstream
            async def bad(n: int):
                async for item in bad(n - 1):
                    yield other
                return

    def test_rejects_stream_missing_terminal_return(self) -> None:
        with self.assertRaises(TailCallError):

            @tailstream
            async def bad(n: int):
                async for item in bad(n - 1):
                    yield item

    def test_rejects_stream_nonterminal_recursive_call(self) -> None:
        with self.assertRaises(TailCallError):

            @tailstream
            async def bad(n: int):
                async for item in bad(n - 1):
                    yield item
                yield "after"
                return


async def source():
    if False:
        yield None


if __name__ == "__main__":
    unittest.main()
