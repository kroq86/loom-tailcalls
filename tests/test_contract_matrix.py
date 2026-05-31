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

    def test_rejects_recursive_call_in_try_block(self) -> None:
        with self.assertRaises(TailCallError):

            @tailrec
            async def bad(n: int) -> int:
                try:
                    return await bad(n - 1)
                except RuntimeError:
                    return 0

    def test_rejects_recursive_call_in_with_block(self) -> None:
        with self.assertRaises(TailCallError):

            @tailrec
            async def bad(n: int) -> int:
                with open(__file__):
                    return await bad(n - 1)

    def test_rejects_recursive_call_in_loop_body(self) -> None:
        with self.assertRaises(TailCallError):

            @tailrec
            async def bad(n: int) -> int:
                for _ in range(1):
                    return await bad(n - 1)
                return 0

    def test_rejects_recursive_call_in_async_for_body(self) -> None:
        with self.assertRaises(TailCallError):

            @tailrec
            async def bad(n: int) -> int:
                async for _ in source():
                    return await bad(n - 1)
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
