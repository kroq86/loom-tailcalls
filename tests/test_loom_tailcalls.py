import inspect
import unittest

from loom import TailCallError, explain_tailcalls, tailrec, tailstream


class TestAsyncTailRec(unittest.IsolatedAsyncioTestCase):
    async def test_deep_async_tail_recursion_does_not_grow_stack(self) -> None:
        @tailrec
        async def countdown(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            return await countdown(n - 1, acc + 1)

        self.assertEqual(await countdown(100_000), 100_000)

    async def test_await_inside_body_works(self) -> None:
        async def one() -> int:
            return 1

        @tailrec
        async def loop(n: int, acc: int = 0) -> int:
            if n == 0:
                return acc
            value = await one()
            return await loop(n - 1, acc + value)

        self.assertEqual(await loop(10), 10)

    async def test_argument_evaluation_order_is_preserved(self) -> None:
        seen: list[str] = []

        def first(value: int) -> int:
            seen.append("first")
            return value - 1

        def second(value: int) -> int:
            seen.append("second")
            return value + 10

        @tailrec
        async def loop(n: int, acc: int) -> int:
            if n <= 0:
                return acc
            return await loop(first(n), second(acc))

        self.assertEqual(await loop(2, 0), 20)
        self.assertEqual(seen, ["first", "second", "first", "second"])

    async def test_keywords_and_defaults_follow_python_call_semantics(self) -> None:
        @tailrec
        async def loop(n: int, *, step: int = 1, acc: int = 0) -> int:
            if n <= 0:
                return acc
            return await loop(n - step, acc=acc + step)

        self.assertEqual(await loop(3), 3)
        self.assertEqual(await loop(6, step=2), 6)

    async def test_introspection_metadata_is_preserved(self) -> None:
        @tailrec
        async def sample(n: int, acc: int = 0) -> int:
            """sample doc"""
            if n == 0:
                return acc
            return await sample(n - 1, acc + 1)

        self.assertEqual(sample.__name__, "sample")
        self.assertEqual(sample.__doc__, "sample doc")
        self.assertEqual(str(inspect.signature(sample)), "(n: int, acc: int = 0) -> int")
        text = explain_tailcalls(sample)
        self.assertIn("optimized tail calls", text)
        data = explain_tailcalls(sample, as_json=True)
        self.assertEqual(data["mode"], "tailrec")

    async def test_implicit_none_fallthrough_is_preserved(self) -> None:
        @tailrec
        async def maybe_loop(n: int) -> int | None:
            if n > 0:
                return await maybe_loop(n - 1)

        self.assertIsNone(await maybe_loop(0))
        self.assertIsNone(await maybe_loop(3))

    async def test_nested_functions_are_not_part_of_outer_tail_analysis(self) -> None:
        @tailrec
        async def outer(n: int) -> int:
            async def helper() -> object:
                return await outer(n - 1)

            if n <= 0:
                return 0
            _ = helper
            return await outer(n - 1)

        self.assertEqual(await outer(3), 0)


class TestTailStream(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_tail_recursion(self) -> None:
        @tailstream
        async def stream(n: int):
            if n <= 0:
                yield {"type": "final", "n": n}
                return
            yield {"type": "token", "n": n}
            async for item in stream(n - 1):
                yield item
            return

        events = []
        async for item in stream(10_000):
            events.append(item)

        self.assertEqual(events[0], {"type": "token", "n": 10_000})
        self.assertEqual(events[-1], {"type": "final", "n": 0})
        self.assertEqual(len(events), 10_001)

    async def test_stream_fallthrough_terminates(self) -> None:
        @tailstream
        async def stream(n: int):
            if n > 0:
                async for item in stream(n - 1):
                    yield item
                return

        events = []
        async for item in stream(2):
            events.append(item)

        self.assertEqual(events, [])


    async def test_kwargs_expansion_matches_explicit_keywords(self) -> None:
        @tailrec
        async def baseline(n: int, *, step: int = 1) -> int:
            if n <= 0:
                return 0
            return await baseline(n - step, step=step)

        @tailrec
        async def optimized(n: int, *, step: int = 1) -> int:
            if n <= 0:
                return 0
            kwargs = {"step": step}
            return await optimized(n - step, **kwargs)

        self.assertEqual(await optimized(10, step=2), await baseline(10, step=2))

        report = explain_tailcalls(optimized, as_json=True)
        self.assertEqual(report["binding_sites"], ["bind"])


class TestRejections(unittest.TestCase):
    def test_rejects_non_tail_return_expression(self) -> None:
        with self.assertRaises(TailCallError) as ctx:

            @tailrec
            async def bad(n: int) -> int:
                if n == 0:
                    return 0
                return 1 + await bad(n - 1)

        self.assertIn("fix:", str(ctx.exception))
        self.assertIn("return await fn(...)", str(ctx.exception))

    def test_rejects_await_without_return(self) -> None:
        with self.assertRaises(TailCallError) as ctx:

            @tailrec
            async def bad(n: int) -> int:
                if n == 0:
                    return 0
                await bad(n - 1)
                return n

        self.assertIn("fix:", str(ctx.exception))
        self.assertIn("return await fn(...)", str(ctx.exception))

    def test_rejects_recursive_call_in_comprehension(self) -> None:
        with self.assertRaises(TailCallError):

            @tailrec
            async def bad(n: int) -> list[object]:
                if n == 0:
                    return []
                return [bad(n - 1)]

    def test_rejects_recursive_call_in_assignment(self) -> None:
        with self.assertRaises(TailCallError) as ctx:

            @tailrec
            async def bad(n: int) -> int:
                if n == 0:
                    return 0
                value = await bad(n - 1)
                return value

        self.assertIn("fix:", str(ctx.exception))
        self.assertIn("assigning it to a variable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
