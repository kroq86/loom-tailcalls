import unittest

from loom import explain_tailcalls, tailrec, tailstream


class TestBindingPlan(unittest.IsolatedAsyncioTestCase):
    async def test_direct_binding_matches_full_positional_call(self) -> None:
        @tailrec
        async def loop(n: int, acc: int) -> int:
            if n <= 0:
                return acc
            return await loop(n - 1, acc + 1)

        report = explain_tailcalls(loop, as_json=True)

        self.assertEqual(report["binding"], "direct")
        self.assertEqual(report["binding_sites"], ["direct"])
        self.assertEqual(await loop(5, 0), 5)

    async def test_fast_binding_matches_positional_and_defaults(self) -> None:
        @tailrec
        async def loop(n: int, acc: int = 0, step: int = 1) -> int:
            if n <= 0:
                return acc
            return await loop(n - step, acc + step)

        report = explain_tailcalls(loop, as_json=True)

        self.assertEqual(report["binding"], "fast")
        self.assertEqual(report["binding_sites"], ["bind"])
        self.assertEqual(await loop(5), 5)

    async def test_fast_binding_matches_keyword_only_defaults(self) -> None:
        @tailrec
        async def loop(n: int, *, step: int = 1, acc: int = 0) -> int:
            if n <= 0:
                return acc
            return await loop(n - step, step=step, acc=acc + step)

        report = explain_tailcalls(loop, as_json=True)

        self.assertEqual(report["binding"], "fast")
        self.assertEqual(await loop(8, step=2), 8)

    async def test_direct_binding_uses_temps_for_aliasing(self) -> None:
        @tailrec
        async def rotate(left: int, right: int, steps: int) -> tuple[int, int]:
            if steps <= 0:
                return left, right
            return await rotate(right, left, steps - 1)

        report = explain_tailcalls(rotate, as_json=True)

        self.assertEqual(report["binding"], "direct")
        self.assertEqual(await rotate(1, 2, 1), (2, 1))
        self.assertEqual(await rotate(1, 2, 2), (1, 2))

    async def test_direct_binding_evaluates_arguments_left_to_right_once(self) -> None:
        seen: list[str] = []

        def first(value: int) -> int:
            seen.append("first")
            return value - 1

        def second(value: int) -> int:
            seen.append("second")
            return value + 1

        @tailrec
        async def loop(n: int, acc: int) -> int:
            if n <= 0:
                return acc
            return await loop(first(n), second(acc))

        self.assertEqual(await loop(2, 0), 2)
        self.assertEqual(seen, ["first", "second", "first", "second"])

    async def test_direct_binding_preserves_exception_during_argument_evaluation(self) -> None:
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
            await loop(1, 0)
        self.assertEqual(seen, ["first", "explode"])

    async def test_fast_binding_raises_type_error_for_unknown_keyword(self) -> None:
        @tailrec
        async def loop(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            return await loop(n - 1, acc=acc + 1, nope=True)

        with self.assertRaises(TypeError):
            await loop(1)

    async def test_fast_binding_raises_type_error_for_duplicate_argument(self) -> None:
        @tailrec
        async def loop(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            return await loop(n - 1, n=0)

        with self.assertRaises(TypeError):
            await loop(1)

    async def test_fast_binding_raises_type_error_for_missing_required_argument(self) -> None:
        @tailrec
        async def loop(n: int, acc: int) -> int:
            if n <= 0:
                return acc
            return await loop(n - 1)

        with self.assertRaises(TypeError):
            await loop(1, 0)

    async def test_complex_signature_uses_signature_fallback(self) -> None:
        @tailrec
        async def loop(n: int, acc: int = 0, *items: object) -> int:
            if n <= 0:
                return acc + len(items)
            return await loop(n - 1, acc + 1, *items)

        report = explain_tailcalls(loop, as_json=True)

        self.assertEqual(report["binding"], "signature")
        self.assertTrue(report["binding_reasons"])
        self.assertEqual(await loop(3, 0, "x", "y"), 5)

    async def test_tailstream_direct_binding_preserves_events(self) -> None:
        @tailstream
        async def stream(n: int):
            if n <= 0:
                yield ("final", n)
                return
            yield ("token", n)
            async for item in stream(n - 1):
                yield item
            return

        report = explain_tailcalls(stream, as_json=True)
        events = [item async for item in stream(3)]

        self.assertEqual(report["binding"], "direct")
        self.assertEqual(events, [("token", 3), ("token", 2), ("token", 1), ("final", 0)])


if __name__ == "__main__":
    unittest.main()
