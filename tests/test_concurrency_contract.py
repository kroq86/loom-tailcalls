import asyncio
import concurrent.futures
import importlib.util
import multiprocessing
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


_TEMP_DIRS: list[object] = []


def load_module(source: str):
    tempdir = tempfile.TemporaryDirectory()
    _TEMP_DIRS.append(tempdir)
    path = Path(tempdir.name) / "concurrency_case.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"concurrency_case_{len(_TEMP_DIRS)}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.__tempdir = tempdir
    return module


def concurrency_source() -> str:
    return """
        from loom import tailrec, tailstream

        async def asyncio_sleep():
            import asyncio
            await asyncio.sleep(0)

        @tailrec
        async def countdown(n: int, acc: int = 0) -> int:
            if n <= 0:
                return acc
            return await countdown(n - 1, acc + 1)

        @tailrec
        async def traced(n: int, label: str, seen: list[str]):
            if n <= 0:
                seen.append(f"{label}:done")
                return list(seen)
            seen.append(f"{label}:{n}")
            await asyncio_sleep()
            return await traced(n - 1, label, seen)

        @tailstream
        async def stream(n: int):
            if n <= 0:
                yield ("final", n)
                return
            yield ("token", n)
            async for item in stream(n - 1):
                yield item
            return
    """


def process_entry(module_path: str, n: int) -> int:
    spec = importlib.util.spec_from_file_location("process_case", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return asyncio.run(module.countdown(n))


class TestConcurrencyContract(unittest.TestCase):
    def test_asyncio_gather_runs_independent_coroutines(self) -> None:
        module = load_module(concurrency_source())

        async def run():
            return await asyncio.gather(
                module.countdown(1000),
                module.countdown(1500),
                module.countdown(2000),
            )

        self.assertEqual(asyncio.run(run()), [1000, 1500, 2000])

    def test_asyncio_interleaving_preserves_per_task_state(self) -> None:
        module = load_module(concurrency_source())

        async def run():
            left, right = await asyncio.gather(
                module.traced(3, "left", []),
                module.traced(3, "right", []),
            )
            return left, right

        left, right = asyncio.run(run())

        self.assertEqual(left, ["left:3", "left:2", "left:1", "left:done"])
        self.assertEqual(right, ["right:3", "right:2", "right:1", "right:done"])

    def test_thread_pool_runs_independent_event_loops(self) -> None:
        module = load_module(concurrency_source())

        def run(n: int) -> int:
            return asyncio.run(module.countdown(n))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(run, [500, 750, 1000, 1250]))

        self.assertEqual(results, [500, 750, 1000, 1250])

    def test_multiprocessing_spawn_runs_source_backed_module(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        _TEMP_DIRS.append(tempdir)
        path = Path(tempdir.name) / "process_case.py"
        path.write_text(textwrap.dedent(concurrency_source()), encoding="utf-8")

        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=2,
            mp_context=context,
        ) as executor:
            results = list(executor.map(process_entry, [str(path)] * 3, [300, 450, 600]))

        self.assertEqual(results, [300, 450, 600])

    def test_greenlet_runs_when_installed(self) -> None:
        try:
            from greenlet import greenlet
        except ModuleNotFoundError:
            self.skipTest("greenlet is not installed")

        module = load_module(concurrency_source())

        def run() -> int:
            return asyncio.run(module.countdown(700))

        worker = greenlet(run)
        self.assertEqual(worker.switch(), 700)


if __name__ == "__main__":
    unittest.main()
