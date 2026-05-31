import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import textwrap
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from loom import TailCallError


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
_TEMP_DIRS: list[object] = []


def load_module(source: str):
    tempdir = tempfile.TemporaryDirectory()
    _TEMP_DIRS.append(tempdir)
    path = Path(tempdir.name) / "ollama_case.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"ollama_case_{len(_TEMP_DIRS)}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.__tempdir = tempdir
    return module


def request_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def selected_model() -> str:
    if model := os.environ.get("OLLAMA_MODEL"):
        return model
    tags = request_json("/api/tags")
    models = tags.get("models") or []
    if not models:
        raise AssertionError("Ollama is reachable but returned no models")
    name = models[0].get("name")
    if not isinstance(name, str) or not name:
        raise AssertionError("Ollama /api/tags returned a model without a valid name")
    return name


def generate_cases(model: str, count: int) -> list[dict[str, Any]]:
    prompt = f"""
Return JSON only, with this exact shape:
{{"cases":[{{"template":"tailrec_countdown","n":5,"step":1}}]}}

Choose exactly {count} cases. Use only these template names:
- tailrec_countdown
- tailrec_keywords
- tailrec_fallthrough
- tailrec_reject_assignment
- tailstream_events
- tailstream_reject_mismatch

For every case, include integer n in [0, 20] and integer step in [1, 5].
Do not include Python code.
"""
    response = request_json(
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
    )
    try:
        data = json.loads(response["response"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AssertionError(f"Malformed Ollama response: {response!r}") from exc

    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != count:
        raise AssertionError(f"Expected exactly {count} cases, got {data!r}")
    return cases


def normalized_case(raw: dict[str, Any]) -> tuple[str, int, int]:
    if not isinstance(raw, dict):
        raise AssertionError(f"Malformed case: {raw!r}")
    template = raw.get("template")
    n = raw.get("n")
    step = raw.get("step")
    if template not in TEMPLATES:
        raise AssertionError(f"Unknown template selected by Ollama: {template!r}")
    if not isinstance(n, int) or not 0 <= n <= 20:
        raise AssertionError(f"Invalid n selected by Ollama: {raw!r}")
    if not isinstance(step, int) or not 1 <= step <= 5:
        raise AssertionError(f"Invalid step selected by Ollama: {raw!r}")
    return template, n, step


async def collect(async_iterable):
    return [item async for item in async_iterable]


def source_tailrec_countdown(_: int, __: int) -> str:
    return """
        from loom import tailrec

        async def baseline(n, acc=0):
            if n <= 0:
                return acc
            return await baseline(n - 1, acc + 1)

        @tailrec
        async def optimized(n, acc=0):
            if n <= 0:
                return acc
            return await optimized(n - 1, acc + 1)
    """


def source_tailrec_keywords(_: int, __: int) -> str:
    return """
        from loom import tailrec

        async def baseline(n, *, step=1, acc=0):
            if n <= 0:
                return acc
            return await baseline(n - step, step=step, acc=acc + step)

        @tailrec
        async def optimized(n, *, step=1, acc=0):
            if n <= 0:
                return acc
            return await optimized(n - step, step=step, acc=acc + step)
    """


def source_tailrec_fallthrough(_: int, __: int) -> str:
    return """
        from loom import tailrec

        async def baseline(n):
            if n > 0:
                return await baseline(n - 1)

        @tailrec
        async def optimized(n):
            if n > 0:
                return await optimized(n - 1)
    """


def source_tailrec_reject_assignment(_: int, __: int) -> str:
    return """
        from loom import tailrec

        @tailrec
        async def optimized(n):
            if n <= 0:
                return 0
            value = await optimized(n - 1)
            return value
    """


def source_tailstream_events(_: int, __: int) -> str:
    return """
        from loom import tailstream

        async def baseline(n):
            if n <= 0:
                yield ("final", n)
                return
            yield ("token", n)
            async for item in baseline(n - 1):
                yield item
            return

        @tailstream
        async def optimized(n):
            if n <= 0:
                yield ("final", n)
                return
            yield ("token", n)
            async for item in optimized(n - 1):
                yield item
            return
    """


def source_tailstream_reject_mismatch(_: int, __: int) -> str:
    return """
        from loom import tailstream

        @tailstream
        async def optimized(n):
            async for item in optimized(n - 1):
                yield other
            return
    """


TEMPLATES = {
    "tailrec_countdown": source_tailrec_countdown,
    "tailrec_keywords": source_tailrec_keywords,
    "tailrec_fallthrough": source_tailrec_fallthrough,
    "tailrec_reject_assignment": source_tailrec_reject_assignment,
    "tailstream_events": source_tailstream_events,
    "tailstream_reject_mismatch": source_tailstream_reject_mismatch,
}

REJECT_TEMPLATES = {"tailrec_reject_assignment", "tailstream_reject_mismatch"}
STREAM_TEMPLATES = {"tailstream_events"}


class TestOllamaContract(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("LOOM_RUN_OLLAMA") == "1",
        "set LOOM_RUN_OLLAMA=1 to run optional Ollama contract fuzzing",
    )
    def test_ollama_selected_trusted_templates(self) -> None:
        try:
            model = selected_model()
            count = int(os.environ.get("LOOM_OLLAMA_CASES", "25"))
            cases = generate_cases(model, count)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise AssertionError(f"Ollama request failed: {exc}") from exc

        for raw in cases:
            with self.subTest(raw=raw):
                template, n, step = normalized_case(raw)
                source = TEMPLATES[template](n, step)

                if template in REJECT_TEMPLATES:
                    with self.assertRaises(TailCallError):
                        load_module(source)
                    continue

                module = load_module(source)
                if template in STREAM_TEMPLATES:
                    expected = asyncio.run(collect(module.baseline(n)))
                    actual = asyncio.run(collect(module.optimized(n)))
                elif template == "tailrec_keywords":
                    expected = asyncio.run(module.baseline(n, step=step))
                    actual = asyncio.run(module.optimized(n, step=step))
                else:
                    expected = asyncio.run(module.baseline(n))
                    actual = asyncio.run(module.optimized(n))
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
