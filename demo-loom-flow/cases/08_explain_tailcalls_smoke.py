"""Case 08: smoke gate for explain_tailcalls binding paths."""

from __future__ import annotations

from pathlib import Path

from loom import explain_tailcalls, tailrec


@tailrec
async def direct_loop(n: int, acc: int) -> int:
    if n <= 0:
        return acc
    return await direct_loop(n - 1, acc + 1)


@tailrec
async def fast_loop(n: int, *, step: int = 1, acc: int = 0) -> int:
    if n <= 0:
        return acc
    return await fast_loop(n - step, step=step, acc=acc + step)


@tailrec
async def fallback_loop(n: int, acc: int = 0, *items: object) -> int:
    if n <= 0:
        return acc + len(items)
    return await fallback_loop(n - 1, acc + 1, *items)


def _assert_report(fn, *, binding: str, binding_sites: list[str]) -> None:
    report = explain_tailcalls(fn, as_json=True)
    assert report["binding"] == binding, report
    assert report["binding_sites"] == binding_sites, report
    assert report["optimized"], report
    assert not report["rejected"], report


def run(output_dir: Path) -> str:
    _ = output_dir
    _assert_report(direct_loop, binding="direct", binding_sites=["direct"])
    _assert_report(fast_loop, binding="fast", binding_sites=["bind"])
    _assert_report(fallback_loop, binding="signature", binding_sites=["bind"])
    return "direct/fast/signature explain_tailcalls smoke passed"
