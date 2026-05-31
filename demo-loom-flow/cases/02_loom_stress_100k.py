"""Case 02: @tailrec stress run without LLM (100k steps)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from loom import tailrec


@dataclass(frozen=True)
class AgentState:
    remaining: int
    total: int = 0

    @property
    def done(self) -> bool:
        return self.remaining <= 0

    def apply(self, value: int) -> AgentState:
        return AgentState(self.remaining - 1, self.total + value)


async def run_tool(_: AgentState) -> int:
    await asyncio.sleep(0)
    return 1


@tailrec
async def agent_loop(state: AgentState) -> AgentState:
    if state.done:
        return state
    value = await run_tool(state)
    return await agent_loop(state.apply(value))


def run(output_dir: Path) -> str:
    _ = output_dir
    started = time.perf_counter()
    result = asyncio.run(agent_loop(AgentState(100_000)))
    elapsed = time.perf_counter() - started
    if result.remaining != 0 or result.total != 100_000:
        raise AssertionError(f"unexpected result: {result!r}")
    return f"{result} in {elapsed:.2f}s"
