"""Case 03: plain recursion fails at 10k; Loom survives 100k."""

from __future__ import annotations

import asyncio
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


async def plain_agent_loop(state: AgentState) -> AgentState:
    if state.done:
        return state
    value = await run_tool(state)
    return await plain_agent_loop(state.apply(value))


@tailrec
async def loom_agent_loop(state: AgentState) -> AgentState:
    if state.done:
        return state
    value = await run_tool(state)
    return await loom_agent_loop(state.apply(value))


def run(output_dir: Path) -> str:
    _ = output_dir
    try:
        asyncio.run(plain_agent_loop(AgentState(10_000)))
    except RecursionError:
        plain_status = "RecursionError"
    else:
        raise AssertionError("plain recursion unexpectedly survived")

    result = asyncio.run(loom_agent_loop(AgentState(100_000)))
    if result.remaining != 0:
        raise AssertionError(f"loom loop failed: {result!r}")
    return f"plain={plain_status} loom={result}"
