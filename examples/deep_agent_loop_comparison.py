"""Show the problem Loom solves: deep async recursion without stack growth."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loom import tailrec


@dataclass(frozen=True)
class AgentState:
    remaining: int
    total: int = 0

    @property
    def done(self) -> bool:
        return self.remaining <= 0

    def apply(self, value: int) -> "AgentState":
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


async def main() -> None:
    try:
        await plain_agent_loop(AgentState(10_000))
    except RecursionError as exc:
        print(f"plain recursion failed: {exc.__class__.__name__}")
    else:
        raise AssertionError("plain recursion unexpectedly survived")

    result = await loom_agent_loop(AgentState(100_000))
    print(f"loom recursion survived: {result}")


if __name__ == "__main__":
    asyncio.run(main())
