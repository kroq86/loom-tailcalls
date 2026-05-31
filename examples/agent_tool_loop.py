"""Minimal async tool loop demo for Loom."""

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


@tailrec
async def agent_loop(state: AgentState) -> AgentState:
    if state.done:
        return state

    tool_result = await run_tool(state)
    return await agent_loop(state.apply(tool_result))


async def main() -> None:
    result = await agent_loop(AgentState(100_000))
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
