"""Minimal recursive async-stream demo for Loom."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loom import tailstream


@dataclass(frozen=True)
class StreamState:
    remaining: int

    @property
    def done(self) -> bool:
        return self.remaining <= 0

    def next(self) -> "StreamState":
        return StreamState(self.remaining - 1)


async def run_step(state: StreamState) -> AsyncIterator[dict[str, int | str]]:
    await asyncio.sleep(0)
    yield {"type": "token", "remaining": state.remaining}


@tailstream
async def stream_agent(state: StreamState):
    if state.done:
        yield {"type": "final", "remaining": state.remaining}
        return

    async for event in run_step(state):
        yield event

    async for event in stream_agent(state.next()):
        yield event
    return


async def main() -> None:
    async for event in stream_agent(StreamState(3)):
        print(event)


if __name__ == "__main__":
    asyncio.run(main())
