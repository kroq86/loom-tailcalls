"""Streaming agent with typed events and an external consumer loop."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loom import tailstream


class TokenEvent(TypedDict):
    type: Literal["token"]
    text: str
    step: int


class ToolCallEvent(TypedDict):
    type: Literal["tool_call"]
    name: str
    step: int


class FinalEvent(TypedDict):
    type: Literal["final"]
    steps_completed: int


StreamEvent = TokenEvent | ToolCallEvent | FinalEvent


@dataclass(frozen=True)
class StreamState:
    remaining: int
    completed: int = 0

    @property
    def done(self) -> bool:
        return self.remaining <= 0

    def next(self, *, tool_value: int) -> "StreamState":
        return StreamState(self.remaining - 1, self.completed + tool_value)


async def run_step(state: StreamState) -> AsyncIterator[StreamEvent]:
    await asyncio.sleep(0)
    yield {"type": "token", "text": f"thinking at step {state.completed + 1}", "step": state.completed + 1}
    yield {"type": "tool_call", "name": "lookup", "step": state.completed + 1}


@tailstream
async def stream_agent(state: StreamState):
    if state.done:
        yield {"type": "final", "steps_completed": state.completed}
        return

    async for event in run_step(state):
        yield event

    async for event in stream_agent(state.next(tool_value=1)):
        yield event
    return


async def consume(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    collected: list[StreamEvent] = []
    async for event in events:
        collected.append(event)
        print(event)
    return collected


async def main() -> None:
    await consume(stream_agent(StreamState(2)))


if __name__ == "__main__":
    asyncio.run(main())
