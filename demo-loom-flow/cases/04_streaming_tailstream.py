"""Case 04: @tailstream streaming agent with flow-xray trace."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Literal, TypedDict

from flow_xray import trace
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

    def next(self, *, tool_value: int) -> StreamState:
        return StreamState(self.remaining - 1, self.completed + tool_value)


@trace
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


async def collect(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [item async for item in events]


def run(output_dir: Path) -> str:
    trace_path = output_dir / "04_streaming_tailstream.html"
    trace_result = trace.run(lambda: asyncio.run(collect(stream_agent(StreamState(2)))))
    trace_result.to_html(str(trace_path))
    events = trace_result.return_value
    if not events or events[-1]["type"] != "final":
        raise AssertionError(f"unexpected stream events: {events!r}")
    return f"events={len(events)} trace={trace_path.name}"
