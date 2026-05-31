"""Case 05: hooks + budget guard with flow-xray trace."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from flow_xray import trace
from loom import tailrec


@dataclass(frozen=True)
class AgentState:
    remaining_tools: int
    remaining_steps: int
    total: int = 0

    @property
    def done(self) -> bool:
        return self.remaining_tools <= 0 or self.remaining_steps <= 0

    def apply(self, value: int) -> AgentState:
        return AgentState(
            self.remaining_tools - 1,
            self.remaining_steps - 1,
            self.total + value,
        )


class StepHook(Protocol):
    async def before_step(self, state: AgentState) -> None: ...

    async def after_step(self, state: AgentState, value: int) -> int: ...


@dataclass
class LogStepsHook:
    log: list[str] = field(default_factory=list)

    async def before_step(self, state: AgentState) -> None:
        self.log.append(f"step start tools={state.remaining_tools} steps={state.remaining_steps}")

    async def after_step(self, state: AgentState, value: int) -> int:
        self.log.append(f"step end value={value}")
        return value


@trace
async def run_tool(_: AgentState) -> int:
    await asyncio.sleep(0)
    return 1


async def run_step_with_hooks(state: AgentState, hooks: list[StepHook]) -> int:
    for hook in hooks:
        await hook.before_step(state)
    value = await run_tool(state)
    for hook in hooks:
        value = await hook.after_step(state, value)
    return value


@tailrec
async def agent_loop(state: AgentState, hooks: list[StepHook]) -> AgentState:
    if state.done:
        return state
    tool_result = await run_step_with_hooks(state, hooks)
    return await agent_loop(state.apply(tool_result), hooks)


def run(output_dir: Path) -> str:
    trace_path = output_dir / "05_hooks_and_budget.html"
    log_hook = LogStepsHook()

    def execute() -> tuple[AgentState, list[str]]:
        result = asyncio.run(agent_loop(AgentState(remaining_tools=3, remaining_steps=5), [log_hook]))
        return result, log_hook.log

    trace_result = trace.run(execute)
    trace_result.to_html(str(trace_path))
    final, log = trace_result.return_value
    if final.remaining_tools != 0 or final.remaining_steps != 2:
        raise AssertionError(f"unexpected budget result: {final!r}")
    return f"{final} log_entries={len(log)} trace={trace_path.name}"
