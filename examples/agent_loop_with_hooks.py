"""Agent loop with injectable hooks composed outside @tailrec."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

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


class StepHook(Protocol):
    async def before_step(self, state: AgentState) -> None: ...

    async def after_step(self, state: AgentState, value: int) -> int: ...


@dataclass
class LogStepsHook:
    log: list[str] = field(default_factory=list)

    async def before_step(self, state: AgentState) -> None:
        self.log.append(f"step start remaining={state.remaining}")

    async def after_step(self, state: AgentState, value: int) -> int:
        self.log.append(f"step end remaining={state.remaining} value={value}")
        return value


@dataclass
class WarnLowBudgetHook:
    threshold: int = 3

    async def before_step(self, state: AgentState) -> None:
        if 0 < state.remaining <= self.threshold:
            print(f"warning: only {state.remaining} steps left")

    async def after_step(self, state: AgentState, value: int) -> int:
        return value


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


async def main() -> None:
    log_hook = LogStepsHook()
    hooks: list[StepHook] = [log_hook, WarnLowBudgetHook(threshold=2)]
    result = await agent_loop(AgentState(5), hooks)
    print(result)
    print("hook log:", log_hook.log[-4:])


if __name__ == "__main__":
    asyncio.run(main())
