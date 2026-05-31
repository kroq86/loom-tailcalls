"""Async agent loop: Loom v0.2 shapes + flow-xray @trace + local Ollama.

Showcase in one loop (see ``explain_tailcalls(agent_loop)``):

- ``return await agent_loop(state, **tail)`` — kwargs expansion in tail-call
- ``try`` / ``except TransientError`` — retry without leaving ``@tailrec``
- ``async with session.step()`` — per-step scope inside the optimized loop

``@trace`` stays on leaves only; ``@tailrec`` wraps the loop only.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from flow_xray import trace
from loom import explain_tailcalls, tailrec

from ollama_client import OllamaError, chat


class TransientError(OllamaError):
    """Retryable LLM / network flake (tail-call in ``except`` is valid in v0.2)."""


@dataclass(frozen=True)
class LoopOpts:
    max_llm_retries: int = 2


@dataclass(frozen=True)
class AgentState:
    remaining_steps: int
    query: str
    total: int = 0
    step_index: int = 0
    llm_attempt: int = 0

    @property
    def done(self) -> bool:
        return self.remaining_steps <= 0

    def after_tool(self, value: int) -> AgentState:
        return AgentState(
            remaining_steps=self.remaining_steps - 1,
            query=self.query,
            total=self.total + value,
            step_index=self.step_index + 1,
            llm_attempt=0,
        )

    def retry_llm(self) -> AgentState:
        return AgentState(
            remaining_steps=self.remaining_steps,
            query=self.query,
            total=self.total,
            step_index=self.step_index,
            llm_attempt=self.llm_attempt + 1,
        )


class AgentSession:
    """One in-flight step at a time — visible in the trace via ``step()``."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.completed = 0

    @asynccontextmanager
    async def step(self, index: int) -> AsyncIterator[None]:
        async with self._lock:
            trace.meta(session_step=index, session_completed=self.completed)
            yield
            self.completed += 1


@trace
async def pick_action(state: AgentState) -> str:
    await asyncio.sleep(0)
    actions = ("search", "summarize", "plan")
    return actions[state.step_index % len(actions)]


@trace
async def call_ollama(prompt: str) -> str:
    content, meta = await chat(prompt)
    if meta:
        trace.meta(**meta)
    return content


@trace
async def run_tool(state: AgentState, action: str) -> int:
    await asyncio.sleep(0.005)
    return len(action) + (state.remaining_steps % 5)


def _tail_bindings(session: AgentSession, opts: LoopOpts) -> dict[str, object]:
    return {"session": session, "opts": opts}


@tailrec
async def agent_loop(
    state: AgentState,
    *,
    session: AgentSession,
    opts: LoopOpts,
) -> AgentState:
    if state.done:
        return state

    tail = _tail_bindings(session, opts)
    action = await pick_action(state)
    prompt = (
        f"Reply in one short sentence. Task: plan for {state.query!r} "
        f"using action {action!r} (step {state.step_index + 1})."
    )

    try:
        # Demo flake on first LLM call of step 2 — retry path shows up in HTML trace.
        if state.step_index == 1 and state.llm_attempt == 0:
            raise TransientError("simulated rate limit (demo retry)")
        await call_ollama(prompt)
    except TransientError as exc:
        if state.llm_attempt < opts.max_llm_retries:
            trace.meta(retry=str(exc), attempt=state.llm_attempt + 1)
            return await agent_loop(state.retry_llm(), **tail)
        raise

    value = await run_tool(state, action)

    async with session.step(state.step_index):
        return await agent_loop(state.after_tool(value), **tail)


async def run_agent(*, steps: int | None = None, query: str = "weather Tokyo") -> AgentState:
    n = steps if steps is not None else int(os.environ.get("OLLAMA_STEPS", "5"))
    session = AgentSession()
    opts = LoopOpts(max_llm_retries=int(os.environ.get("OLLAMA_MAX_RETRIES", "2")))
    return await agent_loop(
        AgentState(remaining_steps=n, query=query),
        **_tail_bindings(session, opts),
    )


def run_traced(*, steps: int | None = None, query: str = "weather Tokyo", trace_path: Path) -> AgentState:
    trace_result = trace.run(lambda: asyncio.run(run_agent(steps=steps, query=query)))
    trace_result.to_html(str(trace_path))
    return trace_result.return_value


def main() -> None:
    out = Path(__file__).resolve().parent / "trace.html"
    print("Loom transform (agent_loop):")
    print(explain_tailcalls(agent_loop))
    print()
    final = run_traced(trace_path=out)
    print(f"final state: {final}")
    print(f"trace written: {out}")


if __name__ == "__main__":
    main()
