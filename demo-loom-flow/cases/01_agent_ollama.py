"""Case 01: agent loop with real Ollama + flow-xray trace."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_demo import run_traced
from ollama_client import OllamaError, ping


def run(output_dir: Path) -> str:
    if not ping():
        raise SkipCase("Ollama not reachable at OLLAMA_URL")

    trace_path = output_dir / "01_agent_ollama.html"
    final = run_traced(steps=5, query="weather Tokyo", trace_path=trace_path)
    if final.remaining_steps != 0:
        raise AssertionError(f"expected remaining_steps=0, got {final!r}")
    return f"{final} trace={trace_path.name}"


class SkipCase(Exception):
    pass
