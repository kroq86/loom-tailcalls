# demo-loom-flow

Loom + flow-xray + local Ollama demo and case runner.

- **Loom** (`@tailrec` / `@tailstream`) — stack-safe async loops
- **flow-xray** (`@trace`) — local HTML execution graph
- **Ollama** — real LLM in the short agent case (5 steps)

**Loom stack:** [hub](https://kroq86.github.io/loom-stack/) · [ECOSYSTEM.md](https://github.com/kroq86/loom-stack/blob/main/docs/ECOSYSTEM.md) · CLIs: [loom-run](https://github.com/kroq86/loom-run) · [loom-ops](https://github.com/kroq86/loom-ops)

## Roadmap

Дальний план для **loom-tailcalls** и **flow-xray** (vision, integration contract, v0.2+): [ROADMAP.md](ROADMAP.md)

Case matrix and v0.3 decision log: [CASE_MATRIX.md](CASE_MATRIX.md)

Blockers and gaps from growth probes: [FINDINGS.md](FINDINGS.md)

## Prerequisites

- Python 3.13+ (3.14 may break on this machine)
- [Ollama](https://ollama.com/) running locally with at least one model:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

Optional env:

- `OLLAMA_URL` (default `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (default: first model from `/api/tags`)
- `OLLAMA_STEPS` (default `5` for agent demo)
- `LOOM_FAST=1` — skip Ollama integration cases 01/12/15 and fuzz 07
- `LOOM_OLLAMA_CASES` (default `10` for case 07 fuzz)
- `LOOM_BENCH_MAX_RATIO` (default `1.25` for case 10)
- `LOOM_CONCURRENT_AGENTS` / `LOOM_CONCURRENT_STEPS` (case 11)
- `OLLAMA_TRACE_STEPS` / `OLLAMA_REALISM_STEPS` (cases 12 / 15, default `20`)

## Setup

```bash
cd demo-loom-flow
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run everything

```bash
# Fast gate (~20s): no Ollama — daily / CI
LOOM_FAST=1 python3.13 run_all_cases.py

# Full gate (~5–15 min): run yourself when Ollama is up
python3.13 run_all_cases.py

# Single case or subset
LOOM_FAST=1 python3.13 run_all_cases.py --case 10
python3.13 run_all_cases.py --only 10,11,14

# Deadlock probe (no Ollama, ~3s) — plain async + session lock without @tailrec
.venv/bin/python scripts/diag_baseline_deadlock.py
python3.13 -m unittest tests.test_v02_agent_baseline_equiv -v   # from repo root
```

| Case | What |
|------|------|
| 01_agent_ollama | Real Ollama + `@tailrec` + flow-xray HTML |
| 02_loom_stress_100k | 100k `@tailrec` steps, no LLM |
| 03_plain_recursion_fails | plain RecursionError @10k vs Loom @100k |
| 04_streaming_tailstream | `@tailstream` + trace HTML |
| 05_hooks_and_budget | hooks + budget guard + trace HTML |
| 06_loom_unittest | `unittest discover` in parent loom repo |
| 07_ollama_fuzz | Ollama contract fuzz (auto if Ollama up) |
| 08_explain_tailcalls_smoke | `explain_tailcalls` binding smoke |
| 10_bench_direct_100k | bench gate via `scripts/bench_tailcalls.py` |
| 11_concurrent_agents | parallel `@tailrec` via `asyncio.gather` |
| 12_trace_sanity_20 | flow-xray trace sanity at 20 Ollama steps |
| 13_streaming_trace_meta | `@tailstream` + `trace.meta(stream_step=...)` |
| 14_expect_rejects | documented Loom rejects still fail with hints |
| 15_agent_realism_20 | 20-step Ollama run with v0.2 loop shapes |

For the actual interpretation of each case, read [CASE_MATRIX.md](CASE_MATRIX.md). The table above is only an index.

v0.2 loop shapes (kwargs, try/with) — [`agent_demo.py`](agent_demo.py) + case 01/15. **F-003:** source shape without `@tailrec` deadlocks on `async with session.step()` — [`FINDINGS.md`](FINDINGS.md), [`scripts/diag_baseline_deadlock.py`](scripts/diag_baseline_deadlock.py), [`tests/test_v02_agent_baseline_equiv.py`](../tests/test_v02_agent_baseline_equiv.py).

## Single demos

```bash
python agent_demo.py          # Ollama agent (v0.2 loop shapes), writes trace.html
open output/traces/01_agent_ollama.html   # after run_all_cases.py
```

Case 01 trace shows **kwargs tail-call, LLM retry in `except`, and `async with` step scope** — see [`agent_demo.py`](agent_demo.py) header.
