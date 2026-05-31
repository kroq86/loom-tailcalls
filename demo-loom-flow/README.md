# demo-loom-flow

Loom + flow-xray + local Ollama demo and case runner.

- **Loom** (`@tailrec` / `@tailstream`) — stack-safe async loops
- **flow-xray** (`@trace`) — local HTML execution graph
- **Ollama** — real LLM in the short agent case (5 steps)

## Roadmap

Дальний план для **loom-tailcalls** и **flow-xray** (vision, integration contract, v0.2+): [ROADMAP.md](ROADMAP.md)

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
- `LOOM_SKIP_OLLAMA=1` — skip case 07 when Ollama is up but you want a fast run
- `LOOM_OLLAMA_CASES` (default `10` for case 07 fuzz)

## Setup

```bash
cd demo-loom-flow
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run everything

```bash
python run_all_cases.py
```

Cases 01–05 live in `cases/`. Traces go to `output/traces/`.

| Case | What |
|------|------|
| 01_agent_ollama | Real Ollama + `@tailrec` + flow-xray HTML |
| 02_loom_stress_100k | 100k `@tailrec` steps, no LLM |
| 03_plain_recursion_fails | plain RecursionError @10k vs Loom @100k |
| 04_streaming_tailstream | `@tailstream` + trace HTML |
| 05_hooks_and_budget | hooks + budget guard + trace HTML |
| 06_loom_unittest | `unittest discover` in parent loom repo |
| 07_ollama_fuzz | Ollama contract fuzz (auto if Ollama up) |

## Single demos

```bash
python agent_demo.py          # Ollama agent (v0.2 loop shapes), writes trace.html
open output/traces/01_agent_ollama.html   # after run_all_cases.py
```

Case 01 trace shows **kwargs tail-call, LLM retry in `except`, and `async with` step scope** — see [`agent_demo.py`](agent_demo.py) header.
