# Changelog

## 0.2.0 — 2026-05-31

### Added

- `**kwargs` expansion in tail-call (bind path)
- Tail-call inside `try`/`except`, `with`/`async with`, `for`/`while`/`async for`
- Loop `break_flag` epilogue for correct `continue` semantics inside nested loops
- `demo-loom-flow` integration lab (cases 01–08)
- `explain_tailcalls` smoke gate (case 08)

### Changed

- Ollama contract fuzz: resilient case count with deterministic padding; case 07 runs automatically when Ollama is reachable
- `docs/formal-core.md`: structured tail positions and kwargs merge rule

### Fixed

- `@tailstream`: terminal `async for` pattern is checked before generic loop reject
