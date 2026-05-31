#!/usr/bin/env python3
"""Run all demo-loom-flow cases + optional loom repo tests."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOOM_ROOT = ROOT.parent
OUTPUT_DIR = ROOT / "output" / "traces"


@dataclass
class CaseOutcome:
    name: str
    status: str  # OK | SKIP | FAIL
    detail: str = ""


def load_case_module(filename: str):
    path = ROOT / "cases" / filename
    module_name = f"demo_cases.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(module)
    return module


def run_python_cases() -> list[CaseOutcome]:
    outcomes: list[CaseOutcome] = []
    case_files = sorted((ROOT / "cases").glob("*.py"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in case_files:
        module = load_case_module(path.name)
        name = path.stem
        try:
            detail = module.run(OUTPUT_DIR)
            outcomes.append(CaseOutcome(name, "OK", detail))
        except Exception as exc:
            if exc.__class__.__name__ == "SkipCase":
                outcomes.append(CaseOutcome(name, "SKIP", str(exc)))
            else:
                outcomes.append(CaseOutcome(name, "FAIL", str(exc)))
    return outcomes


def run_loom_unittest() -> CaseOutcome:
    python = sys.executable
    tests_dir = LOOM_ROOT / "tests"
    if not tests_dir.is_dir():
        return CaseOutcome("06_loom_unittest", "SKIP", f"tests dir not found: {tests_dir}")

    proc = subprocess.run(
        [python, "-m", "unittest", "discover", "-s", str(tests_dir), "-q"],
        cwd=str(LOOM_ROOT),
        capture_output=True,
        text=True,
    )
    detail = (proc.stdout + proc.stderr).strip().splitlines()[-1] if proc.stdout or proc.stderr else ""
    if proc.returncode == 0:
        return CaseOutcome("06_loom_unittest", "OK", detail or "unittest discover passed")
    return CaseOutcome("06_loom_unittest", "FAIL", detail or proc.stderr or "unittest failed")


def _ollama_reachable() -> bool:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from ollama_client import ping
    except ImportError:
        return False
    return ping()


def run_ollama_fuzz() -> CaseOutcome:
    if os.environ.get("LOOM_SKIP_OLLAMA") == "1":
        return CaseOutcome("07_ollama_fuzz", "SKIP", "LOOM_SKIP_OLLAMA=1")

    if not _ollama_reachable():
        return CaseOutcome("07_ollama_fuzz", "SKIP", "Ollama not reachable at OLLAMA_URL")

    python = sys.executable
    env = os.environ.copy()
    env["LOOM_OLLAMA_FUZZ"] = "1"
    env.setdefault("LOOM_OLLAMA_CASES", "10")
    proc = subprocess.run(
        [python, "-m", "unittest", "tests.test_ollama_contract", "-q"],
        cwd=str(LOOM_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    combined = (proc.stdout + proc.stderr).strip()
    detail = combined.splitlines()[-1] if combined else ""
    if proc.returncode == 0:
        return CaseOutcome("07_ollama_fuzz", "OK", detail or "ollama contract fuzz passed")
    return CaseOutcome("07_ollama_fuzz", "FAIL", detail or combined or "ollama fuzz failed")


def main() -> int:
    outcomes = run_python_cases()
    outcomes.append(run_loom_unittest())
    outcomes.append(run_ollama_fuzz())

    failed = False
    for item in outcomes:
        line = f"{item.status:<4}  {item.name}"
        if item.detail:
            line = f"{line}  —  {item.detail}"
        print(line)
        if item.status == "FAIL":
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
