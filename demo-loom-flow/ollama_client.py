"""Minimal async client for local Ollama (http://127.0.0.1:11434)."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


class OllamaError(RuntimeError):
    pass


def _request_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc


def ping() -> bool:
    try:
        _request_json("/api/tags")
        return True
    except OllamaError:
        return False


def selected_model() -> str:
    if model := os.environ.get("OLLAMA_MODEL"):
        return model
    tags = _request_json("/api/tags")
    models = tags.get("models") or []
    if not models:
        raise OllamaError("Ollama is reachable but returned no models")
    name = models[0].get("name")
    if not isinstance(name, str) or not name:
        raise OllamaError("Ollama /api/tags returned a model without a valid name")
    return name


def chat_sync(prompt: str, *, model: str | None = None) -> tuple[str, dict[str, Any]]:
    model_name = model or selected_model()
    response = _request_json(
        "/api/chat",
        {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
    )
    message = response.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise OllamaError(f"Unexpected Ollama chat response: {response!r}")

    meta: dict[str, Any] = {"model": model_name}
    if isinstance(response.get("prompt_eval_count"), int):
        meta["prompt_tokens"] = response["prompt_eval_count"]
    if isinstance(response.get("eval_count"), int):
        meta["completion_tokens"] = response["eval_count"]
    return content, meta


async def chat(prompt: str, *, model: str | None = None) -> tuple[str, dict[str, Any]]:
    return await asyncio.to_thread(chat_sync, prompt, model=model)
