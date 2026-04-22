"""Shared LLM grading helpers for eval modules."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from amplihack_eval.llm import completion

DEFAULT_GRADER_MODEL = "claude-opus-4-6"

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_BRACE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM response text."""
    stripped = text.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = _FENCED_JSON_RE.search(stripped)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_match = _BRACE_JSON_RE.search(stripped)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(f"No valid JSON found in response: {stripped[:200]}", stripped, 0)


def get_grader_model(explicit_model: str | None = None) -> str:
    """Resolve the grader model from an explicit override or environment."""
    return explicit_model or os.environ.get("GRADER_MODEL", DEFAULT_GRADER_MODEL)


async def call_grader_json(
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 1000,
    temperature: float | None = None,
    system: str | None = None,
) -> dict[str, Any]:
    """Run a grading prompt through the LLM adapter and parse the JSON response."""
    resolved_model = get_grader_model(model)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response_text = await completion(
        messages,
        model=resolved_model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return extract_json(response_text)


__all__ = [
    "DEFAULT_GRADER_MODEL",
    "call_grader_json",
    "extract_json",
    "get_grader_model",
]
