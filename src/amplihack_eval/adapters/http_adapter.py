"""Adapter for any agent accessible via HTTP API.

Communicates with agents via REST endpoints:
  POST /learn   - Feed content
  POST /answer  - Ask a question
  POST /reset   - Reset state

Usage::

    from amplihack_eval.adapters.http_adapter import HttpAdapter

    adapter = HttpAdapter(base_url="http://localhost:8000")
    adapter.learn("The capital of France is Paris.")
    response = adapter.answer("What is the capital of France?")
    print(response.answer)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .base import AgentAdapter, AgentResponse, ToolCall

logger = logging.getLogger(__name__)


class HttpAdapter(AgentAdapter):
    """Adapter for agents accessible via HTTP API.

    The agent must expose REST endpoints for learn, answer, and reset.
    Uses urllib (no external dependencies) for HTTP communication.

    Args:
        base_url: Base URL of the agent API (e.g. "http://localhost:8000")
        learn_path: Path for the learn endpoint
        answer_path: Path for the answer endpoint
        reset_path: Path for the reset endpoint
        timeout: HTTP request timeout in seconds
        headers: Additional HTTP headers
    """

    def __init__(
        self,
        base_url: str,
        learn_path: str = "/learn",
        answer_path: str = "/answer",
        reset_path: str = "/reset",
        timeout: float = 120.0,
        headers: dict[str, str] | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._learn_path = learn_path
        self._answer_path = answer_path
        self._reset_path = reset_path
        self._timeout = timeout
        self._headers = headers or {}

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Send a POST request and return JSON response."""
        url = f"{self._base_url}{path}"
        body = json.dumps(data).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **self._headers,
        }
        req = Request(url, data=body, headers=headers, method="POST")

        try:
            with urlopen(req, timeout=self._timeout) as resp:
                response_body = resp.read().decode("utf-8")
                if response_body:
                    return json.loads(response_body)
                return {}
        except URLError as e:
            logger.error("HTTP request failed for %s: %s", url, e)
            return {"error": str(e)}
        except json.JSONDecodeError:
            logger.warning("Non-JSON response from %s", url)
            return {"raw": response_body}

    def learn(self, content: str) -> None:
        """Send content to the agent for learning via HTTP."""
        result = self._post(self._learn_path, {"content": content})
        if "error" in result:
            logger.warning("Learn request failed: %s", result["error"])

    def answer(self, question: str) -> AgentResponse:
        """Ask the agent a question via HTTP."""
        start = time.time()
        result = self._post(self._answer_path, {"question": question})
        elapsed = time.time() - start

        if "error" in result:
            return AgentResponse(
                answer=f"Error: {result['error']}",
                metadata={"elapsed_s": elapsed},
            )

        # Parse tool calls if present
        tool_calls = []
        for tc in result.get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    tool_name=tc.get("tool", ""),
                    arguments=tc.get("args", {}),
                    result=tc.get("result", ""),
                )
            )

        return AgentResponse(
            answer=result.get("answer", str(result)),
            tool_calls=tool_calls,
            reasoning_trace=result.get("reasoning", ""),
            confidence=float(result.get("confidence", 0.0)),
            metadata={"elapsed_s": elapsed},
        )

    def reset(self) -> None:
        """Reset agent state via HTTP."""
        result = self._post(self._reset_path, {})
        if "error" in result:
            logger.warning("Reset request failed: %s", result["error"])

    def close(self) -> None:
        """No persistent HTTP resources to clean up."""

    @property
    def name(self) -> str:
        return f"HTTP({self._base_url})"


__all__ = ["HttpAdapter"]
