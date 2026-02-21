"""Adapter for any agent accessible via CLI subprocess.

Runs the agent as a subprocess, sends content via stdin,
and reads responses from stdout. Supports any agent that
can be invoked from the command line.

Usage::

    from amplihack_eval.adapters.subprocess_adapter import SubprocessAdapter

    adapter = SubprocessAdapter(
        command=["python", "-m", "my_agent"],
        learn_flag="--learn",
        answer_flag="--answer",
    )
    adapter.learn("The capital of France is Paris.")
    response = adapter.answer("What is the capital of France?")
    print(response.answer)
"""

from __future__ import annotations

import json
import logging
import subprocess
import time

from .base import AgentAdapter, AgentResponse, ToolCall

logger = logging.getLogger(__name__)


class SubprocessAdapter(AgentAdapter):
    """Adapter for agents accessible via CLI subprocess.

    The agent is invoked as a subprocess for each learn/answer call.
    Communication is via stdin/stdout with optional JSON formatting.

    Args:
        command: Base command to invoke the agent (e.g. ["python", "-m", "my_agent"])
        learn_flag: CLI flag for learning mode (e.g. "--learn")
        answer_flag: CLI flag for answering mode (e.g. "--answer")
        reset_flag: CLI flag for reset (e.g. "--reset"). If None, reset is a no-op.
        timeout: Timeout in seconds for each subprocess call
        json_output: If True, parse stdout as JSON
        env: Additional environment variables for the subprocess
    """

    def __init__(
        self,
        command: list[str],
        learn_flag: str = "--learn",
        answer_flag: str = "--answer",
        reset_flag: str | None = "--reset",
        timeout: float = 120.0,
        json_output: bool = False,
        env: dict[str, str] | None = None,
    ):
        self._command = command
        self._learn_flag = learn_flag
        self._answer_flag = answer_flag
        self._reset_flag = reset_flag
        self._timeout = timeout
        self._json_output = json_output
        self._env = env

    def _run(self, args: list[str], stdin_text: str = "") -> str:
        """Run a subprocess command and return stdout."""
        import os

        full_cmd = self._command + args
        env = dict(os.environ)
        if self._env:
            env.update(self._env)

        try:
            result = subprocess.run(
                full_cmd,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=env,
            )
            if result.returncode != 0:
                logger.warning("Subprocess returned %d: %s", result.returncode, result.stderr[:500])
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.error("Subprocess timed out after %.1fs", self._timeout)
            return ""
        except FileNotFoundError:
            logger.error("Command not found: %s", full_cmd)
            return ""

    def learn(self, content: str) -> None:
        """Send content to the agent for learning."""
        self._run([self._learn_flag], stdin_text=content)

    def answer(self, question: str) -> AgentResponse:
        """Ask the agent a question via subprocess."""
        start = time.time()
        output = self._run([self._answer_flag], stdin_text=question)
        elapsed = time.time() - start

        if self._json_output and output:
            try:
                data = json.loads(output)
                tool_calls = [
                    ToolCall(
                        tool_name=tc.get("tool", ""),
                        arguments=tc.get("args", {}),
                        result=tc.get("result", ""),
                    )
                    for tc in data.get("tool_calls", [])
                ]
                return AgentResponse(
                    answer=data.get("answer", output),
                    tool_calls=tool_calls,
                    reasoning_trace=data.get("reasoning", ""),
                    confidence=float(data.get("confidence", 0.0)),
                    metadata={"elapsed_s": elapsed},
                )
            except (json.JSONDecodeError, KeyError):
                pass

        return AgentResponse(
            answer=output,
            metadata={"elapsed_s": elapsed},
        )

    def reset(self) -> None:
        """Reset the agent state."""
        if self._reset_flag:
            self._run([self._reset_flag])

    def close(self) -> None:
        """No persistent resources to clean up."""

    @property
    def name(self) -> str:
        return f"Subprocess({' '.join(self._command[:2])})"


__all__ = ["SubprocessAdapter"]
