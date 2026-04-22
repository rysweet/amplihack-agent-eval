"""Lazy shim re-exporting :func:`amplihack.llm.completion`.

Issue #48 ports several modules out of the upstream ``amplihack`` package
into this repository. Some of those modules call ``amplihack.llm.completion``
to grade answers via an LLM. We do **not** want to take a hard runtime
dependency on the ``amplihack`` package at import time — many of this
repo's consumers (CI smoke tests, library imports, recipe wiring) never
actually invoke a grading call.

This module exposes a single name, :func:`completion`, which behaves as a
lazy proxy:

* Importing ``amplihack_eval.llm`` always succeeds, even when the
  ``amplihack`` package is not installed.
* The first call to ``completion(...)`` resolves
  ``amplihack.llm.completion`` and forwards arguments to it.
* If ``amplihack.llm`` cannot be imported, a clear :class:`ImportError`
  is raised at *call* time (not import time) telling the user how to
  install the missing dependency.

This keeps the eval modules' public API stable while letting test
environments without ``amplihack`` installed still import them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["completion"]

_INSTALL_HINT = (
    "amplihack_eval.llm.completion requires the upstream `amplihack` "
    "package providing `amplihack.llm.completion`. Install it with: "
    "pip install amplihack"
)


async def completion(*args: Any, **kwargs: Any) -> Any:
    """Forward to ``amplihack.llm.completion``.

    Resolved lazily so that importing this module never imports
    ``amplihack``. Raises :class:`ImportError` with an actionable install
    hint if the upstream package is not available at call time.
    """
    try:
        from amplihack.llm import completion as _real_completion
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return await _real_completion(*args, **kwargs)
