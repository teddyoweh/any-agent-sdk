"""Provider adapters.

Importing ``any_agent_sdk.providers`` doesn't pull any specific provider —
each one is imported lazily by ``resolve()`` so a user who only wants
Anthropic doesn't pay the boto3 import cost.
"""

from __future__ import annotations

from .base import Provider, register, resolve

__all__ = ["Provider", "register", "resolve"]
