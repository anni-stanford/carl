"""Abstract LLM client interface — keeps Opus, Sonnet, and GPT behind one type.

Three concrete implementations live here:

- :class:`AnthropicClient` — wraps the official ``anthropic`` SDK.
- :class:`OpenAIClient` — wraps the official ``openai`` SDK.
- :class:`FakeLLMClient` — deterministic, in-memory; used by every unit test
  that exercises a code path involving an LLM call. No API key required.

The point of the abstraction is two-fold: (1) the bias-controlled judge
rotates families through this interface, and (2) every LLM-dependent piece
of CARL is unit-testable without a network call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    """Uniform response shape across providers."""

    text: str
    model: str
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    """Provider-agnostic LLM call interface used by judge / mutator / diagnosis."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: str | None = None,  # "json" | None
    ) -> LLMResponse: ...

    @abstractmethod
    def family(self, model: str) -> str:
        """Return the family bucket for ``model`` (used for rotation enforcement).

        Examples: ``"anthropic"`` for any ``claude-*`` model, ``"openai"`` for
        ``gpt-*``, ``"open"`` for any local / open-weight model served via
        Cloudflare Workers AI.
        """


# ----- Fake LLM client for tests --------------------------------------------


class FakeLLMClient(LLMClient):
    """Deterministic in-memory client.

    Configure it with a ``script`` mapping prompt-prefix → canned response.
    If a prompt does not match any prefix, the configured ``default`` is
    returned. Every call increments ``call_count`` so tests can assert how
    many times the LLM was invoked.
    """

    def __init__(
        self,
        script: dict[str, str] | None = None,
        default: str = "{}",
        family_map: dict[str, str] | None = None,
    ) -> None:
        self.script = script or {}
        self.default = default
        self.family_map = family_map or {}
        self.calls: list[tuple[str, str]] = []  # (model, prompt)

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: str | None = None,
    ) -> LLMResponse:
        self.calls.append((model, prompt))
        for prefix, response in self.script.items():
            if prefix in prompt:
                return LLMResponse(text=response, model=model)
        return LLMResponse(text=self.default, model=model)

    def family(self, model: str) -> str:
        if model in self.family_map:
            return self.family_map[model]
        if model.startswith("claude-"):
            return "anthropic"
        if model.startswith("gpt-"):
            return "openai"
        return "unknown"

    @property
    def call_count(self) -> int:
        return len(self.calls)
