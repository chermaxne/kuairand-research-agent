"""Provider-agnostic chat call + token accounting.

`LLMClient.complete()` takes the assembled prompt (static system blocks first, so providers can cache
the static prefix) and returns text + usage read from the API response. The mock client is
deterministic and offline: it is used by tests and by `--mock` runs when no API key is present.
The real Anthropic client lives in `AnthropicClient` (Phase 3).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .schemas import TokenUsage


@dataclass
class LLMResponse:
    text: str
    usage: TokenUsage
    model: str
    latency_s: float
    stop_reason: str = ""
    estimated_usage: bool = False     # True when the client had no API usage field (mock)


class LLMClient(Protocol):
    provider: str

    def complete(self, *, role: str, model: str, system_blocks: Sequence[str], messages: Sequence[Dict[str, str]],
                 max_tokens: int) -> LLMResponse: ...


class LLMError(RuntimeError):
    """Transport-level failure after retries (never a content problem)."""


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


Handler = Callable[[str, List[str], List[Dict[str, str]]], str]   # (role, system_blocks, messages) -> text


class MockLLMClient:
    """Deterministic offline client. `handlers[role]` produce the text; usage is estimated from
    character counts and flagged as estimated so accounting stays honest."""
    provider = "mock"

    def __init__(self, handlers: Optional[Dict[str, Handler]] = None, default: Optional[Handler] = None):
        self.handlers: Dict[str, Handler] = dict(handlers or {})
        self.default = default
        self.calls: List[Dict[str, Any]] = []

    def complete(self, *, role: str, model: str, system_blocks: Sequence[str], messages: Sequence[Dict[str, str]],
                 max_tokens: int) -> LLMResponse:
        t0 = time.time()
        h = self.handlers.get(role) or self.handlers.get(role.split("_")[0]) or self.default
        if h is None:
            raise LLMError(f"mock client has no handler for role {role!r}")
        text = h(role, list(system_blocks), [dict(m) for m in messages])
        prompt_chars = sum(len(b) for b in system_blocks) + sum(len(m.get("content", "")) for m in messages)
        usage = TokenUsage(input_tokens=_approx_tokens("x" * prompt_chars), output_tokens=_approx_tokens(text))
        self.calls.append({"role": role, "model": model, "n_messages": len(messages), "usage": usage.to_dict()})
        return LLMResponse(text=text, usage=usage, model=f"mock:{model}", latency_s=time.time() - t0,
                           stop_reason="end_turn", estimated_usage=True)


class AnthropicClient:
    """Thin wrapper over the Anthropic Messages API with prompt caching on the static prefix and
    retry with exponential backoff. Usage is read from `response.usage` (never estimated)."""
    provider = "anthropic"

    def __init__(self, api_key: str, request_timeout_s: float = 300, max_retries: int = 3, prompt_caching: bool = True):
        import anthropic  # imported lazily so tests never need the package configured
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, timeout=request_timeout_s, max_retries=0)
        self.max_retries = int(max_retries)
        self.prompt_caching = bool(prompt_caching)

    def complete(self, *, role: str, model: str, system_blocks: Sequence[str], messages: Sequence[Dict[str, str]],
                 max_tokens: int) -> LLMResponse:
        system: List[Dict[str, Any]] = []
        for i, block in enumerate(system_blocks):
            if not block:
                continue
            b: Dict[str, Any] = {"type": "text", "text": block}
            if self.prompt_caching and i == len(system_blocks) - 1:
                b["cache_control"] = {"type": "ephemeral"}     # cache the whole static prefix
            system.append(b)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self._client.messages.create(model=model, max_tokens=int(max_tokens), system=system or None,
                                                    messages=[{"role": m["role"], "content": m["content"]} for m in messages])
                text = "".join(getattr(part, "text", "") for part in resp.content)
                u = resp.usage
                usage = TokenUsage(input_tokens=int(getattr(u, "input_tokens", 0) or 0),
                                   output_tokens=int(getattr(u, "output_tokens", 0) or 0),
                                   cache_creation_input_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
                                   cache_read_input_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0))
                return LLMResponse(text=text, usage=usage, model=resp.model, latency_s=time.time() - t0,
                                   stop_reason=str(resp.stop_reason or ""))
            except (self._anthropic.RateLimitError, self._anthropic.APIConnectionError, self._anthropic.APITimeoutError,
                    self._anthropic.InternalServerError) as e:      # transient -> retry
                last_err = e
                time.sleep(min(60, 2 ** attempt * 2))
            except self._anthropic.APIStatusError as e:               # 4xx other than rate limit: not retryable
                raise LLMError(f"{type(e).__name__}: {e}") from e
        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last_err}")


def make_client(cfg: Dict[str, Any], mock_handlers: Optional[Dict[str, Handler]] = None, force_mock: bool = False) -> LLMClient:
    """Build the client from config. The API key is read from the env var named in config and is
    never written anywhere."""
    llm = cfg["llm"]
    provider = str(llm.get("provider", "anthropic")).lower()
    if force_mock or provider == "mock" or llm.get("mock"):
        if mock_handlers is None:
            from .stub_roles import default_mock_handlers
            mock_handlers = default_mock_handlers()
        return MockLLMClient(mock_handlers)
    if provider == "anthropic":
        key_env = llm.get("api_key_env", "ANTHROPIC_API_KEY")
        key = os.environ.get(key_env, "")
        if not key:
            raise LLMError(f"environment variable {key_env} is not set (run with --mock to use the offline client)")
        return AnthropicClient(api_key=key, request_timeout_s=float(llm.get("request_timeout_s", 300)),
                               max_retries=int(llm.get("max_retries", 3)), prompt_caching=bool(llm.get("prompt_caching", True)))
    raise LLMError(f"unknown llm.provider {provider!r}")


class CallLog:
    """Append-only JSONL accounting of every LLM call (role, model, usage, latency). No prompt text, no keys."""

    def __init__(self, path: str):
        self.path = path

    def record(self, iteration: int, role: str, resp: LLMResponse, attempt: int = 1, purpose: str = "") -> None:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "iteration": iteration, "role": role,
               "purpose": purpose, "attempt": attempt, "model": resp.model, "latency_s": round(resp.latency_s, 2),
               "stop_reason": resp.stop_reason, "estimated_usage": resp.estimated_usage, **resp.usage.to_dict()}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
