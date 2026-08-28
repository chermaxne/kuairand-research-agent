"""Provider-agnostic chat call + token accounting.

`LLMClient.complete()` takes the assembled prompt (static system blocks first, so the provider can
cache the static prefix) and returns text + usage read from the API response. The mock client is
deterministic and offline (tests, `--mock` runs). `AnthropicClient` follows the current Anthropic
Messages API: streaming (long outputs), adaptive thinking + `output_config.effort` per role, prompt
caching on the static prefix, server-side refusal fallbacks (beta), typed error chain.
Model names / effort / thinking modes all come from config.yaml — nothing is hardcoded here.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .schemas import TokenUsage

FALLBACK_BETA = "server-side-fallback-2026-07-01"


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
    """Transport-level failure after retries, or a refusal (never a content-format problem)."""


def role_key(role: str) -> str:
    return "scribe" if role.startswith("scribe") else role


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
        h = self.handlers.get(role) or self.handlers.get(role_key(role)) or self.default
        if h is None:
            raise LLMError(f"mock client has no handler for role {role!r}")
        text = h(role, list(system_blocks), [dict(m) for m in messages])
        prompt_chars = sum(len(b) for b in system_blocks) + sum(len(m.get("content", "")) for m in messages)
        usage = TokenUsage(input_tokens=_approx_tokens("x" * prompt_chars), output_tokens=_approx_tokens(text))
        self.calls.append({"role": role, "model": model, "n_messages": len(messages), "usage": usage.to_dict()})
        return LLMResponse(text=text, usage=usage, model=f"mock:{model}", latency_s=time.time() - t0,
                           stop_reason="end_turn", estimated_usage=True)


class AnthropicClient:
    """Thin wrapper over the Anthropic Messages API (streaming). Usage is read from the final message's
    `usage` field — never estimated. `role_params[role_key] = {"effort": ..., "thinking": ...}`."""
    provider = "anthropic"
    TRANSIENT = ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError", "OverloadedError",
                 "ServiceUnavailableError")

    def __init__(self, api_key: Optional[str], *, request_timeout_s: float = 300, max_retries: int = 3, prompt_caching: bool = True,
                 refusal_fallbacks: bool = True, role_params: Optional[Dict[str, Dict[str, Any]]] = None):
        import anthropic  # imported lazily so tests never need the package configured
        self._anthropic = anthropic
        kwargs: Dict[str, Any] = {"timeout": request_timeout_s, "max_retries": 0}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**kwargs)
        self.max_retries = int(max_retries)
        self.prompt_caching = bool(prompt_caching)
        self.refusal_fallbacks = bool(refusal_fallbacks)
        self.role_params = role_params or {}

    # -- request assembly (pure; unit-tested without network) --------------
    def build_request(self, *, role: str, model: str, system_blocks: Sequence[str], messages: Sequence[Dict[str, str]],
                      max_tokens: int) -> Dict[str, Any]:
        system: List[Dict[str, Any]] = []
        blocks = [b for b in system_blocks if b]
        for i, block in enumerate(blocks):
            b: Dict[str, Any] = {"type": "text", "text": block}
            if self.prompt_caching and i == len(blocks) - 1:
                b["cache_control"] = {"type": "ephemeral"}     # cache the whole static prefix (role prompt + knowledge)
            system.append(b)
        req: Dict[str, Any] = {"model": model, "max_tokens": int(max_tokens),
                               "messages": [{"role": m["role"], "content": m["content"]} for m in messages]}
        if system:
            req["system"] = system
        p = self.role_params.get(role_key(role), {})
        thinking = (p.get("thinking") or "none")
        if str(thinking).lower() == "adaptive":
            req["thinking"] = {"type": "adaptive"}
        if p.get("effort"):
            req["output_config"] = {"effort": str(p["effort"])}
        if self.refusal_fallbacks:
            req["betas"] = [FALLBACK_BETA]
            req["fallbacks"] = "default"
        return req

    def _stream_final(self, req: Dict[str, Any]):
        api = self._client.beta.messages if self.refusal_fallbacks else self._client.messages
        with api.stream(**req) as stream:
            return stream.get_final_message()

    @staticmethod
    def parse_message(msg: Any) -> Dict[str, Any]:
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        u = msg.usage
        usage = TokenUsage(input_tokens=int(getattr(u, "input_tokens", 0) or 0),
                           output_tokens=int(getattr(u, "output_tokens", 0) or 0),
                           cache_creation_input_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
                           cache_read_input_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0))
        stop = str(getattr(msg, "stop_reason", "") or "")
        details = getattr(msg, "stop_details", None)
        category = getattr(details, "category", None) if details is not None else None
        return {"text": text, "usage": usage, "model": str(getattr(msg, "model", "")), "stop_reason": stop, "refusal_category": category}

    def complete(self, *, role: str, model: str, system_blocks: Sequence[str], messages: Sequence[Dict[str, str]],
                 max_tokens: int) -> LLMResponse:
        req = self.build_request(role=role, model=model, system_blocks=system_blocks, messages=messages, max_tokens=max_tokens)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                msg = self._stream_final(req)
            except self._anthropic.APIStatusError as e:
                if type(e).__name__ in self.TRANSIENT or getattr(e, "status_code", 0) >= 500:
                    last_err = e
                    time.sleep(min(60, 2 ** attempt * 2))
                    continue
                raise LLMError(f"{type(e).__name__}: {getattr(e, 'message', e)}") from e     # 4xx: not retryable
            except (self._anthropic.APIConnectionError, self._anthropic.APITimeoutError) as e:  # network / timeout
                last_err = e
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            parsed = self.parse_message(msg)
            if parsed["stop_reason"] == "refusal":
                raise LLMError(f"refusal (category={parsed['refusal_category']}) on role {role}")
            return LLMResponse(text=parsed["text"], usage=parsed["usage"], model=parsed["model"], latency_s=time.time() - t0,
                               stop_reason=parsed["stop_reason"])
        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {type(last_err).__name__}: {last_err}")


def make_client(cfg: Dict[str, Any], mock_handlers: Optional[Dict[str, Handler]] = None, force_mock: bool = False) -> LLMClient:
    """Build the client from config. The API key is read from the env var named in config and is never
    written anywhere (not to logs, run dirs, or git)."""
    llm = cfg["llm"]
    provider = str(llm.get("provider", "anthropic")).lower()
    if force_mock or provider == "mock" or llm.get("mock"):
        if mock_handlers is None:
            from .stub_roles import default_mock_handlers, kuairand_mock_handlers
            mock_handlers = kuairand_mock_handlers() if str(llm.get("mock_plan", "toy")) == "kuairand" else default_mock_handlers()
        return MockLLMClient(mock_handlers)
    if provider == "anthropic":
        key_env = llm.get("api_key_env", "ANTHROPIC_API_KEY")
        key = os.environ.get(key_env, "")
        if not key and not llm.get("allow_sdk_default_credentials", False):
            raise LLMError(f"environment variable {key_env} is not set (run with --mock for the offline client, "
                           f"or set llm.allow_sdk_default_credentials: true to let the SDK resolve a login profile)")
        role_params = {k: {"effort": (llm.get("effort") or {}).get(k), "thinking": (llm.get("thinking") or {}).get(k)}
                       for k in ("researcher", "engineer", "debugger", "scribe")}
        return AnthropicClient(api_key=key or None, request_timeout_s=float(llm.get("request_timeout_s", 300)),
                               max_retries=int(llm.get("max_retries", 3)), prompt_caching=bool(llm.get("prompt_caching", True)),
                               refusal_fallbacks=bool(llm.get("refusal_fallbacks", True)), role_params=role_params)
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
