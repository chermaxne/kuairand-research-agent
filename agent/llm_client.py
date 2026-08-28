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
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .schemas import TokenUsage

FALLBACK_BETA = "server-side-fallback-2026-07-01"
POE_BASE_URL = "https://api.poe.com"          # Poe's Anthropic-compatible gateway (key in x-api-key, models = Poe bot handles)


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
                 refusal_fallbacks: bool = True, role_params: Optional[Dict[str, Dict[str, Any]]] = None,
                 base_url: Optional[str] = None, compat: bool = False, provider: str = "anthropic"):
        """`base_url` + `compat=True` target an Anthropic-*compatible* gateway (e.g. Poe): plain-string system prompt,
        no cache_control, no thinking/effort/beta parameters — only the core Messages API surface."""
        import anthropic  # imported lazily so tests never need the package configured
        self._anthropic = anthropic
        kwargs: Dict[str, Any] = {"timeout": request_timeout_s, "max_retries": 0}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)
        self.provider = provider
        self.base_url = base_url
        self.compat = bool(compat)
        self.max_retries = int(max_retries)
        self.prompt_caching = bool(prompt_caching) and not self.compat
        self.refusal_fallbacks = bool(refusal_fallbacks) and not self.compat
        self.role_params = {} if self.compat else (role_params or {})

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
        if system and self.compat:
            req["system"] = "\n\n".join(b["text"] for b in system)     # gateways: plain string, no cache_control
        elif system:
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


def load_dotenv(path: str, override: bool = False) -> List[str]:
    """Load KEY=VALUE lines from a .env file into os.environ (comments/blank lines ignored, optional quotes and a
    leading `export ` stripped). Existing variables win unless override=True. Returns the names loaded — never values."""
    loaded: List[str] = []
    if not path or not os.path.isfile(path):
        return loaded
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if not key or not key.replace("_", "").isalnum():
                continue
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            else:
                val = re.split(r"\s+#", val, maxsplit=1)[0].strip()      # drop an inline comment
            if override or key not in os.environ:
                os.environ[key] = val
                loaded.append(key)
    return loaded


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
    if provider in ("anthropic", "poe", "anthropic-compatible"):
        compat = provider != "anthropic"
        key_env = llm.get("api_key_env", "ANTHROPIC_API_KEY" if not compat else "POE_API_KEY")
        key = os.environ.get(key_env, "")
        if not key and not (llm.get("allow_sdk_default_credentials", False) and not compat):
            raise LLMError(f"environment variable {key_env} is not set (export it in the shell that runs the harness; run with "
                           f"--mock for the offline client)")
        role_params = {k: {"effort": (llm.get("effort") or {}).get(k), "thinking": (llm.get("thinking") or {}).get(k)}
                       for k in ("researcher", "engineer", "debugger", "scribe")}
        base_url = llm.get("base_url") or (POE_BASE_URL if provider == "poe" else None)
        if compat and not base_url:
            raise LLMError("llm.base_url is required for provider anthropic-compatible")
        return AnthropicClient(api_key=key or None, request_timeout_s=float(llm.get("request_timeout_s", 300)),
                               max_retries=int(llm.get("max_retries", 3)), prompt_caching=bool(llm.get("prompt_caching", True)),
                               refusal_fallbacks=bool(llm.get("refusal_fallbacks", True)), role_params=role_params,
                               base_url=base_url, compat=compat, provider=provider)
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


def connectivity_check(cfg: Dict[str, Any], roles: Sequence[str] = ("researcher", "engineer", "debugger", "scribe")) -> List[Dict[str, Any]]:
    """Send one tiny request per configured role model and report text/usage/latency. Validates the key, the
    endpoint and every model id BEFORE Phase 0 spends minutes. Never prints the key."""
    try:
        client = make_client(cfg)
    except LLMError as e:
        return [{"role": "*", "model": "*", "ok": False, "error": str(e)}]
    out = []
    for r in roles:
        model = str(cfg["llm"][f"{r}_model"])
        try:
            resp = client.complete(role=r, model=model, system_blocks=["You are a connectivity probe."],
                                   messages=[{"role": "user", "content": "Reply with the single word OK."}], max_tokens=64)
            out.append({"role": r, "model": model, "ok": True, "reply": resp.text.strip()[:40], "served_by": resp.model,
                        "usage": resp.usage.to_dict(), "latency_s": round(resp.latency_s, 2)})
        except Exception as e:  # noqa: BLE001 - report every failure kind
            out.append({"role": r, "model": model, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    return out
