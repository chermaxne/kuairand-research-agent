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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .schemas import TokenUsage

FALLBACK_BETA = "server-side-fallback-2026-07-01"
POE_BASE_URL = "https://api.poe.com"          # Poe's Anthropic-compatible gateway (key in x-api-key, models = Poe bot handles)
# OpenAI-compatible gateways: provider name -> base URL (each also selectable via provider: openai-compatible + base_url)
OPENAI_COMPAT_PROVIDERS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "together": "https://api.together.xyz/v1",
}


@dataclass
class LLMResponse:
    text: str
    usage: TokenUsage
    model: str
    latency_s: float
    stop_reason: str = ""
    estimated_usage: bool = False     # True when the client had no API usage field (mock)
    fallback_notes: List[str] = field(default_factory=list)   # why earlier candidate models were skipped


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
        try:
            import anthropic  # imported lazily so tests never need the package configured
        except ImportError as e:
            raise LLMError(
                "the `anthropic` package is required for the Anthropic provider but is not installed in the interpreter running "
                "the harness. You are probably not using the project venv — run `.venv/bin/python -m agent.harness ...` "
                "(or `source .venv/bin/activate` first), or install it with `pip install -r requirements.txt`."
            ) from e
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


class OpenAICompatClient:
    """Chat-Completions client for any OpenAI-compatible gateway (OpenRouter, Google Gemini, Groq, Cerebras,
    DeepSeek, ...). The four roles only need text in / text out, so the compat surface is enough.

    Every call is STREAMED so that a stalled generation is detected by an inactivity timeout (no chunk for
    `inactivity_timeout_s`) instead of a blind request timeout, a hard per-call cap (`call_timeout_s`) bounds the
    worst case, and a heartbeat reports progress. Timeouts fall back to the next model after `timeout_retries`
    (default 1) — a slow model must never silently eat an hour. Usage is read from the final streamed chunk
    (`stream_options.include_usage`); when a gateway omits it the usage is estimated and flagged as such.
    """
    provider = "openai-compatible"
    TRANSIENT = ("RateLimitError", "APIConnectionError", "InternalServerError")

    def __init__(self, api_key: Optional[str], base_url: str, *, request_timeout_s: float = 300, max_retries: int = 3,
                 provider_name: str = "openai-compatible", max_tokens_field: str = "max_tokens",
                 extra_body: Optional[Dict[str, Any]] = None, extra_headers: Optional[Dict[str, str]] = None,
                 fallback_models: Optional[Dict[str, List[str]]] = None, inactivity_timeout_s: float = 120,
                 call_timeout_s: float = 900, timeout_retries: int = 1, reasoning: Optional[Dict[str, Any]] = None,
                 heartbeat_s: float = 30):
        try:
            import openai  # imported lazily so the Anthropic path never needs this package
        except ImportError as e:
            raise LLMError(
                "the `openai` package is required for OpenAI-compatible providers (OpenRouter, Gemini, Groq, ...) but is not installed in the interpreter running "
                "the harness. You are probably not using the project venv — run `.venv/bin/python -m agent.harness ...` "
                "(or `source .venv/bin/activate` first), or install it with `pip install -r requirements.txt`."
            ) from e
        self._openai = openai
        # The SDK timeout is applied to every socket read: with streaming that is exactly an inactivity timeout.
        self._client = openai.OpenAI(api_key=api_key or "missing", base_url=base_url, timeout=float(inactivity_timeout_s), max_retries=0)
        self.provider = provider_name
        self.base_url = base_url
        self.max_retries = int(max_retries)
        self.timeout_retries = int(timeout_retries)
        self.inactivity_timeout_s = float(inactivity_timeout_s)
        self.call_timeout_s = float(call_timeout_s)
        self.heartbeat_s = float(heartbeat_s)
        self.max_tokens_field = max_tokens_field
        self.extra_body = dict(extra_body or {})
        self.extra_headers = dict(extra_headers or {})
        # role -> ordered alternates tried when the primary model is rate-limited or unavailable (free tiers are flaky)
        self.fallback_models = {k: list(v) for k, v in (fallback_models or {}).items() if v}
        # role -> OpenRouter-style unified `reasoning` object (e.g. {"max_tokens": 4000} or {"effort": "none"})
        self.reasoning = {k: dict(v) for k, v in (reasoning or {}).items() if v}
        self.progress: Optional[Callable[[str], None]] = None      # heartbeat sink (the harness sets its logger)

    # -- request assembly (pure; unit-tested without network) --------------
    def build_request(self, *, role: str, model: str, system_blocks: Sequence[str], messages: Sequence[Dict[str, str]],
                      max_tokens: int) -> Dict[str, Any]:
        msgs: List[Dict[str, str]] = []
        blocks = [b for b in system_blocks if b]
        if blocks:
            msgs.append({"role": "system", "content": "\n\n".join(blocks)})
        msgs += [{"role": m["role"], "content": m["content"]} for m in messages]
        req: Dict[str, Any] = {"model": model, "messages": msgs, self.max_tokens_field: int(max_tokens)}
        body = dict(self.extra_body)
        r = self.reasoning.get(role_key(role))
        if r:
            body["reasoning"] = dict(r)
        if body:
            req["extra_body"] = body
        return req

    @staticmethod
    def parse_response(resp: Any) -> Dict[str, Any]:
        """Non-streaming response object -> parsed dict (kept for gateways/tests that answer in one piece)."""
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        text = (getattr(getattr(choice, "message", None), "content", "") or "") if choice else ""
        finish = str(getattr(choice, "finish_reason", "") or "") if choice else ""
        return {"text": text, "finish_reason": finish, "usage": OpenAICompatClient._usage_from(getattr(resp, "usage", None), text),
                "model": str(getattr(resp, "model", "")), "estimated": getattr(resp, "usage", None) is None}

    @staticmethod
    def _usage_from(u: Any, text: str) -> TokenUsage:
        if u is None:
            return TokenUsage(output_tokens=_approx_tokens(text))
        cached = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        return TokenUsage(input_tokens=int(getattr(u, "prompt_tokens", 0) or 0) - cached,
                          output_tokens=int(getattr(u, "completion_tokens", 0) or 0), cache_read_input_tokens=cached)

    @staticmethod
    def _stop_reason(finish: str) -> str:
        return {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal", "tool_calls": "tool_use"}.get(finish, finish)

    class _CallTimeout(Exception):
        pass

    def _stream_call(self, req: Dict[str, Any], role: str, candidate: str) -> Dict[str, Any]:
        """One streamed generation. Raises openai errors, or _CallTimeout when the hard cap is exceeded."""
        t0 = time.time()
        last_beat = t0
        parts: List[str] = []
        finish, usage_obj, served, reasoning_chars = "", None, "", 0
        kwargs = dict(req)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        stream = self._client.chat.completions.create(**kwargs)
        try:
            for chunk in stream:
                now = time.time()
                if now - t0 > self.call_timeout_s:
                    raise self._CallTimeout(f"hard cap {self.call_timeout_s:.0f}s exceeded after {sum(len(x) for x in parts)} chars")
                served = served or str(getattr(chunk, "model", "") or "")
                if getattr(chunk, "usage", None) is not None:
                    usage_obj = chunk.usage
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    if delta is not None:
                        piece = getattr(delta, "content", None)
                        if piece:
                            parts.append(piece)
                        rd = getattr(delta, "reasoning", None)
                        if rd:
                            reasoning_chars += len(rd)
                    fr = getattr(choices[0], "finish_reason", None)
                    if fr:
                        finish = str(fr)
                if self.progress and now - last_beat >= self.heartbeat_s:
                    last_beat = now
                    self.progress(f"[llm] {role}: {candidate} streaming — {sum(len(x) for x in parts):,} chars"
                                  f"{f' (+{reasoning_chars:,} reasoning chars)' if reasoning_chars else ''}, {now - t0:.0f}s")
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
        text = "".join(parts)
        return {"text": text, "finish_reason": finish, "usage": self._usage_from(usage_obj, text), "model": served or candidate,
                "estimated": usage_obj is None, "latency_s": time.time() - t0, "reasoning_chars": reasoning_chars}

    def candidates(self, role: str, model: str) -> List[str]:
        """Primary model first, then the role's configured alternates (deduplicated, order preserved)."""
        out = [model]
        for m in self.fallback_models.get(role_key(role), []):
            if m not in out:
                out.append(m)
        return out

    def complete(self, *, role: str, model: str, system_blocks: Sequence[str], messages: Sequence[Dict[str, str]],
                 max_tokens: int) -> LLMResponse:
        last_err: Optional[Exception] = None
        notes: List[str] = []
        models = self.candidates(role, model)
        for m_i, candidate in enumerate(models):
            req = self.build_request(role=role, model=candidate, system_blocks=system_blocks, messages=messages, max_tokens=max_tokens)
            timeouts = 0
            attempt = 0
            while True:
                try:
                    parsed = self._stream_call(req, role, candidate)
                except (self._openai.APITimeoutError, self._CallTimeout) as e:      # stalled or over the hard cap
                    last_err = e
                    timeouts += 1
                    if timeouts <= self.timeout_retries:
                        if self.progress:
                            self.progress(f"[llm] {role}: {candidate} timed out ({type(e).__name__}); retrying once")
                        continue
                    notes.append(f"{candidate}: {type(e).__name__} x{timeouts} (inactivity {self.inactivity_timeout_s:.0f}s / cap {self.call_timeout_s:.0f}s)")
                    break                                                            # -> next model
                except self._openai.APIStatusError as e:
                    status = int(getattr(e, "status_code", 0) or 0)
                    if type(e).__name__ in self.TRANSIENT or status >= 500 or status == 429:
                        last_err = e
                        if attempt < self.max_retries:
                            attempt += 1
                            time.sleep(min(90, 2 ** attempt * 3))                    # rate limits: back off generously
                            continue
                        notes.append(f"{candidate}: {type(e).__name__} {status} after {attempt + 1} attempts")
                        break
                    if status in (400, 402, 403, 404) and m_i < len(models) - 1:
                        last_err = e                                                 # unknown/forbidden/unfunded model -> next
                        notes.append(f"{candidate}: HTTP {status} {str(getattr(e, 'message', e))[:120]}")
                        break
                    raise LLMError(f"{type(e).__name__} on {candidate}: {getattr(e, 'message', e)}") from e
                except self._openai.APIConnectionError as e:
                    last_err = e
                    if attempt < self.max_retries:
                        attempt += 1
                        time.sleep(min(90, 2 ** attempt * 3))
                        continue
                    notes.append(f"{candidate}: {type(e).__name__} after {attempt + 1} attempts")
                    break
                stop = self._stop_reason(parsed["finish_reason"])
                if stop == "refusal":
                    raise LLMError(f"content filter refused the {role} call ({candidate})")
                if not parsed["text"].strip():
                    last_err = LLMError(f"empty response from {candidate} (finish_reason={parsed['finish_reason']!r})")
                    notes.append(f"{candidate}: empty response (finish_reason={parsed['finish_reason']!r}, "
                                 f"{parsed['usage'].output_tokens} output tokens, {parsed.get('reasoning_chars', 0)} reasoning chars)")
                    break                                                            # a mute model is a dead model
                if self.progress and notes:
                    self.progress(f"[llm] {role}: served by fallback {candidate} after: " + "; ".join(notes))
                return LLMResponse(text=parsed["text"], usage=parsed["usage"], model=parsed["model"] or candidate,
                                   latency_s=parsed.get("latency_s", 0.0), stop_reason=stop, estimated_usage=bool(parsed.get("estimated")),
                                   fallback_notes=notes)
        raise LLMError(f"all models failed for role {role} ({', '.join(models)}): "
                       f"{type(last_err).__name__}: {str(last_err)[:300]}")


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
            if not val:                                   # an empty placeholder is not a value
                continue
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
    if provider in OPENAI_COMPAT_PROVIDERS or provider == "openai-compatible":
        base_url = llm.get("base_url") or OPENAI_COMPAT_PROVIDERS.get(provider)
        if not base_url:
            raise LLMError("llm.base_url is required for provider openai-compatible")
        key_env = llm.get("api_key_env") or f"{provider.upper().replace('-', '_')}_API_KEY"
        key = os.environ.get(key_env, "")
        if not key:
            raise LLMError(f"environment variable {key_env} is not set (put it in .env or export it; run with --mock for the offline client)")
        return OpenAICompatClient(api_key=key, base_url=base_url, request_timeout_s=float(llm.get("request_timeout_s", 300)),
                                  max_retries=int(llm.get("max_retries", 3)), provider_name=provider,
                                  max_tokens_field=str(llm.get("max_tokens_field", "max_tokens")),
                                  extra_body=llm.get("extra_body") or {}, extra_headers=llm.get("extra_headers") or {},
                                  fallback_models=llm.get("fallback_models") or {},
                                  inactivity_timeout_s=float(llm.get("inactivity_timeout_s", 120)),
                                  call_timeout_s=float(llm.get("call_timeout_s", 900)),
                                  timeout_retries=int(llm.get("timeout_retries", 1)),
                                  reasoning=llm.get("reasoning") or {}, heartbeat_s=float(llm.get("heartbeat_s", 30)))
    raise LLMError(f"unknown llm.provider {provider!r}")


class CallLog:
    """Append-only JSONL accounting of every LLM call (role, model, usage, latency). No prompt text, no keys."""

    def __init__(self, path: str):
        self.path = path

    def record(self, iteration: int, role: str, resp: LLMResponse, attempt: int = 1, purpose: str = "") -> None:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "iteration": iteration, "role": role,
               "purpose": purpose, "attempt": attempt, "model": resp.model, "latency_s": round(resp.latency_s, 2),
               "stop_reason": resp.stop_reason, "estimated_usage": resp.estimated_usage,
               "fallback_notes": list(getattr(resp, "fallback_notes", []) or []), **resp.usage.to_dict()}
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
            # reasoning models spend tokens thinking before the first visible token: don't starve the probe
            probe_tokens = min(2000, int((cfg["llm"].get("max_output_tokens") or {}).get(role_key(r), 2000)))
            resp = client.complete(role=r, model=model, system_blocks=["You are a connectivity probe."],
                                   messages=[{"role": "user", "content": "Reply with the single word OK."}],
                                   max_tokens=max(512, probe_tokens))
            out.append({"role": r, "model": model, "ok": True, "reply": resp.text.strip()[:40], "served_by": resp.model,
                        "usage": resp.usage.to_dict(), "latency_s": round(resp.latency_s, 2)})
        except Exception as e:  # noqa: BLE001 - report every failure kind
            out.append({"role": r, "model": model, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    return out


def list_models(cfg: Dict[str, Any], contains: str = "") -> List[str]:
    """Ask the configured gateway which model ids it serves (works on every OpenAI-compatible provider and on
    Poe). Use it to discover the exact handles to put in the profile."""
    client = make_client(cfg)
    inner = getattr(client, "_client", None)
    if inner is None or not hasattr(inner, "models"):
        raise LLMError(f"provider {getattr(client, 'provider', '?')} does not expose a model listing")
    ids = sorted(str(m.id) for m in inner.models.list())
    return [i for i in ids if contains.lower() in i.lower()] if contains else ids


def provider_balance(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Credit usage/remaining for providers that expose it (currently OpenRouter). Returns None when the
    provider has no such endpoint or the call fails — never raises, never logs the key."""
    llm = cfg.get("llm", {})
    if str(llm.get("provider", "")).lower() != "openrouter":
        return None
    key = os.environ.get(llm.get("api_key_env", "OPENROUTER_API_KEY"), "")
    if not key:
        return None
    try:
        # reuse an SDK's HTTP stack: it ships a CA bundle, which bare urllib does not on every host
        import httpx2 as httpx
    except ImportError:
        try:
            import httpx  # type: ignore[no-redef]
        except ImportError:
            return None
    try:
        r = httpx.get("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        d = r.json().get("data", {})
    except Exception:  # noqa: BLE001 - accounting must never break a run
        return None
    return {"usage_usd": d.get("usage"), "limit_usd": d.get("limit"), "remaining_usd": d.get("limit_remaining"),
            "is_free_tier": d.get("is_free_tier")}
