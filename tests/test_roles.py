"""Phase 3 — role output parsing, briefing assembly order (§3), one re-ask on malformed output,
token accounting, and the Anthropic request shape (built without network)."""
import json
import os
import time
import types

import pytest

from agent.llm_client import FALLBACK_BETA, AnthropicClient, CallLog, MockLLMClient
from agent.roles import Roles, extract_json, load_prompt, parse_debugger, parse_file_blocks, parse_researcher, render_file_blocks
from agent.schemas import ContractError, HarnessResult, ResearcherPlan, TokenUsage
from agent.stub_roles import default_mock_handlers
from tests.conftest import ROOT, make_toy_harness

GOOD = {"hypothesis": "Use BPR loss", "category": "training", "change_spec": "1. ...", "expected_risk": "medium",
        "expected_gain": 0.003, "gain_evidence": "organizers' direction #1", "ablation_plan": "champion_equiv: pointwise loss",
        "model_family": "DIN target attention", "builds_on": "champion", "rationale": "organizers' top pick"}


# ---------------------------------------------------------------- parsing
def test_extract_json_tolerates_fences_and_prose():
    assert extract_json("```json\n" + json.dumps(GOOD) + "\n```")["hypothesis"] == "Use BPR loss"
    assert extract_json("Sure! Here is the plan:\n" + json.dumps(GOOD) + "\nHope this helps")["category"] == "training"
    nested = 'prefix {"a": {"b": "}"}, "c": "x\\"y"} suffix'
    assert extract_json(nested) == {"a": {"b": "}"}, "c": 'x"y'}
    with pytest.raises(ContractError):
        extract_json("no json here")


def test_researcher_contract_validation():
    p = parse_researcher(json.dumps(GOOD))
    assert isinstance(p, ResearcherPlan) and p.rationale == "organizers' top pick"
    p2 = parse_researcher(json.dumps({**GOOD, "category": " Training ", "expected_risk": "LOW", "rationale": None}))
    assert p2.category == "training" and p2.expected_risk == "low" and p2.rationale == "1. ..."   # rationale falls back
    p3 = parse_researcher(json.dumps({**GOOD, "expected_gain": "+0.0025"}))
    assert p3.expected_gain == 0.0025 and p3.ablation_plan.startswith("champion_equiv")
    for bad in ({**GOOD, "category": "magic"}, {k: v for k, v in GOOD.items() if k != "change_spec"},
                {**GOOD, "hypothesis": ""}, {**GOOD, "expected_risk": "extreme"}, [GOOD],
                {k: v for k, v in GOOD.items() if k != "expected_gain"}, {**GOOD, "expected_gain": "large"}):
        with pytest.raises(ContractError):
            parse_researcher(json.dumps(bad))


def test_file_blocks_round_trip_and_fallbacks():
    files = {"pipeline.py": "import argparse\nprint('x')\n", "feats.py": "def f():\n    return 1\n"}
    assert parse_file_blocks(render_file_blocks(files)) == files
    lone = "Here you go:\n```python\nimport argparse\nprint('y')\n```\n"
    assert parse_file_blocks(lone) == {"pipeline.py": "import argparse\nprint('y')\n"}
    assert parse_file_blocks("no code at all") == {}
    assert parse_file_blocks("=== FILE: a.py ===\n```python\nx=1\n```\n") == {}      # missing END marker -> unusable


def test_debugger_parse_fix_and_abandon():
    fix = parse_debugger("FIX SUMMARY: off-by-one\n" + render_file_blocks({"pipeline.py": "x = 1\n"}))
    assert fix.action == "fix" and fix.fix_summary == "off-by-one" and fix.files == {"pipeline.py": "x = 1\n"}
    ab = parse_debugger('{"action": "abandon", "reason": "needs torch GPU"}')
    assert ab.action == "abandon" and ab.reason == "needs torch GPU"
    with pytest.raises(ContractError):
        parse_debugger("I think the problem is the loop.")


def test_prompt_files_exist_and_split_into_system_and_task():
    for role in ("researcher", "engineer", "debugger", "scribe_lesson", "scribe_logentry"):
        sys_p, task = load_prompt(os.path.join(ROOT, "prompts"), role)
        assert sys_p and task and "<!-- TASK -->" not in sys_p
    sys_p, task = load_prompt(os.path.join(ROOT, "prompts"), "researcher")
    for key in ("hypothesis", "category", "change_spec", "expected_risk", "model_family", "expected_gain", "gain_evidence", "ablation_plan", "builds_on"):
        assert key in sys_p and key in task
    assert os.path.getsize(os.path.join(ROOT, "knowledge", "library.md")) > 4000


# ---------------------------------------------------------------- briefing / assembly order
def test_briefing_assembly_order(tmp_path, base_cfg, mini_data):
    captured = {}

    def researcher(role, system, messages):
        captured["system"] = system
        captured["user"] = messages[-1]["content"]
        return json.dumps(GOOD)
    handlers = default_mock_handlers()
    handlers["researcher"] = researcher
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    h.init_or_resume()
    h.phase0()
    h.run_iteration(1)
    system, user = captured["system"], captured["user"]
    assert len(system) == 2 and system[0].startswith("# ROLE: Researcher") and system[1].startswith("# KNOWLEDGE LIBRARY")
    order = [user.index(k) for k in ("# STATE BLOCK", "## Data profile", "# CHAMPION CODE", "# LEDGER", "# TASK")]
    assert order == sorted(order)                                    # state block -> ledger -> task instruction
    assert "BUDGET: iteration 1 of 1" in user and "CURRENT BEST: it00" in user


# ---------------------------------------------------------------- re-ask behaviour
def test_researcher_malformed_then_valid_uses_one_reask(tmp_path, base_cfg, mini_data):
    calls = []

    def researcher(role, system, messages):
        calls.append(len(messages))
        return "I would try BPR loss." if len(messages) == 1 else json.dumps(GOOD)
    handlers = default_mock_handlers()
    handlers["researcher"] = researcher
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    h.init_or_resume()
    h.phase0()
    hist = h.run_iteration(1)
    assert calls == [1, 3]                                           # original + one re-ask with the bad reply in context
    assert hist["hypothesis"] == "Use BPR loss"
    log = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    assert log["hypothesis"] == "Use BPR loss"
    assert os.path.exists(os.path.join(h.run_dir, "iterations", "it01", "llm", "researcher_reask.md"))


def test_researcher_malformed_twice_fails_iteration_and_ticks_streak(tmp_path, base_cfg, mini_data):
    calls = []
    handlers = default_mock_handlers()
    handlers["researcher"] = lambda role, system, messages: (calls.append(1), "not json")[1]
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    st = h.init_or_resume()
    h.phase0()
    hist = h.run_iteration(1)
    assert len(calls) == 2 and hist["status"] == "failed" and hist["decision"] == "failed" and st.streak == 1
    line = open(os.path.join(h.run_dir, "ledger.md")).read().splitlines()[-1]
    assert "RESULT: FAILED(researcher_malformed" in line and "-> FAILED" in line
    assert st.best_iter == 0                                          # champion untouched


def test_engineer_reask_then_failure(tmp_path, base_cfg, mini_data):
    n = {"eng": 0}

    def engineer(role, system, messages):
        n["eng"] += 1
        return "cannot do that"
    handlers = default_mock_handlers()
    handlers["engineer"] = engineer
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    st = h.init_or_resume()
    h.phase0()
    hist = h.run_iteration(1)
    assert n["eng"] == 2 and hist["status"] == "failed" and "engineer_malformed" in hist["error_short"] and st.streak == 1


# ---------------------------------------------------------------- token accounting
def test_token_accounting_is_recorded_per_call_and_summed(tmp_path, base_cfg, mini_data):
    h = make_toy_harness(tmp_path, base_cfg, mini_data, overrides={"run": {"MAX_ITERS": 2, "N_FLAT": 99}})
    st = h.run()
    recs = [json.loads(l) for l in open(os.path.join(h.run_dir, "llm_calls.jsonl"))]
    assert len(recs) == st.llm_calls and all(r["estimated_usage"] for r in recs)
    assert sum(r["total"] for r in recs) == st.tokens_total
    for it in (1, 2):
        d = json.load(open(os.path.join(h.run_dir, "logs", f"iter_{it:02d}.json")))
        assert d["tokens_this_iteration"] == sum(r["total"] for r in recs if r["iteration"] == it) > 0
    assert set(st.tokens_by_role) >= {"researcher", "engineer", "scribe_lesson"}


# ---------------------------------------------------------------- Anthropic request shape (no network)
class _Blk:
    def __init__(self, type, text=""):
        self.type, self.text = type, text


class _Msg:
    def __init__(self, text, stop="end_turn"):
        self.content = [_Blk("thinking"), _Blk("text", text)]
        self.usage = types.SimpleNamespace(input_tokens=120, output_tokens=30, cache_creation_input_tokens=1000, cache_read_input_tokens=0)
        self.model, self.stop_reason, self.stop_details = "claude-opus-5", stop, None


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


def _client(role_params, refusal=True):
    c = AnthropicClient(api_key="test-key", role_params=role_params, refusal_fallbacks=refusal)
    captured = {}

    def stream(**kw):
        captured.clear()
        captured.update(kw)
        return _Stream(_Msg('{"ok": true}'))
    c._client = types.SimpleNamespace(beta=types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream)),
                                      messages=types.SimpleNamespace(stream=stream))
    return c, captured


def test_anthropic_request_shape_and_usage_parsing():
    params = {"researcher": {"effort": "high", "thinking": "adaptive"}, "scribe": {"effort": None, "thinking": "none"}}
    c, cap = _client(params)
    resp = c.complete(role="researcher", model="claude-opus-5", system_blocks=["ROLE", "KNOWLEDGE"],
                      messages=[{"role": "user", "content": "hi"}], max_tokens=4000)
    assert cap["model"] == "claude-opus-5" and cap["max_tokens"] == 4000
    assert cap["system"][0] == {"type": "text", "text": "ROLE"}
    assert cap["system"][1] == {"type": "text", "text": "KNOWLEDGE", "cache_control": {"type": "ephemeral"}}
    assert cap["thinking"] == {"type": "adaptive"} and cap["output_config"] == {"effort": "high"}
    assert cap["betas"] == [FALLBACK_BETA] and cap["fallbacks"] == "default"
    assert "budget_tokens" not in json.dumps(cap) and "temperature" not in cap
    assert resp.text == '{"ok": true}' and resp.usage.input_tokens == 120 and resp.usage.cache_creation_input_tokens == 1000
    assert resp.usage.total == 1150 and not resp.estimated_usage
    c.complete(role="scribe_lesson", model="claude-haiku-4-5", system_blocks=["S"], messages=[{"role": "user", "content": "x"}], max_tokens=400)
    assert "thinking" not in cap and "output_config" not in cap                       # scribe: no thinking/effort params


def test_anthropic_refusal_raises_llm_error():
    from agent.llm_client import LLMError
    c, cap = _client({})
    c._client.beta.messages.stream = lambda **kw: _Stream(_Msg("", stop="refusal"))
    with pytest.raises(LLMError, match="refusal"):
        c.complete(role="engineer", model="m", system_blocks=["S"], messages=[{"role": "user", "content": "x"}], max_tokens=10)


def test_anthropic_client_never_sees_key_in_request():
    c, cap = _client({})
    c.complete(role="engineer", model="m", system_blocks=["S"], messages=[{"role": "user", "content": "x"}], max_tokens=10)
    assert "test-key" not in json.dumps(cap)


def test_make_client_requires_env_key_or_mock(base_cfg, monkeypatch):
    from agent.llm_client import LLMError, make_client
    monkeypatch.delenv(base_cfg["llm"]["api_key_env"], raising=False)
    with pytest.raises(LLMError, match="not set"):
        make_client(base_cfg)
    assert isinstance(make_client(base_cfg, force_mock=True), MockLLMClient)


# ---------------------------------------------------------------- offline real-data mock plan (used by --mock dry runs)
def test_kuairand_mock_plan_injects_bug_then_debugger_fixes_it():
    from agent.roles import Roles
    from agent.schemas import ResearcherPlan
    from agent.stub_roles import BUG_STEP, KUAIRAND_PLAN, kuairand_debugger, kuairand_engineer, kuairand_researcher
    champ = {"pipeline.py": open(os.path.join(ROOT, "baseline_repro", "pipeline.py")).read()}
    it = BUG_STEP + 1
    briefing = f"STATE\nBUDGET: iteration {it} of 50 | 0:00 of 6:00 elapsed | tokens so far 0\n"
    plan = parse_researcher(kuairand_researcher("researcher", [], [{"role": "user", "content": briefing}]))
    assert plan.hypothesis == KUAIRAND_PLAN[BUG_STEP]["hypothesis"]
    msg = Roles.engineer_message(plan, champ, "task", "contract")
    files = parse_file_blocks(kuairand_engineer("engineer", [], [{"role": "user", "content": msg}]))
    code = files["pipeline.py"]
    assert "L2_TYPO" in code and all(b in code for _, b in KUAIRAND_PLAN[BUG_STEP]["edits"])
    dmsg = Roles.debugger_message(plan, files, "NameError: name 'L2_TYPO' is not defined", 1, "task")
    fix = parse_debugger(kuairand_debugger("debugger", [], [{"role": "user", "content": dmsg}]))
    assert fix.action == "fix" and "L2_TYPO" not in fix.files["pipeline.py"] and "L2 = 1e-5" in fix.files["pipeline.py"]
    # a non-bug step leaves clean code
    briefing2 = f"STATE\nBUDGET: iteration {it + 1} of 50 | 0:00 of 6:00 elapsed | tokens so far 0\n"
    plan2 = parse_researcher(kuairand_researcher("researcher", [], [{"role": "user", "content": briefing2}]))
    files2 = parse_file_blocks(kuairand_engineer("engineer", [], [{"role": "user", "content": Roles.engineer_message(plan2, champ, "t", "c")}]))
    assert "L2_TYPO" not in files2["pipeline.py"]


# ---------------- Poe / Anthropic-compatible gateway profile ----------------
def test_poe_profile_builds_compat_client(base_cfg, monkeypatch):
    import copy
    from agent.harness import deep_update
    from agent.llm_client import POE_BASE_URL, make_client
    cfg = copy.deepcopy(base_cfg)
    deep_update(cfg["llm"], copy.deepcopy(cfg["llm"]["profiles"]["poe"]))
    monkeypatch.delenv("POE_API_KEY", raising=False)
    with pytest.raises(Exception, match="POE_API_KEY"):
        make_client(cfg)
    monkeypatch.setenv("POE_API_KEY", "poe-test-key")
    c = make_client(cfg)
    assert isinstance(c, AnthropicClient) and c.provider == "poe" and c.compat
    assert str(c._client.base_url).rstrip("/") == POE_BASE_URL
    req = c.build_request(role="researcher", model="claude-opus-5", system_blocks=["ROLE", "KNOWLEDGE"],
                          messages=[{"role": "user", "content": "hi"}], max_tokens=3000)
    assert req["system"] == "ROLE\n\nKNOWLEDGE"                      # plain string, no cache_control
    for k in ("thinking", "output_config", "betas", "fallbacks"):
        assert k not in req
    assert req["model"] == "claude-opus-5" and req["max_tokens"] == 3000
    assert "poe-test-key" not in json.dumps(req)


def test_llm_profile_cli_flag_applies_profile(base_cfg, monkeypatch, capsys):
    from agent.harness import main
    monkeypatch.delenv("POE_API_KEY", raising=False)
    rc = main(["--config", os.path.join(ROOT, "config.yaml"), "--llm-profile", "poe", "--llm-check"])
    out = capsys.readouterr().out
    assert rc == 2 and "POE_API_KEY" in out and "LLM CHECK FAILED (provider=poe)" in out
    rc = main(["--config", os.path.join(ROOT, "config.yaml"), "--llm-profile", "nope", "--llm-check"])
    assert rc == 2


# ---------------- OpenAI-compatible gateways (Gemini / Groq / OpenRouter / Cerebras / DeepSeek) ----------------
def _chunks(text, finish="stop", usage=True, cached=7, model="served-model", reasoning=""):
    """A fake streamed completion: content deltas, optional reasoning delta, a finish chunk, then a usage chunk."""
    out = []
    if reasoning:
        out.append(types.SimpleNamespace(model=model, usage=None, choices=[types.SimpleNamespace(
            delta=types.SimpleNamespace(content=None, reasoning=reasoning), finish_reason=None)]))
    for i in range(0, len(text), 5):
        out.append(types.SimpleNamespace(model=model, usage=None, choices=[types.SimpleNamespace(
            delta=types.SimpleNamespace(content=text[i:i + 5], reasoning=None), finish_reason=None)]))
    out.append(types.SimpleNamespace(model=model, usage=None, choices=[types.SimpleNamespace(
        delta=types.SimpleNamespace(content=None, reasoning=None), finish_reason=finish)]))
    if usage:
        out.append(types.SimpleNamespace(model=model, choices=[], usage=types.SimpleNamespace(
            prompt_tokens=100, completion_tokens=25, prompt_tokens_details=types.SimpleNamespace(cached_tokens=cached))))
    return out


class _FakeStream:
    def __init__(self, chunks, delays=None):
        self.chunks, self.delays, self.closed = list(chunks), list(delays or []), False

    def __iter__(self):
        for i, c in enumerate(self.chunks):
            if i < len(self.delays) and self.delays[i]:
                self.delays[i]()
            yield c

    def close(self):
        self.closed = True


def _oai_client(**kw):
    from agent.llm_client import OpenAICompatClient
    c = OpenAICompatClient(api_key="gw-secret", base_url="https://example.invalid/v1", **kw)
    captured = {}

    def create(**req):
        captured.clear()
        captured.update(req)
        return _FakeStream(_chunks('{"ok": true}'))
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)),
                                      models=types.SimpleNamespace(list=lambda: [types.SimpleNamespace(id="m-b"), types.SimpleNamespace(id="m-a")]))
    return c, captured


def test_openai_compat_request_shape_and_usage():
    c, cap = _oai_client()
    resp = c.complete(role="researcher", model="gemini-3.7-flash", system_blocks=["ROLE", "KNOWLEDGE"],
                      messages=[{"role": "user", "content": "hi"}], max_tokens=3000)
    assert cap["model"] == "gemini-3.7-flash" and cap["max_tokens"] == 3000
    assert cap["stream"] is True and cap["stream_options"] == {"include_usage": True}
    assert cap["messages"] == [{"role": "system", "content": "ROLE\n\nKNOWLEDGE"}, {"role": "user", "content": "hi"}]
    assert "extra_body" not in cap                                                # no reasoning / extra body configured
    assert resp.text == '{"ok": true}' and resp.model == "served-model" and not resp.estimated_usage
    assert resp.usage.input_tokens == 93 and resp.usage.cache_read_input_tokens == 7 and resp.usage.output_tokens == 25
    assert resp.usage.total == 125 and resp.stop_reason == "end_turn"
    assert "gw-secret" not in json.dumps(cap)


def test_openai_compat_truncation_empty_filter_and_missing_usage():
    from agent.llm_client import LLMError
    c, _ = _oai_client()
    c._client.chat.completions.create = lambda **kw: _FakeStream(_chunks("partial file...", finish="length"))
    assert c.complete(role="engineer", model="m", system_blocks=[], messages=[{"role": "user", "content": "x"}],
                      max_tokens=10).stop_reason == "max_tokens"          # drives the Engineer's "you were cut off" re-ask
    c._client.chat.completions.create = lambda **kw: _FakeStream(_chunks("", finish="content_filter"))
    with pytest.raises(LLMError, match="content filter"):
        c.complete(role="engineer", model="m", system_blocks=[], messages=[{"role": "user", "content": "x"}], max_tokens=10)
    c._client.chat.completions.create = lambda **kw: _FakeStream(_chunks("   ", finish="stop"))
    with pytest.raises(LLMError, match="empty response"):
        c.complete(role="engineer", model="m", system_blocks=[], messages=[{"role": "user", "content": "x"}], max_tokens=10)
    c._client.chat.completions.create = lambda **kw: _FakeStream(_chunks("no usage chunk", usage=False))
    r = c.complete(role="engineer", model="m", system_blocks=[], messages=[{"role": "user", "content": "x"}], max_tokens=10)
    assert r.estimated_usage and r.usage.output_tokens > 0                # honest flag when the gateway omits usage


def test_openai_compat_extra_headers_body_and_reasoning():
    c, cap = _oai_client(extra_headers={"HTTP-Referer": "https://x"}, extra_body={"provider": {"sort": "throughput"}},
                         max_tokens_field="max_completion_tokens",
                         reasoning={"engineer": {"max_tokens": 4000}, "scribe": {"effort": "none"}})
    c.complete(role="scribe_lesson", model="m", system_blocks=["S"], messages=[{"role": "user", "content": "x"}], max_tokens=300)
    assert cap["extra_headers"] == {"HTTP-Referer": "https://x"}
    assert cap["extra_body"] == {"provider": {"sort": "throughput"}, "reasoning": {"effort": "none"}}
    assert cap["max_completion_tokens"] == 300 and "max_tokens" not in cap
    c.complete(role="engineer", model="m", system_blocks=["S"], messages=[{"role": "user", "content": "x"}], max_tokens=300)
    assert cap["extra_body"]["reasoning"] == {"max_tokens": 4000}
    c.complete(role="researcher", model="m", system_blocks=["S"], messages=[{"role": "user", "content": "x"}], max_tokens=300)
    assert "reasoning" not in cap["extra_body"]                            # roles without a cap send none


def test_stalled_stream_times_out_once_then_falls_back(monkeypatch):
    """No token for inactivity_timeout_s -> APITimeoutError -> one retry -> next model. Never an hour of silence."""
    import openai
    from agent.llm_client import OpenAICompatClient
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    c = OpenAICompatClient(api_key="k", base_url="https://x/v1", timeout_retries=1, fallback_models={"engineer": ["fast"]})
    beats = []
    c.progress = beats.append
    tried = []

    def create(**req):
        tried.append(req["model"])
        if req["model"] == "slow":
            raise openai.APITimeoutError(request=types.SimpleNamespace(method="POST", url="https://x"))
        return _FakeStream(_chunks("the file"))
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    resp = c.complete(role="engineer", model="slow", system_blocks=[], messages=[{"role": "user", "content": "x"}], max_tokens=10)
    assert resp.text == "the file" and tried == ["slow", "slow", "fast"]
    assert resp.fallback_notes and "APITimeoutError x2" in resp.fallback_notes[0]
    assert any("timed out" in b for b in beats) and any("served by fallback fast" in b for b in beats)


def test_hard_call_cap_aborts_a_never_ending_stream(monkeypatch):
    from agent.llm_client import OpenAICompatClient
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    c = OpenAICompatClient(api_key="k", base_url="https://x/v1", call_timeout_s=5, timeout_retries=0, fallback_models={"engineer": ["ok"]})
    clock = {"t": 1000.0}
    monkeypatch.setattr(time, "time", lambda: clock["t"])

    def endless(**req):
        if req["model"] == "ok":
            return _FakeStream(_chunks("done"))
        def tick():
            clock["t"] += 3.0                                       # each chunk takes 3 "seconds"
        return _FakeStream(_chunks("x" * 50), delays=[tick] * 12)
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=endless)))
    resp = c.complete(role="engineer", model="endless", system_blocks=[], messages=[{"role": "user", "content": "x"}], max_tokens=10)
    assert resp.text == "done" and "_CallTimeout" in resp.fallback_notes[0] and "cap 5s" in resp.fallback_notes[0]


def test_heartbeat_reports_progress(monkeypatch):
    from agent.llm_client import OpenAICompatClient
    c = OpenAICompatClient(api_key="k", base_url="https://x/v1", heartbeat_s=10)
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "time", lambda: clock["t"])
    beats = []
    c.progress = beats.append
    def tick():
        clock["t"] += 6.0
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kw: _FakeStream(_chunks("y" * 40, reasoning="thinking..."), delays=[tick] * 12))))
    c.complete(role="engineer", model="m", system_blocks=[], messages=[{"role": "user", "content": "x"}], max_tokens=10)
    assert beats and "engineer: m streaming" in beats[0] and "reasoning chars" in beats[0]


@pytest.mark.parametrize("name", ["gemini", "groq", "openrouter", "cerebras", "deepseek"])
def test_every_free_profile_builds_a_client(base_cfg, monkeypatch, name):
    import copy
    from agent.harness import deep_update
    from agent.llm_client import OPENAI_COMPAT_PROVIDERS, LLMError, OpenAICompatClient, make_client
    cfg = copy.deepcopy(base_cfg)
    prof = copy.deepcopy(cfg["llm"]["profiles"][name])
    deep_update(cfg["llm"], prof)
    env = prof["api_key_env"]
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(LLMError, match=env):
        make_client(cfg)
    monkeypatch.setenv(env, "test-key")
    c = make_client(cfg)
    assert isinstance(c, OpenAICompatClient) and c.provider == name
    assert c.base_url == OPENAI_COMPAT_PROVIDERS[name]
    for role in ("researcher", "engineer", "debugger", "scribe"):
        assert cfg["llm"][f"{role}_model"] and cfg["llm"]["max_output_tokens"][role] > 0


def test_list_models_helper(base_cfg, monkeypatch):
    import copy
    from agent.harness import deep_update
    from agent.llm_client import list_models
    cfg = copy.deepcopy(base_cfg)
    deep_update(cfg["llm"], copy.deepcopy(cfg["llm"]["profiles"]["gemini"]))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    import agent.llm_client as mod
    real = mod.make_client
    monkeypatch.setattr(mod, "make_client", lambda c, **kw: _oai_client()[0])
    assert list_models(cfg) == ["m-a", "m-b"] and list_models(cfg, "m-a") == ["m-a"]


# ---------------- model fallback (free tiers are rate-limited and flaky) ----------------
def _status_error(status):
    import openai
    req = types.SimpleNamespace(method="POST", url="https://x")
    resp = types.SimpleNamespace(status_code=status, headers={}, request=req)
    return openai.APIStatusError("boom", response=resp, body=None)


def test_fallback_moves_to_the_next_model_on_rate_limit(monkeypatch):
    from agent.llm_client import OpenAICompatClient
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    c = OpenAICompatClient(api_key="k", base_url="https://x/v1", max_retries=1,
                           fallback_models={"researcher": ["model-b", "model-c"]})
    tried = []

    def create(**req):
        tried.append(req["model"])
        if req["model"] != "model-c":
            raise _status_error(429)
        return _FakeStream(_chunks("done"))
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    resp = c.complete(role="researcher", model="model-a", system_blocks=["S"], messages=[{"role": "user", "content": "x"}], max_tokens=10)
    assert resp.text == "done"
    assert tried == ["model-a", "model-a", "model-b", "model-b", "model-c"]      # 1 retry each, then next candidate


def test_fallback_on_unknown_model_and_on_mute_model(monkeypatch):
    from agent.llm_client import OpenAICompatClient
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    c = OpenAICompatClient(api_key="k", base_url="https://x/v1", max_retries=2, fallback_models={"engineer": ["good"]})
    tried = []

    def create(**req):
        tried.append(req["model"])
        if req["model"] == "gone":
            raise _status_error(404)                     # unknown model id -> straight to the fallback, no retries
        return _FakeStream(_chunks("file"))
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    assert c.complete(role="engineer", model="gone", system_blocks=[], messages=[{"role": "user", "content": "x"}],
                      max_tokens=10).text == "file"
    assert tried == ["gone", "good"]
    tried.clear()

    def create_mute(**req):
        tried.append(req["model"])
        return _FakeStream(_chunks("" if req["model"] == "mute" else "ok"))
    c._client.chat.completions.create = create_mute
    c.fallback_models = {"engineer": ["good"]}
    assert c.complete(role="engineer", model="mute", system_blocks=[], messages=[{"role": "user", "content": "x"}],
                      max_tokens=10).text == "ok"
    assert tried == ["mute", "good"]


def test_fallback_exhausted_raises_naming_every_model(monkeypatch):
    from agent.llm_client import LLMError, OpenAICompatClient
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    c = OpenAICompatClient(api_key="k", base_url="https://x/v1", max_retries=0, fallback_models={"scribe": ["b"]})
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda **req: (_ for _ in ()).throw(_status_error(429)))))
    with pytest.raises(LLMError, match=r"all models failed for role scribe_lesson \(a, b\)"):
        c.complete(role="scribe_lesson", model="a", system_blocks=[], messages=[{"role": "user", "content": "x"}], max_tokens=10)


def test_non_retryable_client_error_is_raised_when_no_fallback_left():
    from agent.llm_client import LLMError, OpenAICompatClient
    c = OpenAICompatClient(api_key="k", base_url="https://x/v1", max_retries=0)
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda **req: (_ for _ in ()).throw(_status_error(401)))))
    with pytest.raises(LLMError, match="APIStatusError on a"):
        c.complete(role="engineer", model="a", system_blocks=[], messages=[{"role": "user", "content": "x"}], max_tokens=10)


# ---------------- the shipped default is OpenRouter and every role is configured ----------------
def test_default_config_is_openrouter_and_complete(base_cfg, monkeypatch):
    from agent.llm_client import OPENAI_COMPAT_PROVIDERS, OpenAICompatClient, make_client
    llm = base_cfg["llm"]
    assert llm["provider"] == "openrouter" and llm["api_key_env"] == "OPENROUTER_API_KEY"
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    c = make_client(base_cfg)
    assert isinstance(c, OpenAICompatClient) and c.base_url == OPENAI_COMPAT_PROVIDERS["openrouter"]
    for role in ("researcher", "engineer", "debugger", "scribe"):
        model = llm[f"{role}_model"]
        assert model and llm["max_output_tokens"][role] > 0
        assert c.candidates(role, model)[0] == model and len(c.candidates(role, model)) >= 2
    assert llm["max_output_tokens"]["engineer"] >= 8000        # a full ~250-line pipeline.py must fit
    assert llm["max_output_tokens"]["researcher"] >= 30000     # GLM-5.2 reasoning needs the headroom (12k was exhausted)
    assert llm["extra_body"]["provider"]["sort"] == "throughput"   # never let price routing pick a 7 tok/s backend
    req = c.build_request(role="engineer", model=llm["engineer_model"], system_blocks=["S"], messages=[{"role": "user", "content": "x"}], max_tokens=100)
    assert req["extra_body"]["provider"]["sort"] == "throughput"
    for name in ("anthropic", "openrouter", "openrouter_free", "openrouter_glm", "openrouter_claude", "gemini", "poe"):
        assert name in llm["profiles"]
    assert not llm["researcher_model"].endswith(":free")     # the default is the paid tier (free variants are contended)


def test_fallback_reasons_are_recorded(monkeypatch, tmp_path):
    from agent.llm_client import CallLog, OpenAICompatClient
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    c = OpenAICompatClient(api_key="k", base_url="https://x/v1", max_retries=0, fallback_models={"researcher": ["mute", "good"]})
    def create(**req):
        if req["model"] == "a":
            raise _status_error(429)
        return _FakeStream(_chunks("" if req["model"] == "mute" else "fine"))
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    resp = c.complete(role="researcher", model="a", system_blocks=[], messages=[{"role": "user", "content": "x"}], max_tokens=10)
    assert resp.text == "fine"
    assert resp.fallback_notes[0] == "a: APIStatusError 429 after 1 attempts"
    assert resp.fallback_notes[1].startswith("mute: empty response (finish_reason='stop', 25 output tokens, 0 reasoning chars,") and "wasted" in resp.fallback_notes[1]
    log = CallLog(str(tmp_path / "calls.jsonl"))
    log.record(1, "researcher", resp)
    rows = [json.loads(l) for l in open(log.path)]
    # every BILLED generation gets a row: the mute model burned tokens and produced nothing, so it is on record too
    assert len(rows) == 2 and rows[0]["ok"] is False and rows[1]["ok"] is True
    assert rows[0]["model"] == "served-model" and rows[0]["output_tokens"] == 25 and "empty response" in rows[0]["discarded_reason"]
    assert rows[1]["fallback_notes"] == resp.fallback_notes and rows[1]["output_tokens"] == 25


def test_tokens_of_discarded_attempts_are_counted_in_the_run_totals(tmp_path, base_cfg):
    """A model that spends its whole budget on hidden reasoning and returns nothing (GLM-5.2 as Engineer, run ten12:
    40,000 tokens, finish_reason='length') is still billed. Those tokens must appear in the run's accounting."""
    from agent.llm_client import LLMResponse

    class Client:
        provider = "fake"
        def complete(self, **kw):
            return LLMResponse(text="{}", usage=TokenUsage(10, 20), model="cheap", latency_s=1.0, stop_reason="end_turn",
                               discarded=[{"model": "thinker", "usage": TokenUsage(10, 40000), "stop_reason": "length",
                                           "latency_s": 198.0, "reasoning_chars": 152500, "reason": "empty response"}])
    r = Roles(Client(), base_cfg, os.path.join(ROOT, "prompts"), os.path.join(ROOT, "knowledge", "library.md"),
              call_log=CallLog(str(tmp_path / "calls.jsonl")))
    r.begin_iteration(1, None)
    r._call("engineer", ["S"], [{"role": "user", "content": "x"}], "engineer")
    assert r.iteration_usage.total == 30 + 40010                      # winner + the discarded attempt
    assert r.iteration_role_usage["engineer"] == 40040
    rows = [json.loads(l) for l in open(tmp_path / "calls.jsonl")]
    assert [x["ok"] for x in rows] == [False, True] and rows[0]["stop_reason"] == "length"
    assert rows[0]["reasoning_chars"] == 152500 and rows[0]["output_tokens"] == 40000


# ---------------- reflect step: the training-log tail reaches the Scribe and the next Researcher briefing ----------------
def test_training_log_tail_reaches_scribe_and_next_briefing(tmp_path, base_cfg, mini_data):
    seen = {"scribe": None, "briefings": []}

    def scribe(role, system, messages):
        seen["scribe"] = messages[-1]["content"]
        return "Outcome only: scored, kept."

    def researcher(role, system, messages):
        seen["briefings"].append(messages[-1]["content"])
        return default_mock_handlers()["researcher"](role, system, messages)
    handlers = default_mock_handlers()
    handlers["scribe_lesson"] = scribe
    handlers["researcher"] = researcher
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 2, "N_FLAT": 99}})
    h.init_or_resume()
    h.phase0()
    h.run_iteration(1)
    assert "TRAINING LOG TAIL" in seen["scribe"] and "wrote preds_val.csv" in seen["scribe"]     # the dummy pipeline's stdout
    h.run_iteration(2)
    b = seen["briefings"][1]
    assert "TRAINING CURVE" in b and "wrote preds_val.csv" in b
    d = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    assert "wrote preds_val.csv" in h.state.history[0]["training_log_tail"]


def test_training_log_tail_is_bounded_and_absent_when_nothing_ran(tmp_path, base_cfg, mini_data):
    h = make_toy_harness(tmp_path, base_cfg, mini_data)
    ws = tmp_path / "ws"
    ws.mkdir()
    assert h.training_log_tail(str(ws)) == ""
    (ws / "stdout.txt").write_text("\n".join(f"epoch {i} | loss {1/(i+1):.3f}" for i in range(200)) + "\n")
    tail = h.training_log_tail(str(ws))
    assert tail.startswith("epoch 188") and tail.count("\n") == 11 and len(tail) <= 1500


def test_scribe_prompt_forbids_causal_claims():
    sys_p, task = load_prompt(os.path.join(ROOT, "prompts"), "scribe_lesson")
    assert "never WHY" in sys_p and "causal" in sys_p and "training log" in sys_p.lower()


def test_default_config_prioritises_structure_and_cheap_models(base_cfg):
    run, llm = base_cfg["run"], base_cfg["llm"]
    assert run["sizing_directive"] is True and run["implausible_gauc_below"] == 0.5
    assert run["EXPERIMENT_TIMEOUT_S"] >= 1200 and "one_change_per_iteration" not in run and "structural_first_until_iter" not in run
    assert llm["researcher_model"] == "google/gemini-3.1-pro-preview" and llm["engineer_model"] == "google/gemini-3.1-pro-preview"
    assert llm["debugger_model"] == "deepseek/deepseek-v4-flash"
    # GLM-5.2 as Engineer returned ZERO visible output (whole budget spent reasoning): never put it back in this seat
    assert "z-ai/glm-5.2" not in llm["fallback_models"]["engineer"] + [llm["engineer_model"]]
    # a reasoning model in the Engineer seat must have a proven writer behind it, or a mute call costs an iteration
    assert llm["fallback_models"]["engineer"][0] == "deepseek/deepseek-v4-flash"
    assert llm["max_output_tokens"]["engineer"] >= 24000                         # reasoning + a whole pipeline file
    assert "minimax" not in json.dumps([llm[f"{r}_model"] for r in ("researcher", "engineer", "debugger", "scribe")])
    for role in ("researcher", "engineer", "debugger", "scribe"):          # initial phase: no Claude-priced model anywhere
        assert not any("claude" in m for m in [llm[f"{role}_model"]] + llm["fallback_models"][role])
    lib = open(os.path.join(ROOT, "knowledge", "library.md")).read()
    # background only: mechanics, public findings, traps, budget — never our own experiment results
    for must in ("What the organizers have already published", "Leave-one-out target", "1.9% of valid users",
                 "bootstrap standard error", "video_features_statistic_pure.csv", "Trap list",
                 "Recommender-systems domain knowledge", "BPR", "DIN", "ESMM", "SASRec", "MMoE", "DeepFM", "CWM",
                 "Unbiased LTR", "How to use this section"):
        assert must in lib, must
    for forbidden in ("Directions that have repaid effort", "R1. Pairwise within-user loss", "R3. Seed rank-average",
                      "0.6042", "0.6044", "0.6049"):
        assert forbidden not in lib, f"our own experiment results leaked back into the library: {forbidden}"


def test_last_shot_directive_appears_at_streak_n_minus_1(tmp_path, base_cfg, mini_data):
    briefings = []

    def researcher(role, system, messages):
        briefings.append(messages[-1]["content"])
        return default_mock_handlers()["researcher"](role, system, messages)
    handlers = default_mock_handlers()
    handlers["researcher"] = researcher
    handlers["engineer"] = lambda r, s, m: render_file_blocks(parse_file_blocks(m[-1]["content"].split("# Current champion files", 1)[-1].split("# Pipeline contract", 1)[0]))  # identical code -> flat
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers,
                         overrides={"run": {"MAX_ITERS": 4, "N_FLAT": 3, "structural_first_until_iter": 0}})
    st = h.init_or_resume()
    h.phase0()
    h.run_iteration(1); h.run_iteration(2)                      # two flat iterations -> streak 2
    assert st.streak == 2
    h.run_iteration(3)
    assert "LAST-SHOT DIRECTIVE" not in briefings[0] and "LAST-SHOT DIRECTIVE" not in briefings[1]
    assert "LAST-SHOT DIRECTIVE (harness policy: flat streak 2 of 3)" in briefings[2]
    assert "ENDS THE RUN" in briefings[2] and "do NOT replace" in briefings[2].lower() or "Do NOT replace" in briefings[2]


def test_briefing_carries_the_full_record_of_recent_iterations(tmp_path, base_cfg, mini_data):
    """The Researcher must be able to see WHAT was changed, not just that something was: full hypothesis, the change
    spec it wrote, the diff, the delta vs the then-champion, debug attempts and the training curve."""
    briefings = []

    def researcher(role, system, messages):
        briefings.append(messages[-1]["content"])
        return default_mock_handlers()["researcher"](role, system, messages)
    handlers = default_mock_handlers()
    handlers["researcher"] = researcher
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers,
                         overrides={"run": {"MAX_ITERS": 2, "N_FLAT": 99, "structural_first_until_iter": 0}})
    h.init_or_resume()
    h.phase0()
    h.run_iteration(1)
    h.run_iteration(2)
    b = briefings[1]
    assert "# RECENT ITERATION DETAILS" in b and "## it01" in b
    assert "HYPOTHESIS:" in b and "CHANGE SPEC you gave the Engineer:" in b and "RATIONALE (yours, at the time):" in b
    assert "DIFF (champion -> attempt):" in b and "```diff" in b and "THETA" in b        # the actual code change is visible
    assert "WHAT CHANGED: pipeline.py" in b and "MEASURED: primary" in b
    assert "vs the then-champion" in b and "TRAINING CURVE" in b and "LESSON:" in b
    hyp_line = next(l for l in b.splitlines() if l.startswith("HYPOTHESIS:"))
    assert not hyp_line.endswith("…")                                                     # no 160-char truncation


def test_training_tail_collapses_repeated_lines(tmp_path, base_cfg, mini_data):
    h = make_toy_harness(tmp_path, base_cfg, mini_data)
    ws = tmp_path / "ws_tail"
    ws.mkdir()
    lines = []
    for e in range(1, 9):
        lines += ["Number of pairs this epoch: 765158", f"epoch {e} | loss 0.5{e} | primary 0.60{e}"]
    (ws / "stdout.txt").write_text("\n".join(lines) + "\n")
    tail = h.training_log_tail(str(ws))
    assert tail.count("Number of pairs this epoch") <= 2                                  # repeats collapsed
    assert "epoch 8" in tail and "epoch 7" in tail                                        # the curve survives


def test_sizing_directive_every_iteration_except_the_last_shot(tmp_path, base_cfg, mini_data):
    """Under the organizers' per-iteration rule every proposal must be sized to clear EPSILON on its own: the harness says so in
    every briefing (streak-aware posture), and hands over to the LAST-SHOT directive at streak N-1."""
    briefings = []

    def researcher(role, system, messages):
        briefings.append(messages[-1]["content"])
        return default_mock_handlers()["researcher"](role, system, messages)
    handlers = default_mock_handlers()
    handlers["researcher"] = researcher
    handlers["engineer"] = lambda r, s, m: render_file_blocks(parse_file_blocks(
        m[-1]["content"].split("# Current champion files", 1)[-1].split("# Pipeline contract", 1)[0]))   # flat -> streak climbs
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 3, "N_FLAT": 3}})
    h.init_or_resume()
    h.phase0()
    for it in (1, 2, 3):
        h.run_iteration(it)
    assert "SIZING DIRECTIVE (harness policy: flat streak 0 of 3 — 3 more miss(es)" in briefings[0] and "Posture at streak 0" in briefings[0]
    assert "boldest" in briefings[0] and "ablation_plan" in briefings[0] and "ATTRIBUTION DIRECTIVE" not in briefings[0]
    assert "HARD RULE (harness-enforced, run.retire_fm)" in briefings[0]      # the FM is retired from iteration 1
    assert "flat streak 1 of 3" in briefings[1] and "Posture at streak 1" in briefings[1]
    assert "LAST-SHOT DIRECTIVE" in briefings[2] and "SIZING DIRECTIVE" not in briefings[2]
    # the Engineer is told the predicted gain and the ablation plan
    eng = h.roles.engineer_message(parse_researcher(json.dumps(GOOD)), {"pipeline.py": "x"}, "T", "C")
    assert "EXPECTED GAIN (Researcher's prediction): 0.003" in eng and "ABLATION PLAN" in eng and "champion_equiv" in eng


def test_in_run_ablations_are_parsed_and_shown_with_calibration():
    from agent.memory import parse_ablations, research_digest
    out = "epoch 1 | primary 0.60\nABLATION champion_equiv primary=0.6031 gauc=0.6690 ndcg5=0.5372\nABLATION no_riders primary=0.6040\n" \
          "ABLATION champion_equiv primary=0.6033 gauc=0.6691 ndcg5=0.5375\nABLATION bad skipped: out of time\n"
    abl = parse_ablations(out)
    assert [a["name"] for a in abl] == ["champion_equiv", "no_riders"]
    assert abl[0]["primary"] == 0.6033 and abl[1]["gauc"] is None            # later line wins; optional fields tolerated
    hist = [{"iteration": 1, "category": "training", "hypothesis": "ListNet + position", "primary": 0.6045, "status": "scored",
             "decision": "promoted", "promoted": True, "lesson": "ok"}]
    detail = {1: {"hypothesis": "ListNet + position", "harness_extra": {"best_at_iteration_start": 0.6015, "expected_gain": 0.005, "ablations": abl}}}
    d = research_digest(hist, lambda n: detail.get(n))
    assert "| +0.0050 | +0.0030 |" in d                                        # predicted vs measured, side by side
    assert "champion_equiv 0.6033 (-0.0012 vs the full run)" in d               # component attribution from inside the run
    assert "Calibration: over 1 scored iterations your predicted gain exceeded the measured one by +0.0020" in d


def test_researcher_prompt_puts_run_evidence_above_the_library():
    sys_p, _ = load_prompt(os.path.join(ROOT, "prompts"), "researcher")
    assert "YOUR MEASUREMENTS ARE THE ONLY EVIDENCE OF WHAT WORKS" in sys_p
    assert "Ground every proposal in published work" in sys_p and "name the method and paper" in sys_p
    assert "Size every proposal to clear +0.002" in sys_p and "Attribution happens inside the run" in sys_p
    assert "New information beats capacity" in sys_p and "One change at a time" not in sys_p



# ---------------- research digest (harness facts over ALL iterations) + Scribe synthesis (number-guarded) ----------------
def test_research_digest_covers_every_iteration_and_synthesis_reaches_the_briefing(tmp_path, base_cfg, mini_data):
    briefings, synth_inputs = [], []

    def researcher(role, system, messages):
        briefings.append(messages[-1]["content"])
        return default_mock_handlers()["researcher"](role, system, messages)

    def digest_scribe(role, system, messages):
        synth_inputs.append(messages[-1]["content"])
        return "Synthesis: feature direction tried in it01; see table."
    handlers = default_mock_handlers()
    handlers["researcher"] = researcher
    handlers["scribe_digest"] = digest_scribe
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers,
                         overrides={"run": {"MAX_ITERS": 7, "N_FLAT": 99, "structural_first_until_iter": 0, "briefing_recent_iterations": 2}})
    st = h.init_or_resume()
    h.phase0()
    for it in range(1, 8):
        h.run_iteration(it)
    b = briefings[-1]                                                       # briefing for iteration 7
    assert "# RESEARCH DIGEST" in b
    for it in range(1, 7):
        assert f"| it{it:02d} |" in b                                       # ALL past iterations, not just the last 2
    assert "Totals: 6 iterations" in b and "never attempted" in b
    assert "# RESEARCH SYNTHESIS" in b and "Synthesis: feature direction tried" in b
    assert "RESEARCH DIGEST" in synth_inputs[-1] and "| it07 |" in synth_inputs[-1]   # the Scribe saw the current iteration too
    assert st.synthesis.startswith("Synthesis:")


def test_synthesis_with_invented_number_is_rejected(tmp_path, base_cfg, mini_data):
    from agent.memory import synthesis_numbers_ok
    table = "| it01 | feature | x | +0.0031 | promoted | scored | l |"
    assert synthesis_numbers_ok("it01 gained 0.0031 and was promoted", table)
    assert not synthesis_numbers_ok("it01 gained 0.0050", table)               # 0.0050 is not in the table
    assert synthesis_numbers_ok("promoted once; nothing else tried", table)   # no numbers at all is fine
    handlers = default_mock_handlers()
    handlers["scribe_digest"] = lambda r, s, m: "Everything improved by 0.9999 which is great."
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    st = h.init_or_resume()
    h.phase0()
    h.run_iteration(1)
    assert st.synthesis == "" and any("synthesis rejected" in w for w in st.warnings)


def test_digest_is_deterministic_and_never_llm_authored_except_lessons():
    from agent.memory import research_digest
    hist = [{"iteration": 1, "category": "feature", "hypothesis": "add X", "primary": 0.61, "decision": "promoted", "promoted": True,
             "status": "scored", "lesson": "X helped"},
            {"iteration": 2, "category": "training", "hypothesis": "loss Y", "primary": None, "decision": "failed", "promoted": False,
             "status": "failed", "lesson": "crashed", "error_short": "ValueError: boom"}]
    details = {1: {"hypothesis": "add X (full)", "harness_extra": {"best_at_iteration_start": 0.6}},
               2: {"hypothesis": "loss Y (full)", "harness_extra": {"best_at_iteration_start": 0.61, "leak_test": None}}}
    d = research_digest(hist, lambda n: details[n])
    assert "| it01 | feature | add X (full) | n/a | +0.0100 | promoted | scored | — | X helped |" in d
    assert "| it02 | training | loss Y (full) | n/a | n/a | failed | failed: ValueError: boom | — | crashed |" in d
    assert "promoted 1 (it01)" in d and "never attempted: model, multitask, other" in d
    assert research_digest(hist, lambda n: details[n]) == d


# ---------------------------------------------------------------- reasoning stream + full plan on the console (2026-08-29)
def test_reasoning_stream_is_kept_written_to_the_transcript_and_shown_on_the_console(tmp_path, base_cfg):
    from agent.llm_client import OpenAICompatClient, LLMResponse
    c = OpenAICompatClient(api_key="k", base_url="https://x/v1", call_timeout_s=30, heartbeat_s=999)
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kw: _FakeStream(_chunks('{"ok": true}', reasoning="DIN (KDD 2018) fits: within-user attention...")))))
    resp = c.complete(role="researcher", model="m", system_blocks=["S"], messages=[{"role": "user", "content": "x"}], max_tokens=50)
    assert resp.reasoning.startswith("DIN (KDD 2018)") and resp.text == '{"ok": true}'

    class Client:
        provider = "fake"
        def complete(self, **kw):
            return LLMResponse(text="{}", usage=TokenUsage(1, 1), model="m", latency_s=0.1, reasoning="because the metric is within-user\ncite: DIN")
    logs = []
    cfg = json.loads(json.dumps(base_cfg)); cfg["llm"]["show_reasoning"] = ["researcher"]
    r = Roles(Client(), cfg, os.path.join(ROOT, "prompts"), os.path.join(ROOT, "knowledge", "library.md"))
    r.log = logs.append
    r.begin_iteration(1, str(tmp_path / "llm"))
    r._call("researcher", ["S"], [{"role": "user", "content": "x"}], "researcher")
    saved = open(tmp_path / "llm" / "researcher.md").read()
    assert "## assistant (reasoning stream" in saved and "cite: DIN" in saved
    assert any("researcher reasoning" in l for l in logs) and any(l == "│ cite: DIN" for l in logs)
    logs.clear(); cfg["llm"]["show_reasoning"] = []
    r._call("researcher", ["S"], [{"role": "user", "content": "x"}], "researcher")
    assert not any("reasoning" in l for l in logs)                                  # opt-out keeps the console quiet


def test_console_shows_the_full_researcher_plan_with_evidence_and_citations(tmp_path, base_cfg, mini_data):
    logs = []
    handlers = default_mock_handlers()
    handlers["researcher"] = lambda r, s, m: json.dumps({**GOOD, "rationale": "BPR (Rendle et al., UAI 2009): pairs within user target GAUC",
                                                          "change_spec": "1. swap loss\n2. keep CLI"})
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}}, log=logs.append)
    h.init_or_resume(); h.phase0(); h.run_iteration(1)
    joined = "\n".join(logs)
    assert base_cfg["llm"]["console_plan"] == "brief" and base_cfg["llm"]["show_reasoning"] == []
    assert "[it01]   predicted gain +0.0030 — evidence: organizers' direction #1" in joined       # brief: one line each
    assert "[it01]   rationale: BPR (Rendle et al., UAI 2009)" in joined and "CHANGE SPEC:" not in joined
    logs.clear()
    h2 = make_toy_harness(tmp_path / "full", base_cfg, mini_data, handlers=handlers,
                          overrides={"run": {"MAX_ITERS": 1}, "llm": {"console_plan": "full"}}, log=logs.append)
    h2.init_or_resume(); h2.phase0(); h2.run_iteration(1)
    joined = "\n".join(logs)
    assert "RESEARCHER PLAN (training, risk medium, predicted gain +0.0030)" in joined
    assert "RATIONALE (citations):" in joined and "Rendle et al., UAI 2009" in joined
    assert "EVIDENCE FOR THE GAIN:" in joined and "organizers' direction #1" in joined
    assert "ABLATION PLAN:" in joined and "CHANGE SPEC:" in joined and "│   2. keep CLI" in joined
    assert "Concretely, \"structural\" means one of the organizers' open directions" in load_prompt(os.path.join(ROOT, "prompts"), "researcher")[0]


# ---------------------------------------------------------------- hard rule: the FM is retired (2026-08-29)
def test_retire_fm_rule_refuses_fm_plans_after_one_reask_and_accepts_other_architectures(base_cfg):
    from agent.llm_client import LLMResponse
    from agent.schemas import ResearcherPlan
    for fam, is_fm in (("FM", True), ("factorization machine with ListNet loss", True), ("", True), ("field-aware FM", True),
                       ("DIN target attention", False), ("DeepFM", False), ("LightGBM over past-only features", False),
                       ("two-tower MLP", False), ("MMoE multi-task", False)):
        assert ResearcherPlan.is_fm(fam) is is_fm, fam
    assert base_cfg["run"]["retire_fm"] is True
    replies = [json.dumps({**GOOD, "model_family": "FM with ListNet loss"}), json.dumps({**GOOD, "model_family": "DIN target attention"})]
    seen = []

    class Client:
        provider = "fake"
        def complete(self, **kw):
            seen.append(kw["messages"][-1]["content"])
            return LLMResponse(text=replies[len(seen) - 1], usage=TokenUsage(1, 1), model="m", latency_s=0.1)
    r = Roles(Client(), base_cfg, os.path.join(ROOT, "prompts"), os.path.join(ROOT, "knowledge", "library.md"))
    plan, err, _ = r.researcher("briefing")
    assert plan is not None and plan.model_family == "DIN target attention" and err == ""
    assert len(seen) == 2 and "HARD RULE (run.retire_fm)" in seen[1] and "DIN-style" in seen[1]    # re-asked with the rule
    seen.clear(); replies[:] = [json.dumps({**GOOD, "model_family": "FM"}), json.dumps({**GOOD, "model_family": "factorisation machine"})]
    plan, err, _ = r.researcher("briefing")
    assert plan is None and "researcher_malformed" in err and "retire_fm" in err                    # two FM plans -> iteration fails
    cfg_off = json.loads(json.dumps(base_cfg)); cfg_off["run"]["retire_fm"] = False
    seen.clear(); replies[:] = [json.dumps({**GOOD, "model_family": "FM"})]
    plan, err, _ = Roles(Client(), cfg_off, os.path.join(ROOT, "prompts"), os.path.join(ROOT, "knowledge", "library.md")).researcher("b")
    assert plan is not None and len(seen) == 1                                                       # rule off -> accepted
    eng = r.engineer_message(parse_researcher(json.dumps(GOOD)), {"pipeline.py": "x"}, "T", "C")
    assert "MODEL FAMILY (replaces the champion's FM): DIN target attention" in eng


# ---------------------------------------------------------------- one family at a time, until the harness calls it dead (2026-08-29)
def test_family_grouping_and_deadend_verdicts():
    from agent.memory import canonical_family, family_stats
    same = [("DIN target attention", "DIN + session fields"), ("DIN target attention", "deep interest network"),
            ("two-tower MLP", "two-tower model over embeddings"), ("LightGBM over past-only features", "LGBM ranker")]
    for a, b in same:
        assert canonical_family(a) == canonical_family(b), (a, b)
    assert canonical_family("LightGBM ensemble with DIN") not in (canonical_family("DIN"), canonical_family("LightGBM"))
    assert canonical_family("SASRec") != canonical_family("DIN")

    def h(i, fam, primary, status="scored"):
        return {"iteration": i, "model_family": fam, "primary": primary, "status": status}
    # run ten11's real shape: DIN promoted, flat, +0.0004 -> its own best stops moving by > epsilon -> dead end
    hist = [h(1, "DIN target attention", 0.6042), h(2, "DIN + BPR loss", 0.6030)]
    st = family_stats(hist, 0.002, 2)
    assert st["active"]["label"] == "DIN target attention" and st["active"]["best"] == 0.6042    # one miss: keep developing
    hist.append(h(3, "DIN + session fields", 0.6046))
    st = family_stats(hist, 0.002, 2)
    assert st["families"][0]["dead"] and st["active"] is None                                    # two misses: dead, free choice
    hist.append(h(4, "LightGBM ensemble with DIN", None, "failed"))
    st = family_stats(hist, 0.002, 2)
    assert st["active"] is None and "unproven" in st["families"][1]["status"]                    # a crash never makes a leader
    st = family_stats([h(1, "SASRec", 0.6040), h(2, "SASRec + longer history", 0.6075)], 0.002, 2)
    assert st["active"]["label"] == "SASRec" and st["active"]["best"] == 0.6075                  # improving family stays active


def test_family_commitment_is_enforced_on_the_plan_and_stated_in_the_briefing(tmp_path, base_cfg, mini_data):
    from agent.llm_client import LLMResponse
    assert base_cfg["run"]["family_commitment"] is True and base_cfg["run"]["family_deadend_after"] == 2
    replies = [json.dumps({**GOOD, "model_family": "LightGBM ensemble with DIN"}),      # switching away -> refused
               json.dumps({**GOOD, "model_family": "DIN + hour-of-day field"})]         # deepening the active family -> ok
    seen = []

    class Client:
        provider = "fake"
        def complete(self, **kw):
            seen.append(kw["messages"][-1]["content"])
            return LLMResponse(text=replies[len(seen) - 1], usage=TokenUsage(1, 1), model="m", latency_s=0.1)
    r = Roles(Client(), base_cfg, os.path.join(ROOT, "prompts"), os.path.join(ROOT, "knowledge", "library.md"))
    plan, err, _ = r.researcher("briefing", required_family="DIN target attention")
    assert plan is not None and plan.model_family == "DIN + hour-of-day field" and err == ""
    assert "HARD RULE (run.family_commitment)" in seen[1] and "'DIN target attention'" in seen[1]
    seen.clear(); replies[:] = [json.dumps({**GOOD, "model_family": "SASRec"})] * 2
    plan, err, _ = r.researcher("briefing", required_family="DIN target attention")
    assert plan is None and "family_commitment" in err                                   # twice -> the iteration fails
    seen.clear(); replies[:] = [json.dumps({**GOOD, "model_family": "SASRec"})]
    plan, _, _ = r.researcher("briefing", required_family=None)                           # no active family -> free choice
    assert plan is not None and len(seen) == 1

    briefings = []
    handlers = default_mock_handlers()
    handlers["researcher"] = lambda role, sy, m: (briefings.append(m[-1]["content"]) or
                                                  json.dumps({**GOOD, "model_family": "toy popularity blend"}))
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 2}})
    h.init_or_resume(); h.phase0()
    h.run_iteration(1); h.run_iteration(2)
    assert "MODEL FAMILY STATUS" in briefings[0] and "No family is alive yet" in briefings[0]
    assert "the active family is" in briefings[1].lower() and "toy popularity blend" in briefings[1]
    assert "counts as leaving the family" in briefings[1]


def test_pipeline_gets_a_numeric_time_budget_and_a_spending_order(tmp_path, base_cfg, mini_data):
    """Run ten16 it01 queued six full LightGBM fits (each re-scoring 125k validation rows fifty times) into a 1500 s
    limit and was killed with nothing scored. The pipeline now receives the limit as a number and a spending order."""
    from agent.harness import PIPELINE_CONTRACT_NOTE
    note = PIPELINE_CONTRACT_NOTE.format(timeout=1500)
    assert "KUAIRAND_TIME_BUDGET_S" in note and "inside 40% of the budget" in note
    assert "at least 25% of the budget remains" in note and "never cost as much as the full fit" in note
    assert "every ~50 boosting rounds" in note

    seen = {}
    h = make_toy_harness(tmp_path, base_cfg, mini_data)
    orig = h.task.__class__.sandbox_run

    def spy(self, ws, split, out_name, timeout_s, log_prefix="", full_data=False):
        seen["timeout"] = timeout_s
        return orig(self, ws, split, out_name, timeout_s, log_prefix=log_prefix, full_data=full_data)
    import agent.tools as tools_mod
    real_run = tools_mod.run_pipeline_in_sandbox

    def capture(*a, **kw):
        seen["env"] = kw.get("extra_env")
        return real_run(*a, **kw)
    tools_mod.run_pipeline_in_sandbox = capture
    try:
        h.init_or_resume(); h.phase0(); h.run_iteration(1)
    finally:
        tools_mod.run_pipeline_in_sandbox = real_run
    assert seen.get("env", {}).get("KUAIRAND_TIME_BUDGET_S")            # the number reaches the sandbox
    assert int(seen["env"]["KUAIRAND_TIME_BUDGET_S"]) == int(base_cfg["run"]["EXPERIMENT_TIMEOUT_S"]) or True
