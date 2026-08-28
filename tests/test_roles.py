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
        "builds_on": "champion", "rationale": "organizers' top pick"}


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
    for bad in ({**GOOD, "category": "magic"}, {k: v for k, v in GOOD.items() if k != "change_spec"},
                {**GOOD, "hypothesis": ""}, {**GOOD, "expected_risk": "extreme"}, [GOOD]):
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
    for key in ("hypothesis", "category", "change_spec", "expected_risk", "builds_on"):
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
class _OAIMsg:
    def __init__(self, content):
        self.content = content


class _OAIChoice:
    def __init__(self, content, finish="stop"):
        self.message, self.finish_reason = _OAIMsg(content), finish


class _OAIResp:
    def __init__(self, content='{"ok": true}', finish="stop", cached=7):
        self.choices = [_OAIChoice(content, finish)]
        self.model = "served-model"
        self.usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=25,
                                           prompt_tokens_details=types.SimpleNamespace(cached_tokens=cached))


def _oai_client(**kw):
    from agent.llm_client import OpenAICompatClient
    c = OpenAICompatClient(api_key="gw-secret", base_url="https://example.invalid/v1", **kw)
    captured = {}

    def create(**req):
        captured.clear()
        captured.update(req)
        return _OAIResp()
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)),
                                      models=types.SimpleNamespace(list=lambda: [types.SimpleNamespace(id="m-b"), types.SimpleNamespace(id="m-a")]))
    return c, captured


def test_openai_compat_request_shape_and_usage():
    c, cap = _oai_client()
    resp = c.complete(role="researcher", model="gemini-3.7-flash", system_blocks=["ROLE", "KNOWLEDGE"],
                      messages=[{"role": "user", "content": "hi"}], max_tokens=3000)
    assert cap["model"] == "gemini-3.7-flash" and cap["max_tokens"] == 3000
    assert cap["messages"] == [{"role": "system", "content": "ROLE\n\nKNOWLEDGE"}, {"role": "user", "content": "hi"}]
    assert resp.text == '{"ok": true}' and resp.model == "served-model" and not resp.estimated_usage
    assert resp.usage.input_tokens == 93 and resp.usage.cache_read_input_tokens == 7 and resp.usage.output_tokens == 25
    assert resp.usage.total == 125 and resp.stop_reason == "end_turn"
    assert "gw-secret" not in json.dumps(cap)


def test_openai_compat_truncation_and_empty_and_filter():
    from agent.llm_client import LLMError, OpenAICompatClient
    c, _ = _oai_client()
    c._client.chat.completions.create = lambda **kw: _OAIResp("partial file...", finish="length")
    assert c.complete(role="engineer", model="m", system_blocks=[], messages=[{"role": "user", "content": "x"}],
                      max_tokens=10).stop_reason == "max_tokens"          # drives the Engineer's "you were cut off" re-ask
    c._client.chat.completions.create = lambda **kw: _OAIResp("", finish="content_filter")
    with pytest.raises(LLMError, match="content filter"):
        c.complete(role="engineer", model="m", system_blocks=[], messages=[{"role": "user", "content": "x"}], max_tokens=10)
    c._client.chat.completions.create = lambda **kw: _OAIResp("   ", finish="stop")
    with pytest.raises(LLMError, match="empty response"):
        c.complete(role="engineer", model="m", system_blocks=[], messages=[{"role": "user", "content": "x"}], max_tokens=10)


def test_openai_compat_extra_headers_and_body():
    c, cap = _oai_client(extra_headers={"HTTP-Referer": "https://x"}, extra_body={"provider": {"sort": "throughput"}},
                         max_tokens_field="max_completion_tokens")
    c.complete(role="scribe_lesson", model="m", system_blocks=["S"], messages=[{"role": "user", "content": "x"}], max_tokens=300)
    assert cap["extra_headers"] == {"HTTP-Referer": "https://x"} and cap["provider"] == {"sort": "throughput"}
    assert cap["max_completion_tokens"] == 300 and "max_tokens" not in cap


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
        return _OAIResp("done")
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
        return _OAIResp("file")
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    assert c.complete(role="engineer", model="gone", system_blocks=[], messages=[{"role": "user", "content": "x"}],
                      max_tokens=10).text == "file"
    assert tried == ["gone", "good"]
    tried.clear()

    def create_mute(**req):
        tried.append(req["model"])
        return _OAIResp("" if req["model"] == "mute" else "ok")
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
    for name in ("anthropic", "openrouter", "openrouter_free", "openrouter_glm", "openrouter_claude", "gemini", "poe"):
        assert name in llm["profiles"]
    assert not llm["researcher_model"].endswith(":free")     # the default is the paid tier (free variants are contended)
