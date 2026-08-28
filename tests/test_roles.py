"""Phase 3 — role output parsing, briefing assembly order (§3), one re-ask on malformed output,
token accounting, and the Anthropic request shape (built without network)."""
import json
import os
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
