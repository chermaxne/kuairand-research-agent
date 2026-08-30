"""Unit tests for agent/research_tools.py's pure logic (no network) and Roles._call_with_tools's
loop mechanics (fake client, no network/API). The real network calls and real streaming
accumulation in llm_client.py were verified live against the shipped model before this was wired
in; these tests lock in the parts that can be tested offline so a refactor doesn't silently break
the dedup/cap/host-restriction contracts or the tool loop's turn/error handling."""
import json
import os

import pytest

from agent.research_tools import MAX_LIBRARY_CHARS, append_knowledge, web_fetch
from agent.roles import Roles
from agent.llm_client import LLMResponse
from agent.schemas import ContractError, ResearcherPlan, TokenUsage


# ---------------------------------------------------------------------------
# append_knowledge
# ---------------------------------------------------------------------------
def test_append_knowledge_appends_and_dedups(tmp_path):
    lib = tmp_path / "library.md"
    lib.write_text("# existing content\n")

    assert append_knowledge(str(lib), "A new finding about BPR loss (source: http://arxiv.org/abs/1)") is True
    text = lib.read_text()
    assert "A new finding about BPR loss" in text

    # same entry again -> deduped (fingerprint = first 80 chars already present)
    assert append_knowledge(str(lib), "A new finding about BPR loss (source: http://arxiv.org/abs/1)") is False
    assert lib.read_text().count("A new finding about BPR loss") == 1


def test_append_knowledge_empty_entry_is_noop(tmp_path):
    lib = tmp_path / "library.md"
    lib.write_text("# existing\n")
    assert append_knowledge(str(lib), "   ") is False
    assert lib.read_text() == "# existing\n"


def test_append_knowledge_respects_size_cap(tmp_path):
    lib = tmp_path / "library.md"
    lib.write_text("x" * (MAX_LIBRARY_CHARS - 10))
    assert append_knowledge(str(lib), "y" * 50) is False   # would exceed the cap
    assert append_knowledge(str(lib), "y" * 5) is True      # fits


# ---------------------------------------------------------------------------
# web_fetch host restriction (no network for the rejected case)
# ---------------------------------------------------------------------------
def test_web_fetch_rejects_non_arxiv_hosts():
    with pytest.raises(ValueError, match="arxiv.org"):
        web_fetch("https://evil.example.com/steal")
    with pytest.raises(ValueError, match="arxiv.org"):
        web_fetch("http://arxiv.org.evil.com/abs/1")   # lookalike host, not a real arxiv.org subdomain


# ---------------------------------------------------------------------------
# Roles construction: research_tools opt-in
# ---------------------------------------------------------------------------
def test_roles_research_tools_disabled_by_default(tmp_path):
    prompts_dir = str(tmp_path / "prompts")
    os.makedirs(prompts_dir)
    r = Roles(client=None, cfg={"llm": {}}, prompts_dir=prompts_dir, knowledge_path="")
    assert r.research_tools_enabled is False
    assert r._research_tools is None


def test_roles_research_tools_enabled_via_config(tmp_path):
    prompts_dir = str(tmp_path / "prompts")
    os.makedirs(prompts_dir)
    cfg = {"llm": {}, "research_tools": {"enabled": True, "max_tool_turns": 3}}
    r = Roles(client=None, cfg=cfg, prompts_dir=prompts_dir, knowledge_path="")
    assert r.research_tools_enabled is True
    assert r.max_tool_turns == 3
    assert r._research_tools is not None
    assert {t["function"]["name"] for t in r._research_tools} == {"arxiv_search", "web_fetch"}


# ---------------------------------------------------------------------------
# _call_with_tools loop mechanics, against a fake client (no network)
# ---------------------------------------------------------------------------
class _FakeClient:
    """Replays a scripted sequence of LLMResponse objects, one per .complete() call, so the loop's
    turn-counting / message-accumulation / forced-final logic can be tested without any network."""
    provider = "fake"

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def complete(self, *, role, model, system_blocks, messages, max_tokens, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.scripted.pop(0)


def _resp(text="", tool_calls=None, stop_reason="end_turn"):
    return LLMResponse(text=text, usage=TokenUsage(input_tokens=10, output_tokens=5), model="fake-model",
                       latency_s=0.01, stop_reason=stop_reason, tool_calls=tool_calls)


def _make_roles(client, max_tool_turns=6):
    cfg = {"llm": {"researcher_model": "fake-model", "max_output_tokens": {"researcher": 100}},
          "research_tools": {"enabled": True, "max_tool_turns": max_tool_turns}}
    r = Roles(client=client, cfg=cfg, prompts_dir="/nonexistent", knowledge_path="")
    return r


def test_call_with_tools_single_call_then_final():
    tc = [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": json.dumps({"q": "bpr"})}}]
    client = _FakeClient([
        _resp(tool_calls=tc, stop_reason="tool_use"),
        _resp(text="final answer here", stop_reason="end_turn"),
    ])
    roles = _make_roles(client)
    executors = {"search": lambda args: f"result for {args['q']}"}
    messages = [{"role": "user", "content": "go"}]
    resp, log = roles._call_with_tools("researcher", [], messages, "researcher", [{"type": "function"}], executors)
    assert resp.text == "final answer here"
    assert resp.tool_calls is None
    assert len(log) == 1
    assert log[0]["tool"] == "search"
    assert log[0]["args"] == {"q": "bpr"}
    assert "result for bpr" in log[0]["result_preview"]
    # the tool result must be wrapped as external data, and the assistant's tool_calls preserved in the transcript
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "EXTERNAL CONTENT" in tool_msgs[0]["content"]
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    assert assistant_msgs[0]["tool_calls"] == tc


def test_call_with_tools_unknown_tool_name_is_recorded_as_error():
    tc = [{"id": "c1", "type": "function", "function": {"name": "nonexistent_tool", "arguments": "{}"}}]
    client = _FakeClient([
        _resp(tool_calls=tc, stop_reason="tool_use"),
        _resp(text="ok, giving up on the tool", stop_reason="end_turn"),
    ])
    roles = _make_roles(client)
    resp, log = roles._call_with_tools("researcher", [], [{"role": "user", "content": "go"}], "researcher",
                                       [{"type": "function"}], executors={})
    assert log[0]["result_preview"].startswith("ERROR: unknown tool")


def test_call_with_tools_malformed_arguments_is_recorded_as_error():
    tc = [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{not valid json"}}]
    client = _FakeClient([
        _resp(tool_calls=tc, stop_reason="tool_use"),
        _resp(text="final", stop_reason="end_turn"),
    ])
    roles = _make_roles(client)
    resp, log = roles._call_with_tools("researcher", [], [{"role": "user", "content": "go"}], "researcher",
                                       [{"type": "function"}], executors={"search": lambda a: "unused"})
    assert log[0]["result_preview"].startswith("ERROR: could not parse arguments")


def test_call_with_tools_executor_exception_is_caught_not_raised():
    tc = [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
    client = _FakeClient([
        _resp(tool_calls=tc, stop_reason="tool_use"),
        _resp(text="final", stop_reason="end_turn"),
    ])
    roles = _make_roles(client)

    def boom(args):
        raise ConnectionError("network is down")

    resp, log = roles._call_with_tools("researcher", [], [{"role": "user", "content": "go"}], "researcher",
                                       [{"type": "function"}], executors={"search": boom})
    assert resp.text == "final"                      # the loop kept going despite the tool erroring
    assert "ConnectionError" in log[0]["result_preview"]


def test_call_with_tools_exhausts_turns_and_forces_final_with_no_tools():
    tc = [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
    # 2 turns allowed; both would call tools, so the loop must force a final (tools=None) 3rd call
    client = _FakeClient([
        _resp(tool_calls=tc, stop_reason="tool_use"),
        _resp(tool_calls=tc, stop_reason="tool_use"),
        _resp(text="forced final answer", stop_reason="end_turn"),
    ])
    roles = _make_roles(client, max_tool_turns=2)
    resp, log = roles._call_with_tools("researcher", [], [{"role": "user", "content": "go"}], "researcher",
                                       [{"type": "function"}], executors={"search": lambda a: "r"})
    assert resp.text == "forced final answer"
    assert len(log) == 2                              # one tool call recorded per exhausted turn
    assert client.calls[-1]["tools"] is None            # the forced final call must not offer tools again


# ---------------------------------------------------------------------------
# SEARCH_CHECK enforcement: "I skipped search" must come with a stated reason, or it's rejected
# ---------------------------------------------------------------------------
_MINIMAL_PLAN = {"hypothesis": "h", "category": "feature", "change_spec": "c", "expected_risk": "low",
                 "builds_on": "champion", "expected_gain": 0.003}


def test_parse_researcher_checked_allows_missing_search_check_when_tools_disabled(tmp_path):
    prompts_dir = str(tmp_path / "prompts")
    os.makedirs(prompts_dir)
    roles = Roles(client=None, cfg={"llm": {}}, prompts_dir=prompts_dir, knowledge_path="")
    assert roles.research_tools_enabled is False
    plan = roles._parse_researcher_checked(json.dumps(_MINIMAL_PLAN))   # no search_check key at all
    assert isinstance(plan, ResearcherPlan)
    assert plan.search_check == ""


def test_parse_researcher_checked_rejects_missing_search_check_when_tools_enabled(tmp_path):
    prompts_dir = str(tmp_path / "prompts")
    os.makedirs(prompts_dir)
    cfg = {"llm": {}, "research_tools": {"enabled": True}}
    roles = Roles(client=None, cfg=cfg, prompts_dir=prompts_dir, knowledge_path="")
    with pytest.raises(ContractError, match="SEARCH_CHECK"):
        roles._parse_researcher_checked(json.dumps(_MINIMAL_PLAN))
    with pytest.raises(ContractError, match="SEARCH_CHECK"):
        roles._parse_researcher_checked(json.dumps({**_MINIMAL_PLAN, "search_check": "   "}))   # whitespace-only


def test_parse_researcher_checked_accepts_stated_search_check_when_tools_enabled(tmp_path):
    prompts_dir = str(tmp_path / "prompts")
    os.makedirs(prompts_dir)
    cfg = {"llm": {}, "research_tools": {"enabled": True}}
    roles = Roles(client=None, cfg=cfg, prompts_dir=prompts_dir, knowledge_path="")
    reason = ("BPR loss: this is a closed-form 2009 result with no active research direction "
             "changing its risk profile; searched arxiv for 'BPR ranking loss' and found nothing newer.")
    plan = roles._parse_researcher_checked(json.dumps({**_MINIMAL_PLAN, "search_check": reason}))
    assert plan.search_check == reason


def test_researcher_reasks_when_search_check_missing_then_accepts_the_retry():
    """End-to-end through Roles.researcher(): a first reply missing SEARCH_CHECK must trigger the
    same re-ask flow as malformed JSON, and a corrected retry must be accepted."""
    first = json.dumps(_MINIMAL_PLAN)                                    # no search_check -> rejected
    second = json.dumps({**_MINIMAL_PLAN, "search_check": "stated a specific reason here"})
    client = _FakeClient([_resp(text=first, stop_reason="end_turn"), _resp(text=second, stop_reason="end_turn")])
    roles = _make_roles(client)   # research_tools.enabled=True via _make_roles's cfg
    plan, err, raw, tool_log = roles.researcher("some briefing")
    assert err == ""
    assert plan is not None
    assert plan.search_check == "stated a specific reason here"
    assert len(client.calls) == 2   # first call (no tools) rejected -> one plain re-ask call, no tool loop on retry


def test_parse_researcher_checked_accepts_uppercase_key_variant(tmp_path):
    """Regression test for a real observed failure (2026-08-30): a live model wrote the JSON key as
    "SEARCH_CHECK" (matching the directive text's label style) instead of the exact lowercase
    "search_check" the schema originally required only exactly, causing 3/3 real iterations to fail
    for a parsing mismatch rather than any real disagreement. The directive text is now unambiguous
    about the required casing (see harness.TOOL_USE_DIRECTIVE); this is the second line of defense."""
    prompts_dir = str(tmp_path / "prompts")
    os.makedirs(prompts_dir)
    cfg = {"llm": {}, "research_tools": {"enabled": True}}
    roles = Roles(client=None, cfg=cfg, prompts_dir=prompts_dir, knowledge_path="")
    plan = roles._parse_researcher_checked(json.dumps({**_MINIMAL_PLAN, "SEARCH_CHECK": "a real reason, wrong case"}))
    assert plan.search_check == "a real reason, wrong case"
