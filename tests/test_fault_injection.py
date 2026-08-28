"""Spec §14.6 / Phase 4 — fault injection: raising pipeline → debugger path capped at DEBUG_RETRIES;
sleeping pipeline → timeout kill; malformed researcher JSON → one re-ask then FAILED (also in
test_roles.py); BLOCKED marking; stall directive; spend guard; policy violations; sandbox isolation."""
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import pytest

from agent.llm_client import MockLLMClient
from agent.roles import parse_file_blocks, render_file_blocks
from agent.sandbox import detect_isolation, make_env, run_command, static_code_check
from agent.schemas import TokenUsage
from agent.stub_roles import default_mock_handlers
from tests.conftest import ROOT, make_toy_harness, sha256_tree


def _champion_files(user):
    return parse_file_blocks(user.split("# Current champion files", 1)[-1].split("# Pipeline contract", 1)[0])


def _engineer_transform(fn):
    def engineer(role, system, messages):
        files = _champion_files(messages[-1]["content"])
        files["pipeline.py"] = fn(files["pipeline.py"])
        return render_file_blocks(files)
    return engineer


RAISE = "S = load(a.data)"
CRASH = "raise RuntimeError('injected crash')\n    S = load(a.data)"


# ---------------------------------------------------------------- raising pipeline -> debugger, capped at 3
def test_raising_pipeline_invokes_debugger_capped_at_retries(tmp_path, base_cfg, mini_data):
    debug_calls = []

    def debugger(role, system, messages):
        debug_calls.append(messages[-1]["content"])
        files = parse_file_blocks(messages[-1]["content"].split("# Failing files", 1)[-1].split("# Error", 1)[0])
        files["pipeline.py"] = files["pipeline.py"].replace("injected crash", f"still broken {len(debug_calls)}")   # never fixes
        return f"FIX SUMMARY: attempt {len(debug_calls)}\n" + render_file_blocks(files)
    handlers = default_mock_handlers()
    handlers["engineer"] = _engineer_transform(lambda c: c.replace(RAISE, CRASH))
    handlers["debugger"] = debugger
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1, "DEBUG_RETRIES": 3}})
    st = h.init_or_resume()
    h.phase0()
    before = sha256_tree(h.best_dir)
    hist = h.run_iteration(1)
    assert len(debug_calls) == 3                                     # capped at DEBUG_RETRIES
    assert "RuntimeError: injected crash" in debug_calls[0] and "Traceback" in debug_calls[0]
    assert hist["status"] == "failed" and hist["decision"] == "failed" and st.streak == 1 and st.iteration == 1
    log = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    assert [a["attempt"] for a in log["errors_and_recovery"]] == [1, 2, 3]
    assert all(a["status_after"] == "failed" for a in log["errors_and_recovery"])
    assert "retries exhausted" in log["result"]["error_excerpt"]
    assert st.blocked and "failed after 3 debug attempts" in st.blocked[0]
    assert sha256_tree(h.best_dir) == before
    for k in (1, 2, 3):
        assert os.path.exists(os.path.join(h.run_dir, "iterations", "it01", "attempts", f"a{k}", "pipeline.py"))


def test_debugger_fix_that_works_yields_scored_iteration(tmp_path, base_cfg, mini_data):
    def debugger(role, system, messages):
        files = parse_file_blocks(messages[-1]["content"].split("# Failing files", 1)[-1].split("# Error", 1)[0])
        files["pipeline.py"] = files["pipeline.py"].replace(CRASH, RAISE)
        return "FIX SUMMARY: removed the injected raise\n" + render_file_blocks(files)
    handlers = default_mock_handlers()
    handlers["engineer"] = _engineer_transform(lambda c: c.replace(RAISE, CRASH).replace("THETA = 0.50", "THETA = 0.55"))
    handlers["debugger"] = debugger
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    st = h.init_or_resume()
    h.phase0()
    hist = h.run_iteration(1)
    assert hist["status"] == "scored" and hist["decision"] == "promoted" and st.best_iter == 1
    log = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    assert len(log["errors_and_recovery"]) == 1 and log["errors_and_recovery"][0]["status_after"] == "scored"
    assert log["errors_and_recovery"][0]["fix_summary"] == "removed the injected raise"
    assert not st.blocked


def test_debugger_abandon_marks_blocked(tmp_path, base_cfg, mini_data):
    handlers = default_mock_handlers()
    handlers["engineer"] = _engineer_transform(lambda c: c.replace(RAISE, CRASH))
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    st = h.init_or_resume()
    h.phase0()
    h.run_iteration(1)
    assert len(st.blocked) == 1 and "abandoned by debugger" in st.blocked[0]
    assert "BLOCKED: it01:" in open(os.path.join(h.run_dir, "state.md")).read()
    log = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    assert log["errors_and_recovery"][0]["status_after"] == "abandoned"


# ---------------------------------------------------------------- sleeping pipeline -> timeout kill
def test_sleeping_pipeline_is_killed_on_timeout(tmp_path, base_cfg, mini_data):
    handlers = default_mock_handlers()
    handlers["engineer"] = _engineer_transform(lambda c: c.replace(RAISE, "import time; time.sleep(60)\n    " + RAISE))
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers,
                         overrides={"run": {"MAX_ITERS": 1, "EXPERIMENT_TIMEOUT_S": 2, "retry_timeouts_with_debugger": False}})
    st = h.init_or_resume()
    h.phase0()
    t0 = time.time()
    hist = h.run_iteration(1)
    assert time.time() - t0 < 20
    assert hist["status"] == "timeout" and hist["decision"] == "failed" and st.streak == 1
    log = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    assert log["result"]["status"] == "timeout" and "TIMEOUT" in log["result"]["error_excerpt"]
    assert "RUNTIME DIAGNOSIS" in log["result"]["error_excerpt"]      # the diagnosis is recorded even when terminal
    assert log["errors_and_recovery"] == []                          # terminal when the retry is disabled
    assert st.blocked and "timeout 2s" in st.blocked[0]
    line = open(os.path.join(h.run_dir, "ledger.md")).read().splitlines()[-1]
    assert "RESULT: FAILED(timeout" in line
    # no orphaned sandbox process keeps running
    ps = subprocess.run(["pgrep", "-f", f"{h.run_dir}/iterations/it01"], capture_output=True, text=True)
    assert ps.stdout.strip() == ""


def test_timeout_retry_is_the_default_and_debugger_sees_the_diagnosis(base_cfg):
    assert base_cfg["run"]["retry_timeouts_with_debugger"] is True


def test_timeout_retry_with_debugger_when_configured(tmp_path, base_cfg, mini_data):
    calls = []

    def debugger(role, system, messages):
        calls.append(1)
        assert "RUNTIME DIAGNOSIS" in messages[-1]["content"] and "Vectorise" in messages[-1]["content"]
        files = parse_file_blocks(messages[-1]["content"].split("# Failing files", 1)[-1].split("# Error", 1)[0])
        files["pipeline.py"] = files["pipeline.py"].replace("time.sleep(60)", "time.sleep(0)")
        return "FIX SUMMARY: removed the sleep\n" + render_file_blocks(files)
    handlers = default_mock_handlers()
    handlers["engineer"] = _engineer_transform(lambda c: c.replace(RAISE, "import time; time.sleep(60)\n    " + RAISE))
    handlers["debugger"] = debugger
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers,
                         overrides={"run": {"MAX_ITERS": 1, "EXPERIMENT_TIMEOUT_S": 2, "retry_timeouts_with_debugger": True}})
    h.init_or_resume()
    h.phase0()
    hist = h.run_iteration(1)
    assert calls == [1] and hist["status"] == "scored"


# ---------------------------------------------------------------- stall directive + consecutive failures
def test_stall_directive_injected_after_three_failures(tmp_path, base_cfg, mini_data):
    briefings = []

    def researcher(role, system, messages):
        briefings.append(messages[-1]["content"])
        return default_mock_handlers()["researcher"](role, system, messages)
    handlers = default_mock_handlers()
    handlers["researcher"] = researcher
    handlers["engineer"] = _engineer_transform(lambda c: c.replace(RAISE, CRASH))
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 4, "N_FLAT": 99, "STALL_FAILURES": 3}})
    st = h.init_or_resume()
    h.phase0()
    for it in range(1, 5):
        h.run_iteration(it)
    assert st.consecutive_failures == 4
    assert "STALL RECOVERY DIRECTIVE" not in briefings[2]           # before the 3rd failure completed
    assert "STALL RECOVERY DIRECTIVE" in briefings[3]               # injected into the 4th briefing
    assert "3 iterations ALL failed" in briefings[3]


def test_consecutive_failures_reset_on_score(tmp_path, base_cfg, mini_data):
    state = {"n": 0}

    def engineer(role, system, messages):
        state["n"] += 1
        files = _champion_files(messages[-1]["content"])
        if state["n"] <= 3:
            files["pipeline.py"] = files["pipeline.py"].replace(RAISE, CRASH)
        return render_file_blocks(files)
    handlers = default_mock_handlers()
    handlers["engineer"] = engineer
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 4, "N_FLAT": 99}})
    st = h.init_or_resume()
    h.phase0()
    for it in range(1, 5):
        h.run_iteration(it)
    assert st.consecutive_failures == 0 and st.streak == 4          # streak still ticks (flat), stall counter reset


# ---------------------------------------------------------------- spend guard
def test_spend_guard_stops_run_and_finalizes(tmp_path, base_cfg, mini_data):
    class ExpensiveMock(MockLLMClient):
        def complete(self, **kw):
            r = super().complete(**kw)
            r.usage = TokenUsage(input_tokens=50_000, output_tokens=1_000)
            return r
    h = make_toy_harness(tmp_path, base_cfg, mini_data, overrides={"run": {"MAX_ITERS": 50, "N_FLAT": 99}, "llm": {"max_total_tokens": 300_000}})
    h.roles.client = ExpensiveMock(default_mock_handlers())
    st = h.run()
    assert st.stop_reason == "spend_guard" and st.iteration == 2 and st.finalize["ok"]
    assert st.tokens_total >= 300_000


# ---------------------------------------------------------------- policy violations (static guard)
def test_policy_violation_never_executes_and_goes_to_debugger(tmp_path, base_cfg, mini_data):
    seen = []

    def debugger(role, system, messages):
        seen.append(messages[-1]["content"])
        return json.dumps({"action": "abandon", "reason": "policy"})
    handlers = default_mock_handlers()
    handlers["engineer"] = _engineer_transform(lambda c: "import subprocess\n" + c)
    handlers["debugger"] = debugger
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    h.init_or_resume()
    h.phase0()
    hist = h.run_iteration(1)
    assert hist["status"] == "failed" and "policy violation" in seen[0] and "forbidden import 'subprocess'" in seen[0]
    assert not os.path.exists(os.path.join(h.run_dir, "iterations", "it01", "stdout.txt"))   # code was never run


def test_static_code_check_rules(base_cfg):
    sb = base_cfg["sandbox"]
    assert static_code_check({"pipeline.py": "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\n"}, sb) == []
    bad = static_code_check({"pipeline.py": "import os\nos.system('pip install x')\nfrom urllib.request import urlopen\n"}, sb)
    assert any("urllib" in b for b in bad) and any("pip install" in b for b in bad)
    assert static_code_check({"notes.md": "import subprocess"}, sb) == []                  # only .py files are checked


# ---------------------------------------------------------------- sandbox isolation
def test_sandbox_env_strips_secrets(monkeypatch, base_cfg):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("MY_TOKEN", "t")
    env = make_env(base_cfg["sandbox"], pythonpath=["/x"])
    assert "ANTHROPIC_API_KEY" not in env and "MY_TOKEN" not in env and env["PYTHONPATH"] == "/x"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.skipif(detect_isolation("auto") != "sandbox-exec", reason="OS sandbox only on macOS with sandbox-exec")
def test_sandbox_exec_blocks_network_and_outside_writes(tmp_path, base_cfg):
    ws = tmp_path / "ws"
    ws.mkdir()
    # pytest's tmp_path lives under the OS temp tree, which the profile allows; probe a repo path instead
    outside = pathlib.Path(ROOT) / "runs" / f".probe_outside_{os.getpid()}.txt"
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / "s.txt").write_text("hidden")
    (ws / "probe.py").write_text(f"""
import socket, json
r = {{}}
try:
    socket.create_connection(("1.1.1.1", 80), timeout=3); r["net"] = "open"
except Exception as e:
    r["net"] = type(e).__name__
try:
    open({str(outside)!r}, "w").write("x"); r["outside_write"] = "ok"
except Exception as e:
    r["outside_write"] = type(e).__name__
try:
    open({str(secret_dir / 's.txt')!r}).read(); r["denied_read"] = "ok"
except Exception as e:
    r["denied_read"] = type(e).__name__
open("inside.txt", "w").write("x"); r["inside_write"] = "ok"
print(json.dumps(r))
""")
    res = run_command([sys.executable, "probe.py"], str(ws), 30, make_env(base_cfg["sandbox"]), isolation="sandbox-exec",
                      deny_read=[str(secret_dir)])
    assert res.ok, res.stderr_tail
    r = json.loads(res.stdout_tail.strip().splitlines()[-1])
    assert r == {"net": "PermissionError", "outside_write": "PermissionError", "denied_read": "PermissionError", "inside_write": "ok"}
    try:
        assert not outside.exists()
    finally:
        if outside.exists():
            outside.unlink()


# ---------------------------------------------------------------- .env handling
def test_load_dotenv_parses_and_never_overrides(tmp_path, monkeypatch):
    from agent.llm_client import load_dotenv
    f = tmp_path / ".env"
    f.write_text("# comment\nPOE_API_KEY='poe-secret'\nexport OTHER=\"x y\"\nBAD LINE\nEXISTING=new\n\n")
    monkeypatch.delenv("POE_API_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.setenv("EXISTING", "old")
    assert load_dotenv(str(f)) == ["POE_API_KEY", "OTHER"]
    assert os.environ["POE_API_KEY"] == "poe-secret" and os.environ["OTHER"] == "x y" and os.environ["EXISTING"] == "old"
    assert load_dotenv(str(tmp_path / "missing.env")) == []


def test_env_file_is_denied_to_the_sandbox(tmp_path, base_cfg, mini_data, monkeypatch):
    """The key file must be invisible to LLM-written pipelines (macOS sandbox) and stripped from their env."""
    secret = tmp_path / ".env"
    secret.write_text("POE_API_KEY=poe-secret\n")
    monkeypatch.setenv("HARNESS_ENV_FILE", str(secret))
    monkeypatch.setenv("POE_API_KEY", "poe-secret")
    h = make_toy_harness(tmp_path, base_cfg, mini_data)
    h.init_or_resume()
    assert str(secret) in h.task.secret_files()
    ws = tmp_path / "probe_ws"
    ws.mkdir()
    (ws / "pipeline.py").write_text(f"""
import os, json
r = {{"env_has_key": "POE_API_KEY" in os.environ}}
try:
    open({str(secret)!r}).read(); r["read"] = "ok"
except Exception as e:
    r["read"] = type(e).__name__
print(json.dumps(r))
""")
    res = h.task.sandbox_run(str(ws), "val", "preds_val.csv", 30)
    assert res.ok, res.stderr_tail
    r = json.loads(res.stdout_tail.strip().splitlines()[-1])
    assert r["env_has_key"] is False
    if detect_isolation("auto") == "sandbox-exec":
        assert r["read"] == "PermissionError"


def test_load_dotenv_strips_inline_comments(tmp_path, monkeypatch):
    from agent.llm_client import load_dotenv
    f = tmp_path / ".env"
    f.write_text('KEY_A=abc123            # trailing comment\nKEY_B="quoted # not a comment"\nKEY_C=x#y\n')
    for k in ("KEY_A", "KEY_B", "KEY_C"):
        monkeypatch.delenv(k, raising=False)
    load_dotenv(str(f))
    assert os.environ["KEY_A"] == "abc123" and os.environ["KEY_B"] == "quoted # not a comment" and os.environ["KEY_C"] == "x#y"


def test_load_dotenv_ignores_empty_placeholders(tmp_path, monkeypatch):
    from agent.llm_client import load_dotenv
    f = tmp_path / ".env"
    f.write_text("OPENROUTER_API_KEY=\nPOE_API_KEY=real\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("POE_API_KEY", raising=False)
    assert load_dotenv(str(f)) == ["POE_API_KEY"]          # the empty slot is not reported as loaded
    assert "OPENROUTER_API_KEY" not in os.environ


# ---------------------------------------------------------------- implausible-score guard (inverted ranking)
def _invert_engineer(role, system, messages):
    files = _champion_files(messages[-1]["content"])
    files["pipeline.py"] = files["pipeline.py"].replace('f"{THETA * vr(x[2]) + (1 - THETA) * ar(x[3]):.6g}"',
                                                        'f"{-(THETA * vr(x[2]) + (1 - THETA) * ar(x[3])):.6g}"')   # negated scores
    return render_file_blocks(files)


def test_inverted_ranking_goes_to_debugger_once_and_fix_is_used(tmp_path, base_cfg, mini_data):
    seen = []

    def debugger(role, system, messages):
        seen.append(messages[-1]["content"])
        files = parse_file_blocks(messages[-1]["content"].split("# Failing files", 1)[-1].split("# Error", 1)[0])
        files["pipeline.py"] = files["pipeline.py"].replace('f"{-(THETA', 'f"{(THETA')
        return "FIX SUMMARY: removed the stray negation\n" + render_file_blocks(files)
    handlers = default_mock_handlers()
    handlers["engineer"] = _invert_engineer
    handlers["debugger"] = debugger
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1, "implausible_gauc_below": 0.5}})
    st = h.init_or_resume()
    h.phase0()
    hist = h.run_iteration(1)
    assert len(seen) == 1 and "IMPLAUSIBLE RESULT" in seen[0] and "inverted" in seen[0]
    assert hist["status"] == "scored" and hist["gauc"] > 0.5
    log = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    assert log["errors_and_recovery"][0]["fix_summary"].startswith("IMPLAUSIBLE -> removed the stray negation")
    assert log["errors_and_recovery"][0]["status_after"] == "scored"


def test_inverted_ranking_kept_when_debugger_abandons(tmp_path, base_cfg, mini_data):
    handlers = default_mock_handlers()
    handlers["engineer"] = _invert_engineer
    handlers["debugger"] = lambda r, s, m: json.dumps({"action": "abandon", "reason": "cannot see it"})
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1, "implausible_gauc_below": 0.5}})
    st = h.init_or_resume()
    h.phase0()
    hist = h.run_iteration(1)
    assert hist["status"] == "scored" and hist["gauc"] < 0.5 and hist["decision"] == "kept_champion" and st.streak == 1
    log = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    assert log["errors_and_recovery"][0]["status_after"] == "scored (implausible)"


def test_guard_disabled_when_config_is_null(tmp_path, base_cfg, mini_data):
    calls = []
    handlers = default_mock_handlers()
    handlers["engineer"] = _invert_engineer
    handlers["debugger"] = lambda r, s, m: (calls.append(1), json.dumps({"action": "abandon", "reason": "x"}))[1]
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1, "implausible_gauc_below": None}})
    h.init_or_resume()
    h.phase0()
    hist = h.run_iteration(1)
    assert calls == [] and hist["status"] == "scored" and hist["gauc"] < 0.5


# ---------------------------------------------------------------- leak test (flipped validation labels gate every promotion)
def test_flipped_labels_dir_flips_only_post_train_feedback(tmp_path, mini_data):
    from agent import tools
    kit = tools.import_kit(os.path.join(ROOT, "starter_kit"))
    loop = str(tmp_path / "loop")
    tools.ensure_loop_data_dir(mini_data, loop, int(kit.data.SPLITS["valid"][1]))
    flipped = str(tmp_path / "flipped")
    info = tools.ensure_flipped_labels_dir(loop, flipped, int(kit.data.SPLITS["train"][1]))
    assert info["rebuilt"]
    import csv
    src = list(csv.DictReader(open(os.path.join(loop, "log_standard_4_22_to_5_08_pure.csv"))))
    dst = list(csv.DictReader(open(os.path.join(flipped, "log_standard_4_22_to_5_08_pure.csv"))))
    flipped_users = set(info["flipped_users"])
    assert len(src) == len(dst) and 0 < len(flipped_users) < len({a["user_id"] for a in src})
    for a, b in zip(src, dst):
        assert a["user_id"] == b["user_id"] and a["video_id"] == b["video_id"] and a["time_ms"] == b["time_ms"] and a["tab"] == b["tab"]
        if a["user_id"] in flipped_users:
            assert a["long_view"] != b["long_view"] and b["play_time_ms"] == "0"                           # flipped users: inverted / zeroed
        else:
            assert a["long_view"] == b["long_view"] and a["play_time_ms"] == b["play_time_ms"]              # everyone else untouched
    assert json.load(open(os.path.join(flipped, "flipped_users.json"))) == sorted(flipped_users)
    tr_src = open(os.path.join(loop, "log_standard_4_08_to_4_21_pure.csv")).read()
    assert open(os.path.join(flipped, "log_standard_4_08_to_4_21_pure.csv")).read() == tr_src                # train untouched
    assert not tools.ensure_flipped_labels_dir(loop, flipped, int(kit.data.SPLITS["train"][1]))["rebuilt"]   # cached


def test_label_leak_is_caught_and_never_promoted(tmp_path, base_cfg, mini_data):
    """An Engineer that scores validation rows with their own label hits the oracle; the flipped-label re-run
    inverts it, so the harness records a LEAK and keeps the champion."""
    def leaky(role, system, messages):
        files = _champion_files(messages[-1]["content"])
        code = files["pipeline.py"]
        code = code.replace('1 if r["long_view"] != "0" else 0))', '1 if r["long_view"] != "0" else 0))')  # keep label in the tuple
        code = code.replace('f"{THETA * vr(x[2]) + (1 - THETA) * ar(x[3]):.6g}"', 'f"{x[4] + 0.001 * vr(x[2]):.6g}"')   # score = own label
        files["pipeline.py"] = code
        return render_file_blocks(files)
    handlers = default_mock_handlers()
    handlers["engineer"] = leaky
    # ceiling disabled here so the expensive flipped-label path is the thing under test
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers,
                         overrides={"run": {"MAX_ITERS": 1, "implausible_primary_above": None}})
    st = h.init_or_resume()
    h.phase0()
    before = sha256_tree(h.best_dir)
    hist = h.run_iteration(1)
    assert hist["status"] == "failed" and hist["decision"] == "failed" and st.best_iter == 0 and st.streak == 1
    assert sha256_tree(h.best_dir) == before
    log = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    assert "LEAK DETECTED" in log["result"]["error_excerpt"] and log["result"]["primary"] > 0.95   # the leaked score is on record
    assert log["harness_extra"]["leak_test"]["verdict"] == "LEAK" and log["harness_extra"]["leak_test"]["subset_primary"] < 0.5
    assert any("leak detected" in b for b in st.blocked)
    line = open(os.path.join(h.run_dir, "ledger.md")).read().splitlines()[-1]
    assert "FAILED(LEAK DETECTED" in line


def test_legitimate_improvement_passes_the_leak_test(tmp_path, base_cfg, mini_data):
    from agent.stub_roles import default_mock_handlers as dmh
    handlers = dmh()
    handlers["engineer"] = _engineer_transform(lambda c: c.replace("THETA = 0.50", "THETA = 0.55"))
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    st = h.init_or_resume()
    h.phase0()
    hist = h.run_iteration(1)
    assert hist["decision"] == "promoted" and st.best_iter == 1
    log = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    assert log["harness_extra"]["leak_test"]["verdict"] == "clean" and log["harness_extra"]["leak_test"]["subset_primary"] >= 0.5
    assert os.path.exists(os.path.join(h.run_dir, "iterations", "it01", "leak_test.json"))


def test_strict_validation_assertion_does_not_cause_a_false_leak(tmp_path, base_cfg, mini_data):
    """A legitimate pipeline that hard-asserts on its validation labels crashes on the 10% copy; the 2% retry must
    still verify it and the promotion must go through (the 2026-08-29 false positive)."""
    def engineer(role, system, messages):
        files = _champion_files(messages[-1]["content"])
        code = files["pipeline.py"].replace("THETA = 0.50", "THETA = 0.55")
        code = code.replace('    with open(a.out, "w", newline="") as fh:',
                            "    _rate = sum(x[4] for x in S[split]) / max(1, len(S[split]))\n"
                            "    _ref = sum(x[4] for x in S['train']) / max(1, len(S['train']))\n"
                            "    assert abs(_rate - _ref) < 0.03, f'label rate looks corrupted: {_rate:.3f} vs {_ref:.3f}'\n"
                            '    with open(a.out, "w", newline="") as fh:')
        files["pipeline.py"] = code
        return render_file_blocks(files)
    handlers = default_mock_handlers()
    handlers["engineer"] = engineer
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    st = h.init_or_resume()
    h.phase0()
    hist = h.run_iteration(1)
    log = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    lt = log["harness_extra"]["leak_test"]
    assert lt["verdict"] == "clean", lt
    assert hist["decision"] == "promoted" and st.best_iter == 1


def test_crash_on_both_flipped_copies_is_inconclusive_and_not_promoted(tmp_path, base_cfg, mini_data):
    def engineer(role, system, messages):
        files = _champion_files(messages[-1]["content"])
        code = files["pipeline.py"].replace("THETA = 0.50", "THETA = 0.55")
        code = code.replace("    S = load(a.data)", "    import os as _o\n    assert not _o.path.exists(_o.path.join(a.data, 'flipped_users.json')), 'refusing corrupted data'\n    S = load(a.data)")
        files["pipeline.py"] = code
        return render_file_blocks(files)
    handlers = default_mock_handlers()
    handlers["engineer"] = engineer
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    st = h.init_or_resume()
    h.phase0()
    hist = h.run_iteration(1)
    assert hist["status"] == "failed" and st.best_iter == 0
    log = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    assert log["harness_extra"]["leak_test"]["verdict"].startswith("INCONCLUSIVE") and "INCONCLUSIVE" in log["result"]["error_excerpt"]
    assert any("inconclusive" in b for b in st.blocked)


def test_leak_check_default_and_off_switch(tmp_path, base_cfg, mini_data):
    assert base_cfg["run"]["leak_check"] == "on_promotion" and base_cfg["run"]["leak_check_min_primary"] == 0.5
    handlers = default_mock_handlers()
    handlers["engineer"] = _engineer_transform(lambda c: c.replace("THETA = 0.50", "THETA = 0.55"))
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1, "leak_check": "off"}})
    h.init_or_resume()
    h.phase0()
    h.run_iteration(1)
    assert not os.path.exists(os.path.join(h.run_dir, "iterations", "it01", "leak_test.json"))


def test_implausible_score_is_flagged_without_a_rerun(tmp_path, base_cfg, mini_data):
    """The oracle-style leak (score far above anything honest) is caught for free, before the expensive re-run."""
    def oracle(role, system, messages):
        files = _champion_files(messages[-1]["content"])
        files["pipeline.py"] = files["pipeline.py"].replace(
            'f"{THETA * vr(x[2]) + (1 - THETA) * ar(x[3]):.6g}"', 'f"{x[4] + 0.001 * vr(x[2]):.6g}"')   # score = own label
        return render_file_blocks(files)
    handlers = default_mock_handlers()
    handlers["engineer"] = oracle
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers,
                         overrides={"run": {"MAX_ITERS": 1, "implausible_primary_above": 0.70}})
    st = h.init_or_resume()
    h.phase0()
    before = sha256_tree(h.best_dir)
    t0 = time.time()
    hist = h.run_iteration(1)
    assert hist["status"] == "failed" and st.best_iter == 0 and sha256_tree(h.best_dir) == before
    log = json.load(open(os.path.join(h.run_dir, "logs", "iter_01.json")))
    lt = log["harness_extra"]["leak_test"]
    assert lt["verdict"] == "LEAK" and lt["reason"] == "implausible score ceiling" and lt["ran"] is False   # no re-run spent
    assert "implausible score" in log["result"]["error_excerpt"] and "0.8484" in log["result"]["error_excerpt"]
    assert any("implausible" in b for b in st.blocked)
