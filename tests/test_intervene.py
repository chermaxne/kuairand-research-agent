"""Spec §11 — manual intervention CLI: appends a row, bumps the counter, can BLOCK a direction that then
appears in the next briefing's state block; resumes are auto-recorded (see test_resume.py)."""
import json
import os

from agent.intervene import main as intervene_main
from agent.memory import load_run_state
from agent.stub_roles import default_mock_handlers
from tests.conftest import make_toy_harness


def test_intervene_cli_records_and_blocks(tmp_path, base_cfg, mini_data):
    briefings = []

    def researcher(role, system, messages):
        briefings.append(messages[-1]["content"])
        return default_mock_handlers()["researcher"](role, system, messages)
    handlers = default_mock_handlers()
    handlers["researcher"] = researcher
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 2, "N_FLAT": 99}})
    st = h.run(session_iteration_limit=1)
    assert st.interventions == 0
    rc = intervene_main(["restarted with a bigger timeout", "--stuck", "iteration 1 hung", "--scope", "config",
                         "--run-dir", h.run_dir, "--block", "torch models (too slow on this box)"])
    assert rc == 0
    st2 = load_run_state(h.run_dir)
    assert st2.interventions == 1 and st2.blocked == ["manual: torch models (too slow on this box)"]
    text = open(os.path.join(h.run_dir, "interventions.md")).read()
    assert "Count: 1" in text and "| iteration 1 hung | restarted with a bigger timeout | config |" in text
    # the next briefing (after a resume, which is itself recorded) shows the manual block
    h2 = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 2, "N_FLAT": 99}}, run_dir=h.run_dir)
    st3 = h2.run()
    assert st3.interventions == 2 and st3.resumes == 1
    assert "BLOCKED: manual: torch models (too slow on this box)" in briefings[-1]
    summary = open(os.path.join(h.run_dir, "results_summary.md")).read()
    assert "manual interventions: 2 (resumes 1)" in summary


def test_tokens_by_role_survive_resume(tmp_path, base_cfg, mini_data):
    h = make_toy_harness(tmp_path, base_cfg, mini_data, overrides={"run": {"MAX_ITERS": 3, "N_FLAT": 99}})
    s1 = h.run(session_iteration_limit=2)
    by_role_after_2 = dict(s1.tokens_by_role)
    h2 = make_toy_harness(tmp_path, base_cfg, mini_data, overrides={"run": {"MAX_ITERS": 3, "N_FLAT": 99}}, run_dir=h.run_dir)
    s2 = h2.run()
    assert all(s2.tokens_by_role[r] > by_role_after_2[r] for r in by_role_after_2)
    assert sum(s2.tokens_by_role.values()) == s2.tokens_total
