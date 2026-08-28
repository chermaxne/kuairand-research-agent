"""Spec §14.5 — kill after iteration k, resume, counters/streak/best continue. Also §13 Phase 1 gate:
5 iterations end to end with ledger/state/logs written, and resume-from-kill."""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from agent.memory import ledger_entries, load_run_state
from tests.conftest import ROOT, make_toy_harness


def test_phase1_gate_five_iterations_all_artifacts(tmp_path, base_cfg, mini_data):
    h = make_toy_harness(tmp_path, base_cfg, mini_data, overrides={"run": {"MAX_ITERS": 5, "N_FLAT": 99}})
    state = h.run()
    assert state.iteration == 5 and state.stop_reason == "iter_cap"
    assert len(ledger_entries(h.run_dir)) == 5
    for it in range(1, 6):
        p = os.path.join(h.run_dir, "logs", f"iter_{it:02d}.json")
        d = json.load(open(p))
        assert set(d) >= {"iteration", "timestamp", "hypothesis", "rationale", "category", "code_diff", "result",
                          "errors_and_recovery", "decision", "streak_after", "tokens_this_iteration", "runtime_s"}
        assert d["iteration"] == it and d["result"]["status"] in ("scored", "failed", "timeout")
        assert os.path.exists(os.path.join(h.run_dir, "logs", f"iter_{it:02d}.md"))
        assert os.path.exists(os.path.join(h.run_dir, "iterations", f"it{it:02d}", "pipeline.py"))
    for f in ("state.md", "run_state.json", "ledger.md", "interventions.md", "submission.csv", "results_summary.md", "llm_calls.jsonl"):
        assert os.path.exists(os.path.join(h.run_dir, f)), f
    assert state.finalize["ok"] and state.tokens_total > 0 and state.llm_calls >= 20


def test_resume_in_process_continues_counters(tmp_path, base_cfg, mini_data):
    h1 = make_toy_harness(tmp_path, base_cfg, mini_data, overrides={"run": {"MAX_ITERS": 5, "N_FLAT": 99}})
    s1 = h1.run(session_iteration_limit=2)                      # "killed" after iteration 2
    assert s1.iteration == 2 and s1.stop_reason is None
    tokens_after_2, streak_after_2, best_after_2 = s1.tokens_total, s1.streak, s1.best_primary
    hist2 = list(s1.history)

    h2 = make_toy_harness(tmp_path, base_cfg, mini_data, overrides={"run": {"MAX_ITERS": 5, "N_FLAT": 99}}, run_dir=h1.run_dir)
    s2 = h2.run()
    assert s2.run_id == s1.run_id and s2.start_ts == s1.start_ts
    assert s2.iteration == 5 and s2.stop_reason == "iter_cap"
    assert [x["iteration"] for x in s2.history] == [1, 2, 3, 4, 5]
    assert s2.history[:2] == hist2
    assert s2.tokens_total > tokens_after_2
    assert s2.resumes == 1 and s2.interventions == 1               # a restart is recorded honestly
    assert "resume" in open(os.path.join(h1.run_dir, "interventions.md")).read()
    assert len(ledger_entries(h1.run_dir)) == 5
    assert not s2.phase0.get("rerun")                               # phase 0 is not repeated
    # streak continuity: iteration 3's streak_after derives from iteration 2's streak
    it3 = json.load(open(os.path.join(h1.run_dir, "logs", "iter_03.json")))
    assert it3["streak_after"] in (0, streak_after_2 + 1)


@pytest.mark.slow
def test_resume_after_sigkill_subprocess(tmp_path, base_cfg, mini_data):
    """Real kill: launch the harness CLI on the toy task, SIGKILL it mid-iteration, resume, finish."""
    run_dir = str(tmp_path / "run_kill")
    champ = str(tmp_path / "champion_kill")
    from agent import toy as toymod
    toymod.write_dummy_champion(champ)
    cmd = [sys.executable, "-m", "agent.harness", "--toy", "--mock", "--run-dir", run_dir, "--max-iters", "4",
           "--set", "run.N_FLAT=99", "--set", "sandbox.extra_env.TOY_SLEEP_S=1.5",
           "--set", f"toy.data_dir={mini_data}", "--set", f"toy.loop_data={tmp_path / 'loop_kill'}", "--set", f"toy.champion_src={champ}"]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 120
    killed_after = None
    while time.time() < deadline:
        st = load_run_state(run_dir)
        if st is not None and st.iteration >= 2:
            time.sleep(0.5)                                     # now inside iteration 3 (pipeline sleeps 1.5s)
            proc.send_signal(signal.SIGKILL)
            proc.wait()
            killed_after = load_run_state(run_dir).iteration
            break
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    assert killed_after is not None and killed_after >= 2, proc.stdout.read() if proc.stdout else ""
    partial = os.path.join(run_dir, "iterations", f"it{killed_after + 1:02d}")
    # resume with the same CLI (also proves --run-dir resume path)
    cp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    st = load_run_state(run_dir)
    assert st.iteration == 4 and st.stop_reason == "iter_cap" and st.resumes == 1 and st.interventions == 1
    assert [x["iteration"] for x in st.history] == [1, 2, 3, 4]
    assert len(ledger_entries(run_dir)) == 4
    assert st.finalize["ok"]
    leftovers = [d for d in os.listdir(os.path.join(run_dir, "iterations")) if "_partial_" in d]
    assert len(leftovers) <= 1                                   # the interrupted workspace was set aside, never reused


def test_ctrl_c_exits_cleanly_with_resume_hint(tmp_path, base_cfg, mini_data, monkeypatch, capsys):
    """Ctrl-C mid-run must not traceback: it names the run dir and the resume command (exit 130)."""
    import agent.harness as H
    from agent import toy as toymod
    champ = tmp_path / "champ"
    toymod.write_dummy_champion(str(champ))
    run_dir = tmp_path / "run_int"

    def boom(self, session_iteration_limit=None):
        self.init_or_resume()
        raise KeyboardInterrupt
    monkeypatch.setattr(H.Harness, "run", boom)
    rc = H.main(["--config", os.path.join(ROOT, "config.yaml"), "--toy", "--mock", "--run-dir", str(run_dir),
                 "--set", f"toy.data_dir={mini_data}", "--set", f"toy.loop_data={tmp_path / 'loop'}", "--set", f"toy.champion_src={champ}"])
    err = capsys.readouterr().err
    assert rc == 130 and "INTERRUPTED" in err and f"--run-dir {run_dir}" in err and "Traceback" not in err
    assert load_run_state(str(run_dir)) is not None                    # the state file survived
