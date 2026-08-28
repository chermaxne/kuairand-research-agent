"""Phase 2 — sealed evaluation, Phase 0 self-checks, submission round-trip (§14.7)."""
import hashlib
import json
import os
import re
import subprocess
import sys

import pytest

from agent import tools
from agent.harness import FinalizeError
from agent.phase0 import Phase0Error
from agent.roles import parse_file_blocks, render_file_blocks
from agent.stub_roles import default_mock_handlers
from agent.toy import DUMMY_PIPELINE
from tests.conftest import HAVE_REAL_DATA, ROOT, make_toy_harness

SEALED = os.path.join(ROOT, "sealed")
KIT = os.path.join(ROOT, "starter_kit")


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_sealed_evaluate_is_verbatim_copy_of_kit():
    assert _sha(os.path.join(SEALED, "evaluate.py")) == _sha(os.path.join(KIT, "evaluate.py"))
    assert _sha(os.path.join(KIT, "evaluate.py")) == "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de"


def test_sealed_evaluate_matches_known_values():
    ev = tools.import_sealed_evaluate(SEALED)
    r = ev(["u1", "u1", "u1", "u2", "u2"], [1, 0, 0, 1, 1], [0.9, 0.8, 0.1, 0.5, 0.4])
    assert r["GAUC"] == 1.0 and r["nDCG@5"] == 1.0 and r["primary"] == 1.0
    r = ev(["u1", "u1", "u1"], [0, 1, 0], [0.9, 0.1, 0.5])   # positive ranked last
    assert r["GAUC"] == 0.0 and r["nDCG@5"] == pytest.approx(1 / 2.0)   # 1/log2(4)


def test_agent_code_never_reimplements_or_imports_another_evaluate():
    for f in [f for f in os.listdir(os.path.join(ROOT, "agent")) if f.endswith(".py")]:
        src = open(os.path.join(ROOT, "agent", f)).read()
        assert "def ndcg" not in src and "def auc(" not in src and "Mann-Whitney" not in src, f
        assert "from evaluate import" not in src and "import evaluate" not in src, f


def test_evaluate_preds_rejects_misaligned_and_nan(tmp_path, mini_data):
    kit = tools.import_kit(KIT)
    ev = tools.import_sealed_evaluate(SEALED)
    rows = kit.data.load(mini_data)["valid"]
    good = str(tmp_path / "good.csv")
    tools.write_preds(good, rows, tools.random_scores(len(rows)))
    s = tools.evaluate_preds(good, rows, ev, kit)
    assert 0 < s.primary < 1 and s.rows == len(rows)
    bad = str(tmp_path / "nan.csv")
    lines = open(good).read().splitlines()
    lines[3] = lines[3].rsplit(",", 1)[0] + ",nan"
    open(bad, "w").write("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="NaN"):
        tools.evaluate_preds(bad, rows, ev, kit)
    short = str(tmp_path / "short.csv")
    open(short, "w").write("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError):
        tools.evaluate_preds(short, rows, ev, kit)
    swapped = str(tmp_path / "swapped.csv")
    lines2 = open(good).read().splitlines()
    lines2[1], lines2[2] = lines2[2].replace(",", ",", 1), lines2[1]
    open(swapped, "w").write("\n".join(lines2) + "\n")
    with pytest.raises(ValueError):
        tools.evaluate_preds(swapped, rows, ev, kit)


def test_sealed_checker_round_trip_and_nan_rejection(tmp_path, mini_data):
    kit = tools.import_kit(KIT)
    rows = kit.data.load(mini_data)["test"]
    sub = str(tmp_path / "submission.csv")
    tools.write_preds(sub, rows, tools.random_scores(len(rows)))
    ok, out = tools.check_submission(SEALED, KIT, mini_data, sub, "test")
    assert ok, out
    lines = open(sub).read().splitlines()
    lines[5] = lines[5].rsplit(",", 1)[0] + ",inf"
    open(sub, "w").write("\n".join(lines) + "\n")
    ok, out = tools.check_submission(SEALED, KIT, mini_data, sub, "test")
    assert not ok and "NaN/Inf" in out


def test_loop_data_dir_masks_test_period(tmp_path, mini_data):
    kit = tools.import_kit(KIT)
    loop = str(tmp_path / "loop")
    info = tools.ensure_loop_data_dir(mini_data, loop, int(kit.data.SPLITS["valid"][1]))
    assert info["rebuilt"]
    s_full, s_loop = kit.data.load(mini_data), kit.data.load(loop)
    assert s_loop["test"] == [] and s_full["test"]
    assert s_loop["valid"] == s_full["valid"] and s_loop["train"] == s_full["train"]      # row order/ids identical
    assert not tools.ensure_loop_data_dir(mini_data, loop, int(kit.data.SPLITS["valid"][1]))["rebuilt"]   # cached


def test_phase0_toy_installs_champion_and_records_rungs(tmp_path, base_cfg, mini_data):
    h = make_toy_harness(tmp_path, base_cfg, mini_data)
    st = h.init_or_resume()
    h.phase0()
    assert st.phase0["passed"] and st.best_iter == 0 and st.best_primary == pytest.approx(st.phase0["champion"]["primary"])
    assert 0.3 < st.phase0["random"]["primary"] < 0.6 and st.phase0["pop"]["primary"] > st.phase0["random"]["primary"]
    assert os.path.exists(os.path.join(h.best_code_dir, "pipeline.py")) and os.path.exists(os.path.join(h.best_dir, "preds_val.csv"))
    assert json.load(open(os.path.join(h.best_dir, "champion.json")))["iteration"] == 0
    assert "# it00 champion installed" in open(os.path.join(h.run_dir, "ledger.md")).read()


def test_phase0_aborts_when_champion_crashes(tmp_path, base_cfg, mini_data):
    champ = tmp_path / "champion"
    champ.mkdir()
    (champ / "pipeline.py").write_text("import sys\nsys.exit(3)\n")
    h = make_toy_harness(tmp_path, base_cfg, mini_data)
    h.init_or_resume()
    with pytest.raises(Phase0Error):
        h.phase0()


def test_phase0_aborts_when_rung_assertion_fails(tmp_path, base_cfg, mini_data):
    h = make_toy_harness(tmp_path, base_cfg, mini_data)
    h.task.assert_rungs = True
    h.task.expected = {"random": 0.99, "pop": 0.99, "fm": 0.99}
    h.init_or_resume()
    with pytest.raises(Phase0Error, match="PHASE 0 FAILED"):
        h.phase0()
    assert os.path.exists(os.path.join(h.run_dir, "PHASE0_FAILED.md"))


def test_finalize_rejects_nan_submission_before_completing(tmp_path, base_cfg, mini_data):
    """NaN injection in the champion's TEST predictions -> sealed checker rejects -> FinalizeError, no submission.csv."""
    champ = tmp_path / "champion"
    champ.mkdir()
    poisoned = DUMMY_PIPELINE.replace('f"{THETA * vr(x[2]) + (1 - THETA) * ar(x[3]):.6g}"',
                                      '("nan" if split == "test" else f"{THETA * vr(x[2]) + (1 - THETA) * ar(x[3]):.6g}")')
    (champ / "pipeline.py").write_text(poisoned)
    h = make_toy_harness(tmp_path, base_cfg, mini_data, overrides={"run": {"MAX_ITERS": 0}})
    with pytest.raises(FinalizeError):
        h.run()
    st = h.state
    assert not st.finalize["ok"] and st.finalize["attempts"][0]["checker_ok"] is False
    assert not os.path.exists(os.path.join(h.run_dir, "submission.csv"))
    assert os.path.exists(os.path.join(h.run_dir, "results_summary.md"))


def test_finalize_falls_back_to_previous_champion(tmp_path, base_cfg, mini_data):
    """A promoted champion that is NaN on the test split is rejected; finalize falls back to it00."""
    def engineer(role, system, messages):
        user = messages[-1]["content"]
        files = parse_file_blocks(user.split("# Current champion files", 1)[-1].split("# Pipeline contract", 1)[0])
        code = re.sub(r"THETA = [0-9.]+", "THETA = 0.55", files["pipeline.py"], count=1)
        code = code.replace('f"{THETA * vr(x[2]) + (1 - THETA) * ar(x[3]):.6g}"',
                            '("nan" if split == "test" else f"{THETA * vr(x[2]) + (1 - THETA) * ar(x[3]):.6g}")')
        files["pipeline.py"] = code
        return render_file_blocks(files)
    handlers = default_mock_handlers()
    handlers["engineer"] = engineer
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    st = h.run()
    assert st.best_iter == 1 and st.finalize["ok"] and st.finalize["champion_iteration"] == 0
    assert [a["iteration"] for a in st.finalize["attempts"]] == [1, 0]
    assert os.path.exists(os.path.join(h.run_dir, "submission.csv"))


@pytest.mark.realdata
@pytest.mark.slow
@pytest.mark.skipif(not HAVE_REAL_DATA, reason="real KuaiRand-Pure data not present")
def test_phase0_passes_on_real_data(tmp_path, base_cfg):
    """Spec §13 Phase 2 gate: Phase 0 (rungs + official baseline + champion) passes on the real starter kit."""
    from agent.harness import build
    run_dir = str(tmp_path / "run_real")
    h = build(base_cfg, ROOT, run_dir, toy=False, mock=True, log=lambda *_: None)
    st = h.init_or_resume()
    h.phase0()
    p0 = st.phase0
    assert p0["passed"]
    assert abs(p0["random"]["primary"] - h.task.expected["random"]) <= 0.01
    assert abs(p0["pop"]["primary"] - h.task.expected["pop"]) <= 0.01
    assert abs(p0["official_fm"]["primary"] - 0.6016) <= 0.005
    assert abs(p0["champion"]["primary"] - 0.6016) <= 0.005
    assert abs(p0["champion_vs_official"]) < 1e-6          # bit-for-bit reproduction of the official recipe
    assert st.best_iter == 0 and st.best_primary == pytest.approx(p0["champion"]["primary"])
