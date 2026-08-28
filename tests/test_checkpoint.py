"""Spec §14.3 — failed / worse experiments leave best/ byte-identical; only a real improvement changes it."""
import json
import os
import re

from agent.roles import parse_file_blocks, render_file_blocks
from agent.stub_roles import default_mock_handlers
from tests.conftest import make_toy_harness, sha256_tree


def _engineer_with_theta(theta_fn):
    def engineer(role, system, messages):
        user = messages[-1]["content"]
        section = user.split("# Current champion files", 1)[-1].split("# Pipeline contract", 1)[0]
        files = parse_file_blocks(section)
        code = files["pipeline.py"]
        it = int(re.search(r"stub it(\d+)", user).group(1)) if re.search(r"stub it(\d+)", user) else 1
        new = theta_fn(it, code)
        files["pipeline.py"] = new
        return render_file_blocks(files)
    return engineer


def test_failed_and_worse_experiments_never_touch_best(tmp_path, base_cfg, mini_data):
    def theta_fn(it, code):
        if it == 1:      # worse: pure author popularity
            return re.sub(r"THETA = [0-9.]+", "THETA = 0.00", code, count=1)
        if it == 2:      # crash
            return code.replace("S = load(a.data)", "raise RuntimeError('injected crash')\n    S = load(a.data)")
        if it == 3:      # NaN scores -> rejected by the sealed checker
            return code.replace('f"{THETA * vr(x[2]) + (1 - THETA) * ar(x[3]):.6g}"', '"nan"')
        return code      # identical -> flat
    handlers = default_mock_handlers()
    handlers["engineer"] = _engineer_with_theta(theta_fn)
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 4, "N_FLAT": 99}})
    state = h.init_or_resume()
    h.phase0()
    before = sha256_tree(h.best_dir)
    assert before and "code/pipeline.py" in before
    for it in range(1, 5):
        h.run_iteration(it)
        assert sha256_tree(h.best_dir) == before, f"best/ changed after iteration {it}"
    decisions = [x["decision"] for x in state.history]
    assert decisions == ["kept_champion", "failed", "failed", "kept_champion"]
    assert state.best_iter == 0 and state.streak == 4
    # the NaN failure is attributed to the sealed checker, not to the sandbox
    assert "rejected by the sealed checker" in json.load(open(os.path.join(h.run_dir, "logs", "iter_03.json")))["result"]["error_excerpt"]


def test_improvement_promotes_and_replaces_best(tmp_path, base_cfg, mini_data):
    handlers = default_mock_handlers()
    handlers["engineer"] = _engineer_with_theta(lambda it, code: re.sub(r"THETA = [0-9.]+", "THETA = 0.55", code, count=1))
    h = make_toy_harness(tmp_path, base_cfg, mini_data, handlers=handlers, overrides={"run": {"MAX_ITERS": 1}})
    state = h.init_or_resume()
    h.phase0()
    before = sha256_tree(h.best_dir)
    h.run_iteration(1)
    after = sha256_tree(h.best_dir)
    assert state.history[-1]["decision"] == "promoted" and state.best_iter == 1
    assert after["code/pipeline.py"] != before["code/pipeline.py"]
    assert json.load(open(os.path.join(h.best_dir, "champion.json")))["iteration"] == 1
    assert "THETA = 0.55" in open(os.path.join(h.best_code_dir, "pipeline.py")).read()
