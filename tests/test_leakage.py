"""Regression test for the data-isolation guarantee audited by hand earlier: the full data directory
(which contains hidden-test rows) must never reach the sandboxed subprocess during the iteration
loop, on any code path, regardless of which OS-level sandbox (if any) is active. This locks in the
answer to "does Task.sandbox_run ever pass the full dir?" so a future refactor of task.py can't
silently flip the default without a test failing.

Complements tests/test_sealed.py::test_loop_data_dir_masks_test_period, which checks the CONTENT of
the masked copy (dates filtered); this file checks WHICH PATH gets handed to the subprocess call.
"""
import os

from agent import tools
from agent.task import Task
from tests.conftest import ROOT


def _make_task(tmp_path, base_cfg, mini_data, mask_test=True):
    loop_dir = str(tmp_path / "loop")
    task = Task(base_cfg, ROOT, kit_dir=os.path.join(ROOT, "starter_kit"), sealed_dir=os.path.join(ROOT, "sealed"),
                data_dir=mini_data, loop_data_dir=loop_dir, champion_src_dir=str(tmp_path / "champion"),
                name="leak_test", expected={}, assert_rungs=False, run_official_baseline=False, mask_test=mask_test)
    task.prepare(log=lambda *_: None)
    return task


def _capture_sandbox_call(monkeypatch):
    """Stub out the actual subprocess launch; record every (data_dir, deny_read) pair it was called with."""
    calls = []

    def fake(workspace, data_dir, split, out_name, timeout_s, sandbox_cfg, pythonpath=(), deny_read=(), log_prefix="", extra_env=None):
        calls.append({"data_dir": data_dir, "deny_read": list(deny_read)})
        from agent.sandbox import SandboxResult
        return SandboxResult(status="ok", returncode=0, runtime_s=0.0, stdout_tail="", stderr_tail="",
                             cmd=[], isolation="none")

    monkeypatch.setattr(tools, "run_pipeline_in_sandbox", fake)
    return calls


def test_loop_iteration_never_passes_the_full_data_dir(tmp_path, base_cfg, mini_data, monkeypatch):
    """The one property that matters regardless of OS-level sandbox enforcement: during the loop
    (full_data=False, the default — this is what harness.py's per-iteration call site uses), the
    subprocess must receive the MASKED loop_data_dir, never task.data_dir, and task.data_dir must be
    in the read-deny list as defense in depth."""
    task = _make_task(tmp_path, base_cfg, mini_data)
    calls = _capture_sandbox_call(monkeypatch)

    task.sandbox_run(str(tmp_path / "ws1"), "val", "preds_val.csv", timeout_s=30)

    assert len(calls) == 1
    assert calls[0]["data_dir"] == task.loop_data_dir
    assert calls[0]["data_dir"] != task.data_dir
    assert task.data_dir in calls[0]["deny_read"]


def test_finalize_is_the_only_path_allowed_to_see_the_full_dir(tmp_path, base_cfg, mini_data, monkeypatch):
    """full_data=True is the finalize-only call shape (harness.py's finalize() is the sole caller that
    passes it). When explicitly requested, the full dir is used and is NOT in the deny list — this is
    the one intentional exception, and the test pins it to the explicit opt-in rather than a default."""
    task = _make_task(tmp_path, base_cfg, mini_data)
    calls = _capture_sandbox_call(monkeypatch)

    task.sandbox_run(str(tmp_path / "ws2"), "test", "preds_test.csv", timeout_s=30, full_data=True)

    assert calls[0]["data_dir"] == task.data_dir
    assert task.data_dir not in calls[0]["deny_read"]


def test_masking_disabled_falls_back_to_full_dir_explicitly(tmp_path, base_cfg, mini_data, monkeypatch):
    """mask_test=False (or no loop_data_dir configured) is a deliberate opt-out, not a silent gap:
    loop_data_dir collapses to data_dir itself, so there is no separate masked copy to leak out of."""
    task = _make_task(tmp_path, base_cfg, mini_data, mask_test=False)
    assert task.loop_data_dir == task.data_dir
    calls = _capture_sandbox_call(monkeypatch)

    task.sandbox_run(str(tmp_path / "ws3"), "val", "preds_val.csv", timeout_s=30)

    assert calls[0]["data_dir"] == task.data_dir
    # the data dir itself isn't denied (no separate masked copy exists) — secrets (.env) still are,
    # regardless of masking, per Task.secret_files()
    assert task.data_dir not in calls[0]["deny_read"]
