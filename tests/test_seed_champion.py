"""Tests for --seed-champion: starting a new run's iteration-0 champion from an existing pipeline.py
directory instead of the sacred baseline_repro/ port. The published-baseline tolerance check must be
skipped for a seeded champion (it's expected to already beat 0.6016, that's the point of seeding),
while the other Phase 0 sanity checks (random/pop/official-FM rungs) stay in force."""
import os

from agent.task import make_task, published_expectations
from tests.conftest import ROOT


def test_make_task_defaults_to_baseline_repro(base_cfg):
    task = make_task(base_cfg, ROOT)
    assert task.champion_src_dir == os.path.realpath(os.path.join(ROOT, base_cfg["paths"]["baseline_repro"]))
    assert task.verify_champion_baseline is True


def test_make_task_with_seed_champion_overrides_source_and_skips_baseline_check(base_cfg, tmp_path):
    seed_dir = tmp_path / "some_prior_champion" / "code"
    os.makedirs(seed_dir)
    (seed_dir / "pipeline.py").write_text("# not real code, just checking path wiring\n")

    task = make_task(base_cfg, ROOT, seed_champion=str(seed_dir))
    assert task.champion_src_dir == os.path.realpath(str(seed_dir))
    assert task.champion_src_dir != os.path.realpath(os.path.join(ROOT, base_cfg["paths"]["baseline_repro"]))
    assert task.verify_champion_baseline is False
    # everything else about the task (data dirs, expectations, masking) must be unaffected by seeding
    assert task.expected == published_expectations(os.path.join(ROOT, base_cfg["paths"]["baseline_scores"]))
    assert task.mask_test is True
