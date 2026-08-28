"""Shared fixtures: the toy task (mini synthetic dataset in kit format) wired into a real Harness with a
mock LLM client, so every loop property is exercised through the production code path."""
from __future__ import annotations

import copy
import os
import shutil
import sys
import time
from typing import Callable, Dict, Optional

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent import toy as toymod                      # noqa: E402
from agent.harness import Harness, deep_update, load_config   # noqa: E402
from agent.llm_client import CallLog, MockLLMClient  # noqa: E402
from agent.roles import Roles                        # noqa: E402
from agent.stub_roles import default_mock_handlers   # noqa: E402
from agent.task import Task                          # noqa: E402

REAL_DATA = os.path.join(ROOT, "starter_kit", "KuaiRand-Pure", "data")
HAVE_REAL_DATA = os.path.exists(os.path.join(REAL_DATA, "log_standard_4_08_to_4_21_pure.csv"))


@pytest.fixture(scope="session")
def project_root() -> str:
    return ROOT


@pytest.fixture(scope="session")
def base_cfg() -> dict:
    return load_config(os.path.join(ROOT, "config.yaml"))


@pytest.fixture(scope="session")
def mini_data(tmp_path_factory) -> str:
    d = str(tmp_path_factory.mktemp("mini") / "data")
    toymod.make_mini_dataset(d, seed=0)
    return d


def toy_cfg(base_cfg: dict, overrides: Optional[dict] = None) -> dict:
    cfg = copy.deepcopy(base_cfg)
    deep_update(cfg, cfg.get("toy", {}).get("overrides", {}))
    cfg["run"].update({"MAX_ITERS": 5, "N_FLAT": 3, "EXPERIMENT_TIMEOUT_S": 30, "FINALIZE_TIMEOUT_S": 30})
    cfg["llm"]["scribe_narrative"] = True
    if overrides:
        deep_update(cfg, overrides)
    return cfg


def make_toy_harness(tmp_path, base_cfg: dict, mini_data: str, *, handlers: Optional[Dict[str, Callable]] = None,
                     overrides: Optional[dict] = None, clock=time.time, run_dir: Optional[str] = None, log=lambda *_: None) -> Harness:
    cfg = toy_cfg(base_cfg, overrides)
    champ = str(tmp_path / "champion")
    if not os.path.exists(os.path.join(champ, "pipeline.py")):
        toymod.write_dummy_champion(champ)
    task = Task(cfg, ROOT, kit_dir=os.path.join(ROOT, "starter_kit"), sealed_dir=os.path.join(ROOT, "sealed"), data_dir=mini_data,
                loop_data_dir=str(tmp_path / "loop"), champion_src_dir=champ, name="toy", expected={}, assert_rungs=False,
                run_official_baseline=False, mask_test=True)
    client = MockLLMClient(handlers or default_mock_handlers())
    run_dir = run_dir or str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)
    roles = Roles(client, cfg, os.path.join(ROOT, "prompts"), os.path.join(ROOT, "knowledge", "library.md"),
                  call_log=CallLog(os.path.join(run_dir, "llm_calls.jsonl")))
    return Harness(cfg, ROOT, run_dir, task, roles, clock=clock, log=log)


def sha256_tree(directory: str) -> Dict[str, str]:
    import hashlib
    out = {}
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            p = os.path.join(root, f)
            out[os.path.relpath(p, directory)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out
