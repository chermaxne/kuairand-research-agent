"""Task wiring shared by the real KuaiRand run and the toy/test runs: data dirs (full + masked loop
copy), kit + sealed imports, validation rows for scoring, sandbox invocation, submission checking."""
from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from . import tools
from .sandbox import SandboxResult
from .schemas import Score


class Task:
    def __init__(self, cfg: Dict[str, Any], root: str, *, kit_dir: str, sealed_dir: str, data_dir: str,
                 loop_data_dir: Optional[str], champion_src_dir: str, name: str = "kuairand",
                 expected: Optional[Dict[str, float]] = None, assert_rungs: bool = True,
                 run_official_baseline: bool = True, mask_test: bool = True):
        self.cfg = cfg
        self.root = os.path.realpath(root)
        self.name = name
        self.kit_dir = os.path.realpath(kit_dir)
        self.sealed_dir = os.path.realpath(sealed_dir)
        self.data_dir = os.path.realpath(data_dir)
        self.loop_data_dir = os.path.realpath(loop_data_dir) if (loop_data_dir and mask_test) else self.data_dir
        self.mask_test = bool(mask_test and loop_data_dir)
        self.champion_src_dir = os.path.realpath(champion_src_dir)
        self.expected = expected or {}
        self.assert_rungs = assert_rungs
        self.run_official_baseline = run_official_baseline
        self.kit = None
        self.sealed_eval = None
        self.rows_train: List[tuple] = []
        self.rows_valid: List[tuple] = []
        self.profile: str = ""
        self.loop_build_info: Dict[str, Any] = {}
        self.prepared = False

    # ------------------------------------------------------------------
    @property
    def baseline_primary(self) -> Optional[float]:
        return self.expected.get("fm")

    @property
    def sandbox_cfg(self) -> Dict[str, Any]:
        return self.cfg.get("sandbox", {})

    def prepare(self, log=print) -> None:
        if self.prepared:
            return
        self.kit = tools.import_kit(self.kit_dir)
        self.sealed_eval = tools.import_sealed_evaluate(self.sealed_dir)
        if self.mask_test:
            max_date = int(self.kit.data.SPLITS["valid"][1])
            self.loop_build_info = tools.ensure_loop_data_dir(self.data_dir, self.loop_data_dir, max_date)
            if self.loop_build_info.get("rebuilt"):
                log(f"[task] built masked loop data dir {self.loop_data_dir} (rows after {max_date} removed)")
        splits = self.kit.data.load(self.loop_data_dir)
        self.rows_train, self.rows_valid = splits["train"], splits["valid"]
        if self.mask_test and splits.get("test"):
            raise RuntimeError("loop data dir still contains test-period rows — masking failed")
        self.profile = tools.data_profile(self.loop_data_dir, splits)
        self.prepared = True

    # ------------------------------------------------------------------
    def score_preds(self, preds_path: str) -> Score:
        """Sealed score of a validation prediction file (raises ValueError on misalignment / NaN)."""
        return tools.evaluate_preds(preds_path, self.rows_valid, self.sealed_eval, self.kit)

    def sandbox_run(self, workspace: str, split: str, out_name: str, timeout_s: float, log_prefix: str = "",
                    full_data: bool = False) -> SandboxResult:
        """Run `pipeline.py` in `workspace`. During the loop the masked dir is used and the full dir is
        read-denied (sandbox-exec); finalize passes full_data=True."""
        data_dir = self.data_dir if full_data else self.loop_data_dir
        deny = [] if (full_data or not self.mask_test) else [self.data_dir]
        deny += self.secret_files()
        return tools.run_pipeline_in_sandbox(workspace, data_dir, split, out_name, timeout_s, self.sandbox_cfg,
                                             pythonpath=[self.sealed_dir], deny_read=deny, log_prefix=log_prefix)

    def secret_files(self) -> List[str]:
        """Files experiments must never read (API keys): the repo's .env variants and the harness's own env file."""
        out = []
        for name in (".env", ".env.local", ".env.poe", ".env.anthropic"):
            p = os.path.join(self.root, name)
            if os.path.isfile(p):
                out.append(p)
        extra = os.environ.get("HARNESS_ENV_FILE")
        if extra and os.path.isfile(extra):
            out.append(extra)
        return out

    def leak_test(self, workspace: str, timeout_s: float, out_name: str = "preds_val_leaktest.csv") -> Dict[str, Any]:
        """Re-run the workspace's pipeline against the flipped-label copy of the loop data and score its predictions
        against the TRUE validation labels. Returns {'ran', 'primary', 'gauc', 'error'}; the harness decides."""
        flipped_dir = self.loop_data_dir + "_flipped_labels"
        train_end = int(self.kit.data.SPLITS["train"][1])
        info = tools.ensure_flipped_labels_dir(self.loop_data_dir, flipped_dir, train_end)
        deny = [self.data_dir] if self.mask_test else []
        deny += self.secret_files()
        res = tools.run_pipeline_in_sandbox(workspace, flipped_dir, "val", out_name, timeout_s, self.sandbox_cfg,
                                            pythonpath=[self.sealed_dir], deny_read=deny, log_prefix="leaktest_")
        out: Dict[str, Any] = {"ran": res.ok, "runtime_s": round(res.runtime_s, 1), "flipped_dir_rebuilt": info.get("rebuilt")}
        if not res.ok:
            out["error"] = res.error_excerpt(15)
            return out
        try:
            sc = self.score_preds(os.path.join(workspace, out_name))
            out.update({"primary": sc.primary, "gauc": sc.gauc, "ndcg5": sc.ndcg5})
        except (ValueError, OSError) as e:
            out["error"] = f"leak-test predictions rejected: {e}"
        return out

    def check_submission(self, path: str, split: str = "test") -> Tuple[bool, str]:
        return tools.check_submission(self.sealed_dir, self.kit_dir, self.data_dir, path, split=split)

    def official_baseline_preds(self, out_path: str) -> Tuple[bool, str, float]:
        return tools.make_official_baseline_preds(self.kit_dir, self.loop_data_dir, out_path, split="valid")


# ----------------------------------------------------------------------
def published_expectations(baseline_scores_path: str) -> Dict[str, float]:
    """VALID-split rungs from the kit's baseline_scores.json (never test values)."""
    j = json.load(open(baseline_scores_path))
    s = j["scores"]
    return {"random": float(s["random"]["valid"]["primary"]),
            "pop": float(s["item_popularity"]["valid"]["primary"]),
            "fm": float(s["fm_official"]["valid"]["primary"]),
            "fm_gauc": float(s["fm_official"]["valid"]["GAUC"]),
            "fm_ndcg5": float(s["fm_official"]["valid"]["nDCG@5"]),
            "oracle": float(s["oracle_ceiling"]["valid"]["primary"]),
            "epsilon": float(j["convergence_rule"]["epsilon"]), "N": int(j["convergence_rule"]["N"])}


def _p(root: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(root, rel)


def make_task(cfg: Dict[str, Any], root: str, toy: bool = False) -> Task:
    paths = cfg["paths"]
    if toy:
        from . import toy as toymod
        t = cfg.get("toy", {})
        data_dir = _p(root, t.get("data_dir", "./data_cache/toy/data"))
        champ = _p(root, t.get("champion_src", "./data_cache/toy/champion"))
        if not os.path.exists(os.path.join(data_dir, "log_standard_4_08_to_4_21_pure.csv")):
            toymod.make_mini_dataset(data_dir, seed=int(t.get("seed", 0)))
        toymod.write_dummy_champion(champ)
        return Task(cfg, root, kit_dir=_p(root, paths["starter_kit"]), sealed_dir=_p(root, paths["sealed"]), data_dir=data_dir,
                    loop_data_dir=_p(root, t.get("loop_data", "./data_cache/toy/loop")), champion_src_dir=champ, name="toy",
                    expected={}, assert_rungs=False, run_official_baseline=False,
                    mask_test=bool(cfg["run"].get("mask_test_period_in_loop", True)))
    expected = published_expectations(_p(root, paths["baseline_scores"]))
    return Task(cfg, root, kit_dir=_p(root, paths["starter_kit"]), sealed_dir=_p(root, paths["sealed"]), data_dir=_p(root, paths["data"]),
                loop_data_dir=_p(root, paths["loop_data"]), champion_src_dir=_p(root, paths["baseline_repro"]), name="kuairand",
                expected=expected, assert_rungs=bool(cfg["phase0"].get("assert_rungs", True)),
                run_official_baseline=bool(cfg["phase0"].get("run_official_baseline", True)),
                mask_test=bool(cfg["run"].get("mask_test_period_in_loop", True)))
