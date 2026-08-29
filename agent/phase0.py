"""Phase 0 — baseline reproduction + harness self-check (spec §7). Runs once, before iteration 1.

1. random + item-popularity predictors scored by sealed evaluate.py → must land near the published
   VALID rungs (tolerance from config);
2. the organizers' own FM (`submit.py --make --split valid`, unchanged code) → primary ≈ 0.6016;
3. the §5.2-shaped champion (`baseline_repro/pipeline.py`) run through the sandbox → same band, and
   installed as iteration 0's champion in runs/RUN_ID/best/.
Any failed assertion aborts loudly: nothing downstream would be trustworthy.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict

from . import tools
from .memory import append_ledger, init_ledger
from .schemas import RunState, atomic_write_json, utc_now_iso


class Phase0Error(RuntimeError):
    pass


def _fmt(s) -> str:
    return f"primary {s.primary:.4f} (GAUC {s.gauc:.4f} / nDCG5 {s.ndcg5:.4f})"


def install_champion(run_dir: str, files: Dict[str, str], preds_path: str, score, iteration: int, source: str) -> str:
    """Atomically (re)place runs/RUN_ID/best/ with code + preds + score. Only the harness calls this."""
    best = os.path.join(run_dir, "best")
    tmp = best + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "code"))
    tools.write_code_files(os.path.join(tmp, "code"), files)
    if preds_path and os.path.exists(preds_path):
        shutil.copy2(preds_path, os.path.join(tmp, tools.PREDS_VAL))
    atomic_write_json(os.path.join(tmp, "score.json"), score.to_dict())
    atomic_write_json(os.path.join(tmp, "champion.json"), {"iteration": iteration, "score": score.to_dict(),
                                                            "installed_at": utc_now_iso(), "source": source})
    prev = best + ".prev"
    shutil.rmtree(prev, ignore_errors=True)
    if os.path.exists(best):
        os.rename(best, prev)
    os.rename(tmp, best)
    shutil.rmtree(prev, ignore_errors=True)
    return best


def run_phase0(task, run_dir: str, state: RunState, cfg: Dict[str, Any], log=print) -> Dict[str, Any]:
    p0cfg = cfg.get("phase0", {})
    tol, btol = float(p0cfg.get("rung_tolerance", 0.01)), float(p0cfg.get("baseline_tolerance", 0.005))
    p0 = os.path.join(run_dir, "phase0")
    os.makedirs(p0, exist_ok=True)
    res: Dict[str, Any] = {"started": utc_now_iso(), "checks": [], "expected": task.expected}
    failures = []

    def check(name: str, value: float, expected, tolerance: float):
        if expected is None:
            res["checks"].append({"name": name, "value": value, "expected": None, "ok": None})
            return
        ok = abs(value - expected) <= tolerance
        res["checks"].append({"name": name, "value": value, "expected": expected, "tolerance": tolerance, "ok": ok})
        if not ok and task.assert_rungs:
            failures.append(f"{name}: got {value:.4f}, expected {expected:.4f} ± {tolerance}")

    # 0. convergence rule: our EPSILON / N_FLAT must BE the organizers' published numbers ------------------
    # The kit ships the rule as data (baseline_scores.json -> convergence_rule), not as code, so faithfulness means
    # asserting our configuration equals theirs rather than trusting a hand-copied constant.
    run_cfg = cfg.get("run", {})
    kit_eps, kit_n = task.expected.get("epsilon"), task.expected.get("N")
    our_eps, our_n = float(run_cfg.get("EPSILON")), int(run_cfg.get("N_FLAT"))
    ok_rule = (kit_eps is None or abs(our_eps - float(kit_eps)) < 1e-12) and (kit_n is None or our_n == int(kit_n))
    res["checks"].append({"name": "convergence_rule_matches_kit", "value": f"EPSILON={our_eps} N_FLAT={our_n}",
                          "expected": f"epsilon={kit_eps} N={kit_n}", "ok": ok_rule})
    log(f"[phase0] convergence rule: EPSILON={our_eps} N_FLAT={our_n} vs kit baseline_scores.json "
        f"epsilon={kit_eps} N={kit_n} -> {'match' if ok_rule else 'MISMATCH'}")
    if not ok_rule:
        failures.append(f"convergence rule differs from the kit: ours EPSILON={our_eps}/N_FLAT={our_n}, "
                        f"kit epsilon={kit_eps}/N={kit_n}")

    # 1. rungs ---------------------------------------------------------
    rows_valid = task.rows_valid
    rnd_path = os.path.join(p0, "random_valid.csv")
    tools.write_preds(rnd_path, rows_valid, tools.random_scores(len(rows_valid), int(p0cfg.get("random_seed", 0))))
    s_rand = task.score_preds(rnd_path)
    res["random"] = s_rand.to_dict()
    log(f"[phase0] random rung: {_fmt(s_rand)}  (published valid {task.expected.get('random')})")
    check("random_valid_primary", s_rand.primary, task.expected.get("random"), tol)

    pop_path = os.path.join(p0, "pop_valid.csv")
    tools.write_preds(pop_path, rows_valid, tools.popularity_scores(task.rows_train, rows_valid, float(p0cfg.get("pop_prior", 20.0))))
    s_pop = task.score_preds(pop_path)
    res["pop"] = s_pop.to_dict()
    log(f"[phase0] item-popularity rung: {_fmt(s_pop)}  (published valid {task.expected.get('pop')})")
    check("pop_valid_primary", s_pop.primary, task.expected.get("pop"), tol)

    # 2. official baseline (organizer code, unchanged) -------------------
    if task.run_official_baseline:
        off_path = os.path.join(p0, "official_fm_valid.csv")
        log("[phase0] running the organizers' FM baseline (submit.py --make --split valid) ...")
        ok, out, rt = task.official_baseline_preds(off_path)
        if not ok:
            failures.append(f"official baseline failed to run: {out[-800:]}")
            res["official_fm"] = {"ok": False, "output": out[-2000:], "runtime_s": rt}
        else:
            s_off = task.score_preds(off_path)
            res["official_fm"] = {**s_off.to_dict(), "runtime_s": round(rt, 1)}
            log(f"[phase0] official FM baseline: {_fmt(s_off)} in {rt:.0f}s  (published valid {task.expected.get('fm')})")
            check("official_fm_valid_primary", s_off.primary, task.expected.get("fm"), btol)

    # 3. champion (pipeline-contract port of the baseline) ---------------
    files = tools.read_code_files(task.champion_src_dir)
    if tools.PIPELINE_MAIN not in files:
        raise Phase0Error(f"no {tools.PIPELINE_MAIN} in champion source dir {task.champion_src_dir}")
    from .sandbox import static_code_check
    violations = static_code_check(files, cfg.get("sandbox", {}))
    if violations:
        raise Phase0Error("champion source violates the sandbox code policy (fix the champion or the policy): " + "; ".join(violations))
    ws = os.path.join(p0, "champion_check")
    shutil.rmtree(ws, ignore_errors=True)
    tools.write_code_files(ws, files)
    log(f"[phase0] running the iteration-0 champion ({task.champion_src_dir}) through the sandbox ...")
    sres = task.sandbox_run(ws, "val", tools.PREDS_VAL, float(cfg["run"]["EXPERIMENT_TIMEOUT_S"]))
    res["champion_run"] = {"status": sres.status, "runtime_s": round(sres.runtime_s, 1), "isolation": sres.isolation}
    if not sres.ok:
        raise Phase0Error(f"champion pipeline failed ({sres.status}):\n{sres.error_excerpt()}")
    try:
        s_ch = task.score_preds(os.path.join(ws, tools.PREDS_VAL))
    except ValueError as e:
        raise Phase0Error(f"champion predictions rejected by the sealed checker: {e}")
    res["champion"] = s_ch.to_dict()
    log(f"[phase0] champion: {_fmt(s_ch)} in {sres.runtime_s:.0f}s")
    check("champion_valid_primary", s_ch.primary, task.expected.get("fm"), btol)
    if task.run_official_baseline and res.get("official_fm", {}).get("primary") is not None:
        res["champion_vs_official"] = round(s_ch.primary - res["official_fm"]["primary"], 6)

    if failures:
        msg = "PHASE 0 FAILED — nothing downstream is trustworthy:\n  - " + "\n  - ".join(failures)
        with open(os.path.join(run_dir, "PHASE0_FAILED.md"), "w") as fh:
            fh.write(msg + "\n\n" + json.dumps(res, indent=1, default=str) + "\n")
        raise Phase0Error(msg)

    install_champion(run_dir, files, os.path.join(ws, tools.PREDS_VAL), s_ch, 0, task.champion_src_dir)
    state.best_primary, state.best_gauc, state.best_ndcg5, state.best_iter = s_ch.primary, s_ch.gauc, s_ch.ndcg5, 0
    res["passed"] = True
    res["finished"] = utc_now_iso()
    state.phase0 = res
    init_ledger(run_dir)
    append_ledger(run_dir, f"# it00 champion installed from {os.path.relpath(task.champion_src_dir, task.root)}: "
                           f"val primary {s_ch.primary:.4f} (GAUC {s_ch.gauc:.4f} / nDCG5 {s_ch.ndcg5:.4f}); "
                           f"published baseline {task.expected.get('fm')}; rungs random {s_rand.primary:.4f} pop {s_pop.primary:.4f}")
    atomic_write_json(os.path.join(p0, "phase0_results.json"), res)
    return res
