"""Main loop + stopping logic + resume + finalize (spec §6, §7, §12).

Deterministic Python owns every guarantee: the LLM roles only return hypotheses, code, fixes and one
lesson sentence. Scores come from sealed/evaluate.py; promotion / streak / stop decisions come from
agent/promotion.py; everything about a run lives in one timestamped run directory.

    python -m agent.harness                 # new run on the real data (needs the API key in env)
    python -m agent.harness --mock          # offline deterministic roles
    python -m agent.harness --toy --mock    # Phase-1 skeleton on the toy task
    python -m agent.harness --run-dir runs/<RUN_ID>   # resume
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import yaml

from . import tools
from .llm_client import CallLog, make_client
from .memory import (prior_runs_digest, format_ablations, parse_ablations, append_intervention, append_ledger, fmt_elapsed, init_interventions, init_ledger, ledger_line, load_run_state,
                     one_line, read_iteration_detail, read_ledger, render_state_block, research_digest, result_string,
                     save_run_state, synthesis_numbers_ok, write_iteration_log, write_iteration_narrative, write_state_block)
from .phase0 import Phase0Error, install_champion, run_phase0
from .promotion import RunLimits, judge_iteration, stop_reason, vs_best_string
from .roles import Roles
from .sandbox import static_code_check
from .schemas import DebugAttempt, HarnessResult, IterationLog, ResearcherPlan, RunState, Score, atomic_write_json, atomic_write_text, utc_now_iso
from .task import Task, make_task

PIPELINE_CONTRACT_NOTE = """`python pipeline.py --data <data_dir> --split val --out preds_val.csv`
- Train ONLY on the train split (dates 20220408-20220421). Validation rows may be used for early stopping / model selection only.
- Write EVERY validation row, in data.load() order, as `row_id,user_id,video_id,score` (row_id from 0, ids echoed exactly as read, finite scores).
- `--split test` must keep working unchanged (it is used once, at finalize, on the champion).
- Exit 0 on success. Single process, no network, no package installs, only pre-installed libraries
  (numpy, pandas, scikit-learn, lightgbm, torch-cpu). Same-row feedback columns are NOT features (leakage).
- Hard wall-clock limit: {timeout}s for the whole run (load + train + predict).
- TIME BUDGET (hard): `KUAIRAND_TIME_BUDGET_S` is in the environment ({timeout}s here) and the process is killed at it.
  Budget it explicitly with arithmetic, not hope:
  1. Fit the FULL bundle first and WRITE ITS PREDICTIONS before anything else. It must finish inside 40% of the budget.
  2. Only then run ablations, and before each one check `time.time() - t0` against the budget: start a variant only if
     at least 25% of the budget remains, else print `ABLATION <name> skipped: <reason>` and move on.
  3. Ablation variants are DIAGNOSTICS, not submissions: make them cheap (a subsample of the training rows, or a fixed
     small number of rounds/epochs, or one seed). A variant must never cost as much as the full fit.
- In-run attribution: for each variant you do run, score it on validation with the official `evaluate()` and print one
  line `ABLATION <name> primary=<f> gauc=<f> ndcg5=<f>` (real numbers from real fits only). The written predictions are
  always the full bundle.
- Evaluating the metric is expensive (~125k rows). For early stopping, score at most every ~50 boosting rounds or once
  per epoch — NOT every few rounds. Scoring after every 10 rounds turned a 30 s fit into 6 minutes in run ten16.
- If this change replaces or adds a LOSS FUNCTION (e.g. pointwise -> pairwise/BPR, adding an auxiliary
  head): the learning rate and any other optimizer constants were tuned for the OLD objective's gradient
  scale and are not guaranteed to transfer. Reusing them unchanged is a common cause of loss divergence
  (loss climbing epoch over epoch instead of falling). If the change_spec does not already address this,
  pick a conservative LR for the new objective (or add gradient clipping) rather than inheriting the old
  value silently."""

STALL_DIRECTIVE = """# STALL RECOVERY DIRECTIVE (injected by the harness)
The last {n} iterations ALL failed (crash / timeout / rejected output). Do NOT propose anything ambitious now.
Propose the SIMPLEST, most reliable change to the champion that is still a real hypothesis (e.g. one
hyperparameter, one well-understood feature, a smaller model variant). Requirements: no new libraries, no
new files, runtime well under the limit, minimal diff. State in `rationale` why it cannot fail the same way."""


SIZING_DIRECTIVE = """# SIZING DIRECTIVE (harness policy: flat streak {streak} of {n_flat} — {lives} more miss(es) end the run)
The convergence rule is per iteration: only a gain > +{epsilon} over the best-so-far ({best}) resets the streak. A
+0.001 gain is promoted and banked, but it still counts as a miss. So every proposal must be SIZED to clear
+{epsilon} on its own: ONE hypothesis whose expected gain you state as a number with evidence (`expected_gain`,
`gain_evidence`), plus every validated rider (a component already measured positive on this run that is not yet in
the champion). Hyperparameter-only proposals cannot clear +{epsilon} and are not allowed.
{posture}
Attribution is free and happens INSIDE the run, never across iterations: write an `ablation_plan` naming the
variants the pipeline should also train and score on validation (at minimum the bundle WITHOUT the new
component, i.e. the champion-equivalent), printed as `ABLATION <name> primary=... gauc=... ndcg5=...`. The
written predictions are the full bundle; only the sealed score counts. The wall-clock limit is {timeout}s, so
budget the extra fits explicitly (the champion fit takes about a minute)."""

POSTURE_BOLD = """Posture at streak 0: take your boldest well-grounded structural bet — a change that gives the model NEW
INFORMATION (the user's past behaviour as a sequence, auxiliary behaviours, watch time, past-only context) or an
objective closer to the metric. Capacity alone is not information: the organizers measured that bigger embeddings
and more static fields do nothing, so a deeper network over the same inputs is a large diff with a small expected
gain. An architecture change earns its place when it is what lets the model consume a new signal (e.g. attention
over the user's history)."""

POSTURE_STEADY = """Posture at streak {streak}: still structural, but choose the variant with the best evidence rather than the highest
ceiling, stack every validated rider, keep the champion's seed averaging, and prefer the implementation with the
fewest new moving parts — a crash or timeout costs a miss just like a flat result."""


STREAK_DIRECTIVE = """# LAST-SHOT DIRECTIVE (harness policy: flat streak {streak} of {n_flat})
One more iteration without a gain > +{epsilon} over the best-so-far ({best}) ENDS THE RUN. Choose the
highest-probability bundle: keep every component of the champion that produced its gain (its loss, its fields,
its seed averaging) exactly as is, add more seeds if the champion uses fewer than 5, stack EVERY validated rider
not yet in the champion, and add ONE genuinely new signal. Do NOT replace or remove a proven component, do NOT
re-try a lever kind whose last result was within ±0.0006 (noise), and state in `rationale` and `gain_evidence`
why this bundle should clear +{epsilon}. Keep the `ablation_plan` minimal (champion-equivalent only)."""


class FinalizeError(RuntimeError):
    pass


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


class Harness:
    def __init__(self, cfg: Dict[str, Any], root: str, run_dir: str, task: Task, roles: Roles, clock=time.time, log=print):
        self.cfg = cfg
        self.root = os.path.realpath(root)
        self.run_dir = os.path.realpath(run_dir)
        self.task = task
        self.roles = roles
        self.clock = clock
        self.log = log
        self.roles.log = log
        self.limits = RunLimits.from_config(cfg)
        self.run_cfg = cfg["run"]
        self.state: Optional[RunState] = None

    # ------------------------------------------------------------------ paths
    @property
    def best_dir(self) -> str:
        return os.path.join(self.run_dir, "best")

    @property
    def best_code_dir(self) -> str:
        return os.path.join(self.best_dir, "code")

    def iteration_dir(self, it: int) -> str:
        return os.path.join(self.run_dir, "iterations", f"it{it:02d}")

    # ------------------------------------------------------------------ init / resume
    def init_or_resume(self) -> RunState:
        self.task.prepare(self.log)
        state = load_run_state(self.run_dir)
        if state is None:
            os.makedirs(self.run_dir, exist_ok=True)
            for d in ("iterations", "logs", "best"):
                os.makedirs(os.path.join(self.run_dir, d), exist_ok=True)
            run_id = os.path.basename(self.run_dir)
            state = RunState(run_id=run_id, start_ts=utc_now_iso(), start_time=float(self.clock()),
                             baseline_primary=self.task.baseline_primary, config_snapshot=copy.deepcopy(self.cfg))
            state.warnings.append(f"sandbox isolation: {self._isolation_note()}")
            from .llm_client import provider_balance
            state.spend_start = provider_balance(self.cfg) or {}
            init_ledger(self.run_dir)
            init_interventions(self.run_dir, run_id)
            atomic_write_text(os.path.join(self.run_dir, "data_profile.md"), self.task.profile)
            save_run_state(self.run_dir, state)
            self.log(f"=== NEW RUN {run_id} in {self.run_dir} (task={self.task.name}, llm={self.roles.client.provider}) ===")
        else:
            state.resumes += 1
            save_run_state(self.run_dir, state)
            append_intervention(self.run_dir, what_stuck=f"harness process ended after iteration {state.iteration} (stop_reason={state.stop_reason})",
                                what_done="harness restarted and resumed from run_state.json", scope="resume (auto-recorded)")
            state = load_run_state(self.run_dir)
            partial = self.iteration_dir(state.iteration + 1)
            if os.path.exists(partial):
                os.rename(partial, partial + f"_partial_{int(self.clock())}")
            self.log(f"=== RESUMED {state.run_id}: {state.iteration} iterations done, streak {state.streak}, "
                     f"best {state.best_primary} (it{state.best_iter:02d}), {fmt_elapsed(state.elapsed_s(self.clock()))} elapsed ===")
        self.state = state
        return state

    def _isolation_note(self) -> str:
        from .sandbox import detect_isolation
        try:
            iso = detect_isolation(self.cfg.get("sandbox", {}).get("isolation", "auto"))
        except RuntimeError as e:
            return f"ERROR {e}"
        return iso if iso != "none" else "none (WARNING: no OS-level network/write confinement on this host)"

    # ------------------------------------------------------------------ main loop
    def run(self, session_iteration_limit: Optional[int] = None) -> RunState:
        state = self.init_or_resume()
        if state.iteration == 0 and not state.phase0.get("passed"):
            self.phase0()
        done = 0
        while True:
            spend_usd = None
            if self.limits.max_total_spend_usd is not None:
                # prefer real per-call cost (immune to teammates sharing the API key); only fall back to the
                # account-wide balance diff (one extra HTTP round-trip) when the provider never reported one
                spend_usd = state.real_spend_usd if state.real_spend_usd is not None else self._current_spend_usd()
            reason = stop_reason(state.streak, state.iteration, state.elapsed_s(self.clock()), state.tokens_total, self.limits,
                                 spend_usd=spend_usd)
            if reason:
                return self.finalize(reason)
            if session_iteration_limit is not None and done >= session_iteration_limit:
                self.log(f"[harness] session limit of {session_iteration_limit} iterations reached; exiting without finalize (resumable)")
                return state
            self.run_iteration(state.iteration + 1)
            done += 1

    def phase0(self) -> None:
        assert self.state is not None
        self.log("[phase0] baseline reproduction + harness self-check")
        try:
            run_phase0(self.task, self.run_dir, self.state, self.cfg, self.log)
        except Phase0Error as e:
            self.state.warnings.append(f"phase0 failed: {e}")
            save_run_state(self.run_dir, self.state)
            self.log(str(e))
            raise
        write_state_block(self.run_dir, render_state_block(self.state, self._limits_dict(), 0, self.clock()))
        save_run_state(self.run_dir, self.state)
        self.log(f"[phase0] PASSED — champion it00 val primary {self.state.best_primary:.4f}")

    def _limits_dict(self) -> Dict[str, Any]:
        return {**self.run_cfg, "categories": self.run_cfg.get("categories")}

    # ------------------------------------------------------------------ briefing
    def assemble_briefing(self, it: int) -> str:
        state = self.state
        assert state is not None
        parts = ["# STATE BLOCK\n" + render_state_block(state, self._limits_dict(), it, self.clock())]
        parts.append(self.task.profile or "")
        champ = tools.read_code_files(self.best_code_dir)
        parts.append("# CHAMPION CODE (current best pipeline; every experiment builds on it)\n" +
                     "\n".join(f"--- {n} ---\n{c}" for n, c in sorted(champ.items())))
        parts.append("# LEDGER (full history, oldest first)\n" + (read_ledger(self.run_dir) or "(empty)"))
        if self.run_cfg.get("cross_run_memory", True):
            prior = prior_runs_digest(os.path.dirname(os.path.abspath(self.run_dir)), self.run_dir,
                                      int(self.run_cfg.get("cross_run_max_rows", 60)))
            if prior:
                parts.append(prior)
        digest = research_digest(state.history, lambda n: read_iteration_detail(self.run_dir, n))
        if digest:
            parts.append(digest)
            if state.synthesis:
                parts.append("# RESEARCH SYNTHESIS (written by the Scribe from the digest above — interpretive; verify any claim "
                             "against the table)\n" + state.synthesis)
        parts.append(self.render_recent_iterations(state))
        last_shot = self.limits.n_flat > 1 and state.streak >= self.limits.n_flat - 1
        best_s = f"{state.best_primary:.4f}" if state.best_primary is not None else "n/a"
        if not last_shot and self.run_cfg.get("sizing_directive", True):
            posture = POSTURE_BOLD if state.streak == 0 else POSTURE_STEADY.format(streak=state.streak)
            parts.append(SIZING_DIRECTIVE.format(streak=state.streak, n_flat=self.limits.n_flat, lives=self.limits.n_flat - state.streak,
                                                 epsilon=self.limits.epsilon, best=best_s, posture=posture,
                                                 timeout=int(self.run_cfg["EXPERIMENT_TIMEOUT_S"])))
        if last_shot:
            parts.append(STREAK_DIRECTIVE.format(streak=state.streak, n_flat=self.limits.n_flat, epsilon=self.limits.epsilon,
                                                 best=f"{state.best_primary:.4f}" if state.best_primary is not None else "n/a"))
        if state.consecutive_failures >= int(self.run_cfg.get("STALL_FAILURES", 3)):
            parts.append(STALL_DIRECTIVE.format(n=state.consecutive_failures))
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------ one iteration
    def run_iteration(self, it: int) -> Dict[str, Any]:
        state = self.state
        assert state is not None
        t_iter = self.clock()
        ws = self.iteration_dir(it)
        shutil.rmtree(ws, ignore_errors=True)
        os.makedirs(ws)
        self.roles.begin_iteration(it, os.path.join(ws, "llm"))
        best_at_start = state.best_primary
        champion_files = tools.read_code_files(self.best_code_dir)
        tools.write_code_files(ws, champion_files)          # fresh workspace seeded with the champion (copy, never reference)
        self.log(f"\n--- iteration {it} (best {best_at_start}, streak {state.streak}) ---")

        attempts: List[DebugAttempt] = []
        files = champion_files
        score: Optional[Score] = None
        status, error_reason, runtime_s, blocked_reason = "failed", "", 0.0, ""

        briefing = self.assemble_briefing(it)
        atomic_write_text(os.path.join(ws, "briefing.md"), briefing)
        plan, err, _raw = self.roles.researcher(briefing)
        if plan is None:
            error_reason = err
            self.log(f"[it{it:02d}] researcher failed: {one_line(err, 200)}")
        else:
            atomic_write_json(os.path.join(ws, "plan.json"), plan.to_dict())
            self.log(f"[it{it:02d}] HYP ({plan.category}, risk {plan.expected_risk}): {one_line(plan.hypothesis, 200)}")
            new_files, err = self.roles.engineer(plan, champion_files, PIPELINE_CONTRACT_NOTE.format(timeout=int(self.run_cfg["EXPERIMENT_TIMEOUT_S"])))
            if new_files is None:
                error_reason = err
                self.log(f"[it{it:02d}] engineer failed: {one_line(err, 200)}")
            else:
                files = {**champion_files, **new_files}
                status, score, error_reason, attempts, files, runtime_s, blocked_reason = self.execute_with_debugging(ws, plan, files)

        primary = score.primary if (score is not None and status == "scored") else None
        ablations: List[Dict[str, Any]] = []
        if status == "scored":
            try:
                with open(os.path.join(ws, "stdout.txt"), encoding="utf-8", errors="replace") as fh:
                    ablations = parse_ablations(fh.read())
            except OSError:
                ablations = []
            if ablations:
                self.log(f"[it{it:02d}] in-run ablations (pipeline-reported): {format_ablations(ablations, primary)}")
        result = HarnessResult(status=status, gauc=score.gauc if score else 0.0, ndcg5=score.ndcg5 if score else 0.0,
                               primary=score.primary if score else 0.0, runtime_s=runtime_s,
                               error_excerpt="" if status == "scored" else error_reason[-4000:],
                               vs_best=vs_best_string(primary, best_at_start))
        atomic_write_json(os.path.join(ws, "result.json"), result.to_dict())

        # --- leak test: a would-be promotion must survive flipped validation labels (harness-measured) ---
        leak: Dict[str, Any] = {}
        leak_mode = str(self.run_cfg.get("leak_check", "on_promotion"))
        if status == "scored" and score is not None and leak_mode != "off":
            would_promote = best_at_start is None or score.primary > best_at_start + self.limits.promote_margin
            improves = best_at_start is None or score.primary > best_at_start
            if leak_mode == "on_improvement":
                would_promote = would_promote or improves            # verify every improvement so none is ever lost
            ceiling = self.run_cfg.get("implausible_primary_above")
            if would_promote and ceiling is not None and score.primary > float(ceiling):
                # Free first line of defence: no honest model on this task reaches here (oracle 0.8484, baseline 0.6015,
                # best known lever +0.003). Flag without spending a re-run.
                leak = {"ran": False, "verdict": "LEAK", "reason": "implausible score ceiling",
                        "ceiling": float(ceiling), "primary": score.primary, "gauc": score.gauc}
            elif would_promote:
                self.log(f"[leak-test] candidate {score.primary:.4f} would be promoted — re-running with 10% of validation users' labels flipped")
                leak = self.task.leak_test(ws, float(self.run_cfg["EXPERIMENT_TIMEOUT_S"]))
            if leak:
                floor = float(self.run_cfg.get("leak_check_min_primary", 0.5))
                if leak.get("reason") == "implausible score ceiling":
                    pass                                             # verdict already set
                elif leak.get("ran") and leak.get("subset_primary") is not None:
                    leak["verdict"] = "clean" if leak["subset_primary"] >= floor else "LEAK"
                else:
                    leak["verdict"] = "INCONCLUSIVE (pipeline failed on the flipped-label copies)"
                if leak.get("reason") == "implausible score ceiling":
                    diag = (f"LEAK DETECTED (implausible score): sealed primary {score.primary:.4f} / GAUC {score.gauc:.4f} exceeds the "
                            f"plausibility ceiling {leak['ceiling']} for this task — the validation oracle (perfect ranking) is 0.8484, the "
                            f"baseline 0.6015 and the best known single lever +0.003. A score this high means the predictions are derived "
                            f"from the rows' own labels. Check every tuple index and feature path for the label.")
                    blocked_reason = "leak detected (implausible score)"
                elif leak["verdict"] == "LEAK":
                    diag = (f"LEAK DETECTED: sealed primary {score.primary:.4f} with real validation labels, but for the "
                            f"{int(round(leak['fraction'] * 100))}% of validation users whose feedback columns were flipped the "
                            f"predictions rank their TRUE labels at primary {leak['subset_primary']:.4f} (GAUC {leak['subset_gauc']:.4f}; "
                            f"random = 0.5, a clean pipeline scores them like everyone else). The predictions depend on the validation "
                            f"rows' own labels — a training/feature leak, not a result.")
                    blocked_reason = "leak detected"
                elif leak["verdict"] != "clean":
                    diag = (f"LEAK TEST INCONCLUSIVE: sealed primary {score.primary:.4f}, but the pipeline crashed on both flipped-label "
                            f"copies (10% and 2% of validation users), so the harness cannot verify that its predictions are independent "
                            f"of the validation labels; not promoted. Make the pipeline robust to partially corrupted validation labels "
                            f"(no strict assertions on the validation metric).\n{leak.get('error', '')}")
                    blocked_reason = "leak test inconclusive"
                if leak["verdict"] != "clean":
                    self.log(f"[leak-test] {diag[:200]}")
                    status, primary, error_reason = "failed", None, diag
                    result = HarnessResult(status="failed", gauc=score.gauc, ndcg5=score.ndcg5, primary=score.primary,
                                           runtime_s=runtime_s, error_excerpt=diag, vs_best=vs_best_string(None, best_at_start))
                    atomic_write_json(os.path.join(ws, "result.json"), result.to_dict())
                    score = None
                else:
                    self.log(f"[leak-test] clean: flipped users score {leak['subset_primary']:.4f} on their true labels "
                             f"(>= {floor}); full-set {leak['full_primary']:.4f}")
                    if score.primary > float((state.best_measured or {}).get("primary", -1)):
                        state.best_measured = {"iteration": it, "primary": score.primary, "gauc": score.gauc, "ndcg5": score.ndcg5,
                                               "workspace": os.path.relpath(ws, self.run_dir)}
            atomic_write_json(os.path.join(ws, "leak_test.json"), leak) if leak else None

        # --- the two separate judgments (pure functions; no LLM involvement) ---
        dec = judge_iteration(status, primary, best_at_start, state.best_iter, state.streak, it, self.limits)
        if dec.promoted:
            install_champion(self.run_dir, files, os.path.join(ws, tools.PREDS_VAL), score, it, ws)
            state.best_primary, state.best_gauc, state.best_ndcg5, state.best_iter = score.primary, score.gauc, score.ndcg5, it
        state.best_history.append(dec.best_primary_after)
        state.streak = dec.streak_after            # organizers' rule (kit README + baseline_scores.json): each
                                                   # iteration vs best-so-far; gains <= EPSILON and failures tick
        if status == "scored":
            state.consecutive_failures = 0
        else:
            state.consecutive_failures += 1
            if blocked_reason:
                state.blocked.append(f"it{it:02d}: {one_line(plan.hypothesis if plan else error_reason, 70)} [{blocked_reason}]")

        plan_for_scribe = plan or ResearcherPlan(hypothesis=f"(no valid plan: {one_line(error_reason, 120)})", category="other",
                                                 change_spec="n/a", expected_risk="high", rationale="n/a")
        train_tail = self.training_log_tail(ws)
        lesson = self.roles.scribe_lesson(plan_for_scribe, result, dec.decision, dec.best_primary_after, train_tail)
        diff_text, change_summary = tools.unified_diff(champion_files, files)
        if plan is None:
            change_summary = "none (no valid plan)"
        line = ledger_line(it, plan_for_scribe.hypothesis, change_summary, result_string(status, primary, error_reason),
                           dec.best_primary_after, dec.decision, lesson)
        append_ledger(self.run_dir, line)

        facts = {"iteration": it, "hypothesis": plan_for_scribe.hypothesis, "category": plan_for_scribe.category,
                 "result": result.to_dict(), "decision": dec.decision, "streak_after": dec.streak_after,
                 "best_primary_after": dec.best_primary_after, "best_iter_after": dec.best_iter_after,
                 "debug_attempts": [a.to_dict() for a in attempts], "change_summary": change_summary, "lesson": lesson,
                 "training_log_tail": train_tail}
        if self.cfg["llm"].get("scribe_narrative", True):
            write_iteration_narrative(self.run_dir, it, self.roles.scribe_logentry(facts))

        # Scribe job (c): research synthesis of the whole run so far, rebuilt from the harness digest every iteration
        # (this iteration included). Numbers are checked against the digest; a synthesis that invents one is dropped.
        hist_preview = {"iteration": it, "hypothesis": one_line(plan_for_scribe.hypothesis, 0), "category": plan_for_scribe.category,
                        "status": status, "primary": primary, "decision": dec.decision, "promoted": dec.promoted, "lesson": lesson,
                        "error_short": one_line(error_reason.splitlines()[-1] if error_reason else "", 200)}
        if self.cfg["llm"].get("scribe_digest", True):
            digest_now = research_digest(state.history + [hist_preview], lambda n: read_iteration_detail(self.run_dir, n) if n != it else
                                         {"hypothesis": plan_for_scribe.hypothesis, "harness_extra": {"best_at_iteration_start": best_at_start, "leak_test": leak or None,
                                                                                                        "expected_gain": plan_for_scribe.expected_gain, "ablations": ablations}})
            synth = self.roles.scribe_digest(digest_now)
            if synth and synthesis_numbers_ok(synth, digest_now):
                state.synthesis = synth
            elif synth:
                state.warnings.append(f"it{it:02d}: scribe synthesis rejected (contained a number not in the digest)")
                self.log(f"[scribe] synthesis rejected: it contained a number not present in the harness digest")
        usage = self.roles.iteration_usage
        state.tokens_total += usage.total
        state.tokens_input += usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
        state.tokens_output += usage.output_tokens
        if usage.cost_usd is not None:
            state.real_spend_usd = (state.real_spend_usd or 0.0) + usage.cost_usd
        for role, n in self.roles.iteration_role_usage.items():
            state.tokens_by_role[role] = state.tokens_by_role.get(role, 0) + n
        state.llm_calls += self.roles.calls_this_iteration

        entry = IterationLog(iteration=it, timestamp=utc_now_iso(), hypothesis=plan_for_scribe.hypothesis,
                             rationale=plan_for_scribe.rationale, category=plan_for_scribe.category, code_diff=diff_text,
                             result=result.to_dict(), errors_and_recovery=[a.to_dict() for a in attempts], decision=dec.decision,
                             streak_after=dec.streak_after, tokens_this_iteration=usage.total, runtime_s=runtime_s, lesson=lesson,
                             harness_extra={"run_id": state.run_id, "best_primary_after": dec.best_primary_after,
                                            "best_iter_after": dec.best_iter_after, "best_at_iteration_start": best_at_start,
                                            "promoted": dec.promoted, "expected_risk": plan_for_scribe.expected_risk,
                                            "change_summary": change_summary, "tokens": usage.to_dict(),
                                            "llm_calls": self.roles.calls_this_iteration, "iteration_wall_s": round(self.clock() - t_iter, 1),
                                            "blocked_added": blocked_reason or None, "consecutive_failures": state.consecutive_failures,
                                            "leak_test": leak or None,
                                            "expected_gain": plan_for_scribe.expected_gain, "gain_evidence": plan_for_scribe.gain_evidence,
                                            "ablation_plan": plan_for_scribe.ablation_plan, "ablations": ablations,
                                            "workspace": os.path.relpath(ws, self.run_dir)})
        write_iteration_log(self.run_dir, entry)

        hist = {"iteration": it, "hypothesis": one_line(plan_for_scribe.hypothesis, 0), "category": plan_for_scribe.category,
                "training_log_tail": train_tail,
                "status": status, "primary": primary, "gauc": score.gauc if (score and status == "scored") else None,
                "ndcg5": score.ndcg5 if (score and status == "scored") else None, "decision": dec.decision, "promoted": dec.promoted,
                "streak_after": dec.streak_after, "tokens": usage.total, "runtime_s": round(runtime_s, 1), "lesson": lesson,
                "error_short": one_line(error_reason.splitlines()[-1] if error_reason else "", 200), "timestamp": utc_now_iso()}
        state.history.append(hist)
        state.iteration = it
        write_state_block(self.run_dir, render_state_block(state, self._limits_dict(), it, self.clock()))
        save_run_state(self.run_dir, state)                       # crash-resume point
        self.log(f"[it{it:02d}] {status} primary={primary} -> {dec.decision} | best {dec.best_primary_after} (it{dec.best_iter_after:02d}) "
                 f"| streak {dec.streak_after}/{self.limits.n_flat} | tokens {usage.total} | {runtime_s:.0f}s")
        return hist

    def render_recent_iterations(self, state: RunState) -> str:
        """Full record of the most recent iterations: what was proposed (hypothesis + change spec + rationale), what
        the Engineer actually changed (diff), what was measured (delta vs the champion at the time), what went wrong
        (debug attempts, leak verdict) and the training curve. Everything here is harness-measured or previously
        LLM-authored; the point is that the Researcher can judge WHICH PART of a bundled change worked."""
        n = int(self.run_cfg.get("briefing_recent_iterations", 5))
        diff_chars = int(self.run_cfg.get("briefing_diff_chars", 2500))
        spec_chars = int(self.run_cfg.get("briefing_spec_chars", 1200))
        recent = state.history[-n:]
        if not recent:
            return ""
        out = ["# RECENT ITERATION DETAILS (harness-measured facts + what was actually changed)",
               "Use these to decide whether to CONTINUE an idea: when a bundled change moved little, the diff shows which",
               "components were in it, so you can keep the part that plausibly worked and drop the rest. State which",
               "component you are keeping or dropping, and why, in `rationale`."]
        for h in recent:
            it = h["iteration"]
            d = read_iteration_detail(self.run_dir, it) or {}
            x = d.get("harness_extra", {})
            r = d.get("result", {})
            best_before = x.get("best_at_iteration_start")
            delta = (f"{r.get('primary', 0) - best_before:+.4f} vs the then-champion {best_before:.4f}"
                     if (best_before is not None and r.get("status") == "scored") else "n/a")
            out.append(f"\n## it{it:02d} [{h.get('category')}] — {h.get('decision')} ({r.get('status')}), {delta}")
            out.append(f"HYPOTHESIS: {d.get('hypothesis') or h.get('hypothesis')}")
            if isinstance(x.get("expected_gain"), (int, float)):
                measured = (f"; measured {r.get('primary') - best_before:+.4f}" if (best_before is not None and r.get("status") == "scored") else "")
                out.append(f"YOUR PREDICTED GAIN: {x['expected_gain']:+.4f}{measured}" + (f" — evidence given: {one_line(x.get('gain_evidence', ''), 300)}" if x.get("gain_evidence") else ""))
            if d.get("rationale"):
                out.append(f"RATIONALE (yours, at the time): {one_line(d['rationale'], 600)}")
            plan_path = os.path.join(self.iteration_dir(it), "plan.json")
            if os.path.exists(plan_path):
                try:
                    spec = json.load(open(plan_path)).get("change_spec", "")
                    if spec:
                        out.append(f"CHANGE SPEC you gave the Engineer:\n{spec[:spec_chars]}" + ("…" if len(spec) > spec_chars else ""))
                except (OSError, ValueError):
                    pass
            out.append(f"WHAT CHANGED: {x.get('change_summary', 'n/a')}")
            diff = d.get("code_diff") or ""
            if diff:
                out.append("DIFF (champion -> attempt):\n```diff\n" + diff[:diff_chars] + ("\n… (diff truncated)" if len(diff) > diff_chars else "") + "\n```")
            if r.get("status") == "scored":
                out.append(f"MEASURED: primary {r.get('primary'):.4f} (GAUC {r.get('gauc'):.4f} / nDCG@5 {r.get('ndcg5'):.4f}), runtime {r.get('runtime_s')}s")
                if x.get("ablations"):
                    out.append("IN-RUN ABLATIONS (pipeline-reported on validation, unsealed — component attribution): " +
                               format_ablations(x["ablations"], r.get("primary")))
                elif x.get("ablation_plan"):
                    out.append("IN-RUN ABLATIONS: none printed although an ablation plan was given — attribution for this iteration is missing.")
            else:
                out.append(f"OUTCOME: {r.get('status')} — {one_line(r.get('error_excerpt', ''), 400)}")
            for a in d.get("errors_and_recovery", []):
                out.append(f"  debug attempt {a['attempt']}: {one_line(a.get('error'), 160)} -> fix: {one_line(a.get('fix_summary'), 160)} ({a.get('status_after')})")
            lt = x.get("leak_test")
            if lt:
                out.append(f"  leak test: {lt.get('verdict')}" + (f" (flipped users scored {lt['subset_primary']:.4f} on their true labels)"
                                                                  if lt.get("subset_primary") is not None else ""))
            if h.get("training_log_tail"):
                out.append("TRAINING CURVE (the experiment's own stdout):\n" + "\n".join("  " + l for l in h["training_log_tail"].splitlines()))
            out.append(f"LESSON: {h.get('lesson', '')}")
        return "\n".join(out)

    def training_log_tail(self, ws: str, n_lines: int = 12, max_chars: int = 1500) -> str:
        """Last lines of the experiment's stdout (epoch curve, early-stop message) — lets the Scribe and the
        Researcher see HOW a run behaved, not just its final score. Empty when nothing ran."""
        p = os.path.join(ws, "stdout.txt")
        if not os.path.exists(p):
            return ""
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                lines = [l.rstrip() for l in fh.read().splitlines() if l.strip()]
        except OSError:
            return ""
        # drop repeated boilerplate (e.g. "Number of pairs this epoch: 765158" printed every epoch), keeping each
        # distinct line's LAST occurrence, so the line budget is spent on the metric curve rather than on repeats
        seen, kept = set(), []
        for l in reversed(lines):
            if l in seen:
                continue
            seen.add(l)
            kept.append(l)
        dedup = list(reversed(kept))
        tail = "\n".join(dedup[-n_lines:])
        return tail[-max_chars:]

    # ------------------------------------------------------------------ run + debug loop
    def execute_with_debugging(self, ws: str, plan: ResearcherPlan, files: Dict[str, str]):
        """Run the experiment; on failure ask the Debugger up to DEBUG_RETRIES times (retries never
        consume iterations). Returns (status, score, error_reason, attempts, files, total_runtime, blocked_reason)."""
        max_retries = int(self.run_cfg["DEBUG_RETRIES"])
        timeout = float(self.run_cfg["EXPERIMENT_TIMEOUT_S"])
        retry_timeouts = bool(self.run_cfg.get("retry_timeouts_with_debugger", False))
        attempts: List[DebugAttempt] = []
        attempt_no, total_runtime = 0, 0.0
        score: Optional[Score] = None
        status, err, blocked = "failed", "", ""
        while True:
            violations = static_code_check(files, self.cfg.get("sandbox", {}))
            if violations:
                status, err = "failed", "policy violation (code was NOT executed):\n" + "\n".join(violations)
            else:
                tools.write_code_files(ws, files)
                res = self.task.sandbox_run(ws, "val", tools.PREDS_VAL, timeout)
                total_runtime += res.runtime_s
                if res.status == "timeout":
                    status = "timeout"
                    err = (res.error_excerpt() + "\n\nRUNTIME DIAGNOSIS (harness): the process was killed at the "
                           f"{int(timeout)}s limit. The champion pipeline runs in ~130s, so this is almost always a "
                           "performance bug, not a slow model: a Python loop over users x rows (per-user masks, "
                           "per-user list comprehensions), pairs rebuilt from scratch every epoch, or an O(n^2) "
                           "post-processing step. Vectorise per-user operations (pandas groupby + rank/transform, "
                           "np.argsort/np.unique with return_inverse), build index structures once, keep the model "
                           "and hypothesis unchanged.\nLast lines of stdout before the kill:\n" + self.training_log_tail(ws))
                elif res.status == "failed":
                    status, err = "failed", res.error_excerpt()
                else:
                    try:
                        score = self.task.score_preds(os.path.join(ws, tools.PREDS_VAL))
                        status, err = "scored", ""
                    except (ValueError, OSError) as e:
                        status, err = "failed", f"prediction file rejected by the sealed checker: {e}"
            if attempts and not attempts[-1].status_after:
                attempts[-1].status_after = status
            if status == "scored":
                # Plausibility guard (harness-measured): a GAUC below the configured floor means the predicted order is
                # INVERTED relative to the labels (random = 0.5) — almost always a sign error, not a research result.
                # Give the Debugger one shot at it; if it abandons, the measured (implausible) score stands.
                floor = self.run_cfg.get("implausible_gauc_below")
                if (floor is not None and score is not None and score.gauc < float(floor) and attempt_no < max_retries
                        and not any(a.fix_summary.startswith("IMPLAUSIBLE") for a in attempts)):
                    diag = (f"IMPLAUSIBLE RESULT (the code ran and was scored, but the ranking is inverted): sealed GAUC "
                            f"{score.gauc:.4f} < {float(floor)} — a random ranking scores 0.5 and the item-popularity rung 0.64, "
                            f"so positives are being ranked BELOW negatives. Typical causes: a sign error in the loss/gradient "
                            f"(e.g. updating with +g where -g is needed, or a pairwise diff computed as neg - pos), predictions "
                            f"written as -score, or labels/probabilities flipped. Keep the hypothesis; fix the implementation.\n"
                            f"Training log tail:\n{self.training_log_tail(ws)}")
                    self.log(f"[debug] implausible score (GAUC {score.gauc:.4f} < {floor}); asking the Debugger for a sign/logic fix")
                    attempt_no += 1
                    self._archive_attempt(ws, attempt_no, files)
                    fix = self.roles.debugger(plan, files, diag, attempt_no)
                    if fix.action == "abandon":
                        attempts.append(DebugAttempt(attempt=attempt_no, error=one_line(diag.splitlines()[0], 200),
                                                     fix_summary=f"IMPLAUSIBLE, debugger abandoned: {fix.reason}", status_after="scored (implausible)"))
                        break
                    prev_files, prev_score = files, score
                    files = {**files, **fix.files}
                    attempts.append(DebugAttempt(attempt=attempt_no, error=one_line(diag.splitlines()[0], 200),
                                                 fix_summary=f"IMPLAUSIBLE -> {fix.fix_summary}"))
                    tools.write_code_files(ws, files)
                    res2 = self.task.sandbox_run(ws, "val", tools.PREDS_VAL, timeout)
                    total_runtime += res2.runtime_s
                    score2 = None
                    if res2.ok:
                        try:
                            score2 = self.task.score_preds(os.path.join(ws, tools.PREDS_VAL))
                        except (ValueError, OSError):
                            score2 = None
                    if score2 is not None and score2.gauc >= float(floor):
                        attempts[-1].status_after = "scored"
                        score, files = score2, files
                        self.log(f"[debug] fix restored a plausible ranking: GAUC {score2.gauc:.4f}, primary {score2.primary:.4f}")
                    else:                                            # the fix did not help: keep the measured original
                        attempts[-1].status_after = f"{res2.status if not res2.ok else 'scored'} (still implausible); original kept"
                        files, score = prev_files, prev_score
                        tools.write_code_files(ws, files)
                        self.log("[debug] fix did not restore a plausible ranking; keeping the original measured result")
                break
            self.log(f"[debug] attempt {attempt_no} -> {status}: {one_line(err.splitlines()[-1] if err else '', 160)}")
            if status == "timeout" and not retry_timeouts:
                blocked = f"timeout {int(timeout)}s"
                break
            if attempt_no >= max_retries:
                err += f"\n[debugger retries exhausted ({max_retries})]"
                blocked = f"failed after {max_retries} debug attempts"
                break
            attempt_no += 1
            self._archive_attempt(ws, attempt_no, files)
            fix = self.roles.debugger(plan, files, err, attempt_no)
            short_err = one_line(err.splitlines()[-1] if err else "", 200)
            if fix.action == "abandon":
                attempts.append(DebugAttempt(attempt=attempt_no, error=short_err, fix_summary=f"ABANDONED: {fix.reason}", status_after="abandoned"))
                err += f"\n[debugger abandoned: {fix.reason}]"
                blocked = "abandoned by debugger"
                break
            files = {**files, **fix.files}
            attempts.append(DebugAttempt(attempt=attempt_no, error=short_err, fix_summary=fix.fix_summary))
        return status, score, err, attempts, files, total_runtime, blocked

    def _archive_attempt(self, ws: str, attempt_no: int, files: Dict[str, str]) -> None:
        d = os.path.join(ws, "attempts", f"a{attempt_no}")
        os.makedirs(d, exist_ok=True)
        tools.write_code_files(d, files)
        for f in ("stdout.txt", "stderr.txt"):
            p = os.path.join(ws, f)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(d, f))

    # ------------------------------------------------------------------ finalize
    def finalize(self, reason: str) -> RunState:
        state = self.state
        assert state is not None
        state.stop_reason = reason
        from .llm_client import provider_balance
        state.spend_end = provider_balance(self.cfg) or {}
        save_run_state(self.run_dir, state)
        self.log(f"\n[finalize] stop reason: {reason} — generating submission from the champion (it{state.best_iter:02d})")
        fin_dir = os.path.join(self.run_dir, "finalize")
        os.makedirs(fin_dir, exist_ok=True)
        candidates: List[Tuple[int, str]] = []
        bm = state.best_measured or {}
        if bm and bm.get("primary", -1) > (state.best_primary or -1) + 1e-12 and bm.get("iteration") != state.best_iter:
            self.log(f"[finalize] best leak-clean measurement it{bm['iteration']:02d} ({bm['primary']:.4f}) beats the champion "
                     f"({state.best_primary:.4f}) — it was below the promotion margin; using it first")
            candidates.append((int(bm["iteration"]), self.iteration_dir(int(bm["iteration"]))))
        candidates.append((state.best_iter, self.best_code_dir))
        for h in reversed(state.history):
            if h.get("promoted") and h["iteration"] != state.best_iter and (h["iteration"], self.iteration_dir(h["iteration"])) not in candidates:
                candidates.append((h["iteration"], self.iteration_dir(h["iteration"])))
        if state.best_iter != 0:
            candidates.append((0, self.task.champion_src_dir))
        record: Dict[str, Any] = {"stop_reason": reason, "attempts": [], "ok": False}
        submission = os.path.join(self.run_dir, "submission.csv")
        for it, code_dir in candidates:
            ws = os.path.join(fin_dir, f"champion_it{it:02d}")
            shutil.rmtree(ws, ignore_errors=True)
            tools.write_code_files(ws, tools.read_code_files(code_dir))
            res = self.task.sandbox_run(ws, "test", tools.PREDS_TEST, float(self.run_cfg.get("FINALIZE_TIMEOUT_S", 1800)), full_data=True)
            att: Dict[str, Any] = {"iteration": it, "run_status": res.status, "runtime_s": round(res.runtime_s, 1)}
            if not res.ok:
                att["error"] = res.error_excerpt(20)
                record["attempts"].append(att)
                self.log(f"[finalize] champion it{it:02d} failed on --split test ({res.status}); trying the previous champion")
                continue
            preds = os.path.join(ws, tools.PREDS_TEST)
            ok, out = self.task.check_submission(preds, "test")
            att["checker_ok"], att["checker_output"] = ok, out[-1500:]
            record["attempts"].append(att)
            if ok:
                shutil.copy2(preds, submission)
                record.update({"ok": True, "submission": submission, "champion_iteration": it})
                self.log(f"[finalize] submission.csv written from it{it:02d}; sealed checker: {one_line(out, 120)}")
                break
            self.log(f"[finalize] checker REJECTED it{it:02d} predictions: {one_line(out, 200)}")
        state.finalize = record
        save_run_state(self.run_dir, state)
        self.write_results_summary()
        self._banner(reason)
        if not record["ok"]:
            raise FinalizeError("no champion produced a submission that passes the sealed checker; see run_state.json['finalize']")
        return state

    def write_results_summary(self) -> str:
        state = self.state
        assert state is not None
        b = state.baseline_primary
        delta = (state.best_primary - b) if (b is not None and state.best_primary is not None) else None
        promoted = [h for h in state.history if h.get("promoted")]
        failed = [h for h in state.history if h["status"] != "scored"]
        lines = [f"# Results summary — {state.run_id}", "",
                 f"- stop reason: **{state.stop_reason}**",
                 f"- best validation: **primary {state.best_primary:.4f}** (GAUC {state.best_gauc:.4f} / nDCG@5 {state.best_ndcg5:.4f}) at it{state.best_iter:02d}"
                 if state.best_primary is not None else "- best validation: n/a",
                 f"- published baseline (valid): {b} → delta **{delta:+.4f}**" if delta is not None else "- published baseline: n/a",
                 f"- iterations used: {state.iteration} (promoted {len(promoted)}, failed {len(failed)}); final streak {state.streak}",
                 (f"- best leak-clean measurement: it{state.best_measured['iteration']:02d} at {state.best_measured['primary']:.4f}"
                  + (" (below the promotion margin; used for the submission)" if state.best_measured['primary'] > (state.best_primary or -1) + 1e-12 else "")
                  if state.best_measured else "- best leak-clean measurement: n/a"),
                 f"- tokens: {state.tokens_total:,} total ({state.tokens_input:,} in / {state.tokens_output:,} out) over {state.llm_calls} LLM calls; by role: {json.dumps(state.tokens_by_role)}",
                 f"- wall-clock: {fmt_elapsed(state.elapsed_s(self.clock()))} (started {state.start_ts})",
                 f"- manual interventions: {state.interventions} (resumes {state.resumes}) — see interventions.md",
                 self._spend_line(),
                 f"- submission: {'OK — ' + os.path.basename(state.finalize.get('submission', '')) + ' from it%02d' % state.finalize.get('champion_iteration', -1) if state.finalize.get('ok') else 'FAILED'} (sealed checker)",
                 f"- phase 0: random {state.phase0.get('random', {}).get('primary')}, pop {state.phase0.get('pop', {}).get('primary')}, "
                 f"official FM {state.phase0.get('official_fm', {}).get('primary')}, champion it00 {state.phase0.get('champion', {}).get('primary')}",
                 "", "## Promotions", ""]
        lines += [f"- it{h['iteration']:02d}: {h['primary']:.4f} — {h['hypothesis']}" for h in promoted] or ["(none beyond the baseline champion)"]
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in state.warnings]
        text = "\n".join(lines) + "\n"
        atomic_write_text(os.path.join(self.run_dir, "results_summary.md"), text)
        return text

    def _current_spend_usd(self) -> Optional[float]:
        """Real dollars this run has consumed so far (start snapshot vs. a fresh balance check), used only
        as a stop-condition input — never raises, returns None if the provider has no credit endpoint."""
        from .llm_client import provider_balance
        s0 = self.state.spend_start or {}
        s1 = provider_balance(self.cfg) or {}
        if s0.get("usage_usd") is None or s1.get("usage_usd") is None:
            return None
        return s1["usage_usd"] - s0["usage_usd"]

    def _spend_line(self) -> str:
        """Provider credit actually consumed by this run. Prefers the sum of real per-call costs (OpenRouter
        `usage.cost`), which stays correct even when the API key is shared with teammates; the account-wide
        balance diff (start snapshot vs. end snapshot) is shown alongside as context only, since on a shared
        key it reflects everyone's activity in the same time window, not just this run's."""
        s0, s1 = (self.state.spend_start or {}), (self.state.spend_end or {})
        acct_ok = s0.get("usage_usd") is not None and s1.get("usage_usd") is not None
        acct_delta = (s1["usage_usd"] - s0["usage_usd"]) if acct_ok else None
        real = self.state.real_spend_usd
        if real is not None:
            line = f"- **provider spend: ${real:.4f}** this run (real per-call cost, sums {json.dumps(self._models())})"
            if acct_ok:
                line += f" — account balance moved ${acct_delta:.4f} in the same window (shared key: may include other activity)"
            return line
        if acct_ok:
            return (f"- provider spend: not tracked per call for this provider; account balance moved ${acct_delta:.4f} "
                    f"this run's window (account total ${s1['usage_usd']:.2f}, remaining ${s1.get('remaining_usd', float('nan')):.2f}) "
                    f"— on a shared key this may include other activity, not only this run")
        return f"- provider spend: not reported by this provider (models: {json.dumps(self._models())})"

    def _models(self) -> Dict[str, str]:
        llm = self.cfg["llm"]
        return {r: llm[f"{r}_model"] for r in ("researcher", "engineer", "debugger", "scribe")}

    def _banner(self, reason: str) -> None:
        s = self.state
        assert s is not None
        self.log("\n" + "=" * 78)
        self.log(f"RUN {s.run_id} FINISHED — stop reason: {reason.upper()}")
        if s.best_primary is not None:
            self.log(f"best val primary {s.best_primary:.4f} (it{s.best_iter:02d}) vs baseline {s.baseline_primary} | iterations {s.iteration} | "
                     f"tokens {s.tokens_total:,} | {fmt_elapsed(s.elapsed_s(self.clock()))} | interventions {s.interventions}")
        self.log(self._spend_line().replace("- ", "").replace("**", ""))
        self.log(f"submission: {'OK' if s.finalize.get('ok') else 'FAILED'} | run dir: {self.run_dir}")
        self.log("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build(cfg: Dict[str, Any], root: str, run_dir: str, *, toy: bool = False, mock: bool = False, clock=time.time, log=print,
          mock_handlers=None) -> Harness:
    task = make_task(cfg, root, toy=toy)
    if mock and mock_handlers is None and not toy:
        cfg.setdefault("llm", {})["mock_plan"] = "kuairand"      # offline plan of real FM edits for real-data dry runs
    client = make_client(cfg, mock_handlers=mock_handlers, force_mock=mock)
    os.makedirs(run_dir, exist_ok=True)
    if hasattr(client, "progress"):
        client.progress = log                                     # streaming heartbeat + fallback notices in the console
    roles = Roles(client, cfg, os.path.join(root, cfg["paths"]["prompts"]), os.path.join(root, cfg["paths"]["knowledge"]),
                  call_log=CallLog(os.path.join(run_dir, "llm_calls.jsonl")))
    return Harness(cfg, root, run_dir, task, roles, clock=clock, log=log)


def new_run_dir(runs_dir: str, label: str = "") -> str:
    rid = time.strftime("%Y%m%d_%H%M%S") + (f"_{label}" if label else "")
    p = os.path.join(runs_dir, rid)
    os.makedirs(p, exist_ok=False)
    with open(os.path.join(runs_dir, "LATEST"), "w") as fh:
        fh.write(p + "\n")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Autonomous ML research agent harness")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-dir", default=None, help="resume this run directory (default: start a new one)")
    ap.add_argument("--label", default="", help="suffix for a new run id")
    ap.add_argument("--mock", action="store_true", help="offline deterministic LLM roles (no API key needed)")
    ap.add_argument("--toy", action="store_true", help="toy task (mini synthetic dataset + dummy pipeline)")
    ap.add_argument("--max-iters", type=int, default=None, help="override run.MAX_ITERS")
    ap.add_argument("--session-iters", type=int, default=None, help="stop this process after N iterations (resumable)")
    ap.add_argument("--set", action="append", default=[], help="override config: a.b.c=value")
    ap.add_argument("--phase0-only", action="store_true", help="run Phase 0 (baseline reproduction + self-checks) and exit")
    ap.add_argument("--env-file", default=None, help="KEY=VALUE file to load into the environment (default: <repo>/.env if present)")
    ap.add_argument("--llm-profile", default=None, help="apply llm.profiles.<name> from config (e.g. poe)")
    ap.add_argument("--llm-check", action="store_true", help="ping every configured role model with a tiny request and exit")
    ap.add_argument("--llm-usage", action="store_true", help="print the provider's credit usage/remaining and exit")
    ap.add_argument("--llm-list-models", nargs="?", const="", default=None, metavar="FILTER",
                    help="list the model ids the configured provider serves (optionally filtered) and exit")
    a = ap.parse_args(argv)

    root = os.path.dirname(os.path.abspath(a.config))
    from .llm_client import load_dotenv
    env_file = a.env_file or os.path.join(root, ".env")
    loaded = load_dotenv(env_file)
    if os.path.isfile(env_file):
        os.environ["HARNESS_ENV_FILE"] = os.path.abspath(env_file)   # the sandbox denies reads of this file
    if loaded:
        print(f"[env] loaded {', '.join(loaded)} from {os.path.relpath(env_file, root)} (values never logged)")
    cfg = load_config(a.config)
    if a.toy:
        deep_update(cfg, cfg.get("toy", {}).get("overrides", {}))
    if a.llm_profile:
        profiles = cfg.get("llm", {}).get("profiles", {}) or {}
        if a.llm_profile not in profiles:
            print(f"unknown llm profile {a.llm_profile!r}; available: {sorted(profiles)}", file=sys.stderr)
            return 2
        deep_update(cfg["llm"], copy.deepcopy(profiles[a.llm_profile]))
    if a.max_iters is not None:
        cfg["run"]["MAX_ITERS"] = a.max_iters
    for s in a.set:
        k, v = s.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(v)
    if a.llm_list_models is not None:
        from .llm_client import LLMError as _E, list_models
        try:
            for m in list_models(cfg, a.llm_list_models):
                print(m)
        except _E as e:
            print(f"could not list models: {e}", file=sys.stderr)
            return 2
        return 0
    if a.llm_usage:
        from .llm_client import provider_balance
        b = provider_balance(cfg)
        print(json.dumps(b, indent=1) if b else f"provider {cfg['llm'].get('provider')} reports no credit endpoint")
        return 0
    if a.llm_check:
        from .llm_client import connectivity_check
        results = connectivity_check(cfg)
        for r in results:
            print(json.dumps(r))
        ok = all(r["ok"] for r in results)
        print("LLM CHECK " + ("PASSED" if ok else "FAILED") + f" (provider={cfg['llm'].get('provider')})")
        return 0 if ok else 2
    runs_dir = os.path.join(root, cfg["paths"]["runs"])
    os.makedirs(runs_dir, exist_ok=True)
    run_dir = os.path.abspath(a.run_dir) if a.run_dir else new_run_dir(runs_dir, a.label or ("toy" if a.toy else ""))
    h = build(cfg, root, run_dir, toy=a.toy, mock=a.mock)
    try:
        if a.phase0_only:
            st = h.init_or_resume()
            if st.iteration == 0 and not st.phase0.get("passed"):
                h.phase0()
            print(json.dumps({k: v for k, v in st.phase0.items() if k != "expected"}, indent=1, default=str))
            return 0
        h.run(session_iteration_limit=a.session_iters)
    except (Phase0Error, FinalizeError) as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        st = load_run_state(run_dir)
        done = st.iteration if st else 0
        print(f"\nINTERRUPTED — {done} iteration(s) are safely on disk in {run_dir}.\n"
              f"  resume (keeps Phase 0 and all iterations; the restart is logged as an intervention):\n"
              f"    {os.path.basename(sys.executable)} -m agent.harness --run-dir {run_dir}\n"
              f"  or discard it:  rm -rf {run_dir}", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
