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
from .memory import (append_intervention, append_ledger, fmt_elapsed, init_interventions, init_ledger, ledger_line, load_run_state,
                     one_line, read_ledger, render_state_block, result_string, save_run_state, write_iteration_log,
                     write_iteration_narrative, write_state_block)
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
- Hard wall-clock limit: {timeout}s for the whole run (load + train + predict)."""

STALL_DIRECTIVE = """# STALL RECOVERY DIRECTIVE (injected by the harness)
The last {n} iterations ALL failed (crash / timeout / rejected output). Do NOT propose anything ambitious now.
Propose the SIMPLEST, most reliable change to the champion that is still a real hypothesis (e.g. one
hyperparameter, one well-understood feature, a smaller model variant). Requirements: no new libraries, no
new files, runtime well under the limit, minimal diff. State in `rationale` why it cannot fail the same way."""


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
            reason = stop_reason(state.streak, state.iteration, state.elapsed_s(self.clock()), state.tokens_total, self.limits)
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
        recent = [h for h in state.history[-3:]]
        if recent:
            lines = ["# RECENT ITERATION DETAILS"]
            for h in recent:
                lines.append(f"- it{h['iteration']:02d} [{h['category']}] {h['hypothesis']}\n  result: {h['status']} primary={h.get('primary')} "
                             f"decision={h['decision']} runtime={h.get('runtime_s')}s\n  lesson: {h.get('lesson', '')}")
                if h.get("error_short"):
                    lines.append(f"  error: {h['error_short']}")
            parts.append("\n".join(lines))
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
        result = HarnessResult(status=status, gauc=score.gauc if score else 0.0, ndcg5=score.ndcg5 if score else 0.0,
                               primary=score.primary if score else 0.0, runtime_s=runtime_s,
                               error_excerpt="" if status == "scored" else error_reason[-4000:],
                               vs_best=vs_best_string(primary, best_at_start))
        atomic_write_json(os.path.join(ws, "result.json"), result.to_dict())

        # --- the two separate judgments (pure functions; no LLM involvement) ---
        dec = judge_iteration(status, primary, best_at_start, state.best_iter, state.streak, it, self.limits)
        if dec.promoted:
            install_champion(self.run_dir, files, os.path.join(ws, tools.PREDS_VAL), score, it, ws)
            state.best_primary, state.best_gauc, state.best_ndcg5, state.best_iter = score.primary, score.gauc, score.ndcg5, it
        state.streak = dec.streak_after
        if status == "scored":
            state.consecutive_failures = 0
        else:
            state.consecutive_failures += 1
            if blocked_reason:
                state.blocked.append(f"it{it:02d}: {one_line(plan.hypothesis if plan else error_reason, 70)} [{blocked_reason}]")

        plan_for_scribe = plan or ResearcherPlan(hypothesis=f"(no valid plan: {one_line(error_reason, 120)})", category="other",
                                                 change_spec="n/a", expected_risk="high", rationale="n/a")
        lesson = self.roles.scribe_lesson(plan_for_scribe, result, dec.decision, dec.best_primary_after)
        diff_text, change_summary = tools.unified_diff(champion_files, files)
        if plan is None:
            change_summary = "none (no valid plan)"
        line = ledger_line(it, plan_for_scribe.hypothesis, change_summary, result_string(status, primary, error_reason),
                           dec.best_primary_after, dec.decision, lesson)
        append_ledger(self.run_dir, line)

        facts = {"iteration": it, "hypothesis": plan_for_scribe.hypothesis, "category": plan_for_scribe.category,
                 "result": result.to_dict(), "decision": dec.decision, "streak_after": dec.streak_after,
                 "best_primary_after": dec.best_primary_after, "best_iter_after": dec.best_iter_after,
                 "debug_attempts": [a.to_dict() for a in attempts], "change_summary": change_summary, "lesson": lesson}
        if self.cfg["llm"].get("scribe_narrative", True):
            write_iteration_narrative(self.run_dir, it, self.roles.scribe_logentry(facts))

        usage = self.roles.iteration_usage
        state.tokens_total += usage.total
        state.tokens_input += usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
        state.tokens_output += usage.output_tokens
        for role, n in self.roles.role_usage.items():
            state.tokens_by_role[role] = n
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
                                            "workspace": os.path.relpath(ws, self.run_dir)})
        write_iteration_log(self.run_dir, entry)

        hist = {"iteration": it, "hypothesis": one_line(plan_for_scribe.hypothesis, 160), "category": plan_for_scribe.category,
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
                    status, err = "timeout", res.error_excerpt()
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
        save_run_state(self.run_dir, state)
        self.log(f"\n[finalize] stop reason: {reason} — generating submission from the champion (it{state.best_iter:02d})")
        fin_dir = os.path.join(self.run_dir, "finalize")
        os.makedirs(fin_dir, exist_ok=True)
        candidates: List[Tuple[int, str]] = [(state.best_iter, self.best_code_dir)]
        for h in reversed(state.history):
            if h.get("promoted") and h["iteration"] != state.best_iter:
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
                 f"- tokens: {state.tokens_total:,} total ({state.tokens_input:,} in / {state.tokens_output:,} out) over {state.llm_calls} LLM calls; by role: {json.dumps(state.tokens_by_role)}",
                 f"- wall-clock: {fmt_elapsed(state.elapsed_s(self.clock()))} (started {state.start_ts})",
                 f"- manual interventions: {state.interventions} (resumes {state.resumes}) — see interventions.md",
                 f"- submission: {'OK — ' + os.path.basename(state.finalize.get('submission', '')) + ' from it%02d' % state.finalize.get('champion_iteration', -1) if state.finalize.get('ok') else 'FAILED'} (sealed checker)",
                 f"- phase 0: random {state.phase0.get('random', {}).get('primary')}, pop {state.phase0.get('pop', {}).get('primary')}, "
                 f"official FM {state.phase0.get('official_fm', {}).get('primary')}, champion it00 {state.phase0.get('champion', {}).get('primary')}",
                 "", "## Promotions", ""]
        lines += [f"- it{h['iteration']:02d}: {h['primary']:.4f} — {h['hypothesis']}" for h in promoted] or ["(none beyond the baseline champion)"]
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in state.warnings]
        text = "\n".join(lines) + "\n"
        atomic_write_text(os.path.join(self.run_dir, "results_summary.md"), text)
        return text

    def _banner(self, reason: str) -> None:
        s = self.state
        assert s is not None
        self.log("\n" + "=" * 78)
        self.log(f"RUN {s.run_id} FINISHED — stop reason: {reason.upper()}")
        if s.best_primary is not None:
            self.log(f"best val primary {s.best_primary:.4f} (it{s.best_iter:02d}) vs baseline {s.baseline_primary} | iterations {s.iteration} | "
                     f"tokens {s.tokens_total:,} | {fmt_elapsed(s.elapsed_s(self.clock()))} | interventions {s.interventions}")
        self.log(f"submission: {'OK' if s.finalize.get('ok') else 'FAILED'} | run dir: {self.run_dir}")
        self.log("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build(cfg: Dict[str, Any], root: str, run_dir: str, *, toy: bool = False, mock: bool = False, clock=time.time, log=print,
          mock_handlers=None) -> Harness:
    task = make_task(cfg, root, toy=toy)
    client = make_client(cfg, mock_handlers=mock_handlers, force_mock=mock)
    os.makedirs(run_dir, exist_ok=True)
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
    a = ap.parse_args(argv)

    root = os.path.dirname(os.path.abspath(a.config))
    cfg = load_config(a.config)
    if a.toy:
        deep_update(cfg, cfg.get("toy", {}).get("overrides", {}))
    if a.max_iters is not None:
        cfg["run"]["MAX_ITERS"] = a.max_iters
    for s in a.set:
        k, v = s.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(v)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
