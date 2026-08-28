# IMPLEMENTATION SPEC — Autonomous ML Research Agent (TechJam 2026, Track 2: KuaiRand)

You (Claude Code) are implementing a complete, runnable hackathon project. Read this
entire document before writing any code. Where this spec conflicts with the actual
starter kit contents, the starter kit wins — flag the discrepancy in `NOTES.md` and adapt.

---

## 1. What we are building (context)

An **autonomous ML research agent**: an LLM-driven system that improves a recommender
pipeline on its own. The human team builds the *machinery*; the LLM makes the *research
decisions* during the run. The agent runs the classic MLE loop — hypothesize → edit code
→ train → evaluate → reflect → log — repeatedly, against the KuaiRand-Pure benchmark,
trying to beat a fixed official baseline.

The competition task the agent works on:
- **Dataset:** KuaiRand-Pure short-video logs. Fixed date-based splits (train / validation /
  hidden test). We only ever touch train + validation.
- **Task:** rank each user's logged impressions; positive label = `long_view`.
- **Metrics:** GAUC and nDCG@5, primary = mean of the two, computed ONLY by the
  organizers' `evaluate.py` (ships in the starter kit).
- **Official baseline:** a numpy Factorization Machine (k=16, lr=0.001, 5 categorical
  fields), ~40s on one CPU core. Published validation scores: GAUC 0.6674 / nDCG@5
  0.5357 / primary 0.6016. Reference sanity rungs: random ≈ 0.475 primary, item
  popularity ≈ 0.5715 primary (verify exact split conventions against starter kit docs).
- **Grading reality:** the score delta matters, but a huge share of marks is read from the
  per-iteration logs (hypotheses, diffs, metrics, error/recovery events) and from the
  **manual intervention count**. Logging and robustness are first-class product features,
  not afterthoughts.

## 2. Non-negotiable ground rules (encode these as hard properties)

1. **`evaluate.py` is sealed.** Copy it verbatim from the starter kit into the repo.
   Wrap it; never edit it; never reimplement it. All promotion/convergence decisions use
   its output only.
2. **The LLM never grades itself.** Scores, promotion, streaks, budgets, and log facts
   are computed and written by harness code from measured values. The LLM writes only
   hypotheses, code, debug fixes, and one constrained "lesson" sentence.
3. **The checkpoint is sacred.** Only the harness promotes; failed/worse experiments can
   never overwrite the best-known pipeline.
4. **No external training data.** The agent's sandbox has no network access for data;
   training uses only the provided splits. (Library installs are pre-provisioned, not
   agent-initiated.)
5. **Stopping rules enforced by the harness:** convergence (no improvement > EPSILON=0.002
   over N=3 consecutive iterations), MAX_ITERS=50, WALL_CLOCK=6h. Conservative reading:
   **failed iterations tick the flat streak.** Do not start a new iteration past the
   wall-clock ceiling.
6. **Promotion ≠ convergence.** Two separate comparisons: promotion uses
   PROMOTE_MARGIN (default 0.0010, configurable); streak reset requires improvement
   > EPSILON over best-so-far. Keep the logic visibly separate.
7. **Everything about a run lives in one timestamped run directory.** Logs, ledger,
   state, checkpoints, submission, resource counters, interventions. Designating the
   official run = pointing at one folder.
8. **Honest accounting.** Token usage per LLM call (from API usage fields), wall-clock,
   iteration count, and a manual-intervention log are recorded automatically where
   possible and prominently where manual.

## 3. Architecture (one harness, four LLM roles)

Deterministic Python harness runs the loop and owns all guarantees. Four role prompts to
an LLM API (same provider, models configurable per role):

- **Researcher** — reads briefing (knowledge library + state block + full ledger),
  outputs next hypothesis + change specification. Strategy rules live in its prompt:
  explore structurally new ideas early; refine winners mid-run; when flat-streak ≥ 2,
  prefer the most reliable promising idea; never re-propose a failed idea without a
  stated new reason; route around directions marked BLOCKED.
- **Engineer** — receives change spec + current champion code, outputs the full modified
  pipeline file(s). Edits must be minimal and targeted (we diff champion vs. new for the
  log). No mock results, no editing anything outside the pipeline workspace.
- **Debugger** — receives failing code + full traceback, outputs a fixed version or
  `{"action":"abandon","reason":...}`. Max DEBUG_RETRIES=3 per iteration; retries do NOT
  consume iterations.
- **Scribe** — two constrained jobs: (a) the one-sentence LESSON (≤ 20 words) after the
  measured result is known; (b) rendering the required per-iteration log entry from
  harness-supplied structured facts. May use a cheaper model.

Prompt assembly order (for provider prompt-caching): static role prompt → static
knowledge library → dynamic state block → dynamic ledger → task instruction.

## 4. Repository layout to create

```
.
├── IMPLEMENTATION_SPEC.md        # this file
├── NOTES.md                      # discrepancies, decisions, open questions (you maintain)
├── README.md                     # setup, run, reproduce, limitations (write in Phase 6)
├── pyproject.toml / requirements.txt
├── config.yaml                   # all constants + per-role model names + paths
├── starter_kit/                  # UNZIPPED organizer kit (human places it here)
├── sealed/
│   ├── evaluate.py               # verbatim copy from starter_kit (never edited)
│   └── submit_check.py           # wrapper invoking starter kit's submit.py --check
├── agent/
│   ├── harness.py                # main loop + stopping logic + resume
│   ├── llm_client.py             # provider-agnostic chat call + token accounting
│   ├── roles.py                  # briefing assembly + role call functions + parsing
│   ├── tools.py                  # run_experiment, evaluate_preds, data_profile,
│   │                             # read_iteration_detail, make_submission
│   ├── sandbox.py                # subprocess execution with timeout, cwd isolation
│   ├── memory.py                 # ledger append, state-block render, detail JSON I/O
│   ├── promotion.py              # promotion + streak logic (pure functions, unit-tested)
│   └── schemas.py                # dataclasses / pydantic models for all contracts
├── prompts/
│   ├── researcher.md
│   ├── engineer.md
│   ├── debugger.md
│   ├── scribe_lesson.md
│   └── scribe_logentry.md
├── knowledge/
│   └── library.md                # domain playbook injected into Researcher briefings
├── baseline_repro/               # Phase 0 artifacts (agent's reproduction of baseline)
├── runs/                         # one timestamped dir per run (gitignored except example)
│   └── RUN_ID/
│       ├── run_state.json        # resumable: iter, streak, best_score, start_ts, tokens
│       ├── ledger.md             # tier-1 compact history (append-only)
│       ├── state.md              # tier-2 standing block (regenerated each iteration)
│       ├── iterations/itNN/      # workspace: pipeline code, preds, stdout, stderr
│       ├── logs/iter_NN.json     # tier-3 full per-iteration record (judges' schema)
│       ├── best/                 # champion code + score + val predictions
│       ├── interventions.md      # manual intervention log + counter
│       └── submission.csv        # generated at finalize from best/
└── tests/                        # pytest suite (see Phase gates)
```

## 5. Frozen contracts (implement exactly; all roles parse/emit these)

### 5.1 Researcher output (JSON)
```json
{
  "hypothesis": "one sentence: what change and why it should raise primary",
  "category": "feature | model | training | multitask | other",
  "change_spec": "precise instructions for the Engineer",
  "expected_risk": "low | medium | high",
  "builds_on": "champion"
}
```

### 5.2 Experiment pipeline contract (what Engineer's code must satisfy)
Each iteration's workspace contains `pipeline.py` runnable as:
```
python pipeline.py --data <starter_kit_data_dir> --split val --out preds_val.csv
```
It must: train using ONLY the train split; write predictions for every validation row to
`preds_val.csv` with columns `row_id,user_id,video_id,score` aligned to the split's
`data.load()` order; exit 0 on success. Same script with `--split test` produces test
predictions (used only at finalize, from the champion). The harness — not the pipeline —
calls sealed `evaluate.py` on the output. Hard timeout: EXPERIMENT_TIMEOUT (default 900s).

### 5.3 Harness result (fed back to roles)
```json
{
  "status": "scored | failed | timeout",
  "gauc": 0.0, "ndcg5": 0.0, "primary": 0.0,
  "runtime_s": 0, "error_excerpt": "last 60 lines if failed",
  "vs_best": "+0.0021 | n/a"
}
```

### 5.4 Ledger line (tier-1 memory; harness-written except LESSON)
```
[itNN] HYP: <short> | CHANGE: <files/summary> | RESULT: <primary or FAILED(reason)> (best <x>) -> PROMOTED|kept|FAILED | LESSON: <scribe, ≤20 words>
```

### 5.5 State block (tier-2; regenerated fresh each iteration)
```
CURRENT BEST: itNN | val primary X (GAUC a / nDCG5 b) | baseline 0.6016 | margin +Y
BUDGET: iteration K of 50 | H:MM of 6:00 elapsed | tokens so far T
CONVERGENCE: streak S of 3 flat (EPSILON=0.002)
BLOCKED: <comma list or none>
ACTIVE THEMES: <winning / losing / untried directions, one line>
```

### 5.6 Per-iteration log entry (tier-3; the judges' deliverable) — `logs/iter_NN.json`
```json
{
  "iteration": 1, "timestamp": "...",
  "hypothesis": "...", "rationale": "...", "category": "...",
  "code_diff": "unified diff champion -> attempt",
  "result": {"status": "...", "gauc": 0, "ndcg5": 0, "primary": 0},
  "errors_and_recovery": [{"attempt": 1, "error": "...", "fix_summary": "..."}],
  "decision": "promoted | kept_champion | failed",
  "streak_after": 0, "tokens_this_iteration": 0, "runtime_s": 0
}
```

## 6. Main loop (reference pseudocode — implement in `harness.py`)

```python
def run(run_dir, config):
    state = load_or_init_run_state(run_dir)          # resumable after every iteration
    if state.iteration == 0:
        phase0_baseline_repro(state)                 # see Phase 0 below

    while True:
        if state.streak >= 3 or state.iteration >= 50 or elapsed(state) >= 6h:
            return finalize(state)                   # submission from best/, validate, stop

        it = state.iteration + 1
        briefing = assemble_researcher_briefing(state)      # order per §3
        plan     = call_role("researcher", briefing)         # parse §5.1; 1 retry on bad JSON
        code     = call_role("engineer", plan, champion())
        result   = sandbox_run(code, timeout=cfg.timeout)
        attempts = []
        while result.failed and len(attempts) < 3:
            fix = call_role("debugger", code, result.error)
            if fix.action == "abandon": break
            code, result = fix.code, sandbox_run(fix.code, timeout=cfg.timeout)
            attempts.append(...)

        score = sealed_evaluate(result.preds) if result.ok else None

        # --- separate judgments (unit-tested pure functions in promotion.py) ---
        if score is not None and score.primary > state.best + cfg.PROMOTE_MARGIN:
            promote(code, score, state)
        if score is not None and score.primary > state.best_at_iter_start + cfg.EPSILON:
            state.streak = 0
        else:
            state.streak += 1                        # flat, tiny, or failed all tick

        lesson = call_role("scribe_lesson", plan, score_or_error)   # ≤20 words, truncate
        append_ledger(run_dir, it, plan, code_diff, score, decision, lesson)
        write_logjson(run_dir, it, ...)              # §5.6, includes attempts
        write_state_block(run_dir, state)            # §5.5
        persist_run_state(run_dir, state)            # crash-resume point
        state.iteration = it
```

## 7. Phase 0 — baseline reproduction + harness self-check (runs once at run start)

Before iteration 1 the harness must:
1. Score a **random** predictor and an **item-popularity** predictor with sealed
   `evaluate.py`; assert primaries land near the published rungs (tolerance ±0.01; treat
   exact expectations as "verify against starter kit docs" and record in NOTES.md).
2. Run the starter kit's official baseline (`python starter_kit/baseline.py --model fm`
   or per its README), score its validation predictions, assert primary ≈ 0.6016 within
   seed noise (±0.005).
3. Install the baseline as **iteration 0's champion** (copy into `runs/RUN_ID/best/` in
   the §5.2 pipeline contract shape — the Engineer's first edits build on this file).
If any assertion fails: abort loudly with a diagnostic. Nothing downstream is trustworthy.

## 8. Knowledge library — create `knowledge/library.md` with this starter content

Write it as a ~2-page playbook (prose + short lists), covering, in this priority order:
1. **Task facts:** label `long_view` (sparse, logged on every impression — no selection
   bias); 12 feedback columns exist; only 5 categorical ID fields used by baseline;
   date-based split ⇒ test period is AFTER train (temporal shift is real).
2. **Direction ladder (with reasons):**
   a. Multi-task heads: start long_view + click (one aux weight), then + like, then
      play_time as a regression head; escalate to MMoE/PLE-style partial sharing only if
      seesaw symptoms (aux improves, primary stalls).
   b. History features, PAST-DATES-ONLY: user's historical long_view rate, per-category
      engagement rates, item's rolling long_view rate, recency features.
   c. Model ladder: FM (champion) → DeepFM-style / wider embeddings → GBDT (LightGBM)
      on engineered features → small ensemble of champion family.
   d. Training: class weighting for sparse positives, LR schedule, early stopping on val.
3. **Trap list (bold):** same-row feedback columns as input features = LEAKAGE, forbidden;
   whole-dataset aggregates leak future ⇒ compute rolling/past-only; a sudden huge jump
   ⇒ suspect leakage, re-verify before trusting; only sealed evaluate.py scores count.
4. **Strategy rules:** explore structurally different ideas early; refine winners; at
   streak ≥ 2 pick the most reliable promising idea; gains < 0.002 don't reset the
   streak — prefer bigger structural swings over micro-tuning; never retry BLOCKED items.

## 9. Configuration (`config.yaml`)

```yaml
llm:
  provider: anthropic            # thin wrapper; keep swappable
  researcher_model: <strong model>     # placeholders — human fills in current names
  engineer_model:  <strong model>
  debugger_model:  <strong model>
  scribe_model:    <cheap model>
  max_output_tokens: {researcher: 1500, engineer: 6000, debugger: 6000, scribe: 200}
run:
  EPSILON: 0.002
  N_FLAT: 3
  MAX_ITERS: 50
  WALL_CLOCK_HOURS: 6
  PROMOTE_MARGIN: 0.0010
  DEBUG_RETRIES: 3
  EXPERIMENT_TIMEOUT_S: 900
paths:
  starter_kit: ./starter_kit
  data: ./starter_kit/<data-subdir>    # resolve after inspecting kit
```
API key from env `ANTHROPIC_API_KEY` (or provider equivalent). Never write keys to disk,
logs, or git. Add a spend guard: abort the run if cumulative tokens exceed a configurable
ceiling.

## 10. Sandbox (`sandbox.py`) — hackathon-scale, deliberate simplicity

`subprocess.run([...python pipeline.py...], cwd=iteration_workspace,
timeout=EXPERIMENT_TIMEOUT_S, capture_output=True)`. Kill on timeout; return stdout tail
+ stderr tail. Pre-install allowed libraries (numpy, pandas, scikit-learn, lightgbm,
torch-cpu) in the project venv — the generated code must not install packages or access
the network. Each iteration gets a fresh workspace seeded with the champion's files;
the champion directory itself is read-only to experiments (copy, never reference-edit).

## 11. Interventions & resources

- `interventions.md`: template with columns (timestamp, what was stuck, what the human
  did, scope). A CLI helper `python -m agent.intervene "description"` appends and bumps
  the counter in `run_state.json` — make honesty low-friction.
- Token counter: read usage from every API response; accumulate per-iteration and total
  into `run_state.json` and each `iter_NN.json`.
- Wall-clock from run_state start timestamp (survives resume).
- `finalize()` writes `results_summary.md`: best val GAUC/nDCG5/primary, delta vs
  baseline (0.6016 val reference), iterations used, tokens, wall-clock, interventions.

## 12. Finalize

1. Run champion `pipeline.py --split test` → `submission.csv` (§5.2 columns).
2. Validate via the starter kit's own checker (`submit.py --check` pathway) — must pass.
3. Write `results_summary.md`; print a clear end-of-run banner with stop reason
   (converged | iter_cap | wall_clock) — the stop reason also goes in run_state.

## 13. Implementation phases with acceptance gates (build in this order)

- **Phase 1 — skeleton loop:** harness + sandbox + a STUB agent (random tiny perturbation
  of a dummy pipeline). Gate: loop runs 5 iterations end-to-end on a toy scorer, ledger/
  state/logs all written, resume-from-kill works.
- **Phase 2 — sealed evaluation + Phase 0:** wrap evaluate.py; implement baseline repro +
  rung self-checks against real starter kit. Gate: Phase 0 passes on the real data.
- **Phase 3 — real roles:** llm_client with token accounting; four prompts; JSON parsing
  with one re-ask on malformed output. Gate: a 3-iteration real run completes with ≥1
  scored experiment; ledger lines match §5.4 exactly.
- **Phase 4 — robustness:** debugger retry loop, BLOCKED marking, timeout handling,
  spend guard, stall directive (3 consecutive failures ⇒ inject canned recovery
  instruction into next briefing). Gate: fault-injection tests pass (see tests).
- **Phase 5 — full dry run:** complete run on real data to natural stop. Gate: stop
  reason correct per rules; submission.csv passes checker; results_summary written.
- **Phase 6 — packaging:** README (setup, one-command run `python -m agent.harness`,
  reproduce steps, limitations), example run folder, .gitignore (runs/, keys).

## 14. Required tests (`pytest`; write alongside each phase, not at the end)

1. **Convergence boundaries:** three fabricated score sequences → assert stop at streak,
   at iter cap, at wall clock (mock time), with correct champion each time.
2. **Promotion vs streak separation:** a +0.0015 gain promotes (if > margin) but does
   NOT reset streak; a failure ticks streak; a +0.003 gain resets it.
3. **Checkpoint safety:** failed/worse experiment leaves best/ byte-identical.
4. **Ledger format:** golden-file test for §5.4 line; append-only property.
5. **Resume:** kill after iteration k, resume, assert counters/streak/best continue.
6. **Fault injection:** pipeline that raises → debugger path invoked, capped at 3;
   pipeline that sleeps forever → timeout kill; malformed researcher JSON → one re-ask
   then iteration recorded FAILED.
7. **Submission round-trip:** generated CSV passes the sealed checker; NaN injection is
   rejected before finalize completes.

## 15. Do NOT

- Do not modify, "fix", or reimplement anything under `sealed/` or the starter kit.
- Do not let any LLM output write scores, streaks, budgets, or promotion decisions.
- Do not give the sandbox network access or package-install ability.
- Do not use hidden-test data anywhere except the single finalize prediction pass.
- Do not exclude failed iterations from the flat streak.
- Do not hardcode model names/prices/dates in code — config only.
- Do not commit run artifacts containing API keys or full raw LLM transcripts with keys.

## 16. Assumptions to VERIFY against the actual starter kit before Phase 2 (record in NOTES.md)

1. Exact CLI, file names, and data paths of `baseline.py`, `evaluate.py`, `submit.py`,
   `data.load()` ordering guarantee for row_id alignment.
2. Whether the kit ships reference convergence-rule code — if yes, wrap it verbatim
   instead of our own streak implementation (keep our tests, point them at the wrapper).
3. Exact split conventions behind the published rungs (validation vs hidden-test values)
   and the baseline's seed-noise band.
4. Any pinned library or Python-version constraints in the kit's README.
Open competition questions the humans will ask organizers (do not block on these; build
to the conservative interpretation): whether parallel candidate experiments each count
against the 50-iteration cap; whether failures tick the convergence streak (we assume yes).
