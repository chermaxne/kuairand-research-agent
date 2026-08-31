# Autonomous ML Research Agent — KuaiRand-Pure (TechJam 2026, Track 2)

An LLM-driven system that improves a recommender pipeline on its own. A deterministic Python
**harness** runs the classic MLE loop — hypothesize → edit code → train → evaluate → reflect → log —
against the KuaiRand-Pure within-user ranking benchmark (label `long_view`, metric = mean(GAUC,
nDCG@5)) and tries to beat the organizers' factorization-machine baseline (validation primary 0.6016).
Four LLM roles (Researcher, Engineer, Debugger, Scribe) make the research decisions; the harness owns
every guarantee: sealed scoring, promotion, convergence, budgets, logging, resume, interventions.

Read `IMPLEMENTATION_SPEC.md` for the design contract and `NOTES.md` for every discrepancy/decision.

## What is guaranteed (competition rules encoded as code)
| Rule | Where it lives |
|---|---|
| `evaluate.py` is sealed — copied verbatim, never edited, never reimplemented | `sealed/evaluate.py` (sha256 test), `sealed/submit_check.py` wraps the kit's own `submit.py --check` |
| The LLM never grades itself | scores/promotion/streaks/budgets computed only in `agent/promotion.py` + `agent/harness.py` from measured values |
| The checkpoint is sacred | only `install_champion()` (harness) writes `runs/RUN_ID/best/`; failed/worse experiments leave it byte-identical (test) |
| No external training data; hidden-test rows are invisible during the loop | experiments run on a derived train+valid-only data dir and are read-denied on the full dir (macOS `sandbox-exec`) |
| The experiment sandbox has no network — unaffected by Researcher tool-calling | `sandbox.py`'s network denial is about the sandboxed subprocess only; the harness *process* itself makes real outbound calls when `research_tools.enabled: true` (arxiv.org only, Researcher role only) — a different scope, not a contradiction. See `agent/research_tools.py`. |
| Stopping rules: streak ≥ 3 flat (ε = 0.002), 50 iterations, 6 h wall clock, token spend guard | `agent/promotion.py: stop_reason()`; failed iterations tick the streak |
| Promotion (margin 0.0010) ≠ convergence (ε 0.002) | two separate pure functions, unit-tested |
| One timestamped run directory holds everything | `runs/<RUN_ID>/` — ledger, state block, per-iteration JSON, workspaces, best/, submission, interventions, token log |
| Honest accounting | token usage from API `usage` fields per call (`llm_calls.jsonl`), wall clock from the run's start timestamp, manual-intervention log with a counter (restarts are auto-recorded) |

## Setup
Requirements: Python ≥ 3.10 (3.12 used here), macOS or Linux, ~2 GB disk for data + caches.

```bash
git clone <this repo> && cd kuairand-starter-kit
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # numpy pandas scikit-learn lightgbm pyyaml anthropic pytest
pip install torch                          # optional (CPU build), only if experiments should be allowed to use torch
# macOS only: LightGBM needs OpenMP
brew install libomp

# Organizer data (47 MB download, ~400 MB unpacked) — exactly as the kit README says, inside starter_kit/
cd starter_kit && curl -L -o KuaiRand-Pure.tar.gz https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz \
  && tar xzf KuaiRand-Pure.tar.gz && rm KuaiRand-Pure.tar.gz && cd ..
```

The API key is read from `<repo>/.env` (loaded automatically, gitignored) or from the environment variable
named in `config.yaml` (`llm.api_key_env`, default `OPENROUTER_API_KEY`). It is never written to disk, logs or
git, and the experiment sandbox cannot read it. Model ids live in `config.yaml` — check them with `--llm-check`.

```bash
# .env in the repo root:
OPENROUTER_API_KEY=sk-or-...               # the default provider — key from https://openrouter.ai/keys
```

### Which provider / key
**Default: OpenRouter** — one key, every model. Put `OPENROUTER_API_KEY=...` in `.env` and run; no flag needed.
The roles are text-in/text-out, so any capable model works; only the Engineer is demanding (it emits a complete
~250-line `pipeline.py`, so it needs a large output budget and real coding ability).

Shipped model choice (settled after the first live test on 2026-08-28 — see NOTES.md for the incident):

| role | model | why |
|---|---|---|
| Researcher | `z-ai/glm-5.2` | best plans seen in this project (precise specs that cite the ledger and playbook); ~1–2 min and ~$0.02 per call |
| Engineer / Debugger | `deepseek/deepseek-v4-flash` | ~2.5 min per rewrite, but the only model here that produced correct non-trivial code unaided (qwen3-coder: 2/6, including a label leak) |
| Scribe | `mistralai/codestral-2508` | 0.4 s per call |
| automatic fallbacks | `qwen3-coder` → `codestral-2508` (Engineer/Debugger); `qwen3-coder` → `glm-5.3-flash` (Researcher) | used when the primary stalls, returns 429 or disappears |

Why a non-reasoning Engineer: reasoning models' thinking tokens count against `max_tokens`, and an exhausted budget
returns *empty* content while the provider still bills the generation. `llm.reasoning` caps thinking per role.

Every call is **streamed**: a stalled generation is abandoned after `inactivity_timeout_s` (120 s) without a
token, capped at `call_timeout_s` (900 s), retried once, then the next fallback model is used — and the console
shows a heartbeat (`[llm] engineer: qwen/qwen3-coder streaming — 6,120 chars, 30s`) plus one line per completed
call, so you always know what the agent is doing. Cost ≈ $0.03 per iteration (~$1.50 for a full 50-iteration run); ~6 min per iteration (`--llm-profile fast` trades correctness for ~4 min).

```bash
.venv/bin/python -m agent.harness --llm-profile openrouter_claude  # anthropic/claude-opus-4.8 + haiku-4.5, ~$10-20
```

Other providers (`--llm-profile <name>`, key in `.env`):

| profile | key from | cost | free limits (Aug 2026) | notes |
|---|---|---|---|---|
| `anthropic` | [console.anthropic.com](https://console.anthropic.com) → `ANTHROPIC_API_KEY` | paid | — | native API: prompt caching, adaptive thinking and effort are used only here |
| `gemini` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → `GEMINI_API_KEY` | **free** | 10 rpm · 250k tokens/min · **1500 req/day** · 1M context | the most generous free tier that fits our ~12k-token briefings |
| `groq` | [console.groq.com/keys](https://console.groq.com/keys) → `GROQ_API_KEY` | **free** | 30 rpm · **6k tokens/min** | one briefing exceeds a minute's token budget → long back-offs |
| `cerebras` | [cloud.cerebras.ai](https://cloud.cerebras.ai) → `CEREBRAS_API_KEY` | **free** | 1M tokens/day · **8k context** | context too small for Researcher/Engineer |
| `deepseek` | [platform.deepseek.com](https://platform.deepseek.com) → `DEEPSEEK_API_KEY` | cheap paid | — | strong at code, cents per run |
| `poe` | [poe.com/api/keys](https://poe.com/api/keys) → `POE_API_KEY` | Poe subscription | 500 rpm | Claude models on an existing Poe subscription (Anthropic-compatible gateway) |

```bash
.venv/bin/python -m agent.harness --llm-list-models glm    # what this key can actually serve
.venv/bin/python -m agent.harness --llm-check              # 1 tiny request per role model
.venv/bin/python -m agent.harness --max-iters 1 --label smoke
```
Briefing depth (`config.yaml` → `run`): `briefing_recent_iterations: 5`, `briefing_diff_chars`, `briefing_spec_chars`
control how much of each recent iteration the Researcher sees — full hypothesis, change spec, rationale, code diff,
delta vs the then-champion, debug attempts, leak verdict and training curve, so it can judge which component of a
bundled change worked. A briefing costs ~15–25k tokens of a ~1M-token context window.

Memory across the whole run: every briefing carries a harness-written **research digest** (a fact table over all
iterations — direction, what changed, delta vs the then-champion, decision, failure/leak status) and a ≤150-word
**Scribe synthesis** regenerated from that table each iteration and rejected if it contains a number not in the table.

Banking (`config.yaml` → `run`): `PROMOTE_MARGIN: 0.0002` (seed-noise level) so small real gains compound;
`leak_check: on_improvement` verifies every iteration that beats the champion; the best leak-clean measurement is
tracked as `best_measured` and finalize builds the submission from it when it beats the champion, so a gain that
missed the margin is never lost.

Sizing and in-run attribution (`config.yaml` → `run`): convergence is the organizers' rule verbatim (each
iteration vs best-so-far, ε = 0.002, N = 3; there is no alternative reading in the code), so every briefing carries
a **SIZING DIRECTIVE**: one hypothesis sized to clear ε on its own, a numeric `expected_gain` with evidence, every
validated rider stacked, and an `ablation_plan`. The pipeline scores the ablation variants itself and prints
`ABLATION <name> primary=…` lines; the harness parses them into the digest next to the sealed result (marked
pipeline-reported / unsealed) and reports the Researcher's prediction-vs-measurement calibration, so attribution
costs runtime, not iterations. Posture is streak-aware: boldest structural bet at streak 0 ("new information beats
capacity"), best-evidence variant at streak 1, LAST-SHOT bundle at streak 2.

Research strategy knobs (`config.yaml` → `run`): `structural_first_until_iter: 10` injects a briefing directive
that restricts the first 10 iterations to the knowledge library's measured, ranked recipes (pairwise within-user
loss, session-context field, seed averaging, past-only history fields — bundled first) and forbids
hyperparameter-only proposals; `implausible_gauc_below: 0.5` gives the Debugger one pass at a scored-but-inverted
ranking before it counts.

**The knowledge library is evidence, not folklore.** `knowledge/library.md` was built from a data analysis of the
kit, a literature review, and probe experiments scored on the official validation split with the sealed evaluator,
then audited by an independent evaluation agent that recomputed the facts and ran its own experiments. Every
number in it is reproducible from `knowledge/evidence/`.

Model handles move fast; if `--llm-check` reports an unknown model, `--llm-list-models` shows what the key serves
and you edit `config.yaml`. Every role also has a fallback list, so a rate-limited or delisted primary does not
fail the iteration.

## Run
> **Use the project venv.** A bare `python` is often a system/conda interpreter without these dependencies —
> and the same interpreter runs the generated experiments, so it needs `lightgbm`/`torch` too. Either prefix
> commands with `.venv/bin/python` (as below) or `source .venv/bin/activate` once per shell.

```bash
# 1. self-check: Phase 0 only (rungs + organizers' FM + champion) — ~2 min, no API key needed
.venv/bin/python -m agent.harness --mock --phase0-only

# 2. the official run (needs a key in .env): Phase 0, then up to 50 iterations, then finalize
.venv/bin/python -m agent.harness --label official          # add --llm-profile <name> for a non-default provider

# 3. resume after a crash / Ctrl-C (the restart is recorded as an intervention)
.venv/bin/python -m agent.harness --run-dir runs/<RUN_ID>

# 4. offline dry run on the real data with the deterministic mock roles (no key)
.venv/bin/python -m agent.harness --mock --label dryrun

# 5. the Phase-1 toy loop (synthetic mini dataset, seconds)
.venv/bin/python -m agent.harness --toy --mock

# record a manual intervention (bumps the counter; --block marks a direction BLOCKED for the Researcher)
.venv/bin/python -m agent.intervene "restarted with a longer timeout" --stuck "it07 hung" --scope config --run-dir runs/<RUN_ID>
```
Useful flags: `--max-iters N`, `--session-iters N` (stop this process after N iterations, resumable),
`--set run.EXPERIMENT_TIMEOUT_S=1200` (override any config key), `--config other.yaml`.

Designating the official run = pointing at one folder: `runs/<RUN_ID>/` contains
`submission.csv` (validated by the kit checker at finalize), `results_summary.md`, `ledger.md`,
`state.md`, `logs/iter_NN.json` (+ `.md` narrative), `iterations/itNN/` (code, diff basis, preds,
stdout/stderr, debug attempts, LLM transcripts), `best/` (champion code + score + val preds),
`interventions.md`, `run_state.json`, `llm_calls.jsonl`, `phase0/`, `finalize/`.

## Reproduce
1. `.venv/bin/python -m pytest -q` — the full suite (≈ 3 min; the real-data Phase 0 test is skipped when the
   data is absent).
2. `.venv/bin/python -m agent.harness --mock --phase0-only` — must print `random ≈ 0.483`, `pop ≈ 0.581`,
   `official FM ≈ 0.6015`, `champion ≈ 0.6015` (difference 0.0: the champion is a bit-for-bit port).
3. `.venv/bin/python -m agent.harness --mock --label dryrun` — deterministic offline end-to-end run on the real
   data (see `runs/example_run/` for what it produces and the numbers below).
4. Re-score any run's champion by hand with the organizers' tools:
   `.venv/bin/python starter_kit/submit.py --score --split valid --data_dir starter_kit/KuaiRand-Pure/data runs/<RUN_ID>/best/preds_val.csv`
   and check the submission: `.venv/bin/python sealed/submit_check.py --split test --data_dir starter_kit/KuaiRand-Pure/data runs/<RUN_ID>/submission.csv`.

## Layout
```
agent/        harness.py (loop, resume, finalize) · promotion.py (pure decisions) · memory.py (ledger/state/logs)
              roles.py (briefing assembly, parsing, re-ask) · llm_client.py (Anthropic + mock, token accounting)
              sandbox.py (timeout, env stripping, sandbox-exec) · tools.py · task.py · phase0.py · toy.py · intervene.py
prompts/      researcher.md engineer.md debugger.md scribe_lesson.md scribe_logentry.md (system + task template)
knowledge/    library.md — domain playbook injected into every Researcher briefing; evidence/ — probes that produced it
docs/         tool_loop_sketch.py — design sketch for Researcher tool-calling, superseded by the real
              implementation in agent/roles.py + agent/research_tools.py; kept for its native-Anthropic-vs-
              OpenRouter tool-shape notes
baseline_repro/pipeline.py — iteration-0 champion (organizers' FM in the pipeline contract)
sealed/       evaluate.py (verbatim) · submit_check.py (wraps starter_kit/submit.py --check)
starter_kit/  organizer kit, untouched · data_cache/ derived train+valid-only copy · runs/ one dir per run
tests/        promotion · ledger · checkpoint · resume · sealed/phase0/submission · roles · fault injection · intervene
```

## Example run (`runs/example_run/`, offline mock roles on the real data)
`.venv/bin/python -m agent.harness --mock --label dryrun` — deterministic, no API key. The mock Researcher follows a
fixed plan of FM edits; one Engineer output deliberately contains a `NameError` so the Debugger path is on
record. Prediction CSVs were dropped from the copy; everything else is verbatim.

| it | hypothesis | what happened | val primary | decision | streak |
|---|---|---|---|---|---|
| 00 | organizers' FM ported to the pipeline contract | Phase 0: random 0.4827, pop 0.5807, official FM 0.6015 = champion 0.6015 | 0.6015 | champion | 0 |
| 01 | L2 1e-6 → 1e-5 | crashed (`NameError`), Debugger fixed it in 1 attempt, re-run scored | **0.6025** (+0.0010) | **promoted** (> margin 0.001) but < ε 0.002 → streak still ticks | 1 |
| 02 | K 16 → 32 | scored | 0.6022 (−0.0003) | kept_champion | 2 |
| 03 | LR 0.001 → 0.002 | scored (diverged) | 0.4909 | kept_champion | 3 → **converged** |

Finalize: champion it01 re-trained with `--split test`, `submission.csv` (170,588 rows) accepted by the kit's
checker; `results_summary.md` reports best val primary 0.6025 vs published baseline 0.6016 (+0.0009), 3
iterations, ≈42k tokens over 13 LLM calls (mock usage is estimated from characters and flagged as such),
0 interventions. Note how it01 shows the two separate judgments: promoted, yet the flat streak advanced.

## Limitations
* **Real-API runs**: several completed runs now exist against real models via OpenRouter, e.g.
  `runs/20260830_224430_seeded_0605_v2` (KuaiRand-Pure, `results_summary.md`) and
  `runs/20260831_145457_1k_bonus_test` (KuaiRand-1K bonus benchmark — see its own Warnings section for the
  extra kit-scaffolding this dataset needed, since only Pure ships a starter kit natively). To verify
  connectivity before a new run: `.venv/bin/python -m agent.harness --llm-check`, then check
  `llm_calls.jsonl` for real per-call cost (`estimated_usage: false`).
* **Prompt quality with real models is untested** — whether Researcher change specs are precise enough and
  whether the Engineer keeps diffs minimal. The prompts live in `prompts/` and are meant to be tuned by humans.
* **OS-level sandboxing is macOS-only** (`sandbox-exec`): no network, writes confined to the workspace,
  full data dir read-denied during the loop. On Linux the harness records a warning and relies on the static
  code guard + env stripping only; a container/`unshare -n` wrapper is a known gap.
* **One experiment per iteration**, sequential; no parallel candidates.
* **Timeouts are terminal** for an iteration by default (no Debugger retry); flip
  `run.retry_timeouts_with_debugger` if you prefer the retry.
* **Mock usage numbers are estimates** (characters / 4) and flagged `estimated_usage: true`; only real runs
  carry API-reported usage.
* The knowledge library is a hand-written playbook derived from the kit README and the spec; it is not
  validated by experiments beyond the organizers' published ones.
* Hidden-test rows are masked from experiments (derived data dir + read denial); the champion's final
  `--split test` pass is the only time test-period rows are read, and it is a prediction-only pass.

See `NOTES.md` §5 for what works, what is untested, and what the humans must verify next.

## Iteration category vocabulary
Every iteration is tagged with one `category` (`config.yaml: run.categories`):
`feature` (input signals — historical rates, session context, embeddings of raw fields), `model`
(architecture — e.g. FM → FwFM, FM → DeepFM), `training` (the objective/loss — pointwise, BPR, softmax,
ordinal), `multitask` (auxiliary heads), `other`. This is a 5-value vocabulary specific to this project,
not a generic taxonomy — if a rubric expects different category names, they map onto these directly
(architecture→`model`, training_strategy→`training`, features→`feature`; this project has no distinct
`evaluation_diagnostics` or `hyperparameters` category, since neither has ever been the *substance* of a
proposed change here — hyperparameter-only proposals are explicitly disallowed by the sizing directive in
`agent/harness.py`). Every promoted change across every kept run's lineage carries a real category value —
see any run's `iterations/it*/plan.json`.

## Contributions
Chengke, Chermaine, Benjamin, Aaron, Ryan.
