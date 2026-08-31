# Autonomous ML Research Agent — KuaiRand-Pure (TechJam 2026, Track 2)

An LLM-driven system that improves a recommender pipeline on its own. A deterministic Python
**harness** runs the classic MLE loop — hypothesize → edit code → train → evaluate → reflect → log —
against the KuaiRand-Pure within-user ranking benchmark (label `long_view`, metric = mean(GAUC,
nDCG@5)) and tries to beat the organizers' factorization-machine baseline (validation primary 0.6016).
Four LLM roles (Researcher, Engineer, Debugger, Scribe) make the research decisions; the harness owns
every guarantee: sealed scoring, promotion, convergence, budgets, logging, resume, interventions.

Read `NOTES.md` for every discrepancy/decision.

## What is guaranteed (competition rules encoded as code)
| Rule | Where it lives |
|---|---|
| `evaluate.py` is sealed — copied verbatim, never edited, never reimplemented | `sealed/evaluate.py` (sha256 test), `sealed/submit_check.py` wraps the kit's own `submit.py --check` |
| The LLM never grades itself | scores/promotion/streaks/budgets computed only in `agent/promotion.py` + `agent/harness.py` from measured values |
| The checkpoint is sacred | only `install_champion()` (harness) writes `runs/RUN_ID/best/`; failed/worse experiments leave it byte-identical (test) |
| No external training data; hidden-test rows are physically absent from the data dir experiments run on | experiments run on a derived train+valid-only data dir; the full dir is *additionally* OS-read-denied on macOS (`sandbox-exec`) — this extra layer is a no-op on Linux, see Limitations |
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
pip install -r requirements.txt            # numpy pandas scikit-learn lightgbm pyyaml anthropic openai requests pytest
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
The roles are text-in/text-out, so any capable model works, but Researcher and Engineer are both demanding: the
Researcher has to size a hypothesis against the organizers' strict per-iteration convergence rule and reason about
a whole run's history, and the Engineer emits a complete ~230–250-line `pipeline.py`, so both need a large output
budget and real reasoning/coding ability.

Shipped model choice (current `config.yaml`; earlier open-weight-only choices are in NOTES.md's 2026-08-28/29
entries and are superseded by this one):

| role | model | why |
|---|---|---|
| Researcher | `google/gemini-3.1-pro-preview` | $2.00 / $12.00 per M tokens, 65k max output; reasoning effort medium — see the bake-off below |
| Engineer | `google/gemini-3.1-pro-preview` | same model as Researcher; reasoning helps correctness on a whole-file rewrite |
| Debugger | `deepseek/deepseek-v4-flash` | fast, cheap, reasoning effort low — debugging is a small-diff task, not a planning one |
| Scribe | `mistralai/codestral-2508` | no reasoning; three short, format-constrained writes per iteration |
| automatic fallbacks | researcher: `glm-5.2` → `deepseek-v4-flash`; engineer: `deepseek-v4-flash` → `qwen3-coder`; debugger: `qwen3-coder` → `codestral-2508`; scribe: `qwen3-coder` | used when the primary stalls, returns 429 or disappears (`config.yaml: llm.fallback_models`) |

Why a non-reasoning Debugger/Scribe: reasoning models' thinking tokens count against `max_tokens`, and an exhausted
budget returns *empty* content while the provider still bills the generation. `llm.reasoning` caps thinking per role.

Every call is **streamed**: a stalled generation is abandoned after `inactivity_timeout_s` (120 s) without a
token, capped at `call_timeout_s` (900 s), retried once, then the next fallback model is used — and the console
shows a heartbeat (`[llm] engineer: qwen/qwen3-coder streaming — 6,120 chars, 30s`) plus one line per completed
call, so you always know what the agent is doing.

**Measured cost/speed, real runs, current shipped config:** $0.70 over 16 calls / 11 min (KuaiRand-Pure,
`runs/20260901_005427_gemini_pro_baseline/results_summary.md`, 3 iterations, 0 interventions) and $2.01 over 31
calls / 7:26 (KuaiRand-1K, `runs/20260831_145457_1k_bonus_test/results_summary.md`, 5 iterations, but see that
run's disclosed manual pauses below — wall-clock there is not pure compute time). Every real run so far has
converged well under the 50-iteration cap (the organizers' epsilon rule ends a run in a handful of iterations once
gains stop clearing 0.002), so total cost/time *at* the 50-iteration cap is unmeasured — extrapolating linearly
from the measured $0.23–$0.40 per iteration would put it in the $15–20 range, but that is arithmetic, not a
measurement, so treat it as a rough ceiling, not a claim. `--llm-profile fast` swaps in the cheaper open-weight
stack (`deepseek-v4-flash` / `qwen3-coder` / `codestral-2508`, ≈$0.03/iteration in the 2026-08-29 benchmark —
NOTES.md) at some correctness cost.

#### Researcher-role model selection: Gemini 3.1 Pro vs. Claude Sonnet 5

The Researcher/Engineer model was chosen with a controlled bake-off: two otherwise-identical full runs on
KuaiRand-Pure (same harness, same config, same Debugger `deepseek/deepseek-v4-flash` and Scribe
`mistralai/codestral-2508`), differing only in which model drove Researcher + Engineer.

| Researcher/Engineer model | best val primary | Δ vs baseline (0.6016) | iterations (of 50) | LLM calls | total tokens | researcher-role tokens | wall-clock | provider spend |
|---|---|---|---|---|---|---|---|---|
| `google/gemini-3.1-pro-preview` | 0.6049 | +0.0033 | 3 (converged) | 16 | 134,858 | 72,823 | 0:11 | $0.7011 |
| `anthropic/claude-sonnet-5` | 0.6048 (best-measured 0.6049\*) | +0.0032 | 4 (converged) | 26 | 472,975 | 326,639 | 0:40 | $1.9243 |

\* it02 was the promoted champion at 0.6048; it04's 0.6049 was the best leak-clean measurement and is what the
submission was built from — it sat just under the promotion margin, see `config.yaml: run.PROMOTE_MARGIN`.
Sources: `runs/20260901_005427_gemini_pro_baseline/results_summary.md` and
`runs/20260901_011854_sonnet5_pure_v2/results_summary.md` (also each run's `run_state.json` for the token/call
breakdown).

For essentially identical final quality (+0.0033 vs +0.0032), Sonnet used **~3.5x the total tokens** (~4.5x on the
Researcher role specifically) and **~2.7x the provider spend**, over roughly 3.6x the wall-clock. On that
cost/quality trade-off, Gemini 3.1 Pro was kept as the shipped Researcher and Engineer model.

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
0. **Official baseline** (the organizers' own script, unmodified — the number every delta in this README is
   measured against): `cd starter_kit && python3 baseline.py --model fm` (~40 s, CPU, single core; `--data_dir`
   defaults to `./KuaiRand-Pure/data`). Published validation scores are also pinned in
   `starter_kit/baseline_scores.json`: GAUC 0.6674 / nDCG@5 0.5357 / primary **0.6016**.
1. `.venv/bin/python -m pytest -q` — the full suite (≈ 3 min; the real-data Phase 0 test is skipped when the
   data is absent).
2. `.venv/bin/python -m agent.harness --mock --phase0-only` — must print `random ≈ 0.483`, `pop ≈ 0.581`,
   `official FM ≈ 0.6015`, `champion ≈ 0.6015` (difference 0.0: the champion is a bit-for-bit port).
3. `.venv/bin/python -m agent.harness --mock --label dryrun` — deterministic offline end-to-end run on the real
   data, no API key needed. Useful for inspecting the full loop mechanics (briefing → plan → code → sandbox run →
   sealed score → promote/converge → finalize) without spending a token; see the three real-API runs cited
   throughout this README (below) for what the loop produces against live models.
4. Re-score any run's champion by hand with the organizers' tools:
   `.venv/bin/python starter_kit/submit.py --score --split valid --data_dir starter_kit/KuaiRand-Pure/data runs/<RUN_ID>/best/preds_val.csv`
   and check the submission: `.venv/bin/python sealed/submit_check.py --split test --data_dir starter_kit/KuaiRand-Pure/data runs/<RUN_ID>/submission.csv`.

## Layout
```
agent/        harness.py (loop, resume, finalize) · promotion.py (pure decisions) · memory.py (ledger/state/logs)
              roles.py (briefing assembly, parsing, re-ask) · llm_client.py (Anthropic + mock, token accounting)
              sandbox.py (timeout, env stripping, sandbox-exec) · research_tools.py (Researcher arxiv_search/web_fetch)
              tools.py · task.py · phase0.py · toy.py · intervene.py
prompts/      researcher.md engineer.md debugger.md scribe_lesson.md scribe_logentry.md scribe_digest.md (system + task template)
knowledge/    library.md — domain playbook injected into every Researcher briefing; evidence/ — probes that produced it
docs/         tool_loop_sketch.py — design sketch for Researcher tool-calling, superseded by the real
              implementation in agent/roles.py + agent/research_tools.py; kept for its native-Anthropic-vs-
              OpenRouter tool-shape notes
baseline_repro/pipeline.py — iteration-0 champion (organizers' FM in the pipeline contract)
sealed/       evaluate.py (verbatim) · submit_check.py (wraps starter_kit/submit.py --check)
starter_kit/  organizer kit, untouched (KuaiRand-Pure) · starter_kit_1k/ human-built filename-port for the
              KuaiRand-1K bonus benchmark · data_cache/ derived train+valid-only copy · runs/ one dir per run
dashboard/    index.html — static, no build step; judge-facing run viewer (see "Dashboard" below)
tests/        promotion · ledger · checkpoint · resume · sealed/phase0/submission · roles · fault injection · intervene
```

## Results Summary

### KuaiRand-Pure (primary benchmark, organizer-published baseline)
Source: `runs/20260901_005427_gemini_pro_baseline/results_summary.md` + `run_state.json` — a full run against the
shipped production config (Gemini 3.1 Pro as Researcher + Engineer, DeepSeek V4 Flash as Debugger, Codestral as
Scribe; see the bake-off above).

| | validation GAUC | validation nDCG@5 | validation primary |
|---|---|---|---|
| Published baseline (organizers' FM, `starter_kit/baseline_scores.json`) | 0.6674 | 0.5357 | **0.6016** |
| Agent, best validation (it03) | 0.6716 | 0.5382 | **0.6049** |
| **Delta (validation vs. validation)** | +0.0042 | +0.0025 | **+0.0033** |

Resource usage: **134,858 tokens** total (87,304 in / 47,554 out) over **16 LLM calls**; **3 of 50** iterations
used (stop reason: converged, streak 3); wall-clock **0:11**; **0 manual interventions**; **0 GPU-hours** — every
pipeline here runs on CPU/numpy, and the one PyTorch-based hypothesis (it01, DeepFM) was abandoned by the Debugger
because `torch` isn't installed in this environment, not retried on GPU. `submission.csv` (170,588 rows) passed
the kit's own format/alignment checker (`sealed/submit_check.py`); the organizers' hidden-test score is not
available to us — the harness never has access to hidden-test labels, so it cannot be self-reported here.

This README's evidence base is these three real-API runs: `20260901_005427_gemini_pro_baseline` and
`20260901_011854_sonnet5_pure_v2` (the Researcher-model bake-off, KuaiRand-Pure) and
`20260831_145457_1k_bonus_test` (KuaiRand-1K). Earlier real runs referenced in prior drafts of this README
(e.g. a `--seed-champion` continuation run) have been superseded and are no longer cited here.

### KuaiRand-1K (bonus benchmark — no organizer baseline exists for this dataset)
The brief's Limits row only specifies "same task and metrics" as Pure, with no reference score. The number below
is **self-generated**, not organizer-published: the Starter Kit's own, unmodified FM baseline algorithm
(`baseline.py --model fm`, filename-ported to `starter_kit_1k/` — same code, same hyperparameters, single seed 0)
run directly against the KuaiRand-1K splits, via `starter_kit_1k/submit.py --make`. See
`starter_kit_1k/baseline_scores_1k.json` for the full self-measurement note.

| | validation GAUC | validation nDCG@5 | validation primary |
|---|---|---|---|
| **Self-measured** FM baseline (not organizer-published) | 0.6752 | 0.6105 | 0.6428 |
| Agent, best validation (it05) | 0.6788 | 0.6339 | 0.6563 |

No "delta over baseline" is reported for this benchmark — there is no organizer number to net against; the two
rows above are reported side by side, not subtracted, per the bonus-benchmark's own honesty framing.

Resource usage: **494,477 tokens** total over **31 LLM calls**; **5 of 50** iterations used (stop reason:
iteration cap, not convergence); wall-clock **7:26**; **3 manual interventions**; **0 GPU-hours**. Source:
`runs/20260831_145457_1k_bonus_test/results_summary.md`.

**Not fully autonomous — disclosure required by this benchmark's own warnings section.** KuaiRand-1K is not
natively wired into the harness (only KuaiRand-Pure ships a starter kit + published `baseline_scores.json`).
Before this run's Phase 0, a human/coding-assistant built the parallel kit the run depends on
(`starter_kit_1k/{data,evaluate,baseline,submit}.py`, `baseline_scores_1k.json`, a memory-safety fix to a
seed-champion pipeline, and two additive filename-extension lines in `agent/tools.py`) — this setup step did not
touch the Researcher/Engineer/Debugger loop itself, which then ran unmodified, but it means the harness did not
build its own on-ramp to this dataset unassisted. Of the 3 manual interventions: 1 was a deliberate human-requested
pause/resume (no data lost), 1 was recovery from a real infrastructure failure (`openai.APIError: Upstream idle
timeout exceeded` mid-call — a human relaunched the process from the last saved state), and 1 was the routine
`--phase0-only` → loop transition. Full detail in that run's own `results_summary.md` Warnings section and
`interventions.md`.

## Dashboard
`dashboard/index.html` is a single static file — a run viewer/ledger, no server or build step. Open it directly
in a browser (`file://.../dashboard/index.html`) or serve the repo root with any static file server. Its data is
baked in at generation time (`window.RUNS`, near the bottom of the file), not fetched live, so it needs
regenerating from a run's `run_state.json` + `logs/iter_NN.json` whenever the cited run changes — see the TODO
note under Limitations for its current staleness relative to this README.

## Limitations
* **Real-API runs**: three completed runs against real models via OpenRouter back the numbers in this README —
  `runs/20260901_005427_gemini_pro_baseline` and `runs/20260901_011854_sonnet5_pure_v2` (KuaiRand-Pure,
  Researcher-model bake-off) and `runs/20260831_145457_1k_bonus_test` (KuaiRand-1K bonus benchmark — see its own
  Warnings section for the extra kit-scaffolding this dataset needed and the disclosed manual interventions,
  since only Pure ships a starter kit natively). To verify connectivity before a new run:
  `.venv/bin/python -m agent.harness --llm-check`, then check `llm_calls.jsonl` for real per-call cost.
All three run directories cited above (and this README's own claims about them) are committed to git —
`.gitignore` carves out `runs/20260831_145457_1k_bonus_test/`, `runs/20260901_005427_gemini_pro_baseline/`, and
`runs/20260901_011854_sonnet5_pure_v2/` (prediction CSVs excluded, same pattern used throughout), so a judge
cloning the repo can open every file this README cites by path.
* **Prompt quality with real models**: now measured for two models (Gemini 3.1 Pro, Claude Sonnet 5) on
  KuaiRand-Pure and one (Gemini 3.1 Pro) on KuaiRand-1K — see the bake-off above and Results Summary. Untested
  beyond those: whether a cheaper/faster model than either would hold up on a longer, slower-converging run (all
  three real runs so far converged in 3–5 iterations, well under the 50-iteration cap).
* **OS-level sandboxing is macOS-only** (`sandbox-exec`) and did not engage on any of the three cited runs — all
  three explicitly log `sandbox isolation: none (WARNING: no OS-level network/write confinement on this host)`
  in their own `results_summary.md` (this project's dev/CI host, and this repo's own sandbox environment, are
  Linux). On Linux the harness relies on the static code/import guard (`sandbox.py: static_code_check`) and env
  stripping only — no OS-enforced network denial, write confinement, or read denial. A container/`unshare -n`
  wrapper would restore the guarantee; not implemented.
* **The hidden-test read-denial is a Linux no-op, and the safe-directory's own README names the excluded path.**
  During the loop, experiments are pointed at a derived `data_cache/loop_train_valid/` directory with
  hidden-test rows physically removed — this part is real and unconditional. But the *additional* protection
  described in `sandbox.py`'s docstring (`sandbox-exec`'s explicit read-deny list for the full data directory)
  only ever applies on macOS; on this Linux host it is not just weakened, it does not run at all, so nothing at
  the OS level stops a pipeline from reading outside the derived directory. Compounding this,
  `data_cache/loop_train_valid/README_LOOP_DATA.txt` — a file the sandboxed pipeline *can* read, since it lives
  inside the allowed directory — states in plain text the absolute path to the full data directory it was
  derived from (the one containing hidden-test rows). On Linux, nothing prevents a generated `pipeline.py` from
  reading that path and following it. This has not been observed to happen in any real run (the static import
  guard and the fact that no generated pipeline has attempted it are the only reasons it hasn't), but it is a
  real, currently open gap, not a theoretical one — worth fixing before relying on this isolation on Linux.
* **Researcher web-tool scope**: `research_tools.enabled: true` gives the harness process (not the sandbox)
  outbound network access for the Researcher's `arxiv_search`/`web_fetch` tools. This is host-restricted in code,
  not just by prompt instruction — `web_fetch` raises `ValueError` on any host outside `{arxiv.org,
  export.arxiv.org}` (`agent/research_tools.py`), so the Researcher cannot fetch an arbitrary URL, only ones
  `arxiv_search` itself returned. No broader general-web-search capability exists in this codebase to place
  further discipline around.
* **One experiment per iteration**, sequential; no parallel candidates.
* **Timeouts are terminal** for an iteration by default (no Debugger retry); flip
  `run.retry_timeouts_with_debugger` if you prefer the retry.
* **Mock usage numbers are estimates** (characters / 4) and flagged `estimated_usage: true`; only real runs
  carry API-reported usage.
* The knowledge library is a hand-written playbook derived from the kit README and the spec; it is not
  validated by experiments beyond the organizers' published ones.
* **TODO — the judge-facing dashboard (`dashboard/index.html`) is stale.** Its embedded `window.RUNS` data was baked
  from `runs/20260830_224430_seeded_0605_v2` (KuaiRand-Pure) and `runs/20260831_145457_1k_bonus_test`
  (KuaiRand-1K) — see the comment at the top of its `<script>` block. The first of those two run directories has
  since been removed from the repo, so the dashboard's Pure-benchmark numbers no longer match this README's
  (which now cites `20260901_005427_gemini_pro_baseline` instead) or any run directory currently on disk. It
  needs regenerating against the three runs this README uses before it's shown to a judge.

### What we'd improve with more time
* Fix the Linux sandbox gap for real: a `unshare -n` / minimal container wrapper so network denial, write
  confinement, and read denial actually hold on the host this project has mostly been run on, not just on
  macOS. Right now the isolation guarantee in this README's own table is untested on the platform used for the
  three real runs it cites.
* Extend the Researcher-model bake-off past two models/one comparison run each — the current comparison is a
  single run per model, and per-run variance (seed noise, hypothesis-space luck in 3–4 iterations) has not been
  separated from genuine model-quality difference. A few repeats per model would tighten the cost/quality claim.
* Commit the run directories this README cites (see the gap noted above) and regenerate the dashboard from them,
  so the two are consistent and a judge can verify both without re-running anything.
* Wire KuaiRand-1K into the harness properly (its own `config.yaml` paths / `--dataset` flag) instead of the
  filename-ported `starter_kit_1k/` scaffolding a human built once — see the autonomy disclosure in Results
  Summary. That scaffolding step is the one part of this project's pipeline that did not come from the agent
  loop itself.

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
Benjamin, Chengke, Chermaine, Aaron, Ryan.
