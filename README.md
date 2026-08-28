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

The API key is read from the environment variable named in `config.yaml` (`llm.api_key_env`, default
`ANTHROPIC_API_KEY`). It is never written to disk, logs or git. Model ids live in `config.yaml`
(`llm.*_model`) — verify they are current before the official run.

```bash
export ANTHROPIC_API_KEY=...               # needed only for real (non --mock) runs
```

### Which provider / key
The role models are pure text-in/text-out, so any capable provider works. Pick a profile and pass
`--llm-profile <name>`; each profile sets the endpoint, the key variable and the model ids
(`config.yaml` → `llm.profiles`). Only the Engineer is demanding: it must emit a complete ~250-line
`pipeline.py`, so the model needs a big output budget and decent code generation.

| profile | key from | cost | free limits (Aug 2026) | notes |
|---|---|---|---|---|
| *(default)* | [console.anthropic.com](https://console.anthropic.com) → `ANTHROPIC_API_KEY` | paid | — | best quality; prompt caching, thinking and effort are only used here |
| `gemini` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → `GEMINI_API_KEY` | **free** | 10 rpm · 250k tokens/min · 1500 req/day · 1M context | **best free fit** — the only free tier whose per-minute token budget fits our ~12k-token briefings |
| `groq` | [console.groq.com/keys](https://console.groq.com/keys) → `GROQ_API_KEY` | **free** | 30 rpm · **6k tokens/min** · 14.4k req/day | very fast, but one briefing exceeds the per-minute budget → long back-offs |
| `openrouter` | [openrouter.ai/keys](https://openrouter.ai/keys) → `OPENROUTER_API_KEY` | free / $10 | 20 rpm · 50 req/day (**1000/day after a one-time $10**) | the $10 also unlocks pay-as-you-go Claude via `anthropic/claude-opus-4.8` |
| `cerebras` | [cloud.cerebras.ai](https://cloud.cerebras.ai) → `CEREBRAS_API_KEY` | **free** | 1M tokens/day · **8k context** | context too small for the Researcher/Engineer; Scribe only |
| `deepseek` | [platform.deepseek.com](https://platform.deepseek.com) → `DEEPSEEK_API_KEY` | cheap paid | — | strong at code, cents per run |
| `poe` | [poe.com/api/keys](https://poe.com/api/keys) → `POE_API_KEY` | Poe subscription | 500 rpm | Claude models on an existing Poe subscription (Anthropic-compatible gateway) |

```bash
python -m agent.harness --llm-profile gemini --llm-list-models flash   # exact model handles this key can use
python -m agent.harness --llm-profile gemini --llm-check               # 1 tiny request per role model
python -m agent.harness --llm-profile gemini --max-iters 1 --label smoke
python -m agent.harness --llm-profile gemini --label official
```
Model handles move fast; if `--llm-check` reports an unknown model, `--llm-list-models` shows what the key
actually serves and you edit the profile.

**Using a Poe key** (Poe exposes the Claude bots through an Anthropic-compatible gateway at
`https://api.poe.com`): put it in `.env` as `POE_API_KEY` and add `--llm-profile poe`. The profile sends only
the core Messages API surface (no prompt caching, thinking, effort or beta headers — Poe does not document
them). Poe's API listed `claude-opus-4.8 / 4.7 / 4.6 / 4.5`, `claude-sonnet-4.6 / 4.5`, `claude-haiku-4.5` on
2026-08-28 (no Opus 5).

```bash
export POE_API_KEY=...                                     # or put it in .env (see below)
python -m agent.harness --llm-profile poe --llm-check      # 1 tiny request per role model: validates key + model ids
python -m agent.harness --llm-profile poe --max-iters 1 --label smoke   # one real iteration (~5 min)
python -m agent.harness --llm-profile poe --label official              # the real run
```
`--llm-check` also works for the default Anthropic profile (`python -m agent.harness --llm-check`).

**Keys in a `.env` file:** create `<repo>/.env` containing one line, `POE_API_KEY=...` (or `ANTHROPIC_API_KEY=...`). The harness
loads `<repo>/.env` automatically (or `--env-file PATH`); variables already exported in the shell take precedence.
`.env*` is gitignored, values are never printed or logged, the sandbox strips them from every experiment's
environment, and (on macOS) experiments are denied read access to the file itself.

## Run
```bash
# 1. self-check: Phase 0 only (rungs + organizers' FM + champion) — ~2 min, no API key needed
python -m agent.harness --mock --phase0-only

# 2. the official run (needs the real key): Phase 0, then up to 50 iterations, then finalize
python -m agent.harness --label official

# 3. resume after a crash / Ctrl-C (the restart is recorded as an intervention)
python -m agent.harness --run-dir runs/<RUN_ID>

# 4. offline dry run on the real data with the deterministic mock roles (no key)
python -m agent.harness --mock --label dryrun

# 5. the Phase-1 toy loop (synthetic mini dataset, seconds)
python -m agent.harness --toy --mock

# record a manual intervention (bumps the counter; --block marks a direction BLOCKED for the Researcher)
python -m agent.intervene "restarted with a longer timeout" --stuck "it07 hung" --scope config --run-dir runs/<RUN_ID>
```
Useful flags: `--max-iters N`, `--session-iters N` (stop this process after N iterations, resumable),
`--set run.EXPERIMENT_TIMEOUT_S=1200` (override any config key), `--config other.yaml`.

Designating the official run = pointing at one folder: `runs/<RUN_ID>/` contains
`submission.csv` (validated by the kit checker at finalize), `results_summary.md`, `ledger.md`,
`state.md`, `logs/iter_NN.json` (+ `.md` narrative), `iterations/itNN/` (code, diff basis, preds,
stdout/stderr, debug attempts, LLM transcripts), `best/` (champion code + score + val preds),
`interventions.md`, `run_state.json`, `llm_calls.jsonl`, `phase0/`, `finalize/`.

## Reproduce
1. `python -m pytest -q` — the full suite (≈ 3 min; the real-data Phase 0 test is skipped when the
   data is absent).
2. `python -m agent.harness --mock --phase0-only` — must print `random ≈ 0.483`, `pop ≈ 0.581`,
   `official FM ≈ 0.6015`, `champion ≈ 0.6015` (difference 0.0: the champion is a bit-for-bit port).
3. `python -m agent.harness --mock --label dryrun` — deterministic offline end-to-end run on the real
   data (see `runs/example_run/` for what it produces and the numbers below).
4. Re-score any run's champion by hand with the organizers' tools:
   `python starter_kit/submit.py --score --split valid --data_dir starter_kit/KuaiRand-Pure/data runs/<RUN_ID>/best/preds_val.csv`
   and check the submission: `python sealed/submit_check.py --split test --data_dir starter_kit/KuaiRand-Pure/data runs/<RUN_ID>/submission.csv`.

## Layout
```
agent/        harness.py (loop, resume, finalize) · promotion.py (pure decisions) · memory.py (ledger/state/logs)
              roles.py (briefing assembly, parsing, re-ask) · llm_client.py (Anthropic + mock, token accounting)
              sandbox.py (timeout, env stripping, sandbox-exec) · tools.py · task.py · phase0.py · toy.py · intervene.py
prompts/      researcher.md engineer.md debugger.md scribe_lesson.md scribe_logentry.md (system + task template)
knowledge/    library.md — domain playbook injected into every Researcher briefing
baseline_repro/pipeline.py — iteration-0 champion (organizers' FM in the pipeline contract)
sealed/       evaluate.py (verbatim) · submit_check.py (wraps starter_kit/submit.py --check)
starter_kit/  organizer kit, untouched · data_cache/ derived train+valid-only copy · runs/ one dir per run
tests/        promotion · ledger · checkpoint · resume · sealed/phase0/submission · roles · fault injection · intervene
```

## Example run (`runs/example_run/`, offline mock roles on the real data)
`python -m agent.harness --mock --label dryrun` — deterministic, no API key. The mock Researcher follows a
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
* **No real-API run yet.** This environment had no `ANTHROPIC_API_KEY`, so the Anthropic client is verified only
  against a fake transport (request shape, usage parsing, refusal handling). Before the official run, execute
  `python -m agent.harness --max-iters 1 --label smoke` with the key exported and check `llm_calls.jsonl`
  (real `input_tokens`/`cache_read_input_tokens`, `estimated_usage: false`). If the server-side-fallback beta is
  rejected by the account, set `llm.refusal_fallbacks: false`.
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
