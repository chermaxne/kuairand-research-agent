# Results summary — 20260901_011854_sonnet5_pure_v2

- stop reason: **converged**
- best validation: **primary 0.6048** (GAUC 0.6717 / nDCG@5 0.5379) at it02
- published baseline (valid): 0.6016 → delta **+0.0032**
- iterations used: 4 (promoted 2, failed 0); final streak 3
- best leak-clean measurement: it04 at 0.6049 (below the promotion margin; used for the submission)
- tokens: 472,975 total (346,197 in / 126,778 out) over 26 LLM calls; by role: {"researcher": 326639, "engineer": 131724, "scribe_lesson": 5308, "scribe_logentry": 5234, "scribe_digest": 4070}
- wall-clock: 0:40 (started 2026-08-31T17:18:56Z)
- manual interventions: 0 (resumes 0) — see interventions.md
- **provider spend: $1.9243** this run (real per-call cost, sums {"researcher": "anthropic/claude-sonnet-5", "engineer": "anthropic/claude-sonnet-5", "debugger": "deepseek/deepseek-v4-flash", "scribe": "mistralai/codestral-2508"}) — account balance moved $3.2715 in the same window (shared key: may include other activity)
- submission: OK — submission.csv from it04 (sealed checker)
- phase 0: random 0.482659838170463, pop 0.5807219293342882, official FM 0.601468756352959, champion it00 0.601468756352959

## Promotions

- it01: 0.6038 — Upgrading the FM to a DeepFM (add a 1-hidden-layer MLP over the concatenated field embeddings) and feeding it standardized past-only numerical priors (user/video/author long_view & click rates plus user×tab historical rate) gives the model new, genuinely predictive signal beyond raw id crosses, since these levers were independently validated as the three largest measured wins in this project's history (+0.0078, +0.0037, +0.0035 stacked to 0.6563 from an FM baseline).
- it02: 0.6048 — Adding past-only session/time-context categorical fields (hour-of-day, within-day session depth) plus count-based confidence weights (log1p of exposure counts for user/video/author/user-tab) to the DeepFM+numeric-prior champion gives the model label-free position-bias and reliability signal it currently lacks, without touching the loss or architecture that already validated a promotion this run.

## Warnings

- sandbox isolation: none (WARNING: no OS-level network/write confinement on this host)
