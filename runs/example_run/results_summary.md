# Results summary — 20260828_181352_phase5_dryrun

- stop reason: **converged**
- best validation: **primary 0.6025** (GAUC 0.6685 / nDCG@5 0.5365) at it01
- published baseline (valid): 0.6016 → delta **+0.0009**
- iterations used: 3 (promoted 1, failed 0); final streak 3
- tokens: 42,382 total (32,511 in / 9,871 out) over 13 LLM calls; by role: {"researcher": 18252, "engineer": 16580, "debugger": 5258, "scribe_lesson": 947, "scribe_logentry": 1345}
- wall-clock: 0:03 (started 2026-08-28T10:13:55Z)
- manual interventions: 0 (resumes 0) — see interventions.md
- submission: OK — submission.csv from it01 (sealed checker)
- phase 0: random 0.482659838170463, pop 0.5807219293342882, official FM 0.601468756352959, champion it00 0.601468756352959

## Promotions

- it01: 0.6025 — Stronger L2 (1e-6 -> 1e-5) to regularise sparse id embeddings under temporal shift

## Warnings

- sandbox isolation: sandbox-exec
