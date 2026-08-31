# Results summary — 20260830_224430_seeded_0605_v2

- stop reason: **converged**
- best validation: **primary 0.6050** (GAUC 0.6718 / nDCG@5 0.5383) at it00
- published baseline (valid): 0.6016 → delta **+0.0034**
- iterations used: 3 (promoted 0, failed 0); final streak 3
- best leak-clean measurement: it01 at 0.6051 (below the promotion margin; used for the submission)
- tokens: 184,018 total (124,406 in / 59,612 out) over 16 LLM calls; by role: {"researcher": 118153, "engineer": 55703, "scribe_lesson": 3757, "scribe_logentry": 3743, "scribe_digest": 2662}
- wall-clock: 0:16 (started 2026-08-30T14:44:32Z)
- manual interventions: 0 (resumes 0) — see interventions.md
- **provider spend: $0.9000** this run (real per-call cost, sums {"researcher": "google/gemini-3.1-pro-preview", "engineer": "google/gemini-3.1-pro-preview", "debugger": "deepseek/deepseek-v4-flash", "scribe": "mistralai/codestral-2508"}) — account balance moved $1.4036 in the same window (shared key: may include other activity)
- submission: OK — submission.csv from it01 (sealed checker)
- phase 0: random 0.482659838170463, pop 0.5807219293342882, official FM 0.601468756352959, champion it00 0.6050329621374994

## Promotions

(none beyond the baseline champion)

## Warnings

- sandbox isolation: none (WARNING: no OS-level network/write confinement on this host)
