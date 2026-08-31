# Results summary — 20260901_005427_gemini_pro_baseline

- stop reason: **converged**
- best validation: **primary 0.6049** (GAUC 0.6716 / nDCG@5 0.5382) at it03
- published baseline (valid): 0.6016 → delta **+0.0033**
- iterations used: 3 (promoted 2, failed 1); final streak 3
- best leak-clean measurement: it03 at 0.6049
- tokens: 134,858 total (87,304 in / 47,554 out) over 16 LLM calls; by role: {"researcher": 72823, "engineer": 46541, "debugger": 6479, "scribe_lesson": 3288, "scribe_logentry": 3486, "scribe_digest": 2241}
- wall-clock: 0:11 (started 2026-08-31T16:54:29Z)
- manual interventions: 0 (resumes 0) — see interventions.md
- **provider spend: $0.7011** this run (real per-call cost, sums {"researcher": "google/gemini-3.1-pro-preview", "engineer": "google/gemini-3.1-pro-preview", "debugger": "deepseek/deepseek-v4-flash", "scribe": "mistralai/codestral-2508"}) — account balance moved $1.5222 in the same window (shared key: may include other activity)
- submission: OK — submission.csv from it03 (sealed checker)
- phase 0: random 0.482659838170463, pop 0.5807219293342882, official FM 0.601468756352959, champion it00 0.601468756352959

## Promotions

- it02: 0.6032 — Training the numpy FM with a within-user pairwise BPR loss directly aligns the optimization objective with the evaluation metrics (GAUC, nDCG@5), providing a stronger ranking signal than pointwise logloss.
- it03: 0.6049 — Adding the user's daily session depth and hour-of-day as past-only categorical features captures position and time bias, and combining this with a 5-seed score average will reduce variance and safely raise primary over the threshold.

## Warnings

- sandbox isolation: none (WARNING: no OS-level network/write confinement on this host)
- it03: scribe synthesis rejected (contained a number not in the digest)
