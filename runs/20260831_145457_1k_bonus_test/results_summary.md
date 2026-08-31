# Results summary — 20260831_145457_1k_bonus_test

- stop reason: **iter_cap**
- best validation: **primary 0.6563** (GAUC 0.6788 / nDCG@5 0.6339) at it05
- self-measured baseline (not organizer-published — no baseline_scores.json exists for KuaiRand-1K; 0.6428 is this session's own reproduction of the unchanged organizer FM via `starter_kit_1k/submit.py --make`, single seed 0) (valid): 0.6428 → delta **+0.0135**
- iterations used: 5 (promoted 4, failed 0); final streak 0
- best leak-clean measurement: it05 at 0.6563
- tokens: 494,477 total (361,783 in / 132,694 out) over 31 LLM calls; by role: {"researcher": 363618, "engineer": 114245, "scribe_lesson": 5694, "scribe_logentry": 5525, "scribe_digest": 5395}
- wall-clock: 7:26 (started 2026-08-31T06:55:23Z)
- manual interventions: 3 (resumes 3) — see interventions.md
- **provider spend: $2.0079** this run (real per-call cost, sums {"researcher": "google/gemini-3.1-pro-preview", "engineer": "google/gemini-3.1-pro-preview", "debugger": "deepseek/deepseek-v4-flash", "scribe": "mistralai/codestral-2508"}) — account balance moved $9.9112 in the same window (shared key: may include other activity)
- submission: OK — submission.csv from it05 (sealed checker)
- phase 0: random 0.4333811199379805, pop 0.5426657351054346, official FM 0.6450814148154491, champion it00 0.6410755222093484

## Promotions

- it02: 0.6489 — Extending the FM to a DeepFM by adding a 1-layer MLP over the concatenated embeddings and numerical features will allow the model to learn arbitrary high-order feature interactions, providing a stronger personalization signal on this large 5M-row dataset.
- it03: 0.6492 — Adding user historical long_view rates and item/author auxiliary feedback rates (click, like) as past-only numerical features will provide DeepFM's MLP with rich interaction surfaces, allowing it to learn non-linear personalized generosity-vs-quality crosses and raising the ranking metric.
- it04: 0.6528 — Standardizing past-only numerical features will stabilize DeepFM's gradients against scale imbalances, adding missing user click/like rates will complete the behavioral priors, and within-user rank ensembling will optimally align the predictions with the GAUC evaluation metric, jointly exceeding the threshold.
- it05: 0.6563 — Adding past-only user-tab specific historical impression and positive rates as numerical features will give the DeepFM MLP a highly personalized, context-aware baseline for each user's generosity across different UI tabs, improving within-user ranking.

## Warnings

- sandbox isolation: none (WARNING: no OS-level network/write confinement on this host)
- it04: scribe synthesis rejected (contained a number not in the digest)
- KuaiRand-1K is not natively wired into this harness (only KuaiRand-Pure ships a starter kit + published
  baseline_scores.json). Before this run's Phase 0, a human/coding-assistant built the parallel kit this run
  depends on: `starter_kit_1k/{data,evaluate,baseline,submit}.py` (filename-swapped ports of the organizer's
  own unchanged code), `starter_kit_1k/baseline_scores_1k.json` (self-measured reference rungs, see the
  baseline-label note above), the `--seed-champion` pipeline at `runs/manual_1k_test/seed_champion_1k/`
  (a memory-safety fix — the original loader OOM'd past ~60GB on 1K's larger log files — applied to a
  pre-existing champion, not a new model), and two additive one-line filename extensions in
  `agent/tools.py` (loop-data masking + leak-test defaults). None of this touched the Researcher/Engineer/
  Debugger loop itself, which ran unmodified against this new kit. This is disclosed here because it is
  materially different from the KuaiRand-Pure runs, which use the harness's existing, already-wired kit
  with no comparable setup step.
- of the 3 manual interventions (see interventions.md): 1 was a deliberate human-requested pause (paused
  cleanly after iteration 2 so the user could leave; resumed on request, no data lost) and 1 was recovery
  from a genuine infrastructure failure (`openai.APIError: Upstream idle timeout exceeded` mid-iteration-5
  researcher call — an unhandled exception the harness did not itself catch or retry; a human/coding-
  assistant relaunched the process from the last saved state). The third is the routine transition from the
  `--phase0-only` dry run into the iteration loop proper. interventions.md's auto-generated rows do not
  distinguish these three cases from each other; this line is the disambiguation.
