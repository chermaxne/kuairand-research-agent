# researcher — researcher (model google/gemini-3.1-pro-preview, 25748 tokens)

## system block 1

# ROLE: Researcher

You are the research lead of an autonomous ML research agent working on the KuaiRand-Pure benchmark
(TechJam 2026, Track 2). Your only job is to decide WHAT the next experiment is. A separate Engineer
implements it, a deterministic harness runs it in a sandbox, and the organizers' sealed `evaluate.py`
scores it. You never see raw results before they are measured, and nothing you write is ever treated
as a score, decision or streak — the harness owns all of that.

## Objective
Maximise the validation **primary** metric = mean(GAUC, nDCG@5), computed within-user over the logged
impressions of the validation split, label `long_view`. The champion to beat is a numpy factorization
machine (published validation primary 0.6016). Promotion needs primary > best + 0.0002 (small clean gains are
banked); the convergence streak resets ONLY on an improvement > 0.0020 over the best-so-far, and everything else —
a +0.001 gain, a flat result, a crash, a timeout — counts as a miss. Three consecutive misses end the run. The
threshold is 2.5σ of the seed noise (σ = 0.0008): a gain has to be real AND large to register.

## What you receive each iteration (in this order)
1. STATE BLOCK — current best, budget, streak, BLOCKED list, active themes (all harness-measured).
2. DATA PROFILE — split sizes, positive rate, available columns.
3. CHAMPION CODE — the exact file(s) every experiment must build on.
4. LEDGER — one line per past iteration: hypothesis, change, result, decision, lesson.
5. RESEARCH DIGEST — a harness-written table over every iteration: your predicted gain vs the measured one, the
   in-run ablations the pipeline printed (component attribution), decisions, leak verdicts, and a calibration line
   (how far your predictions have run above the measurements). A RESEARCH SYNTHESIS by the Scribe follows it.
6. RECENT ITERATION DETAILS — full records of the last few iterations: your spec, the diff, the measurement, the
   ablations, debug attempts, the training curve.
7. A SIZING DIRECTIVE (streak-aware posture) or a LAST-SHOT DIRECTIVE, and possibly a STALL RECOVERY DIRECTIVE —
   directives override the strategy rules below.

## Strategy rules (apply in this order)
0. **Size every proposal to clear +0.002 on its own.** The run ends after 3 consecutive misses (no gain > 0.002
   over the best-so-far; crashes count), so an iteration built to test a +0.001 lever is an iteration built to
   lose a life. Propose ONE hypothesis with a predicted gain (`expected_gain`, a number) that you can defend
   with evidence (`gain_evidence`: your own measured deltas from the digest, or published results with the
   assumptions checked), and stack every validated rider — a component already measured positive on this run
   that is not yet in the champion. A previous attempt that scored +0.0005..+0.002 is a rider to stack, never a
   reason to stop; a crashed or inverted attempt (GAUC < 0.5) is a good idea badly implemented — re-propose it
   with the fix. Read the digest's calibration line: if your predictions run high, predict smaller and pick
   bigger changes.
1. **YOUR MEASUREMENTS ARE THE ONLY EVIDENCE OF WHAT WORKS.** The knowledge file gives you task mechanics, the
   dataset's properties, the organizers' published findings, traps and the budget — it does NOT tell you which
   directions succeed on this split, because nobody has measured that. Your ledger and the RECENT ITERATION
   DETAILS are the evidence: read the deltas, read the diffs, and let them decide what to pursue next.
   Hyperparameter-only proposals are not allowed while the briefing carries a STRATEGY DIRECTIVE and are a last
   resort afterwards (they cannot clear +0.002).
1a. **Ground every proposal in published work.** The knowledge file's §8 maps the recommender-systems literature
   onto this task (ranking losses, sequential models, multi-task/ESMM, watch-time modelling, feature-interaction
   models, bias/drift, ensembling). Prefer methods with peer-reviewed or credible industry evidence over
   improvisation; in `rationale`, name the method and paper, state which of its assumptions this dataset satisfies
   (§1–§3 of the knowledge file) and which it does not, and derive the smallest faithful version from it.
   Novel ideas are welcome when the evidence points there — but say why the published alternatives do not apply.
1b. **Attribution happens inside the run, not across iterations.** Give an `ablation_plan`: the variants the
   pipeline should also train and score on validation and print as `ABLATION <name> primary=...` lines — at
   minimum the bundle WITHOUT the new component (champion-equivalent), and one line per rider if time allows.
   The next briefing shows those numbers next to the sealed result, so you learn what each part did without
   spending an iteration on it. Budget the extra fits in `change_spec` (the champion fit takes about a minute;
   the whole run must finish inside the wall-clock limit in the directive).
1c. **New information beats capacity.** The organizers measured that bigger embeddings and more static fields do
   nothing on this data: a deeper network over the same inputs is a large diff with a small expected gain. The
   changes with room to clear +0.002 are the ones that give the model a signal it does not have (the user's past
   behaviour as a sequence, auxiliary behaviours, watch time, past-only context) or an objective closer to the
   metric. An architecture change earns its place when it is what lets the model consume such a signal.
2. **Refine winners mid-run**: once a direction promoted, push it (its next obvious variant) before
   switching. Combine two proven winners when both promoted.
3. **When the flat streak is ≥ 2 this is the last shot**: propose the highest-probability > +0.002 bundle —
   the proven champion components kept intact, more seeds, plus one genuinely new signal. NEVER replace a
   component that is part of the champion's gain (its loss, its fields) at streak ≥ 2; and remember that a
   previous gain below 0.0006 is noise, not a signal to repeat that kind of lever.
4. **Never re-propose a failed or flat idea** unless you state a concrete new reason in `rationale`
   (e.g. "it02 crashed on memory; this variant uses 1/4 of the rows").
5. **Route around BLOCKED directions** entirely.
6. Every experiment must fit the pipeline contract: single self-contained `pipeline.py` (extra helper
   files allowed), train ONLY on the train split (validation may be used for early stopping), run in
   the wall-clock limit on one CPU box, no network, no package installs, only numpy / pandas /
   scikit-learn / lightgbm / torch(cpu). Budget the runtime explicitly in your change spec.
7. Be leakage-paranoid: same-row feedback columns are never features; any aggregate must be computed
   from strictly earlier dates (past-only). Say so in the change spec.

## How to write the change specification
The Engineer sees only your JSON and the champion code. Write `change_spec` as precise, numbered
instructions: which function to change, exact formulas, hyperparameters, feature definitions (with the
past-only rule spelled out), expected runtime, and what must NOT change (the CLI, the output format,
the train-only rule). One experiment = one hypothesis (plus its validated riders); do not bundle unrelated
ideas whose effects you cannot separate in the ablation plan.

## Output contract (strict)
Reply with ONLY one JSON object, no prose, no markdown fences:
{
  "hypothesis": "one sentence: what change and why it should raise primary",
  "category": "feature | model | training | multitask | other",
  "change_spec": "precise numbered instructions for the Engineer",
  "expected_risk": "low | medium | high",
  "expected_gain": 0.003,
  "gain_evidence": "why that number: your measured deltas (digest) and/or published results with assumptions checked",
  "ablation_plan": "named variants the pipeline must also score and print as ABLATION lines, e.g. 'champion_equiv: bundle without X; no_riders: X alone'",
  "builds_on": "champion",
  "rationale": "2-4 sentences citing ledger evidence and the knowledge library"
}

## system block 2

# KNOWLEDGE LIBRARY (domain playbook)

# KuaiRand-Pure within-user ranking — background briefing

What this file is: the task mechanics, the dataset's measurable properties, the organizers' own published findings,
the leakage traps, the runtime budget, and pointers to the relevant literature. What it is **not**: a list of what
works. No one has measured which directions succeed on this split — that is the research, and it is yours to do.
Your ledger is the evidence; this file only saves you from re-deriving public facts and from known failure modes.

## 1. Task, metric, and the noise floor
- Rank each user's logged impressions in the split; label `long_view`; primary = mean(GAUC, nDCG@5). GAUC counts
  only users with 0 < positives < impressions, weighted by #positives; nDCG@5 counts every user (all-negative
  users are stuck at 0). Validation: 124,909 rows, 22,377 users; **30.3% all-negative, 11.9% all-positive, 57.8%
  mixed — the mixed users hold 78.9% of the rows and are the only ones a model can move.** Oracle primary 0.848.
- Rungs: random 0.483 · item popularity 0.581 · **FM champion 0.6015** (5 fields: user_id, video_id, author_id,
  tab, duration decile; pointwise logloss; Adam 1e-3; batch 8192; early stop on validation, best epoch ≈ 7).
- Noise: repeated training runs of the same configuration differ by a few ten-thousandths in primary (seed noise;
  measure it yourself with 2–3 seeds before trusting a small delta). The validation metric itself has a
  user-bootstrap standard error of **0.0022** (95% ≈ ±0.0043): a single-run delta below ~0.002 is not evidence of
  a better model, and a validation gain below ~0.004 may not transfer to the hidden test week.
- Harness rules: promotion needs > PROMOTE_MARGIN over the champion (banked, so later iterations build on it); the
  convergence streak resets only on > EPSILON (0.002) over the best-so-far; **three consecutive misses end the run**
  (crashes count). Both thresholds are stated in your state block each iteration. Realistic final range for a


## 2. The label is a watch-time threshold — mechanics and consequences
- Definition (kuairand.com, verified 99.8% on the data with the feature-file `video_duration`):
  `long_view = 1` iff `play_time_ms >= duration` when duration ≤ 18 s, else `play_time_ms >= 18,000`.
  No negative in train has play_time ≥ 18 s. `is_click` in this single-column feed means *valid play* (≥ 7 s, or
  complete if shorter): **`is_click` and `long_view` are nested thresholds of the same variable.**
  P(long_view | click) = 0.72, P(long_view | no click) = 0.003, corr 0.76. P(long_view | like) = 0.68,
  P(long_view | profile_enter) = 0.76; like/follow/comment/forward are 1.9% / 0.10% / 0.26% / 0.10% sparse.
- Consequence for multi-task learning: an `is_click` head is (almost) the main label again, and the other
  feedbacks are too sparse to teach the shared embeddings anything — this is why every multi-task variant
  measured flat (§4). Watch-time regression heads also add nothing: the label already is a duration-normalised
  threshold of watch time, so the *duration-bias* machinery of the watch-time literature solves a problem this
  metric does not have.
- Duration effect is non-monotone and small (long_view rate: ≤ 7 s videos 0.50, 15–18 s 0.25, 18–25 s 0.32,
  quintiles 0.30–0.36), and the `video_id` embedding may already absorb most of it.
  2.1% of rows have `duration_ms == 0` (rate 0.07) — never divide by duration.
- Within-user ranking: any term constant within a user (user bias, user-only features, the user's own rate)
  cannot change the order. Only item-side terms and **user × item interactions** matter — id embeddings that cross
  are exactly what the FM does well, and why aggregate-rate features struggle to beat it.

## 3. Data facts that matter
- Tiny catalogue, dense interactions: 7,583 videos, 6,510 authors, 26,210 train users, 1.14M train rows.
  Impressions per video: median 44, p90 369, max 9,110. Per user in train: median 31, p90 97. Per user in
  valid: median 4, p90 12 (17.5% of valid users have a single impression).
- Cold start is a non-issue: **1.9% of valid users** (422) and 0.01% of valid rows' videos are unseen in train;
  1.6% of valid rows repeat a (user, video) pair seen in train.
- `tab` (UI scenario) is a strong within-user signal: tab 1 = 73% of rows, long_view rate 0.386; tab 0 = 13%,
  rate 0.042; tab 4 rate 0.49; tab 3 rate 0.004. 39.8% of valid users (51% of the mixed users) have more than one
  tab. Global tab rate alone scores 0.540.
- Item-level rates are stable across time: corr(train rate, valid rate) = 0.86 for videos with ≥ 20 impressions in
  both periods (0.59 without the valid-side filter). Video train rate alone scores 0.581 (= popularity rung).
- Drift: daily long_view rate falls from 0.336 to 0.290 over the three weeks. Recency-weighted training did not
  matter for a model fitted on the train window.
- Session structure is informative and label-free: the long_view rate falls from 0.339 at a user's first impression
  of the day to 0.175 at the 10th+; hour-of-day rates range 0.318–0.376. These come from `time_ms`, `hourmin`,
  `date` only — never from feedback columns.
- All videos have `upload_dt` in 2022-04-09..11 (dataset construction): video age is meaningless.
- `video_features_statistic_pure.csv` = average daily statistics "over one month" — a window that covers the test
  period, and it contains `long_time_play_cnt` / `play_progress`, i.e. the label aggregated over time.
  **Forbidden**, not merely risky. `show_cnt` is 0.88-correlated with train impression counts anyway.
- `log_random_4_22_to_5_08_pure.csv`: random-exposure rows, long_view rate 0.081 (vs 0.313 on the feed); the loop
  copy keeps only 0422–0428 (288k rows). Do not train on it and do not use it for validation in the loop.
- `user_features_pure.csv` (activity degree, follower buckets, 18 one-hots) and the extra video fields
  (music_id, video_type, upload_type, tag): the organizers measured no gain from adding them to the FM; user-side
  fields are ranking no-ops unless crossed with the item.

## 4. What the organizers have already published (their words, not ours)
From the starter-kit README — these are the organizers' own measurements on this dataset, so do not re-derive them:
- **Adding static features does not help.** They wired in all 13 CWM feature fields (+music_id / video_type /
  upload_type + 6 coarse user buckets) and got 0.5940 vs 0.5950 for the 5-field baseline — noise, if not slightly worse.
  `user_id × video_id` crosses already absorb most of the learnable signal, and 1.14M rows do not support more.
- **Adding capacity does not help.** Embedding dimension k = 8 / 16 / 32 gives 0.5895 / 0.5902 / 0.5887. Flat.
- **Pure user-side features contribute exactly zero** to a within-user ranking (see §2): any term constant within a
  user cannot reorder that user's rows. They only act through interactions with the item side.
- Their conclusion: *"the bottleneck is neither features nor capacity."*

The directions they list as unexplored and most promising, in their order:
1. **A loss aligned with the metric** — the objective is pointwise logloss while GAUC and nDCG@5 are ranking metrics;
   pairwise (BPR) or listwise (within-user softmax) alignment is the direction they consider most likely to work.
2. **User behaviour sequences** — every user has tens to hundreds of train interactions and none of it is used;
   DIN/SIM-style interest modelling is completely unexplored.
3. **Multi-objective** — `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `play_time_ms` as auxiliary
   tasks alongside long_view (but read §2 first: click and long_view are nested thresholds of the same variable).
4. **Watch-time modelling** — censored regression of play time (the CWM paper's contribution).
5. **Different model families** — DeepFM / DCN / xDeepFM, ranked below 1–4 because capacity is not the bottleneck.
6. **Time features and distribution drift** — `hourmin`, `date`, and the shift between train and test.
7. **Unbiased validation (advanced)** — `log_random_4_22_to_5_08_pure.csv` is a random-exposure log usable as an
   extra unbiased check on whether a gain is real or an artefact of biased traffic.

**Everything beyond this list is yours to discover.** No one has told you which of these works on this split, how large
any gain is, or in what order to try them: measure, read your ledger, and follow the evidence you generate.

## 5. Strategy under the convergence rule
1. **This file is background, not answers; your ledger is the evidence.** It gives you the task mechanics, the
   organizers' published findings, the traps and the budget. Which directions work on this split is unknown here —
   find out, and let your own measurements decide what to pursue.
   When an attempt failed for implementation reasons (crash, GAUC < 0.5, exploding loss), the idea is untested —
   fix it rather than move on.
2. **Rhythm: size every iteration to clear the threshold; attribute inside the run.** The rule is per iteration
   (each result vs the best-so-far; a +0.001 promotion still counts as a miss), so a lever worth less than the
   threshold cannot keep the run alive on its own. Propose one hypothesis big enough to matter, stack the riders you
   have already measured positive, and let the pipeline score the ablations (the bundle without the new component,
   each rider alone) on validation in the same run. Runtime is cheap; iterations are not.
2b. **New information beats capacity.** The organizers' negative results (§4) are about adding capacity and static
   fields to the same inputs. The directions they left open all add a signal the model does not have (§8.1–§8.4) or
   change the objective; treat "a deeper network over the same five ids" as the lowest-yield large change.
3. **Respect the noise floor.** A delta smaller than the seed-to-seed spread of your own pipeline is not evidence.
   Measure that spread once (same configuration, different seeds) and use it to decide what counts as a signal.
4. **Streak ≥ 2 is the last shot.** Take your highest-probability option and never replace a component that is part
   of the champion's gain — keep what works, add one genuinely new signal.
5. Put a self-check in every change spec that the code must print (train GAUC after epoch 1, pair count, a new
   field's vocabulary size) — a wasted iteration costs a third of the run. Assert only on TRAIN-side quantities:
   the harness re-runs every would-be promotion with ~10% of validation users' labels corrupted (leak test), so a
   hard assertion on the validation metric crashes that re-run and forfeits the promotion.
6. A result below the popularity rung (0.581) or a GAUC < 0.5 is an implementation bug, not a research outcome.
7. If the reachable levers are genuinely exhausted, converging at the plateau is the correct outcome — the rule is
   the organizers' definition of "done". Do not manufacture risky swaps to avoid it.

## 6. Trap list
- Same-row feedback columns (`is_click`, `play_time_ms`, `is_like`, …) as inputs = leakage. Session/time features
  may come only from `time_ms`, `hourmin`, `date`. **0.8484 is the validation oracle** (GAUC 1.0): a run in this
  project scored exactly that after the Engineer let the label into the encoded fields. Anything above ~0.65 is a
  leak until proven otherwise. The harness re-runs every would-be promotion on a copy of the data where 10% of the
  validation users' feedback columns are flipped, and scores those users' true labels; a pipeline that reads the
  labels ranks them inverted and is recorded as LEAK, never promoted. Keep the label out of every feature path: the row tuple's label element must be used only as
  the training target and for early stopping.
- `video_features_statistic_pure.csv` = label aggregates over a window covering the test period: forbidden.
- Whole-dataset aggregates leak the future; compute past-only (time-ordered) statistics. **Leave-one-out target
  encoding inverts the signal** (train features anti-correlate with their own label; validation features do not):
  measured 0.45–0.47 — below random.
- GAUC < 0.5 = inverted ranking (sign error); exploding loss (0.7 → 12) = bug; both burn an iteration.
- User-constant features are ranking no-ops but can dominate a pointwise learner's capacity (GBDT importance was
  led by `u_rate`/`u_n`).
- A weak ensemble member makes the ensemble worse; rank-average only members within ~0.01 of the champion.
- Adding several new fields at once can lower the score; one field per run.
- Seed std 0.0003; validation bootstrap SE 0.0022; do not read a single +0.001 as proof.
- A per-user Python loop over the rows (per-user masks, per-user list comprehensions) takes tens of minutes and hits
  the 900 s kill; vectorise (§7). Pairs rebuilt in Python every epoch quintuple the epoch time.
- The pipeline contract: fit on the train split only (validation only for early stopping); write every row of the
  requested split in file order; `--split test` must keep working; no network, no installs.

## 7. Engineering facts — runtime is a budget, and Python loops are how it gets blown
- Data load 3–4 s (pure-Python CSV); FM epoch ≈ 2 s pointwise (1.14M rows) / ≈ 1.5 s pairwise (≈ 380k pairs);
  sealed `evaluate` on valid 0.2 s; the baseline FM ≈ 30 s end to end; a multi-seed pipeline scales roughly linearly.
  **Hard limit 900 s per experiment** — a timeout is a lost iteration. Budget: ≤ 400 s for a 4–5-seed pipeline.
- Vectorise every per-user operation: `groupby(...).rank/transform/size` in pandas, or `np.unique(..., return_inverse)`
  + `np.argsort` / `np.add.at` in numpy. Never write `for u in users: mask = (users == u)` over the rows — quadratic.
- Build pair pools and index structures ONCE (per-user positive/negative index arrays), then resample per epoch with
  array indexing; rebuilding them in Python each epoch costs ~8 s per epoch on top of 1.8 s of training.
- Print progress that a reader can budget from: per-epoch time, pair count, seed number.
- Memory: the encoded train matrix is 1.14M × F int32 — trivial. Libraries: numpy, pandas, scikit-learn, lightgbm,
  torch (CPU). IDs are strings in the CSVs; echo them as read.

## 8. Recommender-systems domain knowledge — published methods, and where each applies to THIS task
Ground every proposal in a published method or a documented industry practice, and say in `rationale` which one and
why its assumptions hold for this data. The organizers' list in §4 tells you what is untried; this section tells you
what the field knows about each of those directions. Citations are venue + year so the Engineer can look them up.

### 8.1 Objectives aligned with ranking metrics (organizers' direction #1)
- **Pairwise / BPR** — Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009: optimise
  `−log σ(s_pos − s_neg)` over (user, positive, negative) triples. Pairwise losses are ranking-calibrated for AUC
  (Uematsu & Lee, JASA 2017): their minimiser induces the same ordering as the likelihood ratio, which is exactly what
  GAUC rewards. Within-user pairs (both items from the SAME user) are the form that targets a *grouped* AUC; a
  Meituan industry paper (PDAOM, arXiv 2023) reports GAUC gains from exactly that construction. Because all-positive
  and all-negative users contribute no pairs, ~40% of users here (§1) provide no gradient under a pure pairwise loss —
  a hybrid pointwise+pairwise objective or a pointwise warm-up are the documented remedies.
- **Listwise / softmax families** — sampled softmax is a tighter surrogate for top-k metrics than BPR and mines hard
  negatives implicitly (Wu et al., *On the Effectiveness of Sampled Softmax Loss for Item Recommendation*, TOIS 2024);
  PSL (Yang et al., 2024) generalises softmax loss to a family whose members bound DCG more tightly. For within-user
  ranking the natural list is the user's impressions (median 4 in validation, tens in train), so a per-user softmax
  over each user's rows is cheap. Metric-specific losses (LambdaRank-style, RBP-inspired) exist but optimising the
  evaluation metric directly is not always best (Li et al., SIGIR 2021).
- **Negative sampling matters as much as the loss** (Ma et al., *Negative Sampling in Recommendation: A Survey*,
  TOIS 2024; Yang et al., TPAMI 2024). Here negatives are *observed* impressions with label 0 — not missing data — so
  the classic implicit-feedback problem (unseen ≠ disliked) does not arise, and uniform within-user sampling is
  well-founded. Hard-negative and popularity-aware sampling (Prakash et al., 2024; Liu et al., 2023) change what the
  model learns and can reinforce or reduce popularity bias; popular-item negatives push the model toward user-specific
  preference rather than global popularity.

### 8.2 Sequential / user-interest models (organizers' direction #2)
- **Target attention over history** — DIN (Zhou et al., KDD 2018): represent the user by attending over their past
  items *with the candidate as the query*, so the same history yields a different user vector per candidate. This is a
  user × item interaction by construction, so it survives the within-user no-op rule (§2). DIEN adds interest
  evolution (AAAI 2019).
- **Self-attention over sequences** — SASRec (Kang & McAuley, ICDM 2018) and BERT4Rec (Sun et al., CIKM 2019) model
  next-item prediction; SASRec is noted to work on both sparse and dense data. On short histories (median 31 train
  impressions per user here), simpler models are competitive: the session-based evaluation of Ludewig & Jannach
  (UMUAI 2018) found nearest-neighbour and factorised-Markov methods matching or beating GRU4Rec. Start simple.
- **Long-sequence retrieval** — SIM (Pi et al., CIKM 2020), MIMN (KDD 2019), TWIN-V2 (Kuaishou, CIKM 2024) retrieve the
  relevant subset of very long histories; overkill at this dataset's history lengths, but the idea of restricting the
  history to items *related to the candidate* (same author, same tag) is cheap to borrow.
- Sequence features must be built past-only (§6): the user's history at row t = rows with earlier `time_ms`.

### 8.3 Multi-task and multi-behaviour learning (organizers' direction #3)
- **Shared-bottom → MMoE → PLE** — Ma et al., KDD 2018 (MMoE: per-task gates over shared experts) and Tang et al.,
  RecSys 2020 (PLE: explicit shared vs task-specific experts, addressing the *seesaw* effect where one task improves
  at another's expense). Gains require tasks that are related but different; negative transfer is the documented
  failure mode.
- **Entire-space / funnel modelling** — ESMM (Ma et al., SIGIR 2018): when behaviours are nested
  (impression → click → conversion), model p(later | earlier) over the entire impression space via the product
  p(click) · p(later | click). Read §2: `is_click` and `long_view` ARE nested thresholds of play time, so ESMM is the
  principled form here, while a naive auxiliary click head largely restates the main label. The sparse behaviours
  (like 1.9%, follow 0.1%, comment 0.3%) are the genuinely different signals, and they are sparse.

### 8.4 Watch time as a signal (organizers' direction #4)
- D2Q (Zhan et al., KDD 2022) deconfounds duration bias by quantile-normalising watch time within duration groups;
  TPM (Lin et al., KDD 2023) decomposes watch time into ordinal classification tasks arranged as a tree; CWM
  (Zhao et al., KDD 2024) treats complete plays as *censored* observations of the latent watch time — this is the
  paper whose evaluation protocol the organizers' `evaluate.py` follows, and it evaluates on KuaiRand-Pure; DML
  (Zhang et al., CIKM 2023) builds quantile-based labels; D2Co (RecSys 2023) separates interest from duration bias and
  noisy watching. The transferable ideas for a *thresholded* label: ordinal decomposition (predict several thresholds
  jointly) and censoring-aware losses; the duration-debiasing machinery targets a bias the label definition already
  normalises (§2).

### 8.5 Feature-interaction models (organizers' direction #5 — ranked low by them)
- FM (Rendle, ICDM 2010) is the champion's family. DeepFM (Guo et al., IJCAI 2017) adds an MLP over the same
  embeddings; xDeepFM (Lian et al., KDD 2018) adds explicit vector-wise high-order crosses (CIN); DCN/DCN-v2
  (Wang et al., 2017/2021) learn bounded-degree crosses cheaply; FiBiNET (Huang et al., RecSys 2019) adds
  SENET feature-importance gating and bilinear interactions; AutoFIS (Liu et al., KDD 2020) learns which field pairs
  to keep. Field-aware variants (FFM, FwFM, FEFM — Pande 2020) give each field pair its own interaction weight. The
  organizers found capacity (k) and static features flat for the plain FM; these models change the *form* of the
  interaction rather than its size, which is the only version of "more model" worth an iteration.
- Tabular ML comparisons (McElfresh et al., 2023; Borisov et al., TNNLS 2021; Gorishniy et al., 2021): GBDTs are
  strong on tabular data in general and are the classic pairing with FMs (Wang et al., 2019). Whether aggregate
  features can substitute for id-embedding interactions in a *within-user* ranking is an open question for you.

### 8.6 Bias and drift in logged feeds (organizers' directions #6–#7)
- **Position / exposure bias** — clicks depend on where and among what an item was shown (Joachims et al., WSDM 2017;
  Wang et al., WSDM 2018; Wu et al., *Unbiased LTR in Feeds Recommendation*, WSDM 2021 — context bias from
  surrounding items; Oosterhuis, TOIS 2022 — doubly-robust correction). Two consequences here: (i) an impression's
  position within the session is a legitimate, label-free context feature (`time_ms` order), and (ii) the
  random-exposure log (`log_random_*`, organizers' direction #7) is the textbook unbiased set for checking whether a
  gain reflects preference or the logging policy — KuaiRand exists precisely for this (Gao et al., CIKM 2022).
- **Temporal drift** — validation follows train in time (§1, §3); the standard remedies are time-aware training
  (recency weighting, fine-tuning on the latest days) and time-aware features; whether they help here is unmeasured.

### 8.7 Ensembling and variance reduction
- Rank-averaging or score-averaging several models is the most reliable gain in ranking competitions when the members
  are individually strong and different in kind; seed-averaging the same model is the cheapest variance reduction
  (Dietterich, 2000 on ensemble diversity; standard Kaggle practice). Within-user *rank* normalisation before averaging
  keeps the ensemble aligned with a within-user metric.

### How to use this section
Pick the direction, name the paper, state which of its assumptions this dataset satisfies (§1–§3) and which it does
not, then design the smallest faithful version and measure it. "DIN-style target attention over the user's past items,
because the metric is within-user and attention makes the user vector candidate-dependent" is a grounded proposal;
"add a deep network" is not.


## user

# STATE BLOCK
CURRENT BEST: it00 | val primary 0.6050 (GAUC 0.6718 / nDCG5 0.5383) | baseline 0.6016 | margin +0.0034
BUDGET: iteration 3 of 20 | 0:11 of 6:00 elapsed | tokens so far 103304
CONVERGENCE: streak 2 of 3 flat (EPSILON=0.002)
BLOCKED: none
ACTIVE THEMES: winning: none; losing/flat: feature[0 promoted/1 flat/0 failed], model[0 promoted/1 flat/0 failed]; untried: training, multitask, other


## Data profile (measured by the harness)
data dir: `/home/q3user/kuairand-research-agent/data_cache/loop_train_valid`

- train: 1,141,112 rows | 26,210 users | 7,538 videos | long_view rate 0.3366 | dates 20220409–20220421
- valid: 124,909 rows | 22,377 users | 5,951 videos | long_view rate 0.3133 | dates 20220422–20220428
- test: 0 rows (masked during the loop)
- train impressions per user: median 31, p90 97, max 809
- log columns: user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow, is_comment, is_forward, is_hate, long_view, play_time_ms, duration_ms, profile_stay_time, comment_stay_time, is_profile_enter, is_rand, tab
- user_features_pure.csv: 31 columns (user_id, user_active_degree, is_lowactive_period, is_live_streamer, is_video_author, follow_user_num, follow_user_num_range, fans_user_num, fans_user_num_range, friend_user_num, friend_user_num_range, register_days, …)
- video_features_basic_pure.csv: 12 columns (video_id, author_id, video_type, upload_dt, upload_type, visible_status, video_duration, server_width, server_height, music_id, music_type, tag)
- video_features_statistic_pure.csv: 52 columns (video_id, counts, show_cnt, show_user_num, play_cnt, play_user_num, play_duration, complete_play_cnt, complete_play_user_num, valid_play_cnt, valid_play_user_num, long_time_play_cnt, …)


# CHAMPION CODE (current best pipeline; every experiment builds on it)
--- pipeline.py ---
"""Iteration-0 champion: the organizers' FM baseline (starter_kit/baseline.py + data.py, k=16, lr=0.001,
batch 8192, <=40 epochs, patience 4, seed 0) ported UNCHANGED in behaviour to the pipeline contract:

    python pipeline.py --data <data_dir> --split val|test --out preds.csv

- trains ONLY on the train split (dates 20220408-20220421); validation is used for early stopping
  (exactly as the official baseline does); the requested split is scored and written in data order
  as row_id,user_id,video_id,score.
- self-contained on purpose: the Engineer edits THIS file (data loading, features, model, training).
- only numpy + the standard library are required; `evaluate` (the official metric) is imported for
  early stopping only — the harness scores predictions with the sealed copy, never this process.

Section map:  [1] config  [2] data loading  [3] feature encoding  [4] model  [5] training  [6] CLI
"""
import argparse
import collections
import csv
import importlib.util
import os
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------- [1] config
LABEL = "long_view"
SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428), "test": (20220429, 20220508)}
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket", "hour", "sess_depth"]   # 7 categorical fields
K = 16            # embedding dim
LR = 0.001
L2 = 1e-6
EPOCHS = 40
BATCH = 8192
PATIENCE = 4
SEED = 0
N_DUR_BUCKETS = 10


# ----------------------------------------------------------------------------- metric (early stopping only)
def _import_evaluate():
    try:
        from evaluate import evaluate  # sealed/ or starter_kit/ on PYTHONPATH (the harness sets it)
        return evaluate
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(7):
            for cand in (os.path.join(here, "sealed", "evaluate.py"), os.path.join(here, "starter_kit", "evaluate.py")):
                if os.path.exists(cand):
                    spec = importlib.util.spec_from_file_location("evaluate_for_es", cand)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod.evaluate
            here = os.path.dirname(here)
        raise ImportError("evaluate.py not found (needed for early stopping)")


evaluate = _import_evaluate()


# ----------------------------------------------------------------------------- [2] data loading (= starter_kit/data.py)
def load(data_dir):
    """Rows as (date, user_id, video_id, author_id, tab, duration_ms, label, hour, sess_depth);
    file order preserved (no global time sort)."""
    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    # read rows in original file order
    rows_orig = []   # will keep original order
    for f in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows_orig.append((int(r["date"]), r["user_id"], r["video_id"], vid2author.get(r["video_id"], "UNK"), r["tab"],
                                 float(r["duration_ms"]), 1 if r[LABEL] != "0" else 0, int(r["time_ms"]), int(r["hourmin"]) // 100))
    N = len(rows_orig)
    
    # indices sorted by time for historical stats
    time_order = sorted(range(N), key=lambda i: rows_orig[i][7])
    
    v_stats = collections.defaultdict(lambda: [0, 0])
    a_stats = collections.defaultdict(lambda: [0, 0])
    num_features = {}
    
    for i in time_order:
        x = rows_orig[i]
        date, _, vid, aid, _, _, label, _, _ = x
        v_imp, v_pos = v_stats[vid]
        a_imp, a_pos = a_stats[aid]
        
        v_rate = v_pos / v_imp if v_imp > 0 else 0.0
        a_rate = a_pos / a_imp if a_imp > 0 else 0.0
        
        num_features[i] = [np.log1p(v_imp), v_rate, np.log1p(a_imp), a_rate]
        
        if 20220408 <= date <= 20220421:
            v_stats[vid][0] += 1
            a_stats[aid][0] += 1
            if label == 1:
                v_stats[vid][1] += 1
                a_stats[aid][1] += 1

    # Compute session depth and time-gap from time-ordered sequence per (user, date)
    # Use a separate sort: order by user, date, time
    group_order = sorted(range(N), key=lambda i: (rows_orig[i][1], rows_orig[i][0], rows_orig[i][7]))
    user_date_counts = collections.defaultdict(int)
    user_date_last_time = {}
    depth_map = {}
    
    if len(group_order) == 0:
        depth_map = {}
    else:
        for idx in group_order:
            x = rows_orig[idx]
            key = (x[1], x[0])               # (user_id, date)
            curr_time = x[7]
            c = user_date_counts[key]
            user_date_counts[key] += 1
            depth = str(c) if c <= 4 else ("5-9" if c <= 9 else "10+")
            depth_map[idx] = depth
            
            time_gap = curr_time - user_date_last_time.get(key, curr_time)
            user_date_last_time[key] = curr_time
            num_features[idx].append(np.log1p(time_gap))
    
    # Build new rows in original order
    new_rows = []
    for i, x in enumerate(rows_orig):
        new_rows.append(x[:7] + (str(x[8]), depth_map[i], num_features[i]))
    rows = new_rows
    return {name: [x for x in rows if lo <= x[0] <= hi] for name, (lo, hi) in SPLITS.items()}


# ----------------------------------------------------------------------------- [3] feature encoding (= starter_kit/data.py)
def _bucket_edges(durations, n=N_DUR_BUCKETS):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


def encode(splits):
    """Categorical ids -> contiguous ints; unseen values fall into a per-field UNK slot.
    Returns ({split: (X int32 (N,F), X_num float32 (N,5), y float32, users)}, total_dim)."""
    tr = splits["train"]
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5]))), x[7], x[8]]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        X_num = np.empty((len(rws), 5), dtype=np.float32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            X_num[n] = x[9]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, X_num, y, users)
    return enc, int(sum(field_dims))


# ----------------------------------------------------------------------------- [4] model (= starter_kit/baseline.py FM)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Second-order factorization machine, pointwise logloss or BPR, Adam."""

    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.W_num = np.zeros(5, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mW_num = np.zeros_like(self.W_num); self.vW_num = np.zeros_like(self.W_num)
        self.t = 0

    def logits(self, X, X_num):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter, E, S

    def step(self, X, X_num, y):
        B = len(y)
        z, E, S = self.logits(X, X_num)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gW_num = (g[:, None] * X_num).sum(0)
        
        gV += self.l2 * self.V; gW += self.l2 * self.W
        gW_num += self.l2 * self.W_num
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW), (self.W_num, gW_num, self.mW_num, self.vW_num)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def step_bpr(self, X_pos, X_num_pos, X_neg, X_num_neg):
        B = len(X_pos)
        z_pos, E_pos, S_pos = self.logits(X_pos, X_num_pos)
        z_neg, E_neg, S_neg = self.logits(X_neg, X_num_neg)
        z_diff = z_pos - z_neg
        g = ((sigmoid(z_diff) - 1.0) / B).astype(np.float32)
        
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        
        np.add.at(gW, X_pos, g[:, None])
        np.add.at(gW, X_neg, -g[:, None])
        
        np.add.at(gV, X_pos, g[:, None, None] * (S_pos[:, None, :] - E_pos))
        np.add.at(gV, X_neg, -g[:, None, None] * (S_neg[:, None, :] - E_neg))
        
        gW_num = (g[:, None] * (X_num_pos - X_num_neg)).sum(0)
        
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        gW_num += self.l2 * self.W_num
        
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW), (self.W_num, gW_num, self.mW_num, self.vW_num)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
            
        return float(-np.mean(np.log(sigmoid(z_diff) + 1e-9)))

    def predict(self, X, X_num, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs], X_num[i:i + bs])[0] for i in range(0, len(X), bs)])


# ----------------------------------------------------------------------------- [5] training
def train(enc, dim, mode="pure_bpr", max_epochs=EPOCHS, seed=SEED, log=print):
    """Train on train, early-stop on validation primary. Returns the best model and its metrics."""
    Xtr, Xnum_tr, ytr, utr = enc["train"]
    Xva, Xnum_va, yva, uva = enc["valid"]
    
    if mode == "champion_equiv":
        Xnum_tr = np.zeros_like(Xnum_tr)
        Xnum_va = np.zeros_like(Xnum_va)
    
    if mode in ("pure_bpr", "champion_equiv"):
        user2pos = collections.defaultdict(list)
        user2neg = collections.defaultdict(list)
        for i, (u, y) in enumerate(zip(utr, ytr)):
            if y == 1.0:
                user2pos[u].append(i)
            else:
                user2neg[u].append(i)
                
        pos_indices = []
        neg_indices = []
        for u, pos_list in user2pos.items():
            neg_list = user2neg.get(u, [])
            if len(neg_list) > 0:
                for p in pos_list:
                    pos_indices.append(p)
                    neg_indices.append(neg_list)
                    
        pos_indices = np.array(pos_indices, dtype=np.int32)
        lens = np.array([len(n) for n in neg_indices], dtype=np.int32)
        
        if len(neg_indices) > 0:
            flat_negs = np.concatenate(neg_indices).astype(np.int32)
            offsets = np.cumsum([0] + list(lens[:-1]), dtype=np.int32)
        else:
            flat_negs = np.array([], dtype=np.int32)
            offsets = np.array([], dtype=np.int32)
            
        num_pairs = len(pos_indices)
        log(f"[{mode}] Total within-user pairs: {num_pairs}")

    m = FM(dim, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, best_metrics, bad = -1.0, None, None, 0
    
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        losses = []
        
        if mode in ("pure_bpr", "champion_equiv"):
            if num_pairs == 0:
                break
            sampled_neg = flat_negs[offsets + (rng.random(num_pairs) * lens).astype(np.int32)]
            idx = rng.permutation(num_pairs)
            for i in range(0, num_pairs, BATCH):
                b_idx = idx[i:i + BATCH]
                p_idx = pos_indices[b_idx]
                n_idx = sampled_neg[b_idx]
                losses.append(m.step_bpr(Xtr[p_idx], Xnum_tr[p_idx], Xtr[n_idx], Xnum_tr[n_idx]))
        else:
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), BATCH):
                b_idx = idx[i:i + BATCH]
                losses.append(m.step(Xtr[b_idx], Xnum_tr[b_idx], ytr[b_idx]))
                
        va = evaluate(uva, yva, m.predict(Xva, Xnum_va))
        loss_val = np.mean(losses) if losses else 0.0
        log(f"[{mode}] epoch {ep:2d} | loss {loss_val:.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
            f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
            
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), m.W_num.copy(), np.float32(m.b))
            best_metrics = va.copy()
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"[{mode}] early stop at epoch {ep}")
                break
                
    if best_state is not None:
        m.V, m.W, m.W_num, m.b = best_state
    return m, best_metrics


# ----------------------------------------------------------------------------- [6] CLI (pipeline contract)
def write_preds(path, rows, scores):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (x, s) in enumerate(zip(rows, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["val", "valid", "test"])
    ap.add_argument("--out", default="preds_val.csv")
    a = ap.parse_args()
    
    split = "valid" if a.split in ("val", "valid") else "test"
    t_start = time.time()
    budget = float(os.environ.get("KUAIRAND_TIME_BUDGET_S", 1500))
    fast = os.environ.get("KUAIRAND_FAST", "0") == "1"
    
    splits = load(a.data)
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}", f"load {time.time() - t_start:.1f}s")
    enc, dim = encode(splits)
    
    # 1. Full fit on champion (pure_bpr) and write predictions
    X, X_num, _, _ = enc[split]
    Xva, Xnum_va, yva, uva = enc["valid"]
    
    all_scores = []
    all_scores_va = []
    last_single_metrics = None
    
    seeds = [42] if fast else [42, 43, 44, 45, 46]
    for s in seeds:
        model_bpr, metrics_bpr = train(enc, dim, mode="pure_bpr", seed=s)
        all_scores.append(model_bpr.predict(X, X_num))
        if not fast:
            all_scores_va.append(model_bpr.predict(Xva, Xnum_va))
        last_single_metrics = metrics_bpr
    
    mean_scores = np.mean(all_scores, axis=0)
    assert np.all(np.isfinite(mean_scores)), "non-finite scores"
    write_preds(a.out, splits[split], mean_scores)
    print(f"wrote {a.out}: {len(splits[split])} rows for split={split} in {time.time() - t_start:.0f}s")
    
    if last_single_metrics is not None:
        print(f"ABLATION pure_bpr_single primary={last_single_metrics['primary']:.4f} gauc={last_single_metrics['GAUC']:.4f} ndcg5={last_single_metrics['nDCG@5']:.4f}")
    
    if not fast:
        mean_scores_va = np.mean(all_scores_va, axis=0)
        ens_metrics = evaluate(uva, yva, mean_scores_va)
        print(f"ABLATION pure_bpr_ensemble primary={ens_metrics['primary']:.4f} gauc={ens_metrics['GAUC']:.4f} ndcg5={ens_metrics['nDCG@5']:.4f}")
    
    if fast:
        return
        
    # 2. Ablations (champion_equiv)
    rem_budget = budget - (time.time() - t_start)
    if rem_budget >= 0.25 * budget:
        model_ce, metrics_ce = train(enc, dim, mode="champion_equiv", seed=42, max_epochs=EPOCHS)
        if metrics_ce is not None:
            print(f"ABLATION champion_equiv primary={metrics_ce['primary']:.4f} gauc={metrics_ce['GAUC']:.4f} ndcg5={metrics_ce['nDCG@5']:.4f}")
    else:
        print("ABLATION champion_equiv skipped: out of time budget")

if __name__ == "__main__":
    main()


# LEDGER (full history, oldest first)
# Ledger (tier-1 memory, append-only; one line per iteration, harness-written except LESSON)
# it00 champion installed from runs/20260830_165325_seeded_0605/best/code: val primary 0.6050 (GAUC 0.6718 / nDCG5 0.5383); published baseline 0.6016; rungs random 0.4827 pop 0.5807
[it01] HYP: Providing the model with strictly past-only video and author historical click (valid play) and like rates as numerical… | CHANGE: pipeline.py (+27/-13) | RESULT: 0.6051 (best 0.6050) -> kept | LESSON: Primary 0.6051 vs 0.6050, kept; early-stopped at epoch 9.
[it02] HYP: Generalizing the Factorization Machine to a Field-weighted FM (FwFM) will allow the model to learn the importance of di… | CHANGE: pipeline.py (+51/-22) | RESULT: 0.6049 (best 0.6050) -> kept | LESSON: FwFM primary=0.6049 gauc=0.6717 ndcg5=0.5380 kept; early-stopped at epoch 9.


# PRIOR RUNS — every experiment this agent has already measured (harness-recorded, earlier runs only)
These are YOUR OWN sealed measurements from previous runs of this same task, not advice. Do not spend an
iteration re-measuring something below unless you state what is different about your version. The deltas are
against the champion at that iteration's start, so a small delta on top of a strong champion is not the same
as a small delta on top of the baseline.

Best score ever recorded across all runs: **0.6050** (20260830_165325_seeded_0605 it03) — Stacking past-only numerical features (video/author historical rates and impression counts) and 5-seed ensembling (both validated riders) alongside a…

## WHAT WORKED — measured gains, largest first (13 of them)
| Δ vs then-champion | direction | what was tried | result |
|---|---|---|---|
| +0.0029 | training | Training with a within-user pairwise BPR loss directly aligns the objective with the evaluation metric (GAUC, nDCG@5) by optimizing relative ranking rather than absolute pointwise… | 0.6043 promoted |
| +0.0023 | training | Training with BPR loss on within-user positive-negative pairs directly aligns the objective with the ranking metric and eliminates user-bias confounding, increasing the primary me… | 0.6038 promoted |
| +0.0023 | training | Changing the pointwise logloss to a pairwise BPR loss aligned with the within-user ranking metric will directly optimise for the primary evaluation criteria and yield a structural… | 0.6038 promoted |
| +0.0021 | feature | Adding the user's daily session depth and hour-of-day as contextual categorical features will capture position bias and time context, and combining this with a 3-seed ensemble wil… | 0.6048 promoted |
| +0.0018 | feature | Adding the user's daily session depth and hour-of-day as categorical features will capture position bias and time context, and ensembling 3 seeds will reduce variance, jointly yie… | 0.6032 promoted |
| +0.0012 | training | Training with BPR loss on within-user positive-negative pairs directly aligns the objective with the ranking metric, raising primary. | 0.6027 promoted |
| +0.0010 | feature | Adding the user's daily session depth and hour-of-day as categorical context features models time and position bias, while a 3-seed ensemble reduces variance, together yielding a… | 0.6025 promoted |
| +0.0007 | feature | Adding daily session depth and hour-of-day as categorical context features, combined with a 3-seed ensemble, will capture position bias and time context to raise the primary metri… | 0.6022 promoted |
| +0.0005 | feature | Adding the user's daily session depth and hour-of-day as contextual categorical features captures position bias and time context, yielding new ranking signal. | 0.6048 promoted |
| +0.0005 | feature | Adding strictly past-only video and author historical long_view rates and impression counts as numerical features will provide strong item-quality signals, improving the BPR model… | 0.6042 kept_champion |
| +0.0004 | model | An ensemble of 3 BPR FMs trained with different random seeds will reliably reduce variance and boost the ranking metric by aggregating decorrelated predictions. | 0.6042 kept_champion |
| +0.0003 | feature | Stacking past-only numerical features (video/author historical rates and impression counts) and 5-seed ensembling (both validated riders) alongside a genuinely new numerical signa… | 0.6050 promoted |
| +0.0002 | feature | Stacking past-only item/author statistics as numerical features (a validated rider) along with session time-gap (a new signal) and a 5-seed ensemble will push the champion past th… | 0.6034 promoted |

## WHAT DID NOT WORK — measured losses or no movement (11 of them)
| Δ vs then-champion | direction | what was tried | result |
|---|---|---|---|
| -0.0115 | training | Replacing pointwise logloss with within-user pairwise BPR loss — which directly optimizes the same within-user ranking that GAUC and nDCG@5 measure — should raise primary because… | 0.5900 kept_champion |
| -0.0089 | training | Training with a within-user pairwise BPR loss directly aligns the objective with the primary ranking metrics (GAUC, nDCG@5), eliminating user-bias confounding and raising primary. | 0.5925 kept_champion |
| -0.0064 | feature | Adding the user's most recently interacted video IDs as past-only categorical fields will explicitly model sequential item-to-item transitions (Markov chains) and short-term inter… | 0.5984 kept_champion |
| -0.0046 | multitask | Adding an auxiliary MSE regression task on play_progress (play_time_ms / duration_ms) will provide a dense, continuous preference signal to the shared embeddings, improving the pr… | 0.6002 kept_champion |
| -0.0028 | feature | Adding the user's last 3 positively interacted videos mapped directly to the shared video_id embedding space will enable Factorized Personalized Markov Chains (FPMC) item-to-item… | 0.6019 kept_champion |
| -0.0010 | multitask | Adding an auxiliary pointwise logloss for is_click with shared embeddings and a weight of 0.5 will improve the representation of items and users, raising the primary long_view ran… | 0.6028 kept_champion |
| -0.0008 | feature | Adding strictly past-only historical long_view rates and impression counts for videos and authors as bucketed categorical fields will provide a dense item-quality signal that shar… | 0.6039 kept_champion |
| -0.0006 | feature | Ensembling 5 seeds, adding past-only global item/author rates (a validated rider), and injecting past-only user-author interaction rates (a new personalization signal) as numerica… | 0.6042 kept_champion |
| -0.0005 | training | Training with a hybrid pointwise logloss and within-user pairwise BPR loss will directly optimize the relative ordering of items for mixed users while maintaining calibration for… | 0.6027 kept_champion |
| -0.0003 | training | Replacing the pairwise BPR loss with a within-user sampled softmax loss over a list of 1 positive and 7 negatives will provide stronger gradients and implicitly mine hard negative… | 0.6045 kept_champion |
| -0.0001 | model | Implementing a DIN-style target attention over the user's past clicks provides strong explicit interest modeling, yielding significant new ranking signal that static FMs cannot ca… | 0.6048 kept_champion |

## WHAT BROKE — 3 iterations never produced a score (an implementation failure costs the same as a bad idea)
- other: (no valid plan: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction under consideration and either w…) — failed: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction un…
- other: (no valid plan: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction under consideration and either w…) — failed: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction un…
- other: (no valid plan: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction under consideration and either w…) — failed: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction un…

Attempts per direction across all prior runs: feature 12 (8 positive), model 2 (1 positive), multitask 2 (0 positive), training 8 (4 positive).

# RESEARCH DIGEST — every iteration so far, grouped by direction (harness-measured facts)
| it | direction | what changed | predicted Δ | measured Δ vs then-champion | decision | status | in-run ablations (pipeline-reported, unsealed) | lesson |
|---|---|---|---|---|---|---|---|---|
| it01 | feature | Providing the model with strictly past-only video and author historical click (valid play) and like rates as numerical features will inject granular item-engagement priors that differentiate between clickbait and high-p… | +0.0025 | +0.0001 | kept_champion | scored | pure_bpr_single 0.6030 (-0.0021 vs the full run); pure_bpr_ensemble 0.6051 (+0.0000 vs the full run); champion_equiv 0.6033 (-0.0018 vs the full run) | Primary 0.6051 vs 0.6050, kept; early-stopped at epoch 9. |
| it02 | model | Generalizing the Factorization Machine to a Field-weighted FM (FwFM) will allow the model to learn the importance of different field-pair interactions, upweighting critical crosses like user-video while discounting nois… | +0.0030 | -0.0002 | kept_champion | scored | pure_bpr_single 0.6030 (-0.0019 vs the full run); pure_bpr_ensemble 0.6049 (+0.0000 vs the full run); no_fwfm 0.6033 (-0.0016 vs the full run) | FwFM primary=0.6049 gauc=0.6717 ndcg5=0.5380 kept; early-stopped at epoch 9. |

Calibration: over 2 scored iterations your predicted gain exceeded the measured one by +0.0028 on average (predicted − measured); size the next prediction accordingly.
Totals: 2 iterations; promoted 0; attempts per direction: feature 1, model 1; never attempted: training, multitask, other.

# RESEARCH SYNTHESIS (written by the Scribe from the digest above — interpretive; verify any claim against the table)
Two iterations have been scored. In iteration 1, adding granular item-engagement features improved the primary metric by +0.0001, but the pure_bpr_ensemble ablation showed no change. In iteration 2, replacing the Factorization Machine with a Field-weighted FM (FwFM) decreased the primary metric by -0.0002, but the pure_bpr_ensemble ablation showed no change. The FwFM primary metric was 0.6049, and it was kept. No iterations have been promoted. Attempts per direction: feature 1, model 1. Directions never attempted: training, multitask, other.

# RECENT ITERATION DETAILS (harness-measured facts + what was actually changed)
Use these to decide whether to CONTINUE an idea: when a bundled change moved little, the diff shows which
components were in it, so you can keep the part that plausibly worked and drop the rest. State which
component you are keeping or dropping, and why, in `rationale`.

## it01 [feature] — kept_champion (scored), +0.0001 vs the then-champion 0.6050
HYPOTHESIS: Providing the model with strictly past-only video and author historical click (valid play) and like rates as numerical features will inject granular item-engagement priors that differentiate between clickbait and high-preference items, raising primary.
YOUR PREDICTED GAIN: +0.0025; measured +0.0001 — evidence given: Previous runs achieved +0.0005 by injecting past-only long_view rate. Click is a less strict threshold (nested threshold variable) and Like is a sparse but explicit positive behavior. Adding their historical rates provides the BPR model with dense, multi-objective item-quality signals without the n…
RATIONALE (yours, at the time): The knowledge file states that multi-task learning often fails here due to the nested nature of labels and sparsity of likes, but that auxiliary behaviors (clicks, likes) contain genuinely different signals. Explicitly aggregating them as past-only numerical features gives the model access to these multi-behavior signals safely. This satisfies the strategy of adding new information (auxiliary behaviors) while staying within a proven tabular architecture.
CHANGE SPEC you gave the Engineer:
1. In `load()`, modify the `rows_orig.append(...)` tuple to append two more elements: `1 if r['is_click'] != '0' else 0` and `1 if r['is_like'] != '0' else 0`. The tuple should now have 11 elements.
2. In `load()`, update `v_stats` and `a_stats` initializers to `lambda: [0, 0, 0, 0]` (to hold imp, pos_label, pos_click, pos_like).
3. In the `for i in time_order:` loop, unpack `x` to 11 variables: `date, _, vid, aid, _, _, label, _, _, click, like = x`.
4. Compute `v_click_rate = v_click / v_imp` and `v_like_rate = v_like / v_imp` (handling `v_imp == 0`), and similarly for author.
5. Build `num_features[i] = [np.log1p(v_imp), v_rate, v_click_rate, v_like_rate, np.log1p(a_imp), a_rate, a_click_rate, a_like_rate]`.
6. Under the `if 20220408 <= date <= 20220421:` block, increment the click and like stats: `if click == 1: v_stats[vid][2] += 1; a_stats[aid][2] += 1` and similarly for `like == 1`.
7. In `encode()`, change `X_num = np.empty((len(rws), 5), dtype=np.float32)` to use size `9`.
8. In `FM.__init__()`, change `self.W_num = np.zeros(5, dtype=np.float32)` to use size `9`.
9. In `train()`, update the `if mode == 'champion_equiv':` block to zero out only the newly added numerical fea…
WHAT CHANGED: pipeline.py (+27/-13)
DIFF (champion -> attempt):
```diff
--- champion/pipeline.py
+++ attempt/pipeline.py
@@ -71,26 +71,32 @@
         with open(os.path.join(data_dir, f)) as fh:
             for r in csv.DictReader(fh):
                 rows_orig.append((int(r["date"]), r["user_id"], r["video_id"], vid2author.get(r["video_id"], "UNK"), r["tab"],
-                                 float(r["duration_ms"]), 1 if r[LABEL] != "0" else 0, int(r["time_ms"]), int(r["hourmin"]) // 100))
+                                 float(r["duration_ms"]), 1 if r[LABEL] != "0" else 0, int(r["time_ms"]), int(r["hourmin"]) // 100,
+                                 1 if r["is_click"] != "0" else 0, 1 if r["is_like"] != "0" else 0))
     N = len(rows_orig)
     
     # indices sorted by time for historical stats
     time_order = sorted(range(N), key=lambda i: rows_orig[i][7])
     
-    v_stats = collections.defaultdict(lambda: [0, 0])
-    a_stats = collections.defaultdict(lambda: [0, 0])
+    v_stats = collections.defaultdict(lambda: [0, 0, 0, 0])
+    a_stats = collections.defaultdict(lambda: [0, 0, 0, 0])
     num_features = {}
     
     for i in time_order:
         x = rows_orig[i]
-        date, _, vid, aid, _, _, label, _, _ = x
-        v_imp, v_pos = v_stats[vid]
-        a_imp, a_pos = a_stats[aid]
+        date, _, vid, aid, _, _, label, _, _, click, like = x
+        v_imp, v_pos, v_click, v_like = v_stats[vid]
+        a_imp, a_pos, a_click, a_like = a_stats[aid]
         
         v_rate = v_pos / v_imp if v_imp > 0 else 0.0
+        v_click_rate = v_click / v_imp if v_imp > 0 else 0.0
+        v_like_rate = v_like / v_imp if v_imp > 0 else 0.0
+        
         a_rate = a_pos / a_imp if a_imp > 0 else 0.0
-        
-        num_features[i] = [np.log1p(v_imp), v_rate, np.log1p(a_imp), a_rate]
+        a_click_rate = a_click / a_imp if a_imp > 0 else 0.0
+        a_like_rate = a_like / a_imp if a_imp > 0 else 0.0
+        
+        num_features[i] = [np.log1p(v_imp), v_rate, v_click_rate, v_like_rate, np.log1p(a_imp), a_rate, a_click_rate, a_like_rate]
         
         if 20220408 <= date <= 20220421:
             v_stats[vid][0] += 1
@@ -98,6 +104,12 @@
             if label == 1:
                 v_stats[vid][1] += 1
                 a_stats[aid][1] += 1
+            if click == 1:
+                v_stats[vid][2] += 1
+                a_stats[aid][2] += 1
+            if like == 1:
+                v_stats[vid][3] += 1
+                a_stats[aid][3] += 1
 
     # Compute session depth and time-gap from time-order
… (diff truncated)
```
MEASURED: primary 0.6051 (GAUC 0.6719 / nDCG@5 0.5383), runtime 82.1s
IN-RUN ABLATIONS (pipeline-reported on validation, unsealed — component attribution): pure_bpr_single 0.6030 (-0.0021 vs the full run); pure_bpr_ensemble 0.6051 (+0.0000 vs the full run); champion_equiv 0.6033 (-0.0018 vs the full run)
  leak test: clean (flipped users scored 0.6090 on their true labels)
TRAINING CURVE (the experiment's own stdout):
  [champion_equiv] Total within-user pairs: 382579
  [champion_equiv] epoch  1 | loss 0.6654 | valid GAUC 0.6574 nDCG@5 0.5302 primary 0.5938 | 1.1s
  [champion_equiv] epoch  2 | loss 0.5914 | valid GAUC 0.6666 nDCG@5 0.5349 primary 0.6008 | 1.1s
  [champion_equiv] epoch  3 | loss 0.5599 | valid GAUC 0.6691 nDCG@5 0.5370 primary 0.6030 | 1.1s
  [champion_equiv] epoch  4 | loss 0.5513 | valid GAUC 0.6695 nDCG@5 0.5371 primary 0.6033 | 1.1s
  [champion_equiv] epoch  5 | loss 0.5458 | valid GAUC 0.6697 nDCG@5 0.5369 primary 0.6033 | 1.1s
  [champion_equiv] epoch  6 | loss 0.5435 | valid GAUC 0.6691 nDCG@5 0.5366 primary 0.6029 | 1.1s
  [champion_equiv] epoch  7 | loss 0.5398 | valid GAUC 0.6678 nDCG@5 0.5358 primary 0.6018 | 1.1s
  [champion_equiv] epoch  8 | loss 0.5365 | valid GAUC 0.6683 nDCG@5 0.5362 primary 0.6023 | 1.1s
  [champion_equiv] epoch  9 | loss 0.5314 | valid GAUC 0.6685 nDCG@5 0.5361 primary 0.6023 | 1.1s
  [champion_equiv] early stop at epoch 9
  ABLATION champion_equiv primary=0.6033 gauc=0.6697 ndcg5=0.5369
LESSON: Primary 0.6051 vs 0.6050, kept; early-stopped at epoch 9.

## it02 [model] — kept_champion (scored), -0.0002 vs the then-champion 0.6050
HYPOTHESIS: Generalizing the Factorization Machine to a Field-weighted FM (FwFM) will allow the model to learn the importance of different field-pair interactions, upweighting critical crosses like user-video while discounting noisy ones, raising primary.
YOUR PREDICTED GAIN: +0.0030; measured -0.0002 — evidence given: The organizers noted that capacity increases (higher k) and additional static fields do not improve the plain FM, implying uniform field interactions are a bottleneck. Field-weighted FM (FwFM, Pan et al., 2018) specifically addresses this by learning which field pairs matter most. Because R is init…
RATIONALE (yours, at the time): Following the directive to implement structural improvements that clear the +0.002 threshold, adding field-weighted interactions to the FM provides a new modeling capability. The baseline FM treats all field crosses equally, but within-user ranking heavily depends on specific pairs (user-video, user-tab). FwFM changes the *form* of the interaction, directly addressing the limitations noted in the knowledge library for feature-interaction models.
CHANGE SPEC you gave the Engineer:
1. In `FM.__init__`, add `freeze_r=False` to the arguments. Add `self.freeze_r = freeze_r` and `self.num_fields = len(FIELDS)`. Initialize `self.R = np.ones((self.num_fields, self.num_fields), dtype=np.float32)`, alongside `self.mR = np.zeros_like(self.R)` and `self.vR = np.zeros_like(self.R)`.
2. In `FM.logits`, replace the `S` and `inter` calculations with a loop: initialize `inter = np.zeros(len(X), dtype=np.float32)` and `S_weighted = np.zeros_like(E)`. Loop `i` from 0 to `self.num_fields` and `j` from `i+1` to `self.num_fields`. Inside, compute `dot = (E[:, i, :] * E[:, j, :]).sum(axis=1)`, add `self.R[i, j] * dot` to `inter`, and accumulate gradients: `S_weighted[:, i, :] += self.R[i, j] * E[:, j, :]` and `S_weighted[:, j, :] += self.R[i, j] * E[:, i, :]`. Return `z, E, S_weighted`.
3. In `FM.step` and `FM.step_bpr`, update `gV` logic: remove the `- E` part, just use `S_pos` (which is now `S_weighted` from logits) directly in `g[:, None, None] * S_pos`. 
4. In `FM.step` and `FM.step_bpr`, compute `gR = np.zeros_like(self.R)` using the same `i`, `j` loops: `gR[i, j] = (g * dot).sum()` (or `(g * (dot_pos - dot_neg)).sum()` for BPR). Add `self.l2 * self.R` to `gR`.
5. In both st…
WHAT CHANGED: pipeline.py (+51/-22)
DIFF (champion -> attempt):
```diff
--- champion/pipeline.py
+++ attempt/pipeline.py
@@ -176,7 +176,7 @@
 class FM:
     """Second-order factorization machine, pointwise logloss or BPR, Adam."""
 
-    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED):
+    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED, freeze_r=False):
         rng = np.random.default_rng(seed)
         self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
         self.W = np.zeros(dim, dtype=np.float32)
@@ -186,13 +186,24 @@
         self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
         self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
         self.mW_num = np.zeros_like(self.W_num); self.vW_num = np.zeros_like(self.W_num)
+        self.freeze_r = freeze_r
+        self.num_fields = len(FIELDS)
+        self.R = np.ones((self.num_fields, self.num_fields), dtype=np.float32)
+        self.mR = np.zeros_like(self.R)
+        self.vR = np.zeros_like(self.R)
         self.t = 0
 
     def logits(self, X, X_num):
         E = self.V[X]                                   # (B,F,k)
-        S = E.sum(1)                                    # (B,k)
-        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
-        return self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter, E, S
+        inter = np.zeros(len(X), dtype=np.float32)
+        S_weighted = np.zeros_like(E)
+        for i in range(self.num_fields):
+            for j in range(i + 1, self.num_fields):
+                dot = (E[:, i, :] * E[:, j, :]).sum(axis=1)
+                inter += self.R[i, j] * dot
+                S_weighted[:, i, :] += self.R[i, j] * E[:, j, :]
+                S_weighted[:, j, :] += self.R[i, j] * E[:, i, :]
+        return self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter, E, S_weighted
 
     def step(self, X, X_num, y):
         B = len(y)
@@ -200,14 +211,25 @@
         g = ((sigmoid(z) - y) / B).astype(np.float32)
         gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
         np.add.at(gW, X, g[:, None])
-        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
+        np.add.at(gV, X, g[:, None, None] * S)
         gW_num = (g[:, None] * X_num).sum(0)
+        
+        gR = np.zeros_like(self.R)
+        for i in range(self.num_fields):
+            for j in range(i + 1, self.num_fields):
+                dot = (E[:, i, :] * E[:, j, :]).sum(axis=1)
+                gR[i, j] = (g * dot).sum()
         
         gV += self.l2 * self.V; gW += self.l2 * self
… (diff truncated)
```
MEASURED: primary 0.6049 (GAUC 0.6717 / nDCG@5 0.5380), runtime 153.8s
IN-RUN ABLATIONS (pipeline-reported on validation, unsealed — component attribution): pure_bpr_single 0.6030 (-0.0019 vs the full run); pure_bpr_ensemble 0.6049 (+0.0000 vs the full run); no_fwfm 0.6033 (-0.0016 vs the full run)
TRAINING CURVE (the experiment's own stdout):
  [no_fwfm] Total within-user pairs: 382579
  [no_fwfm] epoch  1 | loss 0.6654 | valid GAUC 0.6574 nDCG@5 0.5302 primary 0.5938 | 2.3s
  [no_fwfm] epoch  2 | loss 0.5914 | valid GAUC 0.6666 nDCG@5 0.5349 primary 0.6008 | 2.3s
  [no_fwfm] epoch  3 | loss 0.5599 | valid GAUC 0.6691 nDCG@5 0.5370 primary 0.6030 | 2.4s
  [no_fwfm] epoch  4 | loss 0.5513 | valid GAUC 0.6695 nDCG@5 0.5371 primary 0.6033 | 2.4s
  [no_fwfm] epoch  5 | loss 0.5458 | valid GAUC 0.6697 nDCG@5 0.5369 primary 0.6033 | 2.4s
  [no_fwfm] epoch  6 | loss 0.5435 | valid GAUC 0.6691 nDCG@5 0.5366 primary 0.6029 | 2.4s
  [no_fwfm] epoch  7 | loss 0.5398 | valid GAUC 0.6678 nDCG@5 0.5358 primary 0.6018 | 2.3s
  [no_fwfm] epoch  8 | loss 0.5365 | valid GAUC 0.6683 nDCG@5 0.5362 primary 0.6023 | 2.4s
  [no_fwfm] epoch  9 | loss 0.5314 | valid GAUC 0.6685 nDCG@5 0.5361 primary 0.6023 | 2.3s
  [no_fwfm] early stop at epoch 9
  ABLATION no_fwfm primary=0.6033 gauc=0.6697 ndcg5=0.5369
LESSON: FwFM primary=0.6049 gauc=0.6717 ndcg5=0.5380 kept; early-stopped at epoch 9.

# LAST-SHOT DIRECTIVE (harness policy: flat streak 2 of 3)
One more iteration without a gain > +0.002 over the best-so-far (0.6050) ENDS THE RUN. Choose the
highest-probability bundle: keep every component of the champion that produced its gain (its loss, its fields,
its seed averaging) exactly as is, add more seeds if the champion uses fewer than 5, stack EVERY validated rider
not yet in the champion, and add ONE genuinely new signal. Do NOT replace or remove a proven component, do NOT
re-try a lever kind whose last result was within ±0.0006 (noise), and state in `rationale` and `gain_evidence`
why this bundle should clear +0.002. Keep the `ablation_plan` minimal (champion-equivalent only).

# TOOL USE (arxiv_search / web_fetch available this iteration)
You have access to arxiv_search and web_fetch tools. Budget: at most 3 search/fetch calls this
iteration (a hard cap of 6 tool-call turns applies on top of that as a backstop, not a
target). Paraphrase what you find in your own words; do not quote passages verbatim. web_fetch
only accepts a URL that arxiv_search itself returned (arxiv.org only) — it will refuse anything
else.

Before skipping web_search, you must explicitly justify it. Your JSON output must include an
ADDITIONAL top-level key, spelled EXACTLY "search_check" (all lowercase, a sibling of "hypothesis"
and "change_spec", not nested inside rationale or any other field), whose string value states:
(a) the specific technique/direction you are considering, and (b) why you're confident nothing
published in the last 12 months changes its risk profile or offers a better variant. "I already
know this technique" is not sufficient justification — cite what specifically makes this case
exempt (e.g. "this is a closed-form statistical baseline with no active research direction," not
"MMoE is well-established"). If you cannot articulate a specific reason beyond general
familiarity, search. The "search_check" key is required in every reply this iteration, whether or
not you actually called a tool; a reply missing this exact key is rejected and you will be asked
again.

If you find something durable and reusable beyond this one experiment, add another top-level key
spelled EXACTLY "new_knowledge" (all lowercase, optional): a string with a 2-3 sentence paraphrase
plus the source URL. The harness appends this to the knowledge library for future iterations —
don't restate it at length in gain_evidence beyond what justifies this iteration specifically.

# TASK
Decide the next experiment now. Consider the STATE BLOCK (streak, budget, BLOCKED), what the ledger
says worked / failed / was never tried, and the strategy rules. Reply with ONLY the JSON object
described in your role instructions (keys: hypothesis, category, change_spec, expected_risk, expected_gain,
gain_evidence, ablation_plan, builds_on, rationale). The harness will parse it; any other text makes the
iteration fail.

## assistant (response)


