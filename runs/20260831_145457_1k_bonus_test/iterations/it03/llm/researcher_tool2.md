# researcher — researcher_tool2 (model google/gemini-3.1-pro-preview, 39354 tokens)

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
CURRENT BEST: it02 | val primary 0.6489 (GAUC 0.6762 / nDCG5 0.6215) | baseline 0.6428 | margin +0.0061
BUDGET: iteration 3 of 5 | 3:31 of 6:00 elapsed | tokens so far 126867
CONVERGENCE: streak 0 of 3 flat (EPSILON=0.002)
BLOCKED: none
ACTIVE THEMES: winning: model[1 promoted/1 flat/0 failed]; losing/flat: none; untried: feature, training, multitask, other


## Data profile (measured by the harness)
data dir: `/home/q3user/kuairand-research-agent/data_cache/loop_train_valid_1k`

- train: 5,055,984 rows | 983 users | 2,119,510 videos | long_view rate 0.2635 | dates 20220408–20220421
- valid: 2,524,980 rows | 978 users | 1,159,803 videos | long_view rate 0.2645 | dates 20220422–20220428
- test: 0 rows (masked during the loop)
- train impressions per user: median 3489, p90 11648, max 49242


# CHAMPION CODE (current best pipeline; every experiment builds on it)
--- pipeline.py ---
"""Memory-light rewrite of champion_1k.py for the KuaiRand-1K variant, whose interaction logs
(~11.7M rows across two files) are far larger than KuaiRand-Pure's sample despite the "1K" (user-count)
name. The original load()/encode() build several full-size parallel Python object structures (tuples of
strings, dicts of lists) at once and OOMs past ~60GB RSS on an 11.7M-row dataset.

Same features, same model (FM, pure_bpr / champion_equiv), same splits, same CLI contract as
champion_1k.py -- only load()+encode() are rewritten to use array.array/numpy int-coded columns
instead of Python tuples-of-strings, and vectorized numpy instead of per-row dict/list bookkeeping.
Vocab *index assignment order* differs from the original (np.unique's sorted order vs. first-appearance
order) -- this does not change model behaviour: embedding indices are exchangeable, only the trained
values differ, not what they represent or how well they generalize.

    python champion_1k_lowmem.py --data <data_dir> --split val|test --out preds.csv
"""
import argparse
import array
import csv
import importlib.util
import os
import time

import numpy as np

# ----------------------------------------------------------------------------- [1] config (= champion_1k.py)
LABEL = "long_view"
SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428), "test": (20220429, 20220508)}
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket", "hour", "sess_depth"]
K = 16
LR = 0.001
L2 = 1e-6
EPOCHS = 40
BATCH = 8192
PATIENCE = 4
SEED = 0
N_DUR_BUCKETS = 10


def _import_evaluate():
    try:
        from evaluate import evaluate
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


# ----------------------------------------------------------------------------- [2]+[3] data loading + encoding, fused
class VocabCoder:
    """str -> dense int32 code, order of first appearance (only used for the raw parse pass; index
    order here is irrelevant to model behaviour -- see module docstring)."""

    def __init__(self):
        self.d = {}

    def code(self, s):
        c = self.d.get(s)
        if c is None:
            c = len(self.d)
            self.d[s] = c
        return c

    def __len__(self):
        return len(self.d)


def _train_vocab_map(field_arr, train_mask):
    """Build a train-only vocab (unseen-in-train -> UNK) and apply it to the WHOLE field array,
    fully vectorized (np.searchsorted on the sorted unique-train-values array) instead of per-row
    dict lookups. Mirrors champion_1k.py's encode(): vocabs built from `tr` only, `unk[i] = len(vocab)`."""
    uniq = np.unique(field_arr[train_mask])
    unk_idx = len(uniq)
    pos = np.searchsorted(uniq, field_arr)
    pos_clip = np.clip(pos, 0, len(uniq) - 1)
    known = (pos < len(uniq)) & (uniq[pos_clip] == field_arr)
    mapped = np.where(known, pos_clip, unk_idx).astype(np.int32)
    return mapped, len(uniq) + 1  # field_dim = train vocab size + 1 UNK slot, exactly as champion_1k.py


def load_and_encode(data_dir):
    """Returns ({split: (X int32 (N,F), X_num float32 (N,5), y float32, u int32, uid_str list, vid_str list)},
    total_dim). uid_str/vid_str are only materialized for OUTPUT (write_preds needs real ids)."""
    t0 = time.time()
    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_1k.csv")) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    print(f"  video_features loaded ({len(vid2author)} videos) {time.time() - t0:.1f}s")

    date_a, user_a, video_a, author_a, tab_a = (array.array("i") for _ in range(5))
    dur_a = array.array("f")
    label_a = array.array("b")
    time_a = array.array("q")
    hour_a = array.array("i")

    user_vc, video_vc, author_vc, tab_vc = VocabCoder(), VocabCoder(), VocabCoder(), VocabCoder()
    author_vc.code("UNK")  # matches vid2author.get(vid, "UNK") in the original -- reserve the token up front

    for fname in ("log_standard_4_08_to_4_21_1k.csv", "log_standard_4_22_to_5_08_1k.csv"):
        with open(os.path.join(data_dir, fname)) as fh:
            for r in csv.DictReader(fh):
                vid_s = r["video_id"]
                date_a.append(int(r["date"]))
                user_a.append(user_vc.code(r["user_id"]))
                video_a.append(video_vc.code(vid_s))
                author_a.append(author_vc.code(vid2author.get(vid_s, "UNK")))
                tab_a.append(tab_vc.code(r["tab"]))
                dur_a.append(float(r["duration_ms"]))
                label_a.append(1 if r[LABEL] != "0" else 0)
                time_a.append(int(r["time_ms"]))
                hour_a.append(int(r["hourmin"]) // 100)
        print(f"  {fname} parsed, running total {len(date_a)} rows, {time.time() - t0:.1f}s")

    N = len(date_a)
    date_arr = np.frombuffer(date_a, dtype=np.int32)
    user_arr = np.frombuffer(user_a, dtype=np.int32)
    video_arr = np.frombuffer(video_a, dtype=np.int32)
    author_arr = np.frombuffer(author_a, dtype=np.int32)
    tab_arr = np.frombuffer(tab_a, dtype=np.int32)
    dur_arr = np.frombuffer(dur_a, dtype=np.float32)
    label_arr = np.frombuffer(label_a, dtype=np.int8)
    time_arr = np.frombuffer(time_a, dtype=np.int64)
    hour_arr = np.frombuffer(hour_a, dtype=np.int32)
    del date_a, user_a, video_a, author_a, tab_a, dur_a, label_a, time_a, hour_a
    print(f"  {N} rows -> typed arrays, {time.time() - t0:.1f}s")

    n_videos, n_authors = len(video_vc), len(author_vc)
    train_mask = (date_arr >= SPLITS["train"][0]) & (date_arr <= SPLITS["train"][1])

    # --- video/author historical-rate features, in chronological (time_ms) order; only train rows update
    #     the running counters -- identical semantics to champion_1k.py's load(), just array-backed. ---
    num_features = np.empty((N, 5), dtype=np.float32)
    v_imp = [0] * n_videos; v_pos = [0] * n_videos
    a_imp = [0] * n_authors; a_pos = [0] * n_authors
    time_order = np.argsort(time_arr, kind="stable").tolist()
    video_l, author_l, label_l, train_l = video_arr.tolist(), author_arr.tolist(), label_arr.tolist(), train_mask.tolist()
    log1p = np.log1p
    for i in time_order:
        vid, aid = video_l[i], author_l[i]
        vi, vp, ai, ap = v_imp[vid], v_pos[vid], a_imp[aid], a_pos[aid]
        num_features[i, 0] = log1p(vi)
        num_features[i, 1] = vp / vi if vi > 0 else 0.0
        num_features[i, 2] = log1p(ai)
        num_features[i, 3] = ap / ai if ai > 0 else 0.0
        if train_l[i]:
            v_imp[vid] = vi + 1
            a_imp[aid] = ai + 1
            if label_l[i] == 1:
                v_pos[vid] = vp + 1
                a_pos[aid] = ap + 1
    del v_imp, v_pos, a_imp, a_pos, time_order, video_l, author_l, label_l, train_l
    print(f"  historical-rate features done, {time.time() - t0:.1f}s")

    # --- session depth + time-gap-since-last-event, per (user, date) group ordered by time_ms ---
    depth_code = np.empty(N, dtype=np.int8)
    group_order = np.lexsort((time_arr, date_arr, user_arr)).tolist()  # last key = primary: user, then date, then time
    user_l, date_l, time_l = user_arr.tolist(), date_arr.tolist(), time_arr.tolist()
    counts, last_time = {}, {}
    for idx in group_order:
        key = (user_l[idx], date_l[idx])
        c = counts.get(key, 0)
        counts[key] = c + 1
        depth_code[idx] = c if c <= 4 else (5 if c <= 9 else 6)  # 0..4 raw counts, 5="5-9", 6="10+" (relabeled ints)
        t = time_l[idx]
        num_features[idx, 4] = log1p(t - last_time.get(key, t))
        last_time[key] = t
    del group_order, user_l, date_l, time_l, counts, last_time
    print(f"  session-depth features done, {time.time() - t0:.1f}s")

    # --- dur_bucket: quantile edges fit on TRAIN durations only, applied to every row (vectorized) ---
    edges = np.quantile(dur_arr[train_mask], np.linspace(0, 1, N_DUR_BUCKETS + 1)[1:-1])
    dur_bucket_arr = np.searchsorted(edges, dur_arr).astype(np.int32)

    # --- FIELDS = [user_id, video_id, author_id, tab, dur_bucket, hour, sess_depth]; train-only vocab + UNK,
    #     exactly matching champion_1k.py's encode() (unseen-in-train falls into the field's UNK slot). ---
    raw_fields = [user_arr, video_arr, author_arr, tab_arr, dur_bucket_arr, hour_arr, depth_code.astype(np.int32)]
    mapped_fields, field_dims = [], []
    for f in raw_fields:
        m, d = _train_vocab_map(f, train_mask)
        mapped_fields.append(m)
        field_dims.append(d)
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    X_full = np.stack([m + o for m, o in zip(mapped_fields, offsets)], axis=1).astype(np.int32)
    print(f"  categorical encoding done (dim={sum(field_dims)}), {time.time() - t0:.1f}s")

    # reverse maps for OUTPUT only (real ids the pipeline contract requires in the predictions CSV)
    uid_by_code = [None] * len(user_vc)
    for s, c in user_vc.d.items():
        uid_by_code[c] = s
    vid_by_code = [None] * len(video_vc)
    for s, c in video_vc.d.items():
        vid_by_code[c] = s

    enc = {}
    for name, (lo, hi) in SPLITS.items():
        idx = np.flatnonzero((date_arr >= lo) & (date_arr <= hi))  # ascending -> preserves original file order
        uid_list = [uid_by_code[c] for c in user_arr[idx]]
        vid_list = [vid_by_code[c] for c in video_arr[idx]]
        enc[name] = (X_full[idx], num_features[idx], label_arr[idx].astype(np.float32), user_arr[idx], uid_list, vid_list)
    print(f"  splits sliced, total load+encode {time.time() - t0:.1f}s")
    return enc, int(sum(field_dims))


# ----------------------------------------------------------------------------- [4] model (= champion_1k.py, unchanged)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED, use_mlp=True):
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
        
        self.use_mlp = use_mlp
        self.k = k
        if self.use_mlp:
            self.mlp_in_dim = 7 * k + 5
            self.mlp_h = 64
            self.mlp_w1 = rng.normal(0, 0.05, (self.mlp_in_dim, self.mlp_h)).astype(np.float32)
            self.mlp_b1 = np.zeros(self.mlp_h, dtype=np.float32)
            self.mlp_w2 = rng.normal(0, 0.05, (self.mlp_h, 1)).astype(np.float32)
            self.mlp_b2 = np.zeros(1, dtype=np.float32)
            
            self.mM_w1 = np.zeros_like(self.mlp_w1)
            self.vM_w1 = np.zeros_like(self.mlp_w1)
            self.mM_b1 = np.zeros_like(self.mlp_b1)
            self.vM_b1 = np.zeros_like(self.mlp_b1)
            self.mM_w2 = np.zeros_like(self.mlp_w2)
            self.vM_w2 = np.zeros_like(self.mlp_w2)
            self.mM_b2 = np.zeros_like(self.mlp_b2)
            self.vM_b2 = np.zeros_like(self.mlp_b2)

    def logits(self, X, X_num):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_out = self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter
        
        if self.use_mlp:
            E_flat = E.reshape(len(X), -1)
            mlp_in = np.concatenate([E_flat, X_num], axis=1)
            h1 = np.maximum(0, mlp_in.dot(self.mlp_w1) + self.mlp_b1)
            mlp_out = (h1.dot(self.mlp_w2) + self.mlp_b2).squeeze()
            return fm_out + mlp_out, E, S, h1, mlp_in
        else:
            return fm_out, E, S, None, None

    def step_bpr(self, X_pos, X_num_pos, X_neg, X_num_neg):
        B = len(X_pos)
        z_pos, E_pos, S_pos, h1_pos, mlp_in_pos = self.logits(X_pos, X_num_pos)
        z_neg, E_neg, S_neg, h1_neg, mlp_in_neg = self.logits(X_neg, X_num_neg)
        z_diff = z_pos - z_neg
        g = ((sigmoid(z_diff) - 1.0) / B).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X_pos, g[:, None])
        np.add.at(gW, X_neg, -g[:, None])
        
        gE_pos_fm = g[:, None, None] * (S_pos[:, None, :] - E_pos)
        gE_neg_fm = -g[:, None, None] * (S_neg[:, None, :] - E_neg)

        if self.use_mlp:
            g_out_pos = g[:, None]
            g_out_neg = -g[:, None]
            
            gh1_pos = g_out_pos.dot(self.mlp_w2.T) * (h1_pos > 0)
            g_mlp_w2 = h1_pos.T.dot(g_out_pos)
            g_mlp_b2 = g_out_pos.sum(0)
            g_mlp_w1 = mlp_in_pos.T.dot(gh1_pos)
            g_mlp_b1 = gh1_pos.sum(0)
            g_mlp_in_pos = gh1_pos.dot(self.mlp_w1.T)
            
            gh1_neg = g_out_neg.dot(self.mlp_w2.T) * (h1_neg > 0)
            g_mlp_w2 += h1_neg.T.dot(g_out_neg)
            g_mlp_b2 += g_out_neg.sum(0)
            g_mlp_w1 += mlp_in_neg.T.dot(gh1_neg)
            g_mlp_b1 += gh1_neg.sum(0)
            g_mlp_in_neg = gh1_neg.dot(self.mlp_w1.T)
            
            gE_pos_fm += g_mlp_in_pos[:, :7*self.k].reshape(-1, 7, self.k)
            gE_neg_fm += g_mlp_in_neg[:, :7*self.k].reshape(-1, 7, self.k)
            
            gW_num = (g[:, None] * (X_num_pos - X_num_neg)).sum(0)
            gW_num += g_mlp_in_pos[:, 7*self.k:].sum(0) + g_mlp_in_neg[:, 7*self.k:].sum(0)
            
            g_mlp_w2 += self.l2 * self.mlp_w2
            g_mlp_w1 += self.l2 * self.mlp_w1
        else:
            gW_num = (g[:, None] * (X_num_pos - X_num_neg)).sum(0)
            
        np.add.at(gV, X_pos, gE_pos_fm)
        np.add.at(gV, X_neg, gE_neg_fm)

        gV += self.l2 * self.V
        gW += self.l2 * self.W
        gW_num += self.l2 * self.W_num

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        
        params = [
            (self.V, gV, self.mV, self.vV), 
            (self.W, gW, self.mW, self.vW), 
            (self.W_num, gW_num, self.mW_num, self.vW_num)
        ]
        
        if self.use_mlp:
            params.extend([
                (self.mlp_w1, g_mlp_w1, self.mM_w1, self.vM_w1),
                (self.mlp_b1, g_mlp_b1, self.mM_b1, self.vM_b1),
                (self.mlp_w2, g_mlp_w2, self.mM_w2, self.vM_w2),
                (self.mlp_b2, g_mlp_b2, self.mM_b2, self.vM_b2)
            ])
            
        for P, G, M, Vv in params:
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

        return float(-np.mean(np.log(sigmoid(z_diff) + 1e-9)))

    def predict(self, X, X_num, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs], X_num[i:i + bs])[0] for i in range(0, len(X), bs)])


# ----------------------------------------------------------------------------- [5] training (= champion_1k.py, unchanged)
def train(enc, dim, mode="pure_bpr", max_epochs=EPOCHS, seed=SEED, log=print):
    Xtr, Xnum_tr, ytr, utr = enc["train"][0], enc["train"][1], enc["train"][2], enc["train"][3]
    Xva, Xnum_va, yva, uva = enc["valid"][0], enc["valid"][1], enc["valid"][2], enc["valid"][3]

    if mode == "champion_equiv":
        Xnum_tr = np.zeros_like(Xnum_tr)
        Xnum_va = np.zeros_like(Xnum_va)

    user2pos, user2neg = {}, {}
    for i, (u, y) in enumerate(zip(utr.tolist(), ytr.tolist())):
        (user2pos if y == 1.0 else user2neg).setdefault(u, []).append(i)

    pos_indices, owner = [], []       # owner: per-positive index into neg_lists/lens/offsets (one slot per
    neg_lists = []                    # USER, not one per positive -- avoids a pos_u*neg_u blowup below)
    user2neg_slot = {}
    for u, pos_list in user2pos.items():
        neg_list = user2neg.get(u, [])
        if neg_list:
            slot = user2neg_slot.get(u)
            if slot is None:
                slot = len(neg_lists)
                neg_lists.append(neg_list)
                user2neg_slot[u] = slot
            for p in pos_list:
                pos_indices.append(p)
                owner.append(slot)

    pos_indices = np.array(pos_indices, dtype=np.int32)
    owner = np.array(owner, dtype=np.int32)
    lens = np.array([len(n) for n in neg_lists], dtype=np.int32)
    if len(neg_lists) > 0:
        flat_negs = np.concatenate(neg_lists).astype(np.int32)
        offsets = np.cumsum([0] + list(lens[:-1]), dtype=np.int32)
    else:
        flat_negs = np.array([], dtype=np.int32)
        offsets = np.array([], dtype=np.int32)
    num_pairs = len(pos_indices)
    log(f"[{mode}] Total within-user pairs: {num_pairs}")

    use_mlp = (mode != "champion_equiv")
    m = FM(dim, seed=seed, use_mlp=use_mlp)
    rng = np.random.default_rng(seed)
    best, best_state, best_metrics, bad = -1.0, None, None, 0

    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        losses = []
        if num_pairs == 0:
            break
        sampled_neg = flat_negs[offsets[owner] + (rng.random(num_pairs) * lens[owner]).astype(np.int32)]
        idx = rng.permutation(num_pairs)
        for i in range(0, num_pairs, BATCH):
            b_idx = idx[i:i + BATCH]
            p_idx = pos_indices[b_idx]
            n_idx = sampled_neg[b_idx]
            losses.append(m.step_bpr(Xtr[p_idx], Xnum_tr[p_idx], Xtr[n_idx], Xnum_tr[n_idx]))

        va = evaluate(uva.tolist(), yva.tolist(), m.predict(Xva, Xnum_va))
        loss_val = np.mean(losses) if losses else 0.0
        log(f"[{mode}] epoch {ep:2d} | loss {loss_val:.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
            f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")

        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            if m.use_mlp:
                best_state = (m.V.copy(), m.W.copy(), m.W_num.copy(), np.float32(m.b), m.mlp_w1.copy(), m.mlp_b1.copy(), m.mlp_w2.copy(), m.mlp_b2.copy())
            else:
                best_state = (m.V.copy(), m.W.copy(), m.W_num.copy(), np.float32(m.b))
            best_metrics = va.copy()
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"[{mode}] early stop at epoch {ep}")
                break

    if best_state is not None:
        if m.use_mlp:
            m.V, m.W, m.W_num, m.b, m.mlp_w1, m.mlp_b1, m.mlp_w2, m.mlp_b2 = best_state
        else:
            m.V, m.W, m.W_num, m.b = best_state
    return m, best_metrics


# ----------------------------------------------------------------------------- [6] CLI (= champion_1k.py contract)
def write_preds(path, uid_list, vid_list, scores):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v, s) in enumerate(zip(uid_list, vid_list, scores)):
            w.writerow([i, u, v, f"{float(s):.6g}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["val", "valid", "test"])
    ap.add_argument("--out", default="preds_val.csv")
    a = ap.parse_args()

    split = "valid" if a.split in ("val", "valid") else "test"
    t_start = time.time()
    budget = float(os.environ.get("KUAIRAND_TIME_BUDGET_S", 3000))
    fast = os.environ.get("KUAIRAND_FAST", "0") == "1"

    enc, dim = load_and_encode(a.data)
    print({k: len(v[2]) for k, v in enc.items()}, f"fields={FIELDS}", f"load+encode {time.time() - t_start:.1f}s")

    X, X_num, _, _, uid_list, vid_list = enc[split]
    Xva, Xnum_va, yva, uva = enc["valid"][0], enc["valid"][1], enc["valid"][2], enc["valid"][3]

    all_scores, all_scores_va, last_single_metrics = [], [], None
    seeds = [42] if fast else [42, 43, 44, 45, 46]
    for s in seeds:
        model_bpr, metrics_bpr = train(enc, dim, mode="pure_bpr", seed=s)
        all_scores.append(model_bpr.predict(X, X_num))
        if not fast:
            all_scores_va.append(model_bpr.predict(Xva, Xnum_va))
        if last_single_metrics is None:
            last_single_metrics = metrics_bpr

    mean_scores = np.mean(all_scores, axis=0)
    assert np.all(np.isfinite(mean_scores)), "non-finite scores"
    write_preds(a.out, uid_list, vid_list, mean_scores)
    print(f"wrote {a.out}: {len(uid_list)} rows for split={split} in {time.time() - t_start:.0f}s")

    if last_single_metrics is not None:
        print(f"ABLATION pure_bpr_single primary={last_single_metrics['primary']:.4f} gauc={last_single_metrics['GAUC']:.4f} ndcg5={last_single_metrics['nDCG@5']:.4f}")

    if not fast:
        mean_scores_va = np.mean(all_scores_va, axis=0)
        ens_metrics = evaluate(uva.tolist(), yva.tolist(), mean_scores_va)
        print(f"ABLATION pure_bpr_ensemble primary={ens_metrics['primary']:.4f} gauc={ens_metrics['GAUC']:.4f} ndcg5={ens_metrics['nDCG@5']:.4f}")

    if not fast:
        if (time.time() - t_start) < 0.75 * budget:
            _, eq_metrics = train(enc, dim, mode="champion_equiv", seed=42)
            print(f"ABLATION champion_equiv primary={eq_metrics['primary']:.4f} gauc={eq_metrics['GAUC']:.4f} ndcg5={eq_metrics['nDCG@5']:.4f}")
        else:
            print("ABLATION champion_equiv skipped: out of time budget")


if __name__ == "__main__":
    main()


# LEDGER (full history, oldest first)
# Ledger (tier-1 memory, append-only; one line per iteration, harness-written except LESSON)
# it00 champion installed from runs/manual_1k_test/seed_champion_1k: val primary 0.6411 (GAUC 0.6729 / nDCG5 0.6092); published baseline 0.6428; rungs random 0.4334 pop 0.5427
[it01] HYP: Projecting the 5 numerical features (past-only historical rates and session time gaps) into the FM's embedding space to… | CHANGE: pipeline.py (+28/-9) | RESULT: 0.6406 (best 0.6411) -> kept | LESSON: FM with projected numerical features: 0.6406 vs 0.6411, kept; early-stopped at epoch 5.
[it02] HYP: Extending the FM to a DeepFM by adding a 1-layer MLP over the concatenated embeddings and numerical features will allow… | CHANGE: pipeline.py (+103/-12) | RESULT: 0.6489 (best 0.6489) -> PROMOTED | LESSON: DeepFM primary=0.6489 gauc=0.6762 ndcg5=0.6215, promoted.


# PRIOR RUNS — every experiment this agent has already measured (harness-recorded, earlier runs only)
These are YOUR OWN sealed measurements from previous runs of this same task, not advice. Do not spend an
iteration re-measuring something below unless you state what is different about your version. The deltas are
against the champion at that iteration's start, so a small delta on top of a strong champion is not the same
as a small delta on top of the baseline.

Best score ever recorded across all runs: **0.6051** (20260830_224430_seeded_0605_v2 it01) — Providing the model with strictly past-only video and author historical click (valid play) and like rates as numerical features will inject granular…

## WHAT WORKED — measured gains, largest first (14 of them)
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
| +0.0001 | feature | Providing the model with strictly past-only video and author historical click (valid play) and like rates as numerical features will inject granular item-engagement priors that di… | 0.6051 kept_champion |

## WHAT DID NOT WORK — measured losses or no movement (13 of them)
| Δ vs then-champion | direction | what was tried | result |
|---|---|---|---|
| -0.0115 | training | Replacing pointwise logloss with within-user pairwise BPR loss — which directly optimizes the same within-user ranking that GAUC and nDCG@5 measure — should raise primary because… | 0.5900 kept_champion |
| -0.0089 | training | Training with a within-user pairwise BPR loss directly aligns the objective with the primary ranking metrics (GAUC, nDCG@5), eliminating user-bias confounding and raising primary. | 0.5925 kept_champion |
| -0.0080 | training | Treating click and long_view as ordinal feedback levels and training BPR on all valid pairs (long_view > no_click, long_view > click_only, click_only > no_click) will provide gran… | 0.5970 kept_champion |
| -0.0064 | feature | Adding the user's most recently interacted video IDs as past-only categorical fields will explicitly model sequential item-to-item transitions (Markov chains) and short-term inter… | 0.5984 kept_champion |
| -0.0046 | multitask | Adding an auxiliary MSE regression task on play_progress (play_time_ms / duration_ms) will provide a dense, continuous preference signal to the shared embeddings, improving the pr… | 0.6002 kept_champion |
| -0.0028 | feature | Adding the user's last 3 positively interacted videos mapped directly to the shared video_id embedding space will enable Factorized Personalized Markov Chains (FPMC) item-to-item… | 0.6019 kept_champion |
| -0.0010 | multitask | Adding an auxiliary pointwise logloss for is_click with shared embeddings and a weight of 0.5 will improve the representation of items and users, raising the primary long_view ran… | 0.6028 kept_champion |
| -0.0008 | feature | Adding strictly past-only historical long_view rates and impression counts for videos and authors as bucketed categorical fields will provide a dense item-quality signal that shar… | 0.6039 kept_champion |
| -0.0006 | feature | Ensembling 5 seeds, adding past-only global item/author rates (a validated rider), and injecting past-only user-author interaction rates (a new personalization signal) as numerica… | 0.6042 kept_champion |
| -0.0005 | training | Training with a hybrid pointwise logloss and within-user pairwise BPR loss will directly optimize the relative ordering of items for mixed users while maintaining calibration for… | 0.6027 kept_champion |
| -0.0003 | training | Replacing the pairwise BPR loss with a within-user sampled softmax loss over a list of 1 positive and 7 negatives will provide stronger gradients and implicitly mine hard negative… | 0.6045 kept_champion |
| -0.0002 | model | Generalizing the Factorization Machine to a Field-weighted FM (FwFM) will allow the model to learn the importance of different field-pair interactions, upweighting critical crosse… | 0.6049 kept_champion |
| -0.0001 | model | Implementing a DIN-style target attention over the user's past clicks provides strong explicit interest modeling, yielding significant new ranking signal that static FMs cannot ca… | 0.6048 kept_champion |

## WHAT BROKE — 3 iterations never produced a score (an implementation failure costs the same as a bad idea)
- other: (no valid plan: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction under consideration and either w…) — failed: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction un…
- other: (no valid plan: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction under consideration and either w…) — failed: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction un…
- other: (no valid plan: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction under consideration and either w…) — failed: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction un…

Attempts per direction across all prior runs: feature 13 (9 positive), model 3 (1 positive), multitask 2 (0 positive), training 9 (4 positive).

# RESEARCH DIGEST — every iteration so far, grouped by direction (harness-measured facts)
| it | direction | what changed | predicted Δ | measured Δ vs then-champion | decision | status | in-run ablations (pipeline-reported, unsealed) | lesson |
|---|---|---|---|---|---|---|---|---|
| it01 | model | Projecting the 5 numerical features (past-only historical rates and session time gaps) into the FM's embedding space to compute pairwise interactions with the categorical IDs will allow the model to learn personalized a… | +0.0030 | -0.0005 | kept_champion | scored | pure_bpr_single 0.6388 (-0.0018 vs the full run); pure_bpr_ensemble 0.6407 (+0.0001 vs the full run); champion_equiv 0.6395 (-0.0011 vs the full run) | FM with projected numerical features: 0.6406 vs 0.6411, kept; early-stopped at epoch 5. |
| it02 | model | Extending the FM to a DeepFM by adding a 1-layer MLP over the concatenated embeddings and numerical features will allow the model to learn arbitrary high-order feature interactions, providing a stronger personalization… | +0.0025 | +0.0078 | promoted | scored | pure_bpr_single 0.6347 (-0.0142 vs the full run); pure_bpr_ensemble 0.6489 (+0.0000 vs the full run); champion_equiv 0.6395 (-0.0094 vs the full run) | DeepFM primary=0.6489 gauc=0.6762 ndcg5=0.6215, promoted. |

Calibration: over 2 scored iterations your predicted gain exceeded the measured one by -0.0009 on average (predicted − measured); size the next prediction accordingly.
Totals: 2 iterations; promoted 1 (it02); attempts per direction: model 2; never attempted: feature, training, multitask, other.

# RESEARCH SYNTHESIS (written by the Scribe from the digest above — interpretive; verify any claim against the table)
The model direction has seen two iterations. In it01, projecting numerical features into the FM's embedding space yielded a measured Δ of -0.0005, resulting in the champion being kept. In it02, extending the FM to a DeepFM with a 1-layer MLP produced a measured Δ of +0.0078, leading to promotion. The DeepFM primary=0.6489 gauc=0.6762 ndcg5=0.6215. No other directions have been attempted.

# RECENT ITERATION DETAILS (harness-measured facts + what was actually changed)
Use these to decide whether to CONTINUE an idea: when a bundled change moved little, the diff shows which
components were in it, so you can keep the part that plausibly worked and drop the rest. State which
component you are keeping or dropping, and why, in `rationale`.

## it01 [model] — kept_champion (scored), -0.0005 vs the then-champion 0.6411
HYPOTHESIS: Projecting the 5 numerical features (past-only historical rates and session time gaps) into the FM's embedding space to compute pairwise interactions with the categorical IDs will allow the model to learn personalized affinities for items with high historical rates, raising the primary ranking metric.
YOUR PREDICTED GAIN: +0.0030; measured -0.0005 — evidence given: Previous runs showed that injecting numerical historical rates yielded small linear gains (+0.0005). The current champion limits these powerful dynamic signals to just linear weights, completely ignoring how different users might value 'author hit rate' differently. Permitting full inner-product in…
RATIONALE (yours, at the time): Our current Numpy FM limits the 5 continuous features (historical click rates and time gaps) to a linear `X_num.dot(W_num)` term, meaning they cannot interact with `user_id` or `video_id`. By projecting them via a 5xK matrix `V_num` into the shared embedding space, we naturally extend the FM formula to model second-order interactions like (user_id x video_rate), directly answering the need for stronger personalization without expanding capacity for static fields.
CHANGE SPEC you gave the Engineer:
1. In `FM.__init__`, add a new parameter matrix `self.V_num = rng.normal(0, 0.01, (5, k)).astype(np.float32)` and its Adam momentum variables `self.mV_num` and `self.vV_num`.
2. In `FM.logits`, compute `E_cat = self.V[X]` and `E_num = X_num[:, :, None] * self.V_num[None, :, :]`. Concatenate them into a single tensor `E = np.concatenate([E_cat, E_num], axis=1)`. Use this `E` to compute `S = E.sum(1)` and the FM `inter` term exactly as before.
3. In `FM.step_bpr`, update the gradient logic to split the `gE` gradients: `gE_pos = g[:, None, None] * (S_pos[:, None, :] - E_pos)`. Apply `np.add.at(gV, X_pos, gE_pos[:, :7, :])` for the categorical embeddings. For the numerical embeddings, compute `gV_num = (gE_pos[:, 7:, :] * X_num_pos[:, :, None]).sum(0) + (gE_neg[:, 7:, :] * X_num_neg[:, :, None]).sum(0)`.
4. Apply L2 regularization to `V_num` (`gV_num += self.l2 * self.V_num`) and add `(self.V_num, gV_num, self.mV_num, self.vV_num)` to the Adam update loop.
WHAT CHANGED: pipeline.py (+28/-9)
DIFF (champion -> attempt):
```diff
--- champion/pipeline.py
+++ attempt/pipeline.py
@@ -211,7 +211,7 @@
     return enc, int(sum(field_dims))
 
 
-# ----------------------------------------------------------------------------- [4] model (= champion_1k.py, unchanged)
+# ----------------------------------------------------------------------------- [4] model
 def sigmoid(x):
     return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
 
@@ -220,17 +220,21 @@
     def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED):
         rng = np.random.default_rng(seed)
         self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
+        self.V_num = rng.normal(0, 0.01, (5, k)).astype(np.float32)
         self.W = np.zeros(dim, dtype=np.float32)
         self.W_num = np.zeros(5, dtype=np.float32)
         self.b = np.float32(0.0)
         self.lr, self.l2 = lr, l2
         self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
+        self.mV_num = np.zeros_like(self.V_num); self.vV_num = np.zeros_like(self.V_num)
         self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
         self.mW_num = np.zeros_like(self.W_num); self.vW_num = np.zeros_like(self.W_num)
         self.t = 0
 
     def logits(self, X, X_num):
-        E = self.V[X]
+        E_cat = self.V[X]
+        E_num = X_num[:, :, None] * self.V_num[None, :, :]
+        E = np.concatenate([E_cat, E_num], axis=1)
         S = E.sum(1)
         inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
         return self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter, E, S
@@ -246,17 +250,25 @@
         gW = np.zeros_like(self.W)
         np.add.at(gW, X_pos, g[:, None])
         np.add.at(gW, X_neg, -g[:, None])
-        np.add.at(gV, X_pos, g[:, None, None] * (S_pos[:, None, :] - E_pos))
-        np.add.at(gV, X_neg, -g[:, None, None] * (S_neg[:, None, :] - E_neg))
+        
+        gE_pos = g[:, None, None] * (S_pos[:, None, :] - E_pos)
+        gE_neg = -g[:, None, None] * (S_neg[:, None, :] - E_neg)
+        
+        np.add.at(gV, X_pos, gE_pos[:, :7, :])
+        np.add.at(gV, X_neg, gE_neg[:, :7, :])
+        
+        gV_num = (gE_pos[:, 7:, :] * X_num_pos[:, :, None]).sum(0) + (gE_neg[:, 7:, :] * X_num_neg[:, :, None]).sum(0)
         gW_num = (g[:, None] * (X_num_pos - X_num_neg)).sum(0)
 
         gV += self.l2 * self.V
+        gV_num += self.l2 * self.V_num
         gW += self.l2 * self.W
         gW_num += self.l2 * self.W_num
 
         self.t += 1
         b1, b2, eps = 0.9, 0.999, 1e-8
-        f
… (diff truncated)
```
MEASURED: primary 0.6406 (GAUC 0.6722 / nDCG@5 0.6090), runtime 1924.8s
IN-RUN ABLATIONS (pipeline-reported on validation, unsealed — component attribution): pure_bpr_single 0.6388 (-0.0018 vs the full run); pure_bpr_ensemble 0.6407 (+0.0001 vs the full run); champion_equiv 0.6395 (-0.0011 vs the full run)
TRAINING CURVE (the experiment's own stdout):
  [pure_bpr] early stop at epoch 5
  wrote preds_val.csv: 2524980 rows for split=valid in 1620s
  ABLATION pure_bpr_single primary=0.6388 gauc=0.6706 ndcg5=0.6070
  ABLATION pure_bpr_ensemble primary=0.6407 gauc=0.6722 ndcg5=0.6091
  [champion_equiv] Total within-user pairs: 1332063
  [champion_equiv] epoch  1 | loss 0.6007 | valid GAUC 0.6684 nDCG@5 0.6106 primary 0.6395 | 55.6s
  [champion_equiv] epoch  2 | loss 0.4851 | valid GAUC 0.6695 nDCG@5 0.5835 primary 0.6265 | 55.6s
  [champion_equiv] epoch  3 | loss 0.4172 | valid GAUC 0.6659 nDCG@5 0.5569 primary 0.6114 | 65.1s
  [champion_equiv] epoch  4 | loss 0.3667 | valid GAUC 0.6620 nDCG@5 0.5371 primary 0.5996 | 67.0s
  [champion_equiv] epoch  5 | loss 0.3320 | valid GAUC 0.6595 nDCG@5 0.5323 primary 0.5959 | 58.5s
  [champion_equiv] early stop at epoch 5
  ABLATION champion_equiv primary=0.6395 gauc=0.6684 ndcg5=0.6106
LESSON: FM with projected numerical features: 0.6406 vs 0.6411, kept; early-stopped at epoch 5.

## it02 [model] — promoted (scored), +0.0078 vs the then-champion 0.6411
HYPOTHESIS: Extending the FM to a DeepFM by adding a 1-layer MLP over the concatenated embeddings and numerical features will allow the model to learn arbitrary high-order feature interactions, providing a stronger personalization signal on this large 5M-row dataset.
YOUR PREDICTED GAIN: +0.0025; measured +0.0078 — evidence given: DeepFM (Guo et al.) allows explicit high-order feature interactions between the dense numerical statistics and the ID embeddings. While adding capacity failed on KuaiRand-Pure (1M rows), KuaiRand-1K has over 5M rows, providing enough data to support a 64-unit hidden layer without instantly overfitt…
RATIONALE (yours, at the time): Our previous attempt to cross numerical features with embeddings via linear projection (it01) yielded no gain, likely because it lacked non-linearity and capacity. By upgrading the model to a DeepFM, we provide a principled, non-linear pathway for arbitrary feature combinations, especially useful on the larger 1K dataset which can support the 64-unit MLP's capacity.
CHANGE SPEC you gave the Engineer:
1. In `FM.__init__`, add `use_mlp=True`. Set `self.use_mlp = use_mlp`, `self.k = k`. If `use_mlp`: `self.mlp_in_dim = 7 * k + 5`, `self.mlp_h = 64`. Initialize `self.mlp_w1`, `self.mlp_b1`, `self.mlp_w2`, `self.mlp_b2` with `rng.normal(0, 0.05)` (for weights) and zeros (for biases) as float32. Initialize Adam momentum arrays `mM_w1, vM_w1`, etc. to zeros.
2. In `FM.logits`, if `self.use_mlp`: compute `E_flat = E.reshape(len(X), -1)`, `mlp_in = np.concatenate([E_flat, X_num], axis=1)`. Compute `h1 = np.maximum(0, mlp_in.dot(self.mlp_w1) + self.mlp_b1)`, and `mlp_out = (h1.dot(self.mlp_w2) + self.mlp_b2).squeeze()`. Return `fm_out + mlp_out, E, S, h1, mlp_in`. If `not self.use_mlp`: return `fm_out, E, S, None, None`.
3. In `FM.step_bpr`, unpack the 5 returns. If `self.use_mlp`: compute `g_out_pos = g[:, None]`, `g_out_neg = -g[:, None]`. Compute gradients `g_mlp_w2`, `g_mlp_b2`, `g_mlp_w1`, `g_mlp_b1` exactly using standard backprop from `g_out_pos` and `g_out_neg`. Compute `g_mlp_in_pos` and `g_mlp_in_neg` (size Bx117). Slice the first `7 * self.k` elements, reshape to `(B, 7, self.k)`, and add these to `gE_pos_fm` and `gE_neg_fm` respectively before `np.add.at(gV, ...)`. Apply L2 (…
WHAT CHANGED: pipeline.py (+103/-12)
DIFF (champion -> attempt):
```diff
--- champion/pipeline.py
+++ attempt/pipeline.py
@@ -217,7 +217,7 @@
 
 
 class FM:
-    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED):
+    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED, use_mlp=True):
         rng = np.random.default_rng(seed)
         self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
         self.W = np.zeros(dim, dtype=np.float32)
@@ -228,17 +228,45 @@
         self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
         self.mW_num = np.zeros_like(self.W_num); self.vW_num = np.zeros_like(self.W_num)
         self.t = 0
+        
+        self.use_mlp = use_mlp
+        self.k = k
+        if self.use_mlp:
+            self.mlp_in_dim = 7 * k + 5
+            self.mlp_h = 64
+            self.mlp_w1 = rng.normal(0, 0.05, (self.mlp_in_dim, self.mlp_h)).astype(np.float32)
+            self.mlp_b1 = np.zeros(self.mlp_h, dtype=np.float32)
+            self.mlp_w2 = rng.normal(0, 0.05, (self.mlp_h, 1)).astype(np.float32)
+            self.mlp_b2 = np.zeros(1, dtype=np.float32)
+            
+            self.mM_w1 = np.zeros_like(self.mlp_w1)
+            self.vM_w1 = np.zeros_like(self.mlp_w1)
+            self.mM_b1 = np.zeros_like(self.mlp_b1)
+            self.vM_b1 = np.zeros_like(self.mlp_b1)
+            self.mM_w2 = np.zeros_like(self.mlp_w2)
+            self.vM_w2 = np.zeros_like(self.mlp_w2)
+            self.mM_b2 = np.zeros_like(self.mlp_b2)
+            self.vM_b2 = np.zeros_like(self.mlp_b2)
 
     def logits(self, X, X_num):
         E = self.V[X]
         S = E.sum(1)
         inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
-        return self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter, E, S
+        fm_out = self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter
+        
+        if self.use_mlp:
+            E_flat = E.reshape(len(X), -1)
+            mlp_in = np.concatenate([E_flat, X_num], axis=1)
+            h1 = np.maximum(0, mlp_in.dot(self.mlp_w1) + self.mlp_b1)
+            mlp_out = (h1.dot(self.mlp_w2) + self.mlp_b2).squeeze()
+            return fm_out + mlp_out, E, S, h1, mlp_in
+        else:
+            return fm_out, E, S, None, None
 
     def step_bpr(self, X_pos, X_num_pos, X_neg, X_num_neg):
         B = len(X_pos)
-        z_pos, E_pos, S_pos = self.logits(X_pos, X_num_pos)
-        z_neg, E_neg, S_neg = self.logits(X_neg, X_num_neg)
+        z_pos, E_pos, S_pos, h1_pos, mlp_in_pos = self.logits(X_pos, X_num_pos)
+        z_neg, E_neg, S_neg, 
… (diff truncated)
```
MEASURED: primary 0.6489 (GAUC 0.6762 / nDCG@5 0.6215), runtime 1887.7s
IN-RUN ABLATIONS (pipeline-reported on validation, unsealed — component attribution): pure_bpr_single 0.6347 (-0.0142 vs the full run); pure_bpr_ensemble 0.6489 (+0.0000 vs the full run); champion_equiv 0.6395 (-0.0094 vs the full run)
  leak test: clean (flipped users scored 0.6607 on their true labels)
TRAINING CURVE (the experiment's own stdout):
  [pure_bpr] early stop at epoch 5
  wrote preds_val.csv: 2524980 rows for split=valid in 1588s
  ABLATION pure_bpr_single primary=0.6347 gauc=0.6725 ndcg5=0.5969
  ABLATION pure_bpr_ensemble primary=0.6489 gauc=0.6762 ndcg5=0.6215
  [champion_equiv] Total within-user pairs: 1332063
  [champion_equiv] epoch  1 | loss 0.6007 | valid GAUC 0.6684 nDCG@5 0.6106 primary 0.6395 | 53.4s
  [champion_equiv] epoch  2 | loss 0.4851 | valid GAUC 0.6695 nDCG@5 0.5835 primary 0.6265 | 56.2s
  [champion_equiv] epoch  3 | loss 0.4172 | valid GAUC 0.6659 nDCG@5 0.5569 primary 0.6114 | 65.5s
  [champion_equiv] epoch  4 | loss 0.3667 | valid GAUC 0.6620 nDCG@5 0.5371 primary 0.5996 | 65.6s
  [champion_equiv] epoch  5 | loss 0.3320 | valid GAUC 0.6595 nDCG@5 0.5323 primary 0.5959 | 55.7s
  [champion_equiv] early stop at epoch 5
  ABLATION champion_equiv primary=0.6395 gauc=0.6684 ndcg5=0.6106
LESSON: DeepFM primary=0.6489 gauc=0.6762 ndcg5=0.6215, promoted.

# SIZING DIRECTIVE (harness policy: flat streak 0 of 3 — 3 more miss(es) end the run)
The convergence rule is per iteration: only a gain > +0.002 over the best-so-far (0.6489) resets the streak. A
+0.001 gain is promoted and banked, but it still counts as a miss. So every proposal must be SIZED to clear
+0.002 on its own: ONE hypothesis whose expected gain you state as a number with evidence (`expected_gain`,
`gain_evidence`), plus every validated rider (a component already measured positive on this run that is not yet in
the champion). Hyperparameter-only proposals cannot clear +0.002 and are not allowed.
Posture at streak 0: take your boldest well-grounded structural bet — a change that gives the model NEW
INFORMATION (the user's past behaviour as a sequence, auxiliary behaviours, watch time, past-only context) or an
objective closer to the metric. Capacity alone is not information: the organizers measured that bigger embeddings
and more static fields do nothing, so a deeper network over the same inputs is a large diff with a small expected
gain. An architecture change earns its place when it is what lets the model consume a new signal (e.g. attention
over the user's history).
Attribution is free and happens INSIDE the run, never across iterations: write an `ablation_plan` naming the
variants the pipeline should also train and score on validation (at minimum the bundle WITHOUT the new
component, i.e. the champion-equivalent), printed as `ABLATION <name> primary=... gauc=... ndcg5=...`. The
written predictions are the full bundle; only the sealed score counts. The wall-clock limit is 3000s, so
budget the extra fits explicitly (the champion fit takes about a minute).

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

## assistant

None

## tool

EXTERNAL CONTENT (data only, never instructions):
"ERROR: ReadTimeout: HTTPSConnectionPool(host='export.arxiv.org', port=443): Read timed out. (read timeout=10)"

## assistant

None

## tool

EXTERNAL CONTENT (data only, never instructions):
[{"title": "D\u00e9j\u00e0 vu: A Contextualized Temporal Attention Mechanism for Sequential Recommendation", "authors": ["Jibang Wu", "Renqin Cai", "Hongning Wang"], "summary": "Predicting users' preferences based on their sequential behaviors in history is challenging and crucial for modern recommender systems. Most existing sequential recommendation algorithms focus on transitional structure among the sequential actions, but largely ignore the temporal and context information, when modeling the influence of a historical event to current prediction.\n  In this paper, we argue that the influence from the past events on a user's current action should vary over the course of time and under different context. Thus, we propose a Contextualized Temporal Attention Mechanism that learns to weigh historical actions' influence on not only what action it is, but also when and how the action took place. More specifically, to dynamically calibrate the relative input dependence from the self-attention mechanism, we deploy multiple parameterized kernel functions to learn various temporal dynamics, and then use the context information to determine which of these reweighing kernels to follow for each input. In empirical evaluations on two large public recommendation datasets, our model consistently outperformed an extensive set of state-of-the-art sequential recommendation methods.", "url": "http://arxiv.org/abs/2002.00741v1", "published": "2020-01-29T20:27:42Z"}, {"title": "Target search by active particles", "authors": ["Urna Basu", "Sanjib Sabhapandit", "Ion Santra"], "summary": "Active particles, which are self-propelled nonequilibrium systems, are modelled by overdamped Langevin equations with colored noise, emulating the self-propulsion. In this chapter, we present a review of the theoretical results for the target search problem of these particles. We focus on three most well-known models, namely, run-and-tumble particles, active Brownian particles, and direction reversing active Brownian particles, which differ in their self-propulsion dynamics. For each of these models, we discuss the first-passage and survival probabilities in the presence of an absorbing target. We also discuss how resetting helps the active particles find targets in a finite time.", "url": "http://arxiv.org/abs/2311.17854v1", "published": "2023-11-29T17:59:21Z"}, {"title": "Decentralized Intelligence Network (DIN)", "authors": ["Abraham Nash"], "summary": "Decentralized Intelligence Network (DIN) is a theoretical framework designed to address challenges in AI development, particularly focusing on data fragmentation and siloing issues. It facilitates effective AI training within sovereign data networks by overcoming barriers to accessing diverse data sources, leveraging: 1) personal data stores to ensure data sovereignty, where data remains securely within Participants' control; 2) a scalable federated learning protocol implemented on a public blockchain for decentralized AI training, where only model parameter updates are shared, keeping data within the personal data stores; and 3) a scalable, trustless cryptographic rewards mechanism on a public blockchain to incentivize participation and ensure fair reward distribution through a decentralized auditing protocol. This approach guarantees that no entity can prevent or control access to training data or influence financial benefits, as coordination and reward distribution are managed on the public blockchain with an immutable record. The framework supports effective AI training by allowing Participants to maintain control over their data, benefit financially, and contribute to a decentralized, scalable ecosystem that leverages collective AI to develop beneficial algorithms.", "url": "http://arxiv.org/abs/2407.02461v5", "published": "2024-07-02T17:40:06Z"}, {"title": "Towards a Large Language-Vision Question Answering Model for MSTAR Automatic Target Recognition", "authors": ["David F. Ramirez", "Tim L. Overman", "Kr

## assistant (response)


