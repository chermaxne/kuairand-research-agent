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
