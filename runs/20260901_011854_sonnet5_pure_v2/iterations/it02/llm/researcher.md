# researcher — researcher (model anthropic/claude-sonnet-5, 38355 tokens)

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
CURRENT BEST: it01 | val primary 0.6038 (GAUC 0.6703 / nDCG5 0.5372) | baseline 0.6016 | margin +0.0022
BUDGET: iteration 2 of 50 | 0:09 of 6:00 elapsed | tokens so far 233617
CONVERGENCE: streak 0 of 3 flat (EPSILON=0.002)
BLOCKED: none
ACTIVE THEMES: winning: model[1 promoted/0 flat/0 failed]; losing/flat: none; untried: feature, training, multitask, other


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
"""DeepFM upgrade over the FM champion: adds a 1-hidden-layer MLP over the concatenated field
embeddings plus 7 standardized past-only numerical priors (user/video/author long_view & click
rates + user x tab long_view rate). Implements DeepFM (Guo et al. 2017) sum of FM 2nd-order term
and an MLP branch, trained pointwise BCE with the same Adam-style optimizer as the FM baseline.

    python pipeline.py --data <data_dir> --split val|test --out preds.csv

Section map:  [1] config  [2] data loading  [3] feature encoding  [3b] numeric priors (past-only)
              [4] model  [5] training  [6] CLI / ablations
"""
import argparse
import collections
import csv
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- [1] config
LABEL = "long_view"
SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428), "test": (20220429, 20220508)}
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]   # 5 categorical fields of the baseline
K = 16            # embedding dim
LR = 0.001
L2 = 1e-6
EPOCHS = 40
BATCH = 8192
PATIENCE = 4
SEED = 0
N_DUR_BUCKETS = 10
M_NUM = 7         # numerical prior features
HIDDEN = 128      # MLP hidden width

ABLATION_EPOCHS = 8
ABLATION_PATIENCE = 3
ABLATION_MAX_ROWS = 300_000


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


# ----------------------------------------------------------------------------- [2] data loading (= starter_kit/data.py + is_click)
def load(data_dir):
    """Rows as (date, user_id, video_id, author_id, tab, duration_ms, label, is_click); file order preserved."""
    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    rows = []
    for f in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r["date"]), r["user_id"], r["video_id"], vid2author.get(r["video_id"], "UNK"), r["tab"],
                             float(r["duration_ms"]), 1 if r[LABEL] != "0" else 0,
                             1 if r.get("is_click", "0") != "0" else 0))
    return {name: [x for x in rows if lo <= x[0] <= hi] for name, (lo, hi) in SPLITS.items()}


# ----------------------------------------------------------------------------- [3] feature encoding (= starter_kit/data.py)
def _bucket_edges(durations, n=N_DUR_BUCKETS):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


def encode(splits):
    """Categorical ids -> contiguous ints; unseen values fall into a per-field UNK slot.
    Returns ({split: (X int32 (N,F), y float32, users)}, total_dim)."""
    tr = splits["train"]
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

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
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))


# ----------------------------------------------------------------------------- [3b] numeric priors (past-only, no leakage)
_GROUP_DEFS = collections.OrderedDict([
    ("user", ["user"]), ("video", ["video"]), ("author", ["author"]), ("user_tab", ["user", "tab"]),
])
_COLS = ["date", "user", "video", "author", "tab", "dur", "label", "click"]


def build_numeric(splits, log=print):
    """Returns (Zstd dict of {split: (N,7) float32 standardized array}, stats dict)."""
    df_tr = pd.DataFrame(splits["train"], columns=_COLS)
    global_lv = float(df_tr["label"].mean())
    global_ck = float(df_tr["click"].mean())

    def expanding(keys):
        per_date = (df_tr.groupby(keys + ["date"])
                    .agg(lsum=("label", "sum"), lcnt=("label", "count"), csum=("click", "sum"))
                    .reset_index().sort_values("date"))
        grp = per_date.groupby(keys)
        per_date["cum_l"] = grp["lsum"].cumsum()
        per_date["cum_n"] = grp["lcnt"].cumsum()
        per_date["cum_c"] = grp["csum"].cumsum()
        per_date["excl_l"] = per_date["cum_l"] - per_date["lsum"]
        per_date["excl_n"] = per_date["cum_n"] - per_date["lcnt"]
        per_date["excl_c"] = per_date["cum_c"] - per_date["csum"]
        return per_date[keys + ["date", "excl_l", "excl_n", "excl_c"]]

    def full_train_agg(keys):
        return (df_tr.groupby(keys)
                .agg(pos=("label", "sum"), cnt=("label", "count"), cpos=("click", "sum"))
                .reset_index())

    exp_tables = {name: expanding(keys) for name, keys in _GROUP_DEFS.items()}
    full_tables = {name: full_train_agg(keys) for name, keys in _GROUP_DEFS.items()}

    def train_rates(df):
        out = {}
        for name, keys in _GROUP_DEFS.items():
            merged = df.merge(exp_tables[name], on=keys + ["date"], how="left")
            n = merged["excl_n"].fillna(0).values
            l_ = merged["excl_l"].fillna(0).values
            c_ = merged["excl_c"].fillna(0).values
            lv = np.where(n > 0, l_ / np.where(n > 0, n, 1), global_lv)
            ck = np.where(n > 0, c_ / np.where(n > 0, n, 1), global_ck)
            out[name] = (lv.astype(np.float32), ck.astype(np.float32))
        return out

    def eval_rates(df):
        out = {}
        for name, keys in _GROUP_DEFS.items():
            merged = df.merge(full_tables[name], on=keys, how="left")
            cnt = merged["cnt"].fillna(0).values
            pos = merged["pos"].fillna(0).values
            cpos = merged["cpos"].fillna(0).values
            lv = np.where(cnt > 0, pos / np.where(cnt > 0, cnt, 1), global_lv)
            ck = np.where(cnt > 0, cpos / np.where(cnt > 0, cnt, 1), global_ck)
            out[name] = (lv.astype(np.float32), ck.astype(np.float32))
        return out

    def stack(r):
        return np.stack([
            r["user"][0], r["user"][1],
            r["video"][0], r["video"][1],
            r["author"][0], r["author"][1],
            r["user_tab"][0],
        ], axis=1)

    Zraw = {"train": stack(train_rates(df_tr))}
    for name in ("valid", "test"):
        df = pd.DataFrame(splits[name], columns=_COLS)
        Zraw[name] = stack(eval_rates(df))

    mean = Zraw["train"].mean(0)
    std = Zraw["train"].std(0)
    std = np.where(std < 1e-6, 1.0, std)
    Zstd = {name: np.clip((Z - mean) / std, -5, 5).astype(np.float32) for name, Z in Zraw.items()}

    log(f"numeric priors: n_users={df_tr['user'].nunique()} n_videos={df_tr['video'].nunique()} "
        f"n_authors={df_tr['author'].nunique()} global_lv={global_lv:.4f} global_ck={global_ck:.4f}")
    log(f"numeric priors train mean={np.round(mean, 4).tolist()}")
    log(f"numeric priors train std ={np.round(std, 4).tolist()}")
    return Zstd, {"mean": mean, "std": std, "global_lv": global_lv, "global_ck": global_ck}


# ----------------------------------------------------------------------------- [4] model: DeepFM (FM + optional MLP branch)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class DeepFM:
    """Second-order FM (+ optional 1-hidden-layer MLP over concat(flatten(E), Z)), pointwise logloss, Adam.
    With use_mlp=False this is byte-for-byte the original FM baseline (champion_equiv)."""

    def __init__(self, dim, use_mlp=False, k=K, lr=LR, l2=L2, seed=SEED, m=M_NUM, hidden=HIDDEN):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0
        self.use_mlp = use_mlp
        self.m = m
        self.k = k
        if use_mlp:
            rng2 = np.random.default_rng(seed + 1)
            d_in = len(FIELDS) * k + m
            self.W1 = rng2.normal(0, 0.01, (d_in, hidden)).astype(np.float32)
            self.b1 = np.zeros(hidden, dtype=np.float32)
            self.W2 = rng2.normal(0, 0.01, (hidden, 1)).astype(np.float32)
            self.b2 = np.zeros(1, dtype=np.float32)
            for name in ("W1", "b1", "W2", "b2"):
                setattr(self, "m" + name, np.zeros_like(getattr(self, name)))
                setattr(self, "v" + name, np.zeros_like(getattr(self, name)))
            self.t_mlp = 0

    def logits(self, X, Z=None):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                     # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm = self.b + self.W[X].sum(1) + inter
        H0 = None
        h1 = None
        mlp_out = np.zeros(len(X), dtype=np.float32)
        if self.use_mlp:
            B = len(X)
            flat = E.reshape(B, -1)
            if Z is None:
                Z = np.zeros((B, self.m), dtype=np.float32)
            H0 = np.concatenate([flat, Z], axis=1).astype(np.float32)
            h1 = np.maximum(H0 @ self.W1 + self.b1, 0)
            mlp_out = (h1 @ self.W2 + self.b2).ravel()
        return fm + mlp_out, E, S, H0, h1

    def step(self, X, y, Z=None):
        B = len(y)
        z, E, S, H0, h1 = self.logits(X, Z)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))

        if self.use_mlp:
            F_ = len(FIELDS); Kk = self.k
            gh1_out = g[:, None]                          # (B,1)
            dW2 = h1.T @ gh1_out
            db2 = gh1_out.sum(0)
            dh1 = gh1_out @ self.W2.T
            dh1relu = dh1 * (h1 > 0)
            dW1 = H0.T @ dh1relu
            db1 = dh1relu.sum(0)
            dH0 = dh1relu @ self.W1.T
            dE_mlp = dH0[:, :F_ * Kk].reshape(B, F_, Kk)
            np.add.at(gV, X, dE_mlp)
            dW1 = dW1 + self.l2 * self.W1
            dW2 = dW2 + self.l2 * self.W2

        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1c, b2c, eps = 0.9, 0.999, 1e-8
        for P, G, Mm, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            Mm *= b1c; Mm += (1 - b1c) * G
            Vv *= b2c; Vv += (1 - b2c) * (G * G)
            P -= self.lr * (Mm / (1 - b1c ** self.t)) / (np.sqrt(Vv / (1 - b2c ** self.t)) + eps)
        self.b -= self.lr * g.sum()

        if self.use_mlp:
            self.t_mlp += 1
            tt = self.t_mlp
            for name, G in (("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2)):
                P = getattr(self, name)
                Mm = getattr(self, "m" + name); Vv = getattr(self, "v" + name)
                Mm *= b1c; Mm += (1 - b1c) * G
                Vv *= b2c; Vv += (1 - b2c) * (G * G)
                P -= self.lr * (Mm / (1 - b1c ** tt)) / (np.sqrt(Vv / (1 - b2c ** tt)) + eps)

        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, Z=None, bs=200_000):
        outs = []
        for i in range(0, len(X), bs):
            zb = None if Z is None else Z[i:i + bs]
            outs.append(self.logits(X[i:i + bs], zb)[0])
        return np.concatenate(outs)

    def state(self):
        s = [self.V.copy(), self.W.copy(), np.float32(self.b)]
        if self.use_mlp:
            s += [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]
        return s

    def load_state(self, s):
        self.V, self.W, self.b = s[0], s[1], s[2]
        if self.use_mlp:
            self.W1, self.b1, self.W2, self.b2 = s[3], s[4], s[5], s[6]


# ----------------------------------------------------------------------------- [5] training
def train_deepfm(Xtr, ytr, uva, Xva, yva, dim, use_mlp, Ztr=None, Zva=None,
                  epochs=EPOCHS, patience=PATIENCE, seed=SEED, log=print, tag=""):
    m = DeepFM(dim, use_mlp=use_mlp, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        losses = []
        for i in range(0, len(idx), BATCH):
            bidx = idx[i:i + BATCH]
            zb = Ztr[bidx] if Ztr is not None else None
            losses.append(m.step(Xtr[bidx], ytr[bidx], zb))
        va = evaluate(uva, yva, m.predict(Xva, Zva))
        log(f"[{tag}] epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
            f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = m.state()
            best_metrics = va
        else:
            bad += 1
            if bad >= patience:
                log(f"[{tag}] early stop at epoch {ep}")
                break
    m.load_state(best_state)
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
    budget = float(os.environ.get("KUAIRAND_TIME_BUDGET_S", "1500"))
    fast = os.environ.get("KUAIRAND_FAST", "0") == "1"

    t0 = time.time()
    splits = load(a.data)
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}", f"load {time.time() - t0:.1f}s")
    enc, dim = encode(splits)
    Zdict, _ = build_numeric(splits)

    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]

    # ---- FULL bundle first: DeepFM (MLP) + numeric priors Z. This is the proposed champion. ----
    model_full, va_full = train_deepfm(
        Xtr, ytr, uva, Xva, yva, dim, use_mlp=True,
        Ztr=Zdict["train"], Zva=Zdict["valid"],
        epochs=EPOCHS, patience=PATIENCE, seed=SEED, tag="full",
    )
    print(f"full fit done at {time.time() - t0:.0f}s (budget {budget:.0f}s)")

    Xtarget, _, _ = enc[split]
    Ztarget = Zdict[split]
    scores = model_full.predict(Xtarget, Ztarget)
    assert np.all(np.isfinite(scores)), "non-finite scores"
    write_preds(a.out, splits[split], scores)
    print(f"wrote {a.out}: {len(splits[split])} rows for split={split} in {time.time() - t0:.0f}s")
    print(f"ABLATION full primary={va_full['primary']:.4f} gauc={va_full['GAUC']:.4f} ndcg5={va_full['nDCG@5']:.4f}")

    if fast:
        print("KUAIRAND_FAST=1: skipping in-run ablations (champion_equiv, deepfm_only).")
        return

    # ---- cheap diagnostics: subsample + capped epochs, never as costly as the full fit ----
    n = len(ytr)
    if n > ABLATION_MAX_ROWS:
        sub_rng = np.random.default_rng(SEED)
        sub_idx = sub_rng.choice(n, ABLATION_MAX_ROWS, replace=False)
    else:
        sub_idx = np.arange(n)
    Xtr_sub, ytr_sub = Xtr[sub_idx], ytr[sub_idx]
    Ztr_sub = Zdict["train"][sub_idx]

    elapsed = time.time() - t0
    if elapsed < 0.75 * budget:
        model_ce, va_ce = train_deepfm(
            Xtr_sub, ytr_sub, uva, Xva, yva, dim, use_mlp=False,
            Ztr=None, Zva=None,
            epochs=ABLATION_EPOCHS, patience=ABLATION_PATIENCE, seed=SEED, tag="champion_equiv",
        )
        print(f"ABLATION champion_equiv primary={va_ce['primary']:.4f} gauc={va_ce['GAUC']:.4f} ndcg5={va_ce['nDCG@5']:.4f}")
    else:
        print("ABLATION champion_equiv skipped: out of time budget")

    elapsed = time.time() - t0
    if elapsed < 0.75 * budget:
        zero_ztr = np.zeros((len(ytr_sub), M_NUM), dtype=np.float32)
        zero_zva = np.zeros((len(yva), M_NUM), dtype=np.float32)
        model_do, va_do = train_deepfm(
            Xtr_sub, ytr_sub, uva, Xva, yva, dim, use_mlp=True,
            Ztr=zero_ztr, Zva=zero_zva,
            epochs=ABLATION_EPOCHS, patience=ABLATION_PATIENCE, seed=SEED, tag="deepfm_only",
        )
        print(f"ABLATION deepfm_only primary={va_do['primary']:.4f} gauc={va_do['GAUC']:.4f} ndcg5={va_do['nDCG@5']:.4f}")
    else:
        print("ABLATION deepfm_only skipped: out of time budget")


if __name__ == "__main__":
    main()


# LEDGER (full history, oldest first)
# Ledger (tier-1 memory, append-only; one line per iteration, harness-written except LESSON)
# it00 champion installed from baseline_repro: val primary 0.6015 (GAUC 0.6671 / nDCG5 0.5358); published baseline 0.6016; rungs random 0.4827 pop 0.5807
[it01] HYP: Upgrading the FM to a DeepFM (add a 1-hidden-layer MLP over the concatenated field embeddings) and feeding it standardi… | CHANGE: pipeline.py (+265/-51) | RESULT: 0.6038 (best 0.6038) -> PROMOTED | LESSON: DeepFM with standardized priors: 0.6037 vs 0.5893, promoted; early-stopped at epoch 8.


# PRIOR RUNS — every experiment this agent has already measured (harness-recorded, earlier runs only)
These are YOUR OWN sealed measurements from previous runs of this same task, not advice. Do not spend an
iteration re-measuring something below unless you state what is different about your version. The deltas are
against the champion at that iteration's start, so a small delta on top of a strong champion is not the same
as a small delta on top of the baseline.

Best score ever recorded across all runs: **0.6563** (20260831_145457_1k_bonus_test it05) — Adding past-only user-tab specific historical impression and positive rates as numerical features will give the DeepFM MLP a highly personalized, con…

## WHAT WORKED — measured gains, largest first (14 of them)
| Δ vs then-champion | direction | what was tried | result |
|---|---|---|---|
| +0.0078 | model | Extending the FM to a DeepFM by adding a 1-layer MLP over the concatenated embeddings and numerical features will allow the model to learn arbitrary high-order feature interaction… | 0.6489 promoted |
| +0.0037 | feature | Standardizing past-only numerical features will stabilize DeepFM's gradients against scale imbalances, adding missing user click/like rates will complete the behavioral priors, an… | 0.6528 promoted |
| +0.0035 | feature | Adding past-only user-tab specific historical impression and positive rates as numerical features will give the DeepFM MLP a highly personalized, context-aware baseline for each u… | 0.6563 promoted |
| +0.0029 | training | Training with a within-user pairwise BPR loss directly aligns the objective with the evaluation metric (GAUC, nDCG@5) by optimizing relative ranking rather than absolute pointwise… | 0.6043 promoted |
| +0.0024 | model | Upgrading the FM to a DeepFM (1-hidden-layer MLP over the concatenated field embeddings) and adding standardized past-only numerical priors (user/video/author long_view & click ra… | 0.6039 promoted |
| +0.0021 | feature | Adding the user's daily session depth and hour-of-day as contextual categorical features will capture position bias and time context, and combining this with a 3-seed ensemble wil… | 0.6048 promoted |
| +0.0018 | training | Training the numpy FM with a within-user pairwise BPR loss directly aligns the optimization objective with the evaluation metrics (GAUC, nDCG@5), providing a stronger ranking sign… | 0.6032 promoted |
| +0.0017 | feature | Adding the user's daily session depth and hour-of-day as past-only categorical features captures position and time bias, and combining this with a 5-seed score average will reduce… | 0.6049 promoted |
| +0.0012 | training | Training with BPR loss on within-user positive-negative pairs directly aligns the objective with the ranking metric, raising primary. | 0.6027 promoted |
| +0.0005 | feature | Adding the user's daily session depth and hour-of-day as contextual categorical features captures position bias and time context, yielding new ranking signal. | 0.6048 promoted |
| +0.0003 | feature | Adding user historical long_view rates and item/author auxiliary feedback rates (click, like) as past-only numerical features will provide DeepFM's MLP with rich interaction surfa… | 0.6492 promoted |
| +0.0003 | feature | Stacking past-only numerical features (video/author historical rates and impression counts) and 5-seed ensembling (both validated riders) alongside a genuinely new numerical signa… | 0.6050 promoted |
| +0.0002 | feature | Adding the user's daily session depth and hour-of-day as categorical fields will allow the FM/MLP to learn explicit position-bias and time-context interactions, and averaging 5 se… | 0.6041 promoted |
| +0.0001 | feature | Providing the model with strictly past-only video and author historical click (valid play) and like rates as numerical features will inject granular item-engagement priors that di… | 0.6051 kept_champion |

## WHAT DID NOT WORK — measured losses or no movement (12 of them)
| Δ vs then-champion | direction | what was tried | result |
|---|---|---|---|
| -0.0080 | training | Treating click and long_view as ordinal feedback levels and training BPR on all valid pairs (long_view > no_click, long_view > click_only, click_only > no_click) will provide gran… | 0.5970 kept_champion |
| -0.0064 | feature | Adding the user's most recently interacted video IDs as past-only categorical fields will explicitly model sequential item-to-item transitions (Markov chains) and short-term inter… | 0.5984 kept_champion |
| -0.0046 | multitask | Adding an auxiliary MSE regression task on play_progress (play_time_ms / duration_ms) will provide a dense, continuous preference signal to the shared embeddings, improving the pr… | 0.6002 kept_champion |
| -0.0028 | feature | Adding the user's last 3 positively interacted videos mapped directly to the shared video_id embedding space will enable Factorized Personalized Markov Chains (FPMC) item-to-item… | 0.6019 kept_champion |
| -0.0012 | training | Adding a within-user pairwise BPR term (hybrid with the existing pointwise BCE) on top of the current DeepFM+numeric-feature champion directly optimizes an objective aligned with… | 0.6027 kept_champion |
| -0.0010 | model | Rank-averaging the DeepFM champion with a LightGBM lambdarank model trained on the same past-only numeric+categorical features adds a genuinely different interaction family (tree… | 0.6029 kept_champion |
| -0.0008 | feature | Adding strictly past-only historical long_view rates and impression counts for videos and authors as bucketed categorical fields will provide a dense item-quality signal that shar… | 0.6039 kept_champion |
| -0.0006 | feature | Ensembling 5 seeds, adding past-only global item/author rates (a validated rider), and injecting past-only user-author interaction rates (a new personalization signal) as numerica… | 0.6042 kept_champion |
| -0.0005 | model | Projecting the 5 numerical features (past-only historical rates and session time gaps) into the FM's embedding space to compute pairwise interactions with the categorical IDs will… | 0.6406 kept_champion |
| -0.0003 | training | Replacing the pairwise BPR loss with a within-user sampled softmax loss over a list of 1 positive and 7 negatives will provide stronger gradients and implicitly mine hard negative… | 0.6045 kept_champion |
| -0.0002 | model | Generalizing the Factorization Machine to a Field-weighted FM (FwFM) will allow the model to learn the importance of different field-pair interactions, upweighting critical crosse… | 0.6049 kept_champion |
| -0.0001 | model | Implementing a DIN-style target attention over the user's past clicks provides strong explicit interest modeling, yielding significant new ranking signal that static FMs cannot ca… | 0.6048 kept_champion |

## WHAT BROKE — 1 iterations never produced a score (an implementation failure costs the same as a bad idea)
- model: Replacing the numpy FM with a PyTorch DeepFM and concatenating strictly past-only user/video historical rates as numerical features into th… — failed: [debugger abandoned: PyTorch (torch) is not installed in the environment; cannot implemen…

Attempts per direction across all prior runs: feature 13 (9 positive), model 6 (2 positive), multitask 1 (0 positive), training 6 (3 positive).

# RESEARCH DIGEST — every iteration so far, grouped by direction (harness-measured facts)
| it | direction | what changed | predicted Δ | measured Δ vs then-champion | decision | status | in-run ablations (pipeline-reported, unsealed) | lesson |
|---|---|---|---|---|---|---|---|---|
| it01 | model | Upgrading the FM to a DeepFM (add a 1-hidden-layer MLP over the concatenated field embeddings) and feeding it standardized past-only numerical priors (user/video/author long_view & click rates plus user×tab historical r… | +0.0090 | +0.0023 | promoted | scored | full 0.6038 (+0.0000 vs the full run); champion_equiv 0.5893 (-0.0145 vs the full run); deepfm_only 0.5943 (-0.0095 vs the full run) | DeepFM with standardized priors: 0.6037 vs 0.5893, promoted; early-stopped at epoch 8. |

Calibration: over 1 scored iterations your predicted gain exceeded the measured one by +0.0067 on average (predicted − measured); size the next prediction accordingly.
Totals: 1 iterations; promoted 1 (it01); attempts per direction: model 1; never attempted: feature, training, multitask, other.

# RESEARCH SYNTHESIS (written by the Scribe from the digest above — interpretive; verify any claim against the table)
The only direction tried so far is model. It01 promoted a DeepFM with standardized priors, achieving a full run score of 0.6038, an improvement of +0.0023 over the then-champion. The in-run ablations showed that the full run outperformed the champion equivalent and the DeepFM-only versions. The calibration suggests that predicted gains may be overestimated by +0.0067 on average. No other directions have been attempted.

# RECENT ITERATION DETAILS (harness-measured facts + what was actually changed)
Use these to decide whether to CONTINUE an idea: when a bundled change moved little, the diff shows which
components were in it, so you can keep the part that plausibly worked and drop the rest. State which
component you are keeping or dropping, and why, in `rationale`.

## it01 [model] — promoted (scored), +0.0023 vs the then-champion 0.6015
HYPOTHESIS: Upgrading the FM to a DeepFM (add a 1-hidden-layer MLP over the concatenated field embeddings) and feeding it standardized past-only numerical priors (user/video/author long_view & click rates plus user×tab historical rate) gives the model new, genuinely predictive signal beyond raw id crosses, since these levers were independently validated as the three largest measured wins in this project's history (+0.0078, +0.0037, +0.0035 stacked to 0.6563 from an FM baseline).
YOUR PREDICTED GAIN: +0.0090; measured +0.0023 — evidence given: This project's own ledger (prior runs, same task) measured DeepFM MLP-over-embeddings alone at +0.0078 promoted, then standardized numeric behavioral-rate features at +0.0037, then user-tab historical rates at +0.0035, stacking an FM baseline of 0.6015 up to 0.6563 — the single largest validated le…
RATIONALE (yours, at the time): Per DeepFM (Guo et al., IJCAI 2017), combining a shallow FM component with a deep MLP over the same embeddings lets the model learn arbitrary higher-order interactions without manual feature crossing, addressing the organizers' observation that raw FM capacity is not the bottleneck (§4) — the MLP earns its place here specifically because it is the mechanism that lets the model consume the new past-only numerical priors, not because of added depth alone (§8.5, §2b of the strategy). The numerical features are past-only aggregates of `long_view`/`is_click` per user/video/author/user-tab, which a…
CHANGE SPEC you gave the Engineer:
1. In FM class (or a new DeepFM class), after computing the per-field embedding tensor E (B,F,k) as today, ALSO build a numerical feature vector Z (B,M) per row with M=7 past-only fields: (a) user_long_view_rate, (b) user_click_rate, (c) video_long_view_rate, (d) video_click_rate, (e) author_long_view_rate, (f) author_click_rate, (g) user_tab_long_view_rate. 2. Past-only computation rule (must not leak): for TRAIN rows, sort by date, then for each (user|video|author|user,tab) group compute an EXPANDING count and expanding positive-count using only STRICTLY EARLIER-dated rows in train (use cumulative sums indexed by sorted date, subtract same-day rows entirely -- i.e. only rows with date < this row's date count); rate = expanding_pos/expanding_count if expanding_count>0 else the global train long_view rate (0.3366) / click rate fallback. For VALID (and TEST) rows, use the FULL train-split aggregate (all 13 train dates) per group, since validation is strictly after train in time — no expanding computation needed there. 3. Standardize each of the 7 numerical columns using TRAIN-computed mean/std (subtract mean, divide by std, clip to [-5,5]) — fit these stats only on the train rows' c…
WHAT CHANGED: pipeline.py (+265/-51)
DIFF (champion -> attempt):
```diff
--- champion/pipeline.py
+++ attempt/pipeline.py
@@ -1,16 +1,12 @@
-"""Iteration-0 champion: the organizers' FM baseline (starter_kit/baseline.py + data.py, k=16, lr=0.001,
-batch 8192, <=40 epochs, patience 4, seed 0) ported UNCHANGED in behaviour to the pipeline contract:
+"""DeepFM upgrade over the FM champion: adds a 1-hidden-layer MLP over the concatenated field
+embeddings plus 7 standardized past-only numerical priors (user/video/author long_view & click
+rates + user x tab long_view rate). Implements DeepFM (Guo et al. 2017) sum of FM 2nd-order term
+and an MLP branch, trained pointwise BCE with the same Adam-style optimizer as the FM baseline.
 
     python pipeline.py --data <data_dir> --split val|test --out preds.csv
 
-- trains ONLY on the train split (dates 20220408-20220421); validation is used for early stopping
-  (exactly as the official baseline does); the requested split is scored and written in data order
-  as row_id,user_id,video_id,score.
-- self-contained on purpose: the Engineer edits THIS file (data loading, features, model, training).
-- only numpy + the standard library are required; `evaluate` (the official metric) is imported for
-  early stopping only — the harness scores predictions with the sealed copy, never this process.
-
-Section map:  [1] config  [2] data loading  [3] feature encoding  [4] model  [5] training  [6] CLI
+Section map:  [1] config  [2] data loading  [3] feature encoding  [3b] numeric priors (past-only)
+              [4] model  [5] training  [6] CLI / ablations
 """
 import argparse
 import collections
@@ -21,6 +17,7 @@
 import time
 
 import numpy as np
+import pandas as pd
 
 # ----------------------------------------------------------------------------- [1] config
 LABEL = "long_view"
@@ -34,6 +31,12 @@
 PATIENCE = 4
 SEED = 0
 N_DUR_BUCKETS = 10
+M_NUM = 7         # numerical prior features
+HIDDEN = 128      # MLP hidden width
+
+ABLATION_EPOCHS = 8
+ABLATION_PATIENCE = 3
+ABLATION_MAX_ROWS = 300_000
 
 
 # ----------------------------------------------------------------------------- metric (early stopping only)
@@ -57,9 +60,9 @@
 evaluate = _import_evaluate()
 
 
-# ----------------------------------------------------------------------------- [2] data loading (= starter_kit/data.py)
+# ----------------------------------------------------------------------------- [2] data loading (= starter_kit/data.py + is_click)
 def load(data_dir):
-    """Rows as (date, user_id, video_id, author_id, tab, duration_
… (diff truncated)
```
MEASURED: primary 0.6038 (GAUC 0.6703 / nDCG@5 0.5372), runtime 41.5s
IN-RUN ABLATIONS (pipeline-reported on validation, unsealed — component attribution): full 0.6038 (+0.0000 vs the full run); champion_equiv 0.5893 (-0.0145 vs the full run); deepfm_only 0.5943 (-0.0095 vs the full run)
  leak test: clean (flipped users scored 0.6027 on their true labels)
TRAINING CURVE (the experiment's own stdout):
  [champion_equiv] epoch  7 | loss 0.5060 | valid GAUC 0.6488 nDCG@5 0.5282 primary 0.5885 | 0.5s
  [champion_equiv] epoch  8 | loss 0.4906 | valid GAUC 0.6498 nDCG@5 0.5289 primary 0.5893 | 0.5s
  ABLATION champion_equiv primary=0.5893 gauc=0.6498 ndcg5=0.5289
  [deepfm_only] epoch  1 | loss 0.6688 | valid GAUC 0.6193 nDCG@5 0.5171 primary 0.5682 | 0.9s
  [deepfm_only] epoch  2 | loss 0.5902 | valid GAUC 0.6461 nDCG@5 0.5274 primary 0.5867 | 0.9s
  [deepfm_only] epoch  3 | loss 0.5270 | valid GAUC 0.6529 nDCG@5 0.5298 primary 0.5913 | 0.9s
  [deepfm_only] epoch  4 | loss 0.4958 | valid GAUC 0.6566 nDCG@5 0.5314 primary 0.5940 | 1.1s
  [deepfm_only] epoch  5 | loss 0.4803 | valid GAUC 0.6570 nDCG@5 0.5313 primary 0.5942 | 0.9s
  [deepfm_only] epoch  6 | loss 0.4724 | valid GAUC 0.6567 nDCG@5 0.5314 primary 0.5941 | 0.9s
  [deepfm_only] epoch  7 | loss 0.4680 | valid GAUC 0.6569 nDCG@5 0.5318 primary 0.5943 | 1.1s
  [deepfm_only] epoch  8 | loss 0.4651 | valid GAUC 0.6557 nDCG@5 0.5312 primary 0.5935 | 0.9s
  ABLATION deepfm_only primary=0.5943 gauc=0.6569 ndcg5=0.5318
LESSON: DeepFM with standardized priors: 0.6037 vs 0.5893, promoted; early-stopped at epoch 8.

# SIZING DIRECTIVE (harness policy: flat streak 0 of 3 — 3 more miss(es) end the run)
The convergence rule is per iteration: only a gain > +0.002 over the best-so-far (0.6038) resets the streak. A
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
written predictions are the full bundle; only the sealed score counts. The wall-clock limit is 1500s, so
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

## assistant (response)

{
  "hypothesis": "Adding past-only session/time-context categorical fields (hour-of-day, within-day session depth) plus count-based confidence weights (log1p of exposure counts for user/video/author/user-tab) to the DeepFM+numeric-prior champion gives the model label-free position-bias and reliability signal it currently lacks, without touching the loss or architecture that already validated a promotion this run.",
  "category": "feature",
  "change_spec": "Build on the current champion pipeline.py unchanged except: 1) In load(), also parse 'hourmin' and 'time_ms' per row (already read is_click; add these two fields to the row tuple, keep column order documented). 2) In encode(): add two new categorical fields to FIELDS: 'hour_bucket' and 'session_depth_bucket' (FIELDS becomes 7 entries: user_id, video_id, author_id, tab, dur_bucket, hour_bucket, session_depth_bucket). hour_bucket = int(hourmin // 400) (6 buckets of ~4h, derived purely from hourmin, no label). session_depth_bucket: for every row compute, PAST-AND-CURRENT ONLY (no label used), the 1-indexed rank of this impression among the same user's rows on the same date ordered by time_ms ascending (a pandas groupby(['user','date'])['time_ms'].rank(method='first') computed once per split independently -- this uses only user/date/time_ms, never touches label or is_click, so it's leak-safe even though it's computed 'within-day' since it doesn't use future information about outcomes); then bucket via min(rank,7) into 7 categorical values ('1','2',...,'6','7+'). Build vocab from train only, as with existing fields; unseen bucket falls into UNK slot per current pattern. 3. In build_numeric(): the expanding()/full_train_agg() functions already compute excl_n (train, past-only expanding count) and cnt (valid/test, full-train count) per group for user/video/author/user_tab -- currently these counts are computed but discarded after deriving rates. ADD 4 new numeric columns: log1p(excl_n) for train / log1p(cnt) for valid&test, for each of the 4 groups (user, video, author, user_tab), giving M_NUM=11 total numeric columns (append these 4 after the existing 7, in the same stack() function). Standardize with train-only mean/std exactly as the existing 7 are (extend the mean/std vectors to length 11), clip to [-5,5]. 4. Increase HIDDEN unchanged, K unchanged; the model class already generalizes to len(FIELDS)*k + m for the MLP input, no change needed there besides FIELDS/M_NUM constants. 5. Do NOT change training loop, loss, LR, epochs, patience, or the CLI/output format. 6. Print self-check: vocabulary sizes of hour_bucket and session_depth_bucket, and a sample of 5 log1p-count values before standardization, right after build_numeric. 7. Ablation plan (train on ABLATION_MAX_ROWS=300k subsample, ABLATION_EPOCHS=8, ABLATION_PATIENCE=3, same pattern as existing champion_equiv/deepfm_only ablations): (a) 'champion_equiv': current champion exactly (5 fields, 7 numeric, use_mlp=True) -- no new fields/features; (b) 'no_confidence_counts': 7 fields (with hour/session) + only the original 7 numeric (drop the 4 count columns); (c) 'no_session_fields': 5 fields (no hour/session) + all 11 numeric (with counts). Full run trains the full bundle (7 fields, 11 numeric) as today's 'full'. Expected total runtime: full fit ~60-90s (slightly more fields/features than it01's 41.5s), plus 3 ablation fits at ~15-20s each on the 300k subsample = well under the 1500s budget.",
  "expected_risk": "low",
  "expected_gain": 0.0028,
  "gain_evidence": "Cross-run ledger (PRIOR RUNS table) shows session-depth/hour-of-day categorical features promoted 4 separate times on different champions with deltas +0.0021, +0.0017, +0.0005, +0.0002 (always positive, never negative) -- a clean, low-variance direction. Separately, 'adding past-only user-tab specific historical impression and positive rates' (i.e. counts, not just rates) gave +0.0035 when stacked as a genuinely new signal on a DeepFM+rate champion. Both are additive, label-free, past-only feature signals not yet present in this run's champion (FIELDS=5, M_NUM=7, no counts). Calibration says my predictions run high by ~0.0067 on average, so I am sizing below the naive sum of historical deltas (~0.004) to 0.0028.",
  "ablation_plan": "champion_equiv: current 5-field/7-numeric DeepFM (no new fields); no_confidence_counts: 7 fields (adds hour/session) + original 7 numeric only; no_session_fields: 5 fields (no hour/session) + all 11 numeric (adds counts); full: all 7 fields + 11 numeric (the proposed champion). All four printed as ABLATION <name> primary=... gauc=... ndcg5=... on the same 300k-row/8-epoch harness as today's ablations.",
  "builds_on": "it01 (champion: DeepFM + 7 standardized past-only numeric priors)",
  "rationale": "This directly follows §8.6 (position/exposure bias: session position is a legitimate label-free context feature) and standard count-based confidence smoothing (a row's rate estimate from n=1 exposure is unreliable; giving the model log1p(n) lets it learn to discount noisy rates -- directly relevant given 17.5% of valid users have a single impression, §1). Ledger evidence across prior independent runs of this exact task shows both components consistently non-negative when added as new past-only signal, unlike training-loss changes (BPR hybridized onto a DeepFM+numeric champion measured -0.0012 in a prior run) or model-family swaps (FwFM -0.0002, DIN-attention -0.0001) which plateaued once DeepFM+numerics became champion. Keeping the validated DeepFM+numeric-prior champion fully intact and only adding new, previously-absent information respects rule 1c and the streak-0 sizing directive's call for genuinely new signal over more capacity.",
  "search_check": "Direction considered: past-only session-depth/hour-of-day categorical features and count-based (log1p exposure count) confidence weighting for noisy empirical rate features. Both are basic, decades-old statistical/feature-engineering techniques (position bias features, additive/count smoothing for sparse rate estimates) with no active research frontier that would change their risk profile; nothing published in the last 12 months alters how a within-user-day rank feature or a log-count confidence feature should be computed or evaluated. I already have direct same-task ledger evidence (4 promotions, no negatives) for the session/hour component, which is stronger and more specific than any general literature search would provide."
}
