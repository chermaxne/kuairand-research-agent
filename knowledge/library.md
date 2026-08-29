# KuaiRand-Pure within-user ranking — evidence-based playbook (built 2026-08-28)

How to read this file: every number below was **measured on the official validation split with the sealed
`evaluate.py`** (train 20220408–0421 → valid 0422–0428) unless it is marked *literature*. Claims were
independently re-derived by a second agent; where the two disagreed, the corrected number is given. Treat the
measured table in §4 as ground truth about THIS dataset and metric — it overrides recommender-system folklore.

## 1. Task, metric, and the noise floor
- Rank each user's logged impressions in the split; label `long_view`; primary = mean(GAUC, nDCG@5). GAUC counts
  only users with 0 < positives < impressions, weighted by #positives; nDCG@5 counts every user (all-negative
  users are stuck at 0). Validation: 124,909 rows, 22,377 users; **30.3% all-negative, 11.9% all-positive, 57.8%
  mixed — the mixed users hold 78.9% of the rows and are the only ones a model can move.** Oracle primary 0.848.
- Rungs: random 0.483 · item popularity 0.581 · **FM champion 0.6015** (5 fields: user_id, video_id, author_id,
  tab, duration decile; pointwise logloss; Adam 1e-3; batch 8192; early stop on validation, best epoch ≈ 7).
- Noise: FM seed-to-seed std is **0.0003** (5 seeds: 0.6011–0.6020). But the validation metric itself has a
  user-bootstrap standard error of **0.0022** (95% ≈ ±0.0043): a single-run delta below ~0.002 is not evidence of
  a better model, and a validation gain below ~0.004 may not transfer to the hidden test week.
- Harness rules: promotion needs > +0.0005 over the champion (≈ 2× the 3-seed noise, so +0.0006..+0.002 results
  are banked and stacked); the convergence streak resets only on > +0.0020 over the best-so-far; **three consecutive
  misses end the run** (crashes count). Realistic final range for a
  well-run session: 0.605–0.61, not 0.65.

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
  quintiles 0.30–0.36) and the `video_id` embedding already absorbs it: fine duration buckets measured 0.6013 (flat).
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
  help (tau 10 d: +0.0003; tau 5 d: −0.0005).
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

## 4. Where value has and has not been found
This section combines the organizers' published findings, the recommender-systems literature and offline analysis of
this dataset. It tells you which directions have repaid effort and which have not. It deliberately does **not** predict
how much any change will gain — that is what your iterations measure, and your own measurements override this section.

### Directions that have repaid effort
- **Aligning the objective with the metric.** The baseline optimises pointwise calibration while the metric is a pure
  within-user ranking metric. A pairwise within-user objective — for each train positive, sample a negative from the
  SAME user; loss `softplus(s_neg − s_pos)` — is the most valuable single change found so far, and loss alignment is
  the organizers' top untested direction. Train it **from scratch**: warm-starting from the converged pointwise model
  does not help. It converges fast (best epoch ≈ 4–8), so keep early stopping on validation.
- **Label-free session context as a model field.** Where a row sits in the user's day carries real signal (the
  long_view rate falls from 0.34 at a user's first impression of the day to 0.18 by the tenth). Encoding the
  within-day position as a bucketed categorical field is worth trying; derived from `time_ms`/`date` only, so it
  cannot leak.
- **Variance reduction by seed averaging.** Rank-normalise each model's scores *within user*, then average the ranks
  over a handful of seeds. Small but among the most dependable moves, and it composes with everything else. Rank
  averaging outperformed logit averaging.

### Directions that have not repaid effort here
Do not spend a streak-critical iteration on these without a stated reason your version differs:
- **Adding more categorical fields to the FM.** The organizers measured this for static user/item features; it also
  holds for label-free context fields (hour-of-day, session-boundary flags), item×context crosses, and bucketed
  history-rate fields (user×author, user×tab, user×video). The `user_id × video_id` crosses already carry most of the
  learnable signal and extra fields dilute the embedding budget more often than they add; bucketed history-rate
  fields were the worst offenders. If you try one, try exactly **one**, and measure it.
- **Model capacity.** k = 8/16/32 is flat under the pointwise loss (organizers) and also under a pairwise loss.
- **Gradient-boosted trees** in every form tried (pointwise, user-grouped lambdarank, with and without an out-of-fold
  FM score as a feature, alone and as an ensemble member). Within-user ranking rewards the id-embedding interactions
  an FM learns; aggregate rate features do not substitute, and a weak ensemble member drags a strong one down.
- **Auxiliary heads / multi-task.** `is_click` and `long_view` are nested thresholds of the same play-time variable
  (§2) and the other feedback signals are 0.1–1.9% sparse, so a shared-trunk auxiliary head adds little. Attempt it
  only in a form that exploits the nesting (ESMM-style) or uses genuinely different behaviour.
- **Watch-time regression heads.** The label already *is* a duration-normalised threshold of watch time.
- **Recency weighting, duration re-bucketing, learning-rate / patience / regularisation tuning.** All flat.
- **Ensembling highly correlated members.** Rank-averaging variants of the same model family gains little; an
  ensemble needs members that differ in kind and are individually close to the champion.

### The open frontier
The FM family appears close to exhausted in the low 0.60s. What has **not** been tried on this dataset, in rough order
of promise: user behaviour sequences (per-user histories run to 30–100 impressions; DIN-style target attention over a
user's previous items is completely unexplored and is the organizers' second-ranked direction); listwise /
sampled-softmax objectives, particularly as an *additional* ensemble member rather than a replacement for a working
pairwise loss; a genuinely different model family to ensemble with the FM; and the random-exposure log as an unbiased
check on whether a gain is real or an artefact of biased traffic.

## 5. Reference implementations — how to build the tricky pieces correctly
Not a script and not a ranking: implementation notes for the pieces that are easy to get wrong (two of the first four
autonomous attempts here crashed or produced an inverted ranking). What to try, in what order, and what to combine is
your judgement — informed by §4 and, above all, by your own ledger.
**R1. Pairwise within-user loss, from scratch.** Keep the FM scorer and fields.
Training set = the train rows of *mixed* users (users with both labels). Each epoch: for every positive row of a
mixed user draw one random negative row of the SAME user; shuffle the pairs; batches of 8192 pairs. Loss
`softplus(s_neg − s_pos)`; with `g = sigmoid(s_neg − s_pos) / batch`: apply `+g` to the negative row's W and V
gradients and `−g` to the positive row's (same `np.add.at` pattern as the pointwise step: V gets `g·(S − E)`).
Adam lr 1e-3, l2 1e-6, ≤ 30 epochs, early stopping on validation primary with patience 4 (best epoch 4–8).
Prediction and output unchanged. **Sanity: train GAUC after epoch 1 must be > 0.6** (the first attempt in this
project had the sign flipped and scored 0.39). Do not warm-start from the pointwise model.
**R2. Label-free session-context field.**
Sort rows by (`user_id`, `time_ms`); `pos_day` = rank of the impression within its (user, date); bucket with
edges [1, 2, 3, 4, 6, 10] → {0, 1, 2, 3, 4–5, 6–9, 10+}; add as a 6th categorical FM field (own vocabulary +
UNK, same encoding as the others). Computed identically for train and validation rows from non-feedback columns.
Variants to try one at a time (never two new fields in one run): impressions-in-day count, log minutes since the
user's previous impression (30-min gap = new session), hour-of-day bucket.
**R3. Seed rank-average.** Train the same pipeline with
3–5 seeds inside one run (each ≈ 30–45 s), rank-normalise each model's scores *within user* (percentile rank), average
the ranks. Rank averaging outperformed logit averaging. Always the finisher; combine with R1+R2.
Vectorised, the whole rank step is one line — `pd.Series(scores).groupby(np.asarray(users)).rank(pct=True).values`
(≈ 0.1 s). A Python loop over users that masks the rows (`for u in unique(users): mask = [x == u for x in users]`)
is 22k × 125k comparisons ≈ tens of minutes and **killed the 2026-08-29 run at the 900 s limit**.
**R4. Past-only history fields — deprioritised (§4: extra fields have not paid off).** Bucketed
user × author / user × tab rates duplicate interactions the FM already learns from its id fields, so as extra FM
fields they add nothing. They only make sense for an entity the FM does NOT have as a field (user × tag,
user × duration-bucket history, video impression counts) or as inputs to a different model family — and even
then expect ≤ +0.001. Do not spend a streak-critical iteration on them.
**R5. Untested variants worth considering:** listwise / sampled-softmax within user (a variant of R1;
literature says a tighter DCG surrogate); a video × tab cross field; a DeepFM/DCN trunk over the same fields
(capacity alone is flat — only with the new fields).
**R6. Multi-task — low expected value here (§2, §4), not forbidden.** The variants we tried were
simple (linear heads sharing the FM interaction; click / watch-time targets). Untried forms with a real argument:
ESMM-style p(long_view) = p(click) · p(long_view | click); heads on genuinely different behaviours (like,
profile_enter) with gated sharing (MMoE/PLE); an MLP tower per task. Expect ≤ +0.001 unless you can say why yours
differs.
**Measured flat in our probes — deprioritise, and re-try only with a stated reason why your version differs:**
GBDT (pointwise and user-grouped lambdarank, with and without an out-of-fold FM score); auxiliary click / watch-time
heads; recency weighting; fine duration buckets; K / LR / L2 / patience knobs; cold-start handling (1.9% of users);
video age; static user/video side features (organizers' finding); warm-starting pairwise from pointwise.
The statistic feature file is not "flat" — it is forbidden (§3, §7).

## 6. Strategy under the convergence rule
1. **This file is the prior; your ledger is the posterior.** When your own measured iterations contradict a number
   here, trust the ledger and say so in the rationale. When an attempt failed for implementation reasons (crash,
   GAUC < 0.5, exploding loss), the idea is untested — fix it rather than move on.
2. **Rhythm: bundle where the rule forces it, single changes everywhere else.** Single levers here are ≈ ±0.001 and
   only a > +0.002 step resets the streak, so iteration 1 and the last shot before convergence must bundle
   complementary levers whose individual effects you or §4 already understand. In between, change exactly ONE thing per
   iteration: gains above +0.0005 are banked as the new champion, and a single change's delta is the only way to learn
   which component works. **Never bundle components whose individual effects are unknown** — a bundle that moves
   +0.0001 teaches nothing about its parts, which has already cost this project several iterations. The run that reached 0.6044 died exactly this way: it tested one small
   lever per iteration after the bundle. After a promotion, bundle again — several complementary UNTESTED levers
   plus more seeds in one pipeline (e.g. 5 seeds + a new field type + a loss variant kept alongside the proven
   loss, not instead of it).
3. **Noise floor: a gain below 0.0006 (2× seed std) is noise, not a signal.** Stack on a +0.0006..+0.002 result;
   treat a +0.0002 as "no information" and do not spend the next iteration on the same kind of lever.
4. **Streak ≥ 2 is the last shot.** Take the highest-probability bundle, never a replacement of a proven component
   (the 0.6044 run swapped its working pairwise loss for sampled-softmax at streak 2 and lost 0.004). Re-using
   what works plus more seeds plus one genuinely new signal is the reliable move.
5. Put a self-check in every change spec that the code must print (train GAUC after epoch 1, pair count, a new
   field's vocabulary size) — a wasted iteration costs a third of the run. Assert only on TRAIN-side quantities:
   the harness re-runs every would-be promotion with ~10% of validation users' labels corrupted (leak test), so a
   hard assertion on the validation metric crashes that re-run and forfeits the promotion.
6. A result below the popularity rung (0.581) or a GAUC < 0.5 is an implementation bug, not a research outcome.
7. If the reachable levers are genuinely exhausted, converging at the plateau is the correct outcome — the rule is
   the organizers' definition of "done". Do not manufacture risky swaps to avoid it.

## 7. Trap list
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
  the 900 s kill; vectorise (§8). Pairs rebuilt in Python every epoch quintuple the epoch time.
- The pipeline contract: fit on the train split only (validation only for early stopping); write every row of the
  requested split in file order; `--split test` must keep working; no network, no installs.

## 8. Engineering facts — runtime is a budget, and Python loops are how it gets blown
- Data load 3–4 s (pure-Python CSV); FM epoch ≈ 2 s pointwise (1.14M rows) / ≈ 1.5 s pairwise (≈ 380k pairs);
  sealed `evaluate` on valid 0.2 s; the pointwise champion ≈ 30 s end to end; the 3-seed R1+R2+R3 champion ≈ 130 s.
  **Hard limit 900 s per experiment** — a timeout is a lost iteration. Budget: ≤ 400 s for a 4–5-seed pipeline.
- Vectorise every per-user operation: `groupby(...).rank/transform/size` in pandas, or `np.unique(..., return_inverse)`
  + `np.argsort` / `np.add.at` in numpy. Never write `for u in users: mask = (users == u)` over the rows — quadratic.
- Build pair pools and index structures ONCE (per-user positive/negative index arrays), then resample per epoch with
  array indexing; rebuilding them in Python each epoch costs ~8 s per epoch on top of 1.8 s of training.
- Print progress that a reader can budget from: per-epoch time, pair count, seed number.
- Memory: the encoded train matrix is 1.14M × F int32 — trivial. Libraries: numpy, pandas, scikit-learn, lightgbm,
  torch (CPU). IDs are strings in the CSVs; echo them as read.

## 9. Literature notes (what transfers, what does not)
- Watch-time prediction & duration bias — D2Q (KDD'22), TPM (KDD'23), CWM (KDD'24; evaluated on KuaiRand-Pure
  and cited by the organizers' evaluate.py), DML (CIKM'23), D2Co (RecSys'23): they debias a *continuous* watch-time
  target; our label is already a duration-normalised threshold, so only the ordinal-decomposition idea transfers.
- Multi-task — MMoE (KDD'18), PLE (RecSys'20), ESMM (SIGIR'18): gains need related-but-different tasks; ours are
  nested thresholds of one variable. Seesaw/negative transfer is the documented failure mode.
- Ranking losses — pairwise/listwise objectives are ranking-calibrated for AUC (Uematsu & Lee, JASA 2017);
  within-user pairwise losses raise GAUC in industry; softmax-family losses are tighter DCG surrogates
  (PSL 2024; sampled softmax, TOIS 2024). Consistent with R1 being the best measured lever.
- Long-sequence interest models — DIN, MIMN, SIM, TWIN-V2 (Kuaishou): large gains with long histories; here
  histories are short (median 31) and the catalogue tiny, so the cheap form is R4.
- Tabular ML — GBDTs are strong on tabular data in general (McElfresh 2023; Borisov 2021), but within-user ranking
  rewards user × item embeddings over aggregate rates: measured worse here in every form.
