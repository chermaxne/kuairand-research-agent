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
- Harness rules: promotion needs > +0.0010 over the champion; the convergence streak resets only on > +0.0020
  over the best-so-far; **three consecutive misses end the run** (crashes count). Realistic final range for a
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

## 4. Measured levers (validation primary; champion = 0.6015; ± seed std 0.0003)
| lever | primary | Δ | verdict |
|---|---|---|---|
| **pairwise within-user loss, FM trained from scratch** (3 seeds) | 0.6034 / 0.6026 / 0.6027 | **+0.0013** | best single lever; every seed above every pointwise seed |
| pairwise loss **warm-started** from the pointwise optimum | 0.6012 / 0.6016 | 0 | useless — train from scratch |
| + **within-day position field** (label-free, from `time_ms`) on the pointwise FM (3 seeds) | 0.6028 / 0.6027 / 0.6018 | +0.0008 | reliable small gain |
| pairwise + position field (2 seeds) | 0.6038 / 0.6039 | +0.0023 | stacks |
| **rank-average of 2 pairwise+position seeds** | **0.6042** | **+0.0027** | clears the convergence threshold |
| 5-seed rank-average of the pointwise FM (3-seed: 0.6021; logit-average: 0.6021) | 0.6026 | +0.0011 | near-certain, cheap finisher |
| past-only rolling user×author/tab/video rates as extra FM fields (autonomous run) | 0.6022 | +0.0008 | real; untested on top of pairwise |
| L2 1e-6 → 1e-5 (autonomous run) | 0.6025 | +0.0010 | within noise |
| K = 32 · patience/epochs · LR 0.002 | 0.6009–0.6022 · flat · 0.49 (diverged) | ≤ 0 | not worth iterations |
| multi-task FM: + is_click head (w 0.3) · + censored watch-time head · + both | 0.6016 · 0.6001 · 0.6007 | ≈ 0 / − | flat-to-negative |
| FM trained on `is_click`, scored on long_view · rank-avg with the lv-FM | 0.5831 · 0.5953 | − | click signal does not transfer |
| position + session-position + fine duration fields added at once | 0.6013 | −0.0002 | adding several fields at once can hurt |
| fine duration buckets (0 / 7 / 18 s) | 0.6013 | −0.0002 | absorbed by video_id |
| recency-weighted training tau 10 d / 5 d | 0.6018 / 0.6010 | ≈ 0 | flat |
| LightGBM (logloss) on past-only rate/count features | 0.5842 | −0.017 | worse than FM |
| LightGBM lambdarank grouped by user, past-only features + out-of-fold FM score · rank-avg with FM | 0.5975 · 0.6009 | − | GBDT loses in every form tried |
| LightGBM with leave-one-out target encoding | 0.45–0.47 | below random | **inverted by the LOO leak** |
| rank-avg of FM + the 0.5842 LightGBM | 0.5941 | −0.007 | a weak ensemble member drags the strong one down |
| FM trained on week 1 only | 0.5981 | −0.0034 | half the data costs 0.003 |

## 5. Direction ladder — ranked by measured gain × reliability. A prior, not an order.
The recipes are reference implementations that worked once; use them to avoid implementation failures (two of the
first four autonomous attempts crashed or inverted), not as a script. Deviate whenever the ledger gives a reason.
**R1. Pairwise within-user loss, from scratch (+0.0013, high reliability).** Keep the FM scorer and fields.
Training set = the train rows of *mixed* users (users with both labels). Each epoch: for every positive row of a
mixed user draw one random negative row of the SAME user; shuffle the pairs; batches of 8192 pairs. Loss
`softplus(s_neg − s_pos)`; with `g = sigmoid(s_neg − s_pos) / batch`: apply `+g` to the negative row's W and V
gradients and `−g` to the positive row's (same `np.add.at` pattern as the pointwise step: V gets `g·(S − E)`).
Adam lr 1e-3, l2 1e-6, ≤ 30 epochs, early stopping on validation primary with patience 4 (best epoch 4–8).
Prediction and output unchanged. **Sanity: train GAUC after epoch 1 must be > 0.6** (the first attempt in this
project had the sign flipped and scored 0.39). Do not warm-start from the pointwise model.
**R2. Label-free session-context field (+0.0008 alone, +0.0010 on top of R1, medium-high reliability).**
Sort rows by (`user_id`, `time_ms`); `pos_day` = rank of the impression within its (user, date); bucket with
edges [1, 2, 3, 4, 6, 10] → {0, 1, 2, 3, 4–5, 6–9, 10+}; add as a 6th categorical FM field (own vocabulary +
UNK, same encoding as the others). Computed identically for train and validation rows from non-feedback columns.
Variants to try one at a time (never two new fields in one run): impressions-in-day count, log minutes since the
user's previous impression (30-min gap = new session), hour-of-day bucket.
**R3. Seed rank-average (+0.0010 pointwise, +0.0003 on R1+R2; near-certain).** Train the same pipeline with
3–5 seeds inside one run (each ≈ 30 s), rank-normalise each model's scores *within user* (percentile rank), average
the ranks. Logit-averaging is slightly worse (0.6021 vs 0.6026). Always the finisher; combine with R1+R2.
**R4. Past-only history fields (+0.0008 measured on the pointwise FM; medium).** For each row, statistics from
strictly earlier dates: user × author long_view rate and count, user × tab rate, user × tag rate, video rate and
impression count (smoothed with prior 20 toward the train mean); for validation rows use all of train. Bucket the
rates into ~10 quantile bins and add as categorical fields — one field per run. These are user × item
interactions in disguise, so they survive the within-user rule. Untested on top of R1–R3.
**R5. Untested, plausible small gains (≤ +0.002):** listwise / sampled-softmax within user (a variant of R1;
literature says a tighter DCG surrogate); a video × tab cross field; a DeepFM/DCN trunk over the same fields
(capacity alone is flat — only with the new fields).
**R6. Multi-task — measured flat here (§2, §4), so low expected gain, not forbidden.** The variants we tried were
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
2. Arithmetic you cannot escape: single levers on this data are each ≈ +0.001, the streak resets only on > +0.002,
   and three misses end the run. Early iterations should therefore combine complementary levers rather than test one
   at a time (a loss change + a new field + seed averaging measured 0.6042 together; each alone ≈ 0.602–0.603) —
   which combination, and in what implementation, is your call.
3. A +0.0005..+0.002 result is a signal to keep stacking on it, not to abandon it; adding several new fields at once
   can hurt, so grow the champion one element at a time once it survives.
4. Put a self-check in every change spec that the code must print (train GAUC after epoch 1, pair count, a new
   field's vocabulary size) — a wasted iteration costs a third of the run.
5. A result below the popularity rung (0.581) or a GAUC < 0.5 is an implementation bug, not a research outcome.

## 7. Trap list
- Same-row feedback columns (`is_click`, `play_time_ms`, `is_like`, …) as inputs = leakage. Session/time features
  may come only from `time_ms`, `hourmin`, `date`.
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
- The pipeline contract: fit on the train split only (validation only for early stopping); write every row of the
  requested split in file order; `--split test` must keep working; no network, no installs.

## 8. Engineering facts
Data load 3–4 s (pure-Python CSV); FM epoch ≈ 2 s pointwise (1.14M rows) / ≈ 1.5 s pairwise (≈ 380k pairs);
sealed `evaluate` on valid 0.2 s; the champion run ≈ 30 s end to end; a 5-seed R1+R2+R3 pipeline ≈ 3–4 min;
wall-clock limit 900 s per experiment. Memory: the encoded train matrix is 1.14M × F int32 — trivial. Libraries:
numpy, pandas, scikit-learn, lightgbm, torch (CPU). IDs are strings in the CSVs; echo them as read.

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
