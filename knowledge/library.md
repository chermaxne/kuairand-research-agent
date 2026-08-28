# Knowledge library — KuaiRand-Pure within-user ranking (playbook for the Researcher)

## 1. Task facts (measured; do not re-derive)
- Data: KuaiRand-Pure short-video impression logs. Train = dates 20220408–20220421 (1,141,112 rows),
  validation = 20220422–20220428 (124,909 rows, 22,377 users), hidden test = 20220429–20220508. The split
  is **date-based ⇒ the evaluation period is AFTER training: temporal shift is real** (new videos,
  drifting popularity). During the loop the data dir contains **no test-period rows at all**.
- Label: `long_view` (0/1, native column). It is logged on **every** impression (no selection bias on
  the label itself). Positive rate ≈ 0.337 in train, ≈ 0.313 in validation — moderately sparse.
- Metric: `primary = mean(GAUC, nDCG@5)` computed by the sealed `evaluate.py`, **within user** over that
  user's logged impressions in the split (no full-catalogue retrieval). GAUC counts only users with
  0 < positives < impressions and weights them by #positives; nDCG@5 counts every user (all-negative
  users score 0 no matter what). Validation ceiling: oracle primary 0.8484 (nDCG@5 ceiling 0.6968).
  Baseline FM: 0.6016. Item popularity: 0.5807. Random: 0.4834. FM seed std ≈ 0.0008 ⇒ gains below
  0.002 are noise.
- **Any score term that is constant within a user is a ranking no-op** (user bias, pure user-side
  features): it cannot change within-user order. User-side information only helps through
  **interactions with the item side** (crosses, user-history × item features, sequences).
- Log columns: user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow, is_comment,
  is_forward, is_hate, long_view, play_time_ms, duration_ms, profile_stay_time, comment_stay_time,
  is_profile_enter, is_rand, tab (12 feedback columns). The baseline uses only 5 categorical fields:
  user_id, video_id, author_id, tab, duration bucket (10 train-quantile buckets of duration_ms).
- Side files: `user_features_pure.csv` (activity degree, follower/fan/friend ranges, register days,
  18 one-hot feats), `video_features_basic_pure.csv` (author_id, video_type, upload_dt, upload_type,
  video_duration, music_id, tag), `video_features_statistic_pure.csv` (aggregate engagement counts —
  aggregation window unknown ⇒ treat as potentially future-leaking; prefer train-derived stats),
  `log_random_4_22_to_5_08_pure.csv` (random-exposure log, validation-period part only in the loop —
  usable as an extra unbiased validation set, never for training).

## 2. What the organizers already measured (do not repeat)
- **Adding static features** (all 13 CWM fields: +music_id/video_type/upload_type + 6 coarse user
  buckets) → no gain (0.5940 vs 0.5950 test, inside noise). Reason: user_id × video_id crosses already
  absorb most learnable signal; coarse user buckets are redundant next to user_id.
- **More capacity** (FM k = 8 / 16 / 32) → 0.5895 / 0.5902 / 0.5887: flat. 1.1M rows do not support
  bigger embeddings. **The bottleneck is neither features nor capacity.**
- Organizers' ranked list of UNTESTED headroom: (1) loss aligned with the ranking metric — pairwise
  (BPR) or listwise (within-user softmax) instead of pointwise logloss; (2) user behaviour sequences /
  interest modelling (DIN/SIM-style) — completely unused so far; (3) multi-objective auxiliary tasks
  (is_click, is_like, is_follow, is_comment, is_forward, play_time_ms); (4) watch-time modelling
  (censored regression of play time); (5) DeepFM/DCN/xDeepFM — lower priority given (capacity is not
  the bottleneck); (6) time features and drift (hourmin, date); (7) the random-exposure log as an
  unbiased validation check.

## 3. Direction ladder (with reasons) — climb it, do not skip rungs blindly
a. **Loss / objective aligned with the metric** (cheapest structural swing, top organizer pick):
   sample within-user (positive, negative) pairs from the same user in train and optimise BPR
   (log-sigmoid of score difference), or a within-user softmax over the user's impressions of a
   day/session. Keep the FM scorer; change only the loss and the batch construction. Expect the
   largest single gain; watch runtime (pair sampling per epoch is O(rows)).
b. **Multi-task heads**: start with long_view + is_click (shared embeddings, one auxiliary loss weight
   ≈ 0.3–0.5), then + is_like, then play_time (regression head, log1p, or censored at duration).
   Escalate to MMoE/PLE-style partial sharing only on seesaw symptoms (aux improves, primary stalls).
c. **History features, PAST-DATES-ONLY**: per row, the user's historical long_view rate, per-author /
   per-tab engagement rates, the item's rolling long_view rate and impression count, recency (days
   since the user's last impression / since the video first appeared), computed strictly from earlier
   dates (train rows: earlier train dates; validation rows: all train dates). Smooth with a prior.
   These are user × item interactions in disguise, so they survive the within-user no-op rule.
d. **Sequence / interest models**: the user's last N (20–50) train interactions (video, author, tab,
   label) attended against the candidate item (DIN-style). Costly; do it after a–c produced a champion
   worth attending on top of, and budget the runtime.
e. **Model ladder**: FM (champion) → FM with the new loss → DeepFM-style / wider embeddings (only with
   a new signal, capacity alone is flat) → LightGBM on engineered past-only features (fast, strong on
   count features; needs the history features from c) → small ensemble (rank-average) of the champion
   family — an ensemble of two decorrelated scorers is a reliable, low-risk final gain.
f. **Training tweaks**: class weighting for the sparse positive, LR schedule / warm restarts,
   early-stopping patience, more epochs with a smaller LR. Small, reliable, good streak-≥2 material
   only when the champion has not been tuned yet; gains are usually < 0.002.

## 4. TRAP LIST — read before every proposal
- **Same-row feedback columns as input features = LEAKAGE, forbidden** (`is_click`, `play_time_ms`, …
  of the row being scored). They may only be auxiliary *targets* in multi-task training on TRAIN rows.
- **Whole-dataset aggregates leak the future**: popularity / rates computed over all rows, or over the
  validation rows themselves, are illegal. Compute rolling / past-only statistics; validation rows may
  use train statistics only.
- **A sudden huge jump (> +0.03) ⇒ suspect leakage first.** Re-verify the feature computation before
  trusting it; the ledger should say "verified past-only".
- **Only the sealed `evaluate.py` score counts.** Metrics printed inside a pipeline are for early
  stopping only and may disagree if the pipeline evaluates on something else.
- Validation-based early stopping is fine (the baseline does it) but tuning many knobs against
  validation overfits it; prefer structural changes over knob sweeps.
- `video_features_statistic_pure.csv` may aggregate beyond the train period — if used, say why it is
  safe or restrict to train-derived counts.
- Pure user-side features (and user bias terms) are ranking no-ops — see §1.
- IDs are strings in the CSVs; keep them as read. Output every row of the split, in order, finite.

## 5. Strategy rules (the harness enforces the numbers; you enforce the judgment)
- Explore structurally different ideas early; refine winners once found; combine winners later.
- At flat streak ≥ 2 pick the most reliable promising idea — the one most likely to clear +0.002.
- Gains < 0.002 do not reset the streak: prefer bigger structural swings over micro-tuning.
- Never retry BLOCKED items; never re-propose a failed idea without a stated new reason.
- Budget runtime explicitly (baseline ≈ 30 s; each FM epoch ≈ 2 s on 1.14M rows; loading ≈ 4 s;
  wall-clock limit per experiment 900 s); a timeout is a failed iteration.
- One idea per iteration; write the change spec so the Engineer cannot misread it.
