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

## 3. Direction ladder — biggest plausible upside first; knob-tuning last
The convergence rule gives you only three consecutive misses, so spend them on changes that can move primary by
+0.005 or more, not on ±0.001 knobs. Order of attack (the harness also enforces "structural first" for the
opening iterations):
a. **Multi-task learning, in its STRONG form.** The single is_click auxiliary head was flat here (−0.0005), so
   do not repeat it alone. The strong form is: shared FM/embedding trunk + several heads — long_view (main),
   is_click, is_like, and a **watch-time head** (play_time_ms, log1p, regressed with a censored/one-sided loss
   because play time is truncated at video duration — the CWM idea), each with its own weight (e.g. 1.0 / 0.3 /
   0.2 / 0.3); escalate to gated partial sharing (MMoE / PLE-style: 2–4 experts, per-task gates) when the
   auxiliaries improve but the main task stalls (seesaw). Watch time carries much more signal per row than the
   binary label and is available on every train row. Score with the long_view head only.
b. **User history / sequence features, PAST-DATES-ONLY.** The user's last N (20–50) train interactions
   (video, author, tab, label, play-time ratio) attended against the candidate item (DIN-style target attention),
   or, cheaper, per-user × per-author / per-tab / per-duration-bucket historical long_view rates and recency
   (days since the user's last impression, since the video first appeared). These are user × item interactions,
   so they survive the within-user no-op rule. Nothing about the user's history has been used so far.
c. **GBDT stacking on engineered past-only features** (see §6.1): LightGBM over rolling rates/counts + the FM
   score. Count features are where GBDTs shine; the organizers never tried them.
d. **Loss aligned with the metric**: pairwise BPR or within-user listwise softmax on the FM scorer (§6.3 —
   implement with the sign check; the first attempt here was inverted, not wrong in principle).
e. **Ensembles of the above** (rank-average per user) once two decorrelated scorers exist — reliable
   +0.002..+0.005.
f. **Hyperparameter tuning** (LR, k, epochs, patience, L2, batch, bucket counts) — LAST. Capacity is not the
   bottleneck (organizers: k = 8/16/32 flat); gains here are < 0.002 and do not reset the streak. Not before
   the structural directions are exhausted, and never as a first idea.

## 4. TRAP LIST — read before every proposal
- **A score BELOW the rungs means the implementation is broken, not that the idea failed.** Random scores
  0.4834, item-popularity 0.5807. Sealed GAUC < 0.5 means the ranking is INVERTED (sign error in a loss or
  gradient, negated predictions, flipped labels) — the harness sends such results to the Debugger once; if
  you see one in the ledger, re-propose the idea with the sign fix, do not write it off.
- **An exploding training loss (loss rising epoch over epoch, e.g. 0.7 → 12) is a bug** (wrong gradient sign,
  learning rate far too high for a new loss), never a research outcome.
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
- **Convergence arithmetic — read this first.** The run ENDS after 3 consecutive iterations that do not beat the
  best-so-far by more than 0.002 (failed and crashed iterations count). A whole run can therefore be over after
  three misses. Every proposal must be your single highest-expected-gain idea, implemented in its most reliable
  form; there is no budget for "let's just see". A crash or an inverted implementation burns one of the three.
- **Sub-threshold gains are signals to STACK, not to abandon.** A result of +0.0005..+0.002 (e.g. rolling
  past-only features at +0.0008) is real under σ≈0.0008 only in aggregate: the next iteration should COMBINE
  that change with the next idea (features + GBDT, features + ensemble, features + tuned loss), because
  independent small gains add up and the promotion margin (0.001) / ε (0.002) are cleared by sums, rarely by
  one knob. Say explicitly in the change spec which earlier attempt you are stacking on and why.
- Explore structurally different ideas early — but "structural" must also mean "high expected gain and
  reliable to implement" (see §6). Refine winners once found; combine winners later.
- At flat streak ≥ 2 pick the most reliable promising idea — the one most likely to clear +0.002 — and
  prefer combining two sub-threshold winners over a new unproven direction.
- Never retry BLOCKED items; never re-propose a failed idea without a stated new reason. A crashed or
  inverted implementation of a good idea is NOT a failed idea: re-propose it with the specific fix.
- Budget runtime explicitly (baseline ≈ 30 s; each FM epoch ≈ 2 s on 1.14M rows; loading ≈ 4 s;
  wall-clock limit per experiment 900 s); a timeout is a failed iteration.
- One idea per iteration; write the change spec so the Engineer cannot misread it — include a
  self-check the code must print (e.g. "train GAUC after epoch 1 must be > 0.55; assert it").

## 6. High-expected-gain recipes for THIS dataset (ordered by expected gain / reliability)
0. **Multi-task FM with a watch-time head.** Keep the champion's embeddings V (k=16) as the shared trunk. Heads:
   long_view (logloss, weight 1.0), is_click (logloss, 0.3), is_like (logloss, 0.2), watch-time
   (target = log1p(play_time_ms), predicted from the same interaction term plus a head-specific bias/linear
   term; loss = squared error, but one-sided when play_time_ms >= duration_ms, i.e. the video was watched to
   the end: only penalise under-prediction, because the true watch time is censored at duration; weight 0.3).
   All heads trained jointly on TRAIN rows only, same Adam, same early stopping on validation long_view
   primary. Score the validation rows with the long_view head only. Runtime ≈ 2× the champion (~60 s).
   Expected: +0.003..+0.008 if the auxiliary signals transfer; if the auxiliaries improve but long_view stalls,
   the next iteration escalates to MMoE/PLE gating (2–4 expert embedding sets, softmax gate per task).
1. **Stack: past-only rolling features + LightGBM.** Compute, strictly from earlier dates, per-row features:
   user long_view rate & count, user×author rate, user×tab rate, video rate & impression count, author rate,
   days since the user's last impression, days since the video's first impression, the FM champion's score
   (train it first, use out-of-fold or train-only predictions) — then a LightGBM binary classifier
   (num_leaves 63, lr 0.05, ~400 rounds, early stopping on validation) scores the rows. GBDT on count/rate
   features is where the big jumps usually are; it uses the exact features the organizers never tried.
2. **Rank-average ensemble of two decorrelated scorers** (FM champion + LightGBM from recipe 1, or FM + a
   correctly implemented BPR-FM): per user, rank-normalise each score and average. Reliable +0.002..+0.005 when
   the members are individually near the champion and different in kind. Low risk, low runtime.
3. **BPR done right.** For each TRAIN positive sample a negative from the SAME user and date-neighbourhood,
   loss = softplus(s_neg − s_pos) = −log σ(s_pos − s_neg); gradient on the pos row is −σ(s_neg − s_pos)·∂s,
   on the neg row +σ(s_neg − s_pos)·∂s. Use ALL positives per epoch (≈380k pairs), not a per-user cap. Sanity:
   train GAUC after epoch 1 must exceed 0.55 — if it is below 0.5 the sign is flipped. Warm-start from the
   pointwise champion weights (train pointwise 5 epochs, then BPR) for stability.
4. **Multi-task only as a regulariser on top of a winner** (is_click aux head, weight 0.3) — alone it moved
   primary by < 0.001 here; combined with recipe 1 or 3 it can add a little.
5. Wider embeddings / DeepFM / DCN: capacity is not the bottleneck (organizers' finding) — only after 1–3.
