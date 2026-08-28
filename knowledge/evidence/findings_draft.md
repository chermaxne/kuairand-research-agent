# Draft findings for the KuaiRand-Pure knowledge file — FOR INDEPENDENT EVALUATION
Task: within-user ranking of logged impressions, label `long_view`, metric primary = mean(GAUC, nDCG@5) from the
organizers' sealed evaluate.py. Train 20220408–0421 (1,141,112 rows), valid 0422–0428 (124,909 rows, 22,377 users),
hidden test 0429–0508. Baseline: numpy FM (5 categorical fields user_id, video_id, author_id, tab, duration decile),
valid primary 0.60147 (published 0.6016), seed std 0.0008. Convergence rule: 3 consecutive iterations without a
> +0.002 gain over best-so-far end the run.

## A. Data facts (measured with pandas on the kit CSVs; scripts: analyze.py -> stats.json)
A1. Catalogue is tiny: 7,583 videos, 6,510 authors, 26,210 train users. Train impressions per video: median 44, p90 369,
    max 9,110. Per user: median 31, p90 97. Valid impressions per user: median 4, p90 12.
A2. Cold start is a user phenomenon, not an item one: 1,990 of 22,377 valid users (8.9%) never appear in train; only
    0.01% of valid rows involve a video unseen in train. Same (user, video) pair repeats across train->valid: 1.6% of rows.
A3. All 7,583 videos have upload_dt in 2022-04-09..11 (dataset construction) -> "video age" is uninformative.
A4. Feedback rates (train): is_click 0.463, long_view 0.337, is_like 0.019, is_profile_enter 0.025, is_comment 0.0026,
    is_follow 0.0010, is_forward 0.0010, is_hate 0.0004. Valid long_view rate 0.313 (drift: daily rate 0.34 -> 0.29
    across the three weeks).
A5. Item-level rates are very stable across time: corr(video train rate, video valid rate | n>=20) = 0.86. User-level
    rates less so: 0.41 (and user-level rates are ranking no-ops anyway, see B3).
A6. Validation user composition: 30.3% all-negative users (nDCG fixed at 0, excluded from GAUC), 11.9% all-positive,
    57.8% mixed. Only the mixed users can be moved.
A7. `tab` (UI scenario) is a strong signal: tab 1 = 73% of rows, long_view rate 0.386; tab 0 = 13%, rate 0.042; tab 4
    rate 0.489; tab 3 rate 0.004. 39.8% of valid users have impressions in more than one tab, so tab varies within user.
A8. Duration: median 70 s (p10 12 s, p90 236 s). long_view rate by duration quintile 0.30/0.36/0.36/0.36/0.33 -> duration
    bias on THIS label is mild (the label definition normalises it). 15.6% of train rows are complete plays
    (play_time >= duration): censoring.
A9. Hour-of-day effect on long_view rate: 0.318..0.376 (min..max over hours).
A10. video_features_statistic_pure.csv = "average daily statistics over one month" (kuairand.com): a window that extends
     beyond the train period -> future information; show_cnt correlates 0.88 with train impression counts, i.e. it is
     mostly popularity. Treat as leak-prone; prefer train-derived counts.
A11. log_random_4_22_to_5_08 (random exposure): 288,338 rows in the valid period, 19,091 users, long_view rate 0.081
     (vs 0.313 on the recommended feed). Different distribution -> useful only as an unbiased check, not as training data.
A12. `is_rand` is 0 on every standard-log row.

## B. Label mechanics (from kuairand.com field definitions, verified on the data)
B1. long_view = 1 iff play_time_ms >= duration_ms when duration_ms <= 18,000, else play_time_ms >= 18,000.
    Verified: no negative has play_time >= 18 s (max 17,999 ms); "play >= 18 s OR complete" reproduces 96.9% of labels
    (the remaining 3% presumably use the feature-file video_duration rather than the log duration_ms).
B2. is_click in the single-column feed = "valid play": play_time >= duration if duration <= 7 s else play_time > 7 s.
    So is_click and long_view are NESTED THRESHOLDS of the same variable (play time): P(long_view | click) = 0.723,
    P(long_view | no click) = 0.003, corr 0.76. An is_click auxiliary head therefore carries almost no information the
    main label does not already carry.
B3. Within-user ranking: any score term constant within a user (user bias, user-only features, user's historical rate)
    cannot change the ranking. Only item-side terms and user x item interactions matter.

## C. Probe results on the real validation split (sealed evaluate.py; scripts probe.py / probe2.py / probe3.py)
C1. Rungs: random 0.4834 (published), item popularity 0.5807, FM champion 0.6015.
C2. Single-feature scorers (train-only, smoothed, prior 20): video train rate 0.5819; global tab rate 0.5399;
    user x tab rate 0.5213; user x author rate 0.4823 (~random: most (user, author) pairs are unseen -> falls back to
    the global mean -> no within-user information).
C3. LightGBM (binary logloss) on past-only rolling rate/count features (user, video, author, user x author, user x tab,
    user x duration-bucket, tab, user x tag, video age, duration, tab, hour): 0.5842 — WORSE than the FM. Feature
    importance was dominated by duration_ms and by user-level features (u_rate, u_n) that are ranking no-ops: a pointwise
    GBDT spends its capacity on calibration across users. Rank-average ensemble FM + this LightGBM: 0.5941 — worse than
    FM alone (a weak member drags the ensemble down).
C4. LightGBM lambdarank grouped by user, same features but with LEAVE-ONE-OUT target encoding for train rows: 0.4699;
    pointwise variant 0.4525 — BELOW RANDOM. Diagnosis: leave-one-out encoding makes the train-row feature anti-correlated
    with its own label (subtracting the row's label), the model learns the inverted relation, validation features are not
    LOO -> inverted predictions. Past-only (time-ordered) encoding must be used, never LOO. This is a trap worth recording.
C5. Multi-task FM (shared embeddings V; extra heads = linear + shared interaction; Adam; early stopping on validation
    long_view primary):  control 0.6015 | + is_click head (w 0.3) 0.6016 | + censored watch-time head (log1p play time,
    one-sided loss on complete plays, w 0.5) 0.6001 | + both 0.6007. The autonomous agent's own is_click aux head:
    0.6010 (-0.0005). Conclusion: naive shared-trunk multi-task is flat-to-negative here; consistent with B2 (nested
    labels) and A4 (the other signals — like/follow/comment/forward — are 0.1–1.9% sparse).
C6. Autonomous-agent results (real runs, same harness): BPR pairwise loss implemented with a sign error -> 0.3948
    (GAUC 0.35 = inverted; loss exploding) — the idea is untested, the implementation was wrong; past-only rolling
    user/author/tab/video rates added as extra FM fields -> 0.6022 (+0.0008, the only positive signal so far);
    K=32 -> 0.6009/0.6022 (flat); LR 0.002 -> 0.49 (diverges); L2 1e-5 -> 0.6025 (+0.0010, within seed noise x1.25);
    patience/epochs tweaks -> flat.
C7. (probe3, pending at draft time — the evaluator should read probe3.out): FM seed ensembles (3 and 5 seeds,
    rank-average and logit-average) and recency-weighted training (tau 5 and 10 days).

## D. Literature (Consensus/arXiv; to be checked for mischaracterisation)
D1. Watch-time prediction literature (D2Q KDD'22, TPM KDD'23, CWM KDD'24, DML CIKM'23, D2Co RecSys'23) is about
    predicting/ debiasing CONTINUOUS watch time as a proxy for interest, with duration as a confounder. Our label is
    already a duration-normalised threshold of watch time (B1), so most of that machinery addresses a bias our metric
    does not have; the transferable ideas are (i) ordinal/quantile decomposition of watch time (TPM, DML) and (ii)
    treating complete plays as censored (CWM).
D2. Multi-task recommendation (MMoE KDD'18, PLE RecSys'20) shows gains when tasks are related-but-different and warns of
    the seesaw effect; ESMM (SIGIR'18) models nested funnels (impression -> click -> conversion) as products of
    conditional probabilities. Our labels are nested thresholds (B2): ESMM-style decomposition p(long_view) =
    p(click) * p(long_view | click) is the principled multi-task form, but with p(long_view|no click)=0.003 the
    decomposition is nearly the identity — little to gain.
D3. Ranking losses: pairwise/listwise objectives are ranking-calibrated for AUC (Uematsu et al. 2017); within-user pairwise
    losses improve GAUC in industry (PDAOM, Meituan); RBP/softmax-style losses are tighter surrogates for DCG (PSL 2024;
    SSM TOIS'22). Expected to help GAUC/nDCG@5 modestly when implemented correctly; needs a sign sanity check.
D4. Long user-behaviour sequence models (DIN, MIMN, SIM, TWIN-V2 at Kuaishou) give large CTR gains in industry with
    long histories. Here histories are short (median 31 train impressions per user) and the catalogue is tiny; the
    equivalent cheap form is user x (author/tag/video) historical statistics as model fields (C6: +0.0008).
D5. Tabular benchmarks (McElfresh 2023, Borisov 2021): GBDTs are strong on tabular data in general — but C3 shows a
    pointwise GBDT on aggregate features underperforms an id-embedding FM for WITHIN-USER ranking, because the useful
    signal is user x item interaction that the FM's embeddings capture and aggregate rates do not.

## E. Proposed direction ranking (what the knowledge file will recommend) — TO BE AFFIRMED OR REFUTED
E1. Variance reduction first: multi-seed FM ensembles (rank-average per user) — near-guaranteed, cheap (pending C7).
E2. History-aware interaction fields inside the FM (or a DeepFM-style trunk): user x author / user x tag / user x
    duration-bucket / user x tab past-only rates and counts, item recency counts — the only direction with a measured
    positive (+0.0008); stack several such fields, then combine with E1.
E3. Ranking loss done right (pairwise within-user, warm-started from the pointwise FM, sign check: train GAUC after
    epoch 1 > 0.55). Organizers' top pick; untested cleanly.
E4. Drift handling: recency weighting / fine-tuning on the last train days (pending C7), item-level rate priors.
E5. Multi-task: only ESMM-style or with genuinely different signals (like/profile_enter) and gated sharing — low expected
    gain per C5; not a first move.
E6. GBDT: only as a candidate ensemble member with past-only features AND the FM score as an input; measured worse alone.
E7. Not worth iterations: capacity (k), static side features (organizers), LR/L2 knob turning, video age, whole-period
    statistic features (leak).

## F. Trap list candidates
F1. Same-row feedback columns as inputs = leakage.  F2. Whole-period aggregates (video_features_statistic) = future leak.
F3. Leave-one-out target encoding inverts the signal (C4): use time-ordered past-only encoding.  F4. GAUC < 0.5 =
inverted implementation.  F5. Exploding loss = bug.  F6. User-constant features are no-ops for within-user ranking but
can dominate a pointwise learner's capacity.  F7. Ensembles with a weak member get worse (C3).  F8. Seed noise 0.0008:
one-run differences < 0.002 are not evidence.

## G. Questions for the evaluator
G1. Is the multi-task conclusion (C5) robust, or an artefact of my minimal implementation (linear aux heads sharing the
    FM interaction term, untuned weights, aux heads' own learning rate 0.01)? If you think a stronger form (e.g. a
    small MLP tower per task over the shared embeddings, or an ESMM product) would change the picture, say so — and if
    you can test it within ~10 minutes of compute, do it.
G2. Is the GBDT conclusion (C3) robust? Would adding the FM score as a feature, or a proper user-grouped lambdarank with
    PAST-ONLY (not LOO) features, plausibly beat the FM?
G3. Are the literature characterisations (D1–D5) fair? Flag any misreading.
G4. Any data fact (A1–A12, B1–B3) you can refute by recomputation?
G5. What high-upside direction is missing from E?
