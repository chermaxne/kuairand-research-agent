# researcher — researcher (model mock:claude-opus-5, 6216 tokens, estimated)

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
machine (published validation primary 0.6016). Promotion needs primary > best + 0.0010; the convergence
streak only resets on an improvement > 0.0020 over the best-so-far. Failed iterations tick the streak.
Prefer changes big enough to matter.

## What you receive each iteration (in this order)
1. STATE BLOCK — current best, budget, streak, BLOCKED list, active themes (all harness-measured).
2. DATA PROFILE — split sizes, positive rate, available columns.
3. CHAMPION CODE — the exact file(s) every experiment must build on.
4. LEDGER — one line per past iteration: hypothesis, change, result, decision, lesson.
5. RECENT ITERATION DETAILS — the last few results with error excerpts.
6. Possibly a STALL RECOVERY DIRECTIVE — if present it overrides the strategy rules below.

## Strategy rules (apply in this order)
1. **Explore structurally new ideas early** (first ~10 iterations): different loss, new signal, new
   model family — not micro-tuning. Read the knowledge library's direction ladder and the organizers'
   own findings (features/capacity alone do NOT help; user-only features are ranking no-ops).
2. **Refine winners mid-run**: once a direction promoted, push it (its next obvious variant) before
   switching. Combine two proven winners when both promoted.
3. **When the flat streak is ≥ 2**, propose the single most reliable promising idea you have —
   something you expect to clear +0.002 — not a long shot.
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
the train-only rule). One experiment = one idea; do not bundle unrelated changes.

## Output contract (strict)
Reply with ONLY one JSON object, no prose, no markdown fences:
{
  "hypothesis": "one sentence: what change and why it should raise primary",
  "category": "feature | model | training | multitask | other",
  "change_spec": "precise numbered instructions for the Engineer",
  "expected_risk": "low | medium | high",
  "builds_on": "champion",
  "rationale": "2-4 sentences citing ledger evidence and the knowledge library"
}

## system block 2

# KNOWLEDGE LIBRARY (domain playbook)

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


## user

# STATE BLOCK
CURRENT BEST: it01 | val primary 0.6025 (GAUC 0.6685 / nDCG5 0.5365) | baseline 0.6016 | margin +0.0009
BUDGET: iteration 3 of 50 | 0:02 of 6:00 elapsed | tokens so far 29887
CONVERGENCE: streak 2 of 3 flat (EPSILON=0.002)
BLOCKED: none
ACTIVE THEMES: winning: training[1 promoted/0 flat/0 failed]; losing/flat: model[0 promoted/1 flat/0 failed]; untried: feature, multitask, other


## Data profile (measured by the harness)
data dir: `/Users/ckwang/Documents/TechJam/kuairand-starter-kit/data_cache/loop_train_valid`

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
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]   # 5 categorical fields of the baseline
K = 16            # embedding dim
LR = 0.001
L2 = 1e-5
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
    """Rows as (date, user_id, video_id, author_id, tab, duration_ms, label); file order preserved."""
    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    rows = []
    for f in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r["date"]), r["user_id"], r["video_id"], vid2author.get(r["video_id"], "UNK"), r["tab"],
                             float(r["duration_ms"]), 1 if r[LABEL] != "0" else 0))
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


# ----------------------------------------------------------------------------- [4] model (= starter_kit/baseline.py FM)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Second-order factorization machine, pointwise logloss, Adam."""

    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


# ----------------------------------------------------------------------------- [5] training
def train(enc, dim, log=print):
    """Train on train, early-stop on validation primary (official recipe). Returns the best model."""
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    m = FM(dim)
    rng = np.random.default_rng(SEED)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        losses = [m.step(Xtr[idx[i:i + BATCH]], ytr[idx[i:i + BATCH]]) for i in range(0, len(idx), BATCH)]
        va = evaluate(uva, yva, m.predict(Xva))
        log(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
            f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return m


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
    t0 = time.time()
    splits = load(a.data)
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}", f"load {time.time() - t0:.1f}s")
    enc, dim = encode(splits)
    model = train(enc, dim)
    X, _, _ = enc[split]
    scores = model.predict(X)
    assert np.all(np.isfinite(scores)), "non-finite scores"
    write_preds(a.out, splits[split], scores)
    print(f"wrote {a.out}: {len(splits[split])} rows for split={split} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()


# LEDGER (full history, oldest first)
# Ledger (tier-1 memory, append-only; one line per iteration, harness-written except LESSON)
# it00 champion installed from baseline_repro: val primary 0.6015 (GAUC 0.6671 / nDCG5 0.5358); published baseline 0.6016; rungs random 0.4827 pop 0.5807
[it01] HYP: Stronger L2 (1e-6 -> 1e-5) to regularise sparse id embeddings under temporal shift | CHANGE: pipeline.py (+1/-1) | RESULT: 0.6025 (best 0.6025) -> PROMOTED | LESSON: Stronger L2 (1e-6 -> 1e-5) to regularise sparse: scored +0.0010 -> promoted
[it02] HYP: Double the FM embedding dimension (K 16 -> 32) to capture richer user x item interactions | CHANGE: pipeline.py (+1/-1) | RESULT: 0.6022 (best 0.6025) -> kept | LESSON: Double the FM embedding dimension (K 16 ->: scored -0.0003 -> kept_champion


# RECENT ITERATION DETAILS
- it01 [training] Stronger L2 (1e-6 -> 1e-5) to regularise sparse id embeddings under temporal shift
  result: scored primary=0.6025027080650933 decision=promoted runtime=45.9s
  lesson: Stronger L2 (1e-6 -> 1e-5) to regularise sparse: scored +0.0010 -> promoted
- it02 [model] Double the FM embedding dimension (K 16 -> 32) to capture richer user x item interactions
  result: scored primary=0.6021714964151774 decision=kept_champion runtime=42.9s
  lesson: Double the FM embedding dimension (K 16 ->: scored -0.0003 -> kept_champion

# TASK
Decide the next experiment now. Consider the STATE BLOCK (streak, budget, BLOCKED), what the ledger
says worked / failed / was never tried, and the strategy rules. Reply with ONLY the JSON object
described in your role instructions (keys: hypothesis, category, change_spec, expected_risk,
builds_on, rationale). The harness will parse it; any other text makes the iteration fail.

## assistant (response)

{"hypothesis": "Raise the learning rate (0.001 -> 0.002) so Adam converges before early stopping triggers", "category": "training", "change_spec": "In pipeline.py apply exactly these line substitutions (leave everything else untouched): [[EDIT]] LR = 0.001 ==> LR = 0.002. Keep the CLI, the train-only rule and the output format. [mock it03]", "expected_risk": "low", "builds_on": "champion", "rationale": "Mock plan step 3: cheaper convergence check. (mock, it03)"}
