# engineer — engineer (model anthropic/claude-sonnet-5, 39728 tokens)

## system block 1

# ROLE: Engineer

You implement one experiment on the champion pipeline of an autonomous ML research agent
(KuaiRand-Pure ranking, label `long_view`, metric GAUC/nDCG@5 within user). You receive the
Researcher's change specification and the exact champion file(s). You output the complete modified
file(s). A deterministic harness then runs `python pipeline.py --data <dir> --split val --out
preds_val.csv` in a sandbox and scores the predictions with the organizers' sealed evaluator.

## Hard rules
1. Implement the change specification faithfully and MINIMALLY. Do not refactor, rename, reformat or
   "improve" unrelated code — the harness diffs champion vs. attempt and judges read that diff.
2. Keep the pipeline contract intact: CLI flags `--data`, `--split val|valid|test`, `--out`; write EVERY
   row of the requested split in data order as `row_id,user_id,video_id,score` (ids echoed exactly as
   read from the CSV, finite float scores); exit 0 on success. `--split test` must keep working.
3. Fit ONLY on the train split (dates 20220408–20220421). Validation rows may be used for early stopping
   / model selection only. Never read labels of the split you are predicting for anything else.
4. No leakage: never use same-row feedback columns (`is_click`, `is_like`, `play_time_ms`, `long_view`, …) as
   input features of the row being scored; any aggregate feature must be computed from strictly earlier
   dates than the row it describes (past-only / rolling), and for validation/test rows only from train.
   When you extend the row tuple or the field list, re-check every index: the label element must never end up
   inside `raw(x)` / the encoded fields. The harness re-runs every would-be promotion on a copy where ~10% of the
   validation users have corrupted labels and checks that those users are still ranked well — a leaked score is
   worth nothing. Consequence for self-checks: print validation sanity numbers, but do NOT hard-assert on the
   validation metric (an `assert primary > 0.55` crashed that re-run once); assert on train-side quantities
   (train GAUC after epoch 1, pair count, vocabulary sizes) instead.
5. Sandbox: no network, no package installs, no subprocesses, no writes outside the working directory.
   Only numpy, pandas, scikit-learn, lightgbm, torch (CPU) and the standard library are available.
   Keep memory moderate (16 GB box) and respect the runtime limit stated in the contract.
6. Never print fake metrics or mock results. Never skip work with hardcoded outputs.
7. Keep determinism: fixed seeds, no time-dependent randomness.
8. Runtime is a hard budget (the kill limit is stated in the pipeline contract; aim for well under half of it). Vectorise every per-user operation with pandas
   `groupby(...).rank/transform` or numpy `np.unique(..., return_inverse)` + `np.argsort` / `np.add.at`. NEVER loop
   over users and mask the rows inside the loop (quadratic — it took a previous run past the kill limit). Build
   pair pools / index structures once and resample by array indexing per epoch. Print per-epoch timing.
9. In-run attribution, under an explicit time budget. `KUAIRAND_TIME_BUDGET_S` is in the environment and the process
   is killed at it. Structure every pipeline as: (a) fit the FULL bundle and write its predictions — this must finish
   inside 40% of the budget; (b) then, for each variant in the ABLATION PLAN, check the clock and start it only if at
   least 25% of the budget remains, otherwise print `ABLATION <name> skipped: out of time budget`; (c) score each
   variant you did run with the official `evaluate()` and print exactly one line
   `ABLATION <name> primary=<float> gauc=<float> ndcg5=<float>` — real numbers from real fits, never estimated.
   Ablation variants are DIAGNOSTICS: make them cheap (subsample the training rows, cap the rounds/epochs, one seed).
   A variant must never cost as much as the full fit. The written predictions are always the full bundle.
9b. Do not score the validation metric on a tight loop. `evaluate()` runs over ~125k rows; call it for early stopping
   at most every ~50 boosting rounds or once per epoch. Scoring every 10 rounds turned a 30-second LightGBM fit into
   6 minutes and blew the whole budget on one iteration (run ten16). Print per-fit timing so the cost is visible.
10. Fast path: when the environment variable `KUAIRAND_FAST=1` is set (the harness sets it for its flipped-label
   re-run), skip in-run ablations, sweeps and extra seeds (one seed, the full bundle only) — but NEVER change which
   features or labels are used; the feature/label code path must be identical with and without the flag.
11. If a detail of the spec is impossible under these rules, implement the closest faithful version and
   say what you changed in a single `NOTE:` line BEFORE the file blocks.

## Output format (strict)
For every file you change or add, output the COMPLETE file (not a diff, not a snippet):

=== FILE: pipeline.py ===
```python
<entire file content>
```
=== END FILE ===

Nothing else except an optional leading `NOTE:` line. The harness discards anything it cannot parse.

## user

# Change specification (from the Researcher)
HYPOTHESIS: Fine-tuning the current DeepFM+numeric+session champion with a warm-started, staged within-user BPR pairwise loss (pointwise pretrain to convergence as now, then continue optimizing purely on sampled within-user positive/negative pairs, keeping the best-of-either checkpoint) will push GAUC/nDCG@5 further because it directly targets the ranking objective without the hybrid-loss interference that hurt a prior attempt.
CATEGORY: training
EXPECTED RISK: medium
EXPECTED GAIN (Researcher's prediction): 0.0024
ABLATION PLAN (variants to also score and print as ABLATION lines): champion_equiv: current pointwise-only DeepFM+numeric+session (no BPR stage), subsampled as today; bpr_finetune_1seed: the pretrain+BPR-finetune pipeline at seed=0 only (no ensembling), scored on full validation to isolate the loss effect from the ensembling effect; full: the 3-seed-averaged bpr_finetune bundle (the proposed champion).
CHANGE SPEC:
1. Keep encode()/build_numeric()/FIELDS/M_NUM exactly as in the current champion (7 categorical fields incl. hour/session buckets, 11 numeric priors incl. user-tab rate and log1p counts) -- no feature changes. 2. Add a new training stage in train_deepfm-style flow: (a) train the full DeepFM exactly as today (pointwise BCE) to its early-stopped best state -- call this the pretrain stage, unchanged code path, tag='pretrain'. (b) Build pair pools ONCE (not per epoch): for the TRAIN split only, group row indices by user_id; for every user with >=1 positive AND >=1 negative row in train, store pos_idx array and neg_idx array (numpy int arrays). Concatenate all users' arrays into two big index arrays pos_all, neg_all via vectorized construction (no per-user Python loop over rows -- only a loop over the ~26k distinct users to fill lists, then np.concatenate once; this is a one-time O(n_users) build, not O(rows) per epoch). (c) Each fine-tune epoch: resample neg_all by drawing a fresh random negative index from the SAME user's neg-index array for every pos row (vectorized via array indexing with random column selection per user, or oversample pos_all K=2x by repeating and drawing random negatives) -- must be pure numpy indexing, no Python row loop. (d) New method DeepFM.step_pairwise(Xpos,Zpos,Xneg,Zneg): compute z_pos,*_ = self.logits(Xpos,Zpos) and z_neg,*_ = self.logits(Xneg,Zneg) in one forward each (reuse existing logits()); let d = sigmoid(z_neg - z_pos) (vector over the pair batch); this is dL/dz_pos = -d and dL/dz_neg = +d for L=-log(sigmoid(z_pos-z_neg)). Build a combined batch of concatenated pos and neg rows (X_cat=[Xpos;Xneg], Z_cat=[Zpos;Zneg]) with per-row gradient g_cat=[-d/B; +d/B], and call the EXACT same accumulation/backward code path used in step() (gV/gW/MLP backward) but substitute g_cat for the BCE gradient g -- do not rederive the FM math, just replace the gradient vector's source terms in a refactored helper shared by step() and step_pairwise(). Same Adam state, LR=0.001, L2=1e-6. 3. Fine-tune schedule: start from the pretrain best_state (warm start), run up to 8 epochs of step_pairwise over resampled pairs, batch=8192 pairs, evaluate validation primary via evaluate() each epoch exactly as now, keep the OVERALL best state across BOTH pretrain and fine-tune epochs (patience=3 on the fine-tune stage; if fine-tune never beats the pretrain best, keep the pretrain model -- this makes the stage strictly non-regressive by construction). Print pair_count and per-epoch primary as today's logging format. 4. Riders: after selecting the best single model, run this same pretrain+fine-tune pipeline for seeds 0,1,2 and average their validation/target predictions (rank-safe score average, 3-seed ensemble) as the FULL bundle written to preds.csv -- this is the validated variance-reduction rider from prior runs. 5. Ablations to print: 'champion_equiv' = today's pointwise-only DeepFM+numeric+session, single seed, subsampled as currently coded (already exists, keep it); 'bpr_finetune_1seed' = the pretrain+BPR-finetune pipeline with seed=0 only, no ensembling, scored on FULL validation (not subsampled) so it is directly comparable to 'full'; 'full' = the 3-seed-averaged bundle. 6. Runtime: pretrain ~45-55s/seed (unchanged), fine-tune ~8 epochs * ~1.5-2s pair batches ~15s/seed, so ~70s/seed * 3 seeds ~210s, plus champion_equiv/no_confidence_counts/no_session_fields ablations at reduced rows (~60s total) -- total well under the 1500s wall-clock budget. 7. Leave CLI, output format, train-only rule, and past-only feature computation unchanged; do NOT use any same-row feedback column in step_pairwise beyond the label used to build the pos/neg index split (label used only to partition indices before training, never as an input feature).

# Current champion files
=== FILE: pipeline.py ===
```python
"""DeepFM upgrade over the FM champion: adds a 1-hidden-layer MLP over the concatenated field
embeddings plus standardized past-only numerical priors (user/video/author long_view & click
rates + user x tab long_view rate) AND past-only session/time-context categorical fields
(hour-of-day bucket, within-day session depth bucket) plus log1p exposure-count confidence
features for user/video/author/user_tab. Implements DeepFM (Guo et al. 2017) sum of FM 2nd-order
term and an MLP branch, trained pointwise BCE with the same Adam-style optimizer as the FM
baseline.

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
# 7 categorical fields: 5 baseline + hour-of-day bucket + within-day session-depth bucket (past-only, no label)
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket", "hour_bucket", "session_depth_bucket"]
K = 16            # embedding dim
LR = 0.001
L2 = 1e-6
EPOCHS = 40
BATCH = 8192
PATIENCE = 4
SEED = 0
N_DUR_BUCKETS = 10
M_NUM = 11        # numerical prior features: 7 rate features + 4 log1p exposure-count features
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
    """Rows as (date, user_id, video_id, author_id, tab, duration_ms, label, is_click, hourmin, time_ms);
    file order preserved. hourmin/time_ms are past-only clock fields (no label/feedback), used only for
    deriving hour-of-day and within-day session-depth buckets."""
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
                             1 if r.get("is_click", "0") != "0" else 0,
                             float(r["hourmin"]), float(r["time_ms"])))
    return {name: [x for x in rows if lo <= x[0] <= hi] for name, (lo, hi) in SPLITS.items()}


# ----------------------------------------------------------------------------- [3] feature encoding (= starter_kit/data.py)
def _bucket_edges(durations, n=N_DUR_BUCKETS):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


def encode(splits):
    """Categorical ids -> contiguous ints; unseen values fall into a per-field UNK slot.
    Returns ({split: (X int32 (N,F), y float32, users)}, total_dim)."""
    tr = splits["train"]
    edges = _bucket_edges([x[5] for x in tr])

    # ---- past-only session/time-context fields, computed independently per split (never uses label/is_click) ----
    extra_hour = {}
    extra_sess = {}
    for name, rws in splits.items():
        users = np.array([x[1] for x in rws])
        dates = np.array([x[0] for x in rws])
        tms = np.array([x[9] for x in rws], dtype=np.float64)
        dfx = pd.DataFrame({"user": users, "date": dates, "time_ms": tms})
        rank = dfx.groupby(["user", "date"])["time_ms"].rank(method="first").values
        depth = np.minimum(rank, 7).astype(int)
        sess = np.where(depth >= 7, "7+", depth.astype(str))
        hour = np.array([str(int(x[8] // 400)) for x in rws])
        extra_hour[name] = hour
        extra_sess[name] = sess

    def raw(x, hb, sb):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5]))), hb, sb]

    vocabs = [dict() for _ in FIELDS]
    hb_tr, sb_tr = extra_hour["train"], extra_sess["train"]
    for idx, x in enumerate(tr):
        for i, v in enumerate(raw(x, hb_tr[idx], sb_tr[idx])):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    print(f"vocab sizes: hour_bucket={len(vocabs[5])} session_depth_bucket={len(vocabs[6])}")
    enc = {}
    for name, rws in splits.items():
        hb, sb = extra_hour[name], extra_sess[name]
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x, hb[n], sb[n])):
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
    """Returns (Zstd dict of {split: (N,11) float32 standardized array}, stats dict)."""
    df_tr = pd.DataFrame([r[:8] for r in splits["train"]], columns=_COLS)
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
            out[name] = (lv.astype(np.float32), ck.astype(np.float32), n.astype(np.float32))
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
            out[name] = (lv.astype(np.float32), ck.astype(np.float32), cnt.astype(np.float32))
        return out

    def stack(r):
        base = np.stack([
            r["user"][0], r["user"][1],
            r["video"][0], r["video"][1],
            r["author"][0], r["author"][1],
            r["user_tab"][0],
        ], axis=1)
        counts = np.stack([
            np.log1p(r["user"][2]), np.log1p(r["video"][2]), np.log1p(r["author"][2]), np.log1p(r["user_tab"][2]),
        ], axis=1)
        return np.concatenate([base, counts], axis=1)

    Zraw = {"train": stack(train_rates(df_tr))}
    for name in ("valid", "test"):
        df = pd.DataFrame([r[:8] for r in splits[name]], columns=_COLS)
        Zraw[name] = stack(eval_rates(df))

    log(f"sample log1p counts (pre-std, cols=user/video/author/user_tab): "
        f"{np.round(Zraw['train'][:5, 7:11], 4).tolist()}")

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
    With use_mlp=False this is byte-for-byte the original FM baseline (champion_equiv).
    n_fields lets callers run ablations with a subset of the FIELDS columns of X (defaults to len(FIELDS))."""

    def __init__(self, dim, use_mlp=False, k=K, lr=LR, l2=L2, seed=SEED, m=M_NUM, hidden=HIDDEN, n_fields=None):
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
        self.n_fields = n_fields if n_fields is not None else len(FIELDS)
        if use_mlp:
            rng2 = np.random.default_rng(seed + 1)
            d_in = self.n_fields * k + m
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
            F_ = self.n_fields; Kk = self.k
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
                  epochs=EPOCHS, patience=PATIENCE, seed=SEED, log=print, tag="",
                  m=M_NUM, n_fields=None):
    m_model = DeepFM(dim, use_mlp=use_mlp, seed=seed, m=m, n_fields=n_fields)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        losses = []
        for i in range(0, len(idx), BATCH):
            bidx = idx[i:i + BATCH]
            zb = Ztr[bidx] if Ztr is not None else None
            losses.append(m_model.step(Xtr[bidx], ytr[bidx], zb))
        va = evaluate(uva, yva, m_model.predict(Xva, Zva))
        log(f"[{tag}] epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
            f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = m_model.state()
            best_metrics = va
        else:
            bad += 1
            if bad >= patience:
                log(f"[{tag}] early stop at epoch {ep}")
                break
    m_model.load_state(best_state)
    return m_model, best_metrics


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

    # ---- FULL bundle first: DeepFM (MLP) + 7 fields + 11 numeric priors. This is the proposed champion. ----
    model_full, va_full = train_deepfm(
        Xtr, ytr, uva, Xva, yva, dim, use_mlp=True,
        Ztr=Zdict["train"], Zva=Zdict["valid"],
        epochs=EPOCHS, patience=PATIENCE, seed=SEED, tag="full",
        m=M_NUM, n_fields=len(FIELDS),
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
        print("KUAIRAND_FAST=1: skipping in-run ablations.")
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

    # (a) champion_equiv: original 5 fields, no MLP, no numeric priors at all.
    elapsed = time.time() - t0
    if elapsed < 0.75 * budget:
        model_ce, va_ce = train_deepfm(
            Xtr_sub[:, :5], ytr_sub, uva, Xva[:, :5], yva, dim, use_mlp=False,
            Ztr=None, Zva=None,
            epochs=ABLATION_EPOCHS, patience=ABLATION_PATIENCE, seed=SEED, tag="champion_equiv",
            m=M_NUM, n_fields=5,
        )
        print(f"ABLATION champion_equiv primary={va_ce['primary']:.4f} gauc={va_ce['GAUC']:.4f} ndcg5={va_ce['nDCG@5']:.4f}")
    else:
        print("ABLATION champion_equiv skipped: out of time budget")

    # (b) no_confidence_counts: 7 fields (with hour/session) + only the original 7 numeric rate features.
    elapsed = time.time() - t0
    if elapsed < 0.75 * budget:
        model_ncc, va_ncc = train_deepfm(
            Xtr_sub, ytr_sub, uva, Xva, yva, dim, use_mlp=True,
            Ztr=Ztr_sub[:, :7], Zva=Zdict["valid"][:, :7],
            epochs=ABLATION_EPOCHS, patience=ABLATION_PATIENCE, seed=SEED, tag="no_confidence_counts",
            m=7, n_fields=len(FIELDS),
        )
        print(f"ABLATION no_confidence_counts primary={va_ncc['primary']:.4f} gauc={va_ncc['GAUC']:.4f} ndcg5={va_ncc['nDCG@5']:.4f}")
    else:
        print("ABLATION no_confidence_counts skipped: out of time budget")

    # (c) no_session_fields: 5 fields (no hour/session) + all 11 numeric (with counts).
    elapsed = time.time() - t0
    if elapsed < 0.75 * budget:
        model_nsf, va_nsf = train_deepfm(
            Xtr_sub[:, :5], ytr_sub, uva, Xva[:, :5], yva, dim, use_mlp=True,
            Ztr=Ztr_sub, Zva=Zdict["valid"],
            epochs=ABLATION_EPOCHS, patience=ABLATION_PATIENCE, seed=SEED, tag="no_session_fields",
            m=M_NUM, n_fields=5,
        )
        print(f"ABLATION no_session_fields primary={va_nsf['primary']:.4f} gauc={va_nsf['GAUC']:.4f} ndcg5={va_nsf['nDCG@5']:.4f}")
    else:
        print("ABLATION no_session_fields skipped: out of time budget")


if __name__ == "__main__":
    main()
```
=== END FILE ===

# Pipeline contract
`python pipeline.py --data <data_dir> --split val --out preds_val.csv`
- Train ONLY on the train split (dates 20220408-20220421). Validation rows may be used for early stopping / model selection only.
- Write EVERY validation row, in data.load() order, as `row_id,user_id,video_id,score` (row_id from 0, ids echoed exactly as read, finite scores).
- `--split test` must keep working unchanged (it is used once, at finalize, on the champion).
- Exit 0 on success. Single process, no network, no package installs, only pre-installed libraries
  (numpy, pandas, scikit-learn, lightgbm, torch-cpu). Same-row feedback columns are NOT features (leakage).
- Hard wall-clock limit: 1500s for the whole run (load + train + predict).
- TIME BUDGET (hard): `KUAIRAND_TIME_BUDGET_S` is in the environment (1500s here) and the process is killed at it.
  Budget it explicitly with arithmetic, not hope:
  1. Fit the FULL bundle first and WRITE ITS PREDICTIONS before anything else. It must finish inside 40% of the budget.
  2. Only then run ablations, and before each one check `time.time() - t0` against the budget: start a variant only if
     at least 25% of the budget remains, else print `ABLATION <name> skipped: <reason>` and move on.
  3. Ablation variants are DIAGNOSTICS, not submissions: make them cheap (a subsample of the training rows, or a fixed
     small number of rounds/epochs, or one seed). A variant must never cost as much as the full fit.
- In-run attribution: for each variant you do run, score it on validation with the official `evaluate()` and print one
  line `ABLATION <name> primary=<f> gauc=<f> ndcg5=<f>` (real numbers from real fits only). The written predictions are
  always the full bundle.
- Evaluating the metric is expensive (~125k rows). For early stopping, score at most every ~50 boosting rounds or once
  per epoch — NOT every few rounds. Scoring after every 10 rounds turned a 30 s fit into 6 minutes in run ten16.
- If this change replaces or adds a LOSS FUNCTION (e.g. pointwise -> pairwise/BPR, adding an auxiliary
  head): the learning rate and any other optimizer constants were tuned for the OLD objective's gradient
  scale and are not guaranteed to transfer. Reusing them unchanged is a common cause of loss divergence
  (loss climbing epoch over epoch instead of falling). If the change_spec does not already address this,
  pick a conservative LR for the new objective (or add gradient clipping) rather than inheriting the old
  value silently.

# TASK
Implement the change specification above on the champion files. Output the COMPLETE modified file(s)
in the `=== FILE: name === ... === END FILE ===` format described in your role instructions.
Minimal targeted edits only; keep the CLI contract, the train-only rule and the output format intact.

## assistant (response)

NOTE: Kept the three pre-existing subsampled ablations (champion_equiv, no_confidence_counts, no_session_fields) unchanged in addition to the two new ones required by the ABLATION PLAN (bpr_finetune_1seed, full), since the spec's own runtime section references keeping them and rule 1 forbids removing unrelated existing code.

=== FILE: pipeline.py ===
```python
"""DeepFM upgrade over the FM champion: adds a 1-hidden-layer MLP over the concatenated field
embeddings plus standardized past-only numerical priors (user/video/author long_view & click
rates + user x tab long_view rate) AND past-only session/time-context categorical fields
(hour-of-day bucket, within-day session depth bucket) plus log1p exposure-count confidence
features for user/video/author/user_tab. Implements DeepFM (Guo et al. 2017) sum of FM 2nd-order
term and an MLP branch, trained pointwise BCE with the same Adam-style optimizer as the FM
baseline.

NEW (this experiment): after the pointwise pretrain converges (early-stopped, unchanged), the
model is warm-started into a staged within-user BPR pairwise fine-tune stage that directly
optimizes pairwise ranking on sampled (pos,neg) pairs drawn from the SAME user, using the same
Adam optimizer state and a shared FM/MLP backward helper. The overall best checkpoint (pretrain
OR any fine-tune epoch) is kept, so the stage is non-regressive by construction. The FULL bundle
averages 3 seeds of this pretrain+finetune pipeline (rank-safe score average).

    python pipeline.py --data <data_dir> --split val|test --out preds.csv

Section map:  [1] config  [2] data loading  [3] feature encoding  [3b] numeric priors (past-only)
              [4] model (+pairwise step)  [5] training (pretrain + BPR fine-tune)  [6] CLI / ablations
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
# 7 categorical fields: 5 baseline + hour-of-day bucket + within-day session-depth bucket (past-only, no label)
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket", "hour_bucket", "session_depth_bucket"]
K = 16            # embedding dim
LR = 0.001
L2 = 1e-6
EPOCHS = 40
BATCH = 8192
PATIENCE = 4
SEED = 0
N_DUR_BUCKETS = 10
M_NUM = 11        # numerical prior features: 7 rate features + 4 log1p exposure-count features
HIDDEN = 128      # MLP hidden width

ABLATION_EPOCHS = 8
ABLATION_PATIENCE = 3
ABLATION_MAX_ROWS = 300_000

# BPR fine-tune stage config (change spec section 2/3)
BPR_EPOCHS = 8
BPR_PATIENCE = 3
BPR_BATCH = 8192
FULL_SEEDS = [0, 1, 2]


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
    """Rows as (date, user_id, video_id, author_id, tab, duration_ms, label, is_click, hourmin, time_ms);
    file order preserved. hourmin/time_ms are past-only clock fields (no label/feedback), used only for
    deriving hour-of-day and within-day session-depth buckets."""
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
                             1 if r.get("is_click", "0") != "0" else 0,
                             float(r["hourmin"]), float(r["time_ms"])))
    return {name: [x for x in rows if lo <= x[0] <= hi] for name, (lo, hi) in SPLITS.items()}


# ----------------------------------------------------------------------------- [3] feature encoding (= starter_kit/data.py)
def _bucket_edges(durations, n=N_DUR_BUCKETS):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


def encode(splits):
    """Categorical ids -> contiguous ints; unseen values fall into a per-field UNK slot.
    Returns ({split: (X int32 (N,F), y float32, users)}, total_dim)."""
    tr = splits["train"]
    edges = _bucket_edges([x[5] for x in tr])

    # ---- past-only session/time-context fields, computed independently per split (never uses label/is_click) ----
    extra_hour = {}
    extra_sess = {}
    for name, rws in splits.items():
        users = np.array([x[1] for x in rws])
        dates = np.array([x[0] for x in rws])
        tms = np.array([x[9] for x in rws], dtype=np.float64)
        dfx = pd.DataFrame({"user": users, "date": dates, "time_ms": tms})
        rank = dfx.groupby(["user", "date"])["time_ms"].rank(method="first").values
        depth = np.minimum(rank, 7).astype(int)
        sess = np.where(depth >= 7, "7+", depth.astype(str))
        hour = np.array([str(int(x[8] // 400)) for x in rws])
        extra_hour[name] = hour
        extra_sess[name] = sess

    def raw(x, hb, sb):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5]))), hb, sb]

    vocabs = [dict() for _ in FIELDS]
    hb_tr, sb_tr = extra_hour["train"], extra_sess["train"]
    for idx, x in enumerate(tr):
        for i, v in enumerate(raw(x, hb_tr[idx], sb_tr[idx])):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    print(f"vocab sizes: hour_bucket={len(vocabs[5])} session_depth_bucket={len(vocabs[6])}")
    enc = {}
    for name, rws in splits.items():
        hb, sb = extra_hour[name], extra_sess[name]
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x, hb[n], sb[n])):
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
    """Returns (Zstd dict of {split: (N,11) float32 standardized array}, stats dict)."""
    df_tr = pd.DataFrame([r[:8] for r in splits["train"]], columns=_COLS)
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
            out[name] = (lv.astype(np.float32), ck.astype(np.float32), n.astype(np.float32))
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
            out[name] = (lv.astype(np.float32), ck.astype(np.float32), cnt.astype(np.float32))
        return out

    def stack(r):
        base = np.stack([
            r["user"][0], r["user"][1],
            r["video"][0], r["video"][1],
            r["author"][0], r["author"][1],
            r["user_tab"][0],
        ], axis=1)
        counts = np.stack([
            np.log1p(r["user"][2]), np.log1p(r["video"][2]), np.log1p(r["author"][2]), np.log1p(r["user_tab"][2]),
        ], axis=1)
        return np.concatenate([base, counts], axis=1)

    Zraw = {"train": stack(train_rates(df_tr))}
    for name in ("valid", "test"):
        df = pd.DataFrame([r[:8] for r in splits[name]], columns=_COLS)
        Zraw[name] = stack(eval_rates(df))

    log(f"sample log1p counts (pre-std, cols=user/video/author/user_tab): "
        f"{np.round(Zraw['train'][:5, 7:11], 4).tolist()}")

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
    With use_mlp=False this is byte-for-byte the original FM baseline (champion_equiv).
    n_fields lets callers run ablations with a subset of the FIELDS columns of X (defaults to len(FIELDS))."""

    def __init__(self, dim, use_mlp=False, k=K, lr=LR, l2=L2, seed=SEED, m=M_NUM, hidden=HIDDEN, n_fields=None):
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
        self.n_fields = n_fields if n_fields is not None else len(FIELDS)
        if use_mlp:
            rng2 = np.random.default_rng(seed + 1)
            d_in = self.n_fields * k + m
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

    def _backward_and_update(self, X, E, S, H0, h1, g):
        """Shared accumulation/backward/Adam-update code path. `g` is the per-row gradient of the
        loss w.r.t. the row's logit (dL/dz), already divided by the batch size used for averaging.
        Used identically by step() (pointwise BCE gradient) and step_pairwise() (BPR gradient)."""
        B = len(g)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))

        dW1 = dW2 = db1 = db2 = None
        if self.use_mlp:
            F_ = self.n_fields; Kk = self.k
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

    def step(self, X, y, Z=None):
        B = len(y)
        z, E, S, H0, h1 = self.logits(X, Z)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        self._backward_and_update(X, E, S, H0, h1, g)
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def step_pairwise(self, Xpos, Zpos, Xneg, Zneg):
        """BPR pairwise update: L = -log(sigmoid(z_pos - z_neg)); dL/dz_pos = -d, dL/dz_neg = +d
        where d = sigmoid(z_neg - z_pos). Reuses logits() and the shared backward/update helper by
        building a combined batch [Xpos;Xneg] with per-row gradient [-d/B_total; +d/B_total]."""
        Bp = len(Xpos)
        z_pos, Epos, Spos, H0pos, h1pos = self.logits(Xpos, Zpos)
        z_neg, Eneg, Sneg, H0neg, h1neg = self.logits(Xneg, Zneg)
        d = sigmoid(z_neg - z_pos).astype(np.float32)
        B_total = 2 * Bp
        g_pos = (-d / B_total).astype(np.float32)
        g_neg = (d / B_total).astype(np.float32)

        X_cat = np.concatenate([Xpos, Xneg], axis=0)
        g_cat = np.concatenate([g_pos, g_neg], axis=0)
        E_cat = np.concatenate([Epos, Eneg], axis=0)
        S_cat = np.concatenate([Spos, Sneg], axis=0)
        if self.use_mlp:
            H0_cat = np.concatenate([H0pos, H0neg], axis=0)
            h1_cat = np.concatenate([h1pos, h1neg], axis=0)
        else:
            H0_cat, h1_cat = None, None

        self._backward_and_update(X_cat, E_cat, S_cat, H0_cat, h1_cat, g_cat)
        return float(-np.mean(np.log(sigmoid(z_pos - z_neg) + 1e-9)))

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
                  epochs=EPOCHS, patience=PATIENCE, seed=SEED, log=print, tag="",
                  m=M_NUM, n_fields=None):
    m_model = DeepFM(dim, use_mlp=use_mlp, seed=seed, m=m, n_fields=n_fields)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        losses = []
        for i in range(0, len(idx), BATCH):
            bidx = idx[i:i + BATCH]
            zb = Ztr[bidx] if Ztr is not None else None
            losses.append(m_model.step(Xtr[bidx], ytr[bidx], zb))
        va = evaluate(uva, yva, m_model.predict(Xva, Zva))
        log(f"[{tag}] epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
            f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = m_model.state()
            best_metrics = va
        else:
            bad += 1
            if bad >= patience:
                log(f"[{tag}] early stop at epoch {ep}")
                break
    m_model.load_state(best_state)
    return m_model, best_metrics


def build_pair_pools(Xtr, ytr, users_tr):
    """One-time (O(n_users), not O(rows)) construction of within-user positive/negative index
    pools for the TRAIN split only. Returns:
      pos_all             : (P,) int64 indices into Xtr/ytr/Ztr of positive rows (all eligible users concatenated)
      neg_flat            : (Q,) int64 indices into Xtr/ytr/Ztr of the negative rows of those same users
      neg_offset_per_pos  : (P,) int64 start offset into neg_flat of the owning user's negative slice
      neg_len_per_pos     : (P,) int64 length of the owning user's negative slice
    Only users with >=1 positive AND >=1 negative row in train are included."""
    users_arr = np.asarray(users_tr)
    uniq_users, inv = np.unique(users_arr, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    boundaries = np.searchsorted(inv_sorted, np.arange(len(uniq_users) + 1))

    pos_chunks, neg_flat_chunks = [], []
    offset_chunks, len_chunks = [], []
    cur_neg_offset = 0
    for u in range(len(uniq_users)):        # loop over ~26k distinct users, not over rows
        s, e = boundaries[u], boundaries[u + 1]
        if e - s < 2:
            continue
        ii = order[s:e]
        yy = ytr[ii]
        pos_idx = ii[yy == 1]
        neg_idx = ii[yy == 0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        pos_chunks.append(pos_idx)
        neg_flat_chunks.append(neg_idx)
        offset_chunks.append(np.full(len(pos_idx), cur_neg_offset, dtype=np.int64))
        len_chunks.append(np.full(len(pos_idx), len(neg_idx), dtype=np.int64))
        cur_neg_offset += len(neg_idx)

    pos_all = np.concatenate(pos_chunks).astype(np.int64)
    neg_flat = np.concatenate(neg_flat_chunks).astype(np.int64)
    neg_offset_per_pos = np.concatenate(offset_chunks)
    neg_len_per_pos = np.concatenate(len_chunks)
    return pos_all, neg_flat, neg_offset_per_pos, neg_len_per_pos


def run_pretrain_and_bpr(Xtr, ytr, uva, Xva, yva, dim, Ztr, Zva,
                          pos_all, neg_flat, neg_offset_per_pos, neg_len_per_pos,
                          seed, epochs_pre=EPOCHS, patience_pre=PATIENCE,
                          epochs_ft=BPR_EPOCHS, patience_ft=BPR_PATIENCE, log=print, tag_prefix=""):
    """(a) pointwise pretrain to early-stopped best (unchanged code path). (b)/(c)/(d) warm-started
    within-user BPR fine-tune for up to `epochs_ft` epochs with fresh per-epoch negative resampling
    (pure numpy indexing, no python row loop). Keeps the OVERALL best state across BOTH stages."""
    model, va_pre = train_deepfm(
        Xtr, ytr, uva, Xva, yva, dim, use_mlp=True, Ztr=Ztr, Zva=Zva,
        epochs=epochs_pre, patience=patience_pre, seed=seed, tag=f"{tag_prefix}pretrain",
        m=M_NUM, n_fields=len(FIELDS),
    )
    best = va_pre["primary"]
    best_state = model.state()
    best_metrics = va_pre

    rng = np.random.default_rng(seed + 777)
    n_pos = len(pos_all)
    bad = 0
    for ep in range(1, epochs_ft + 1):
        t0 = time.time()
        neg_local = (rng.random(n_pos) * neg_len_per_pos).astype(np.int64)
        neg_idx_ep = neg_flat[neg_offset_per_pos + neg_local]
        perm = rng.permutation(n_pos)
        pos_ep = pos_all[perm]
        neg_ep = neg_idx_ep[perm]
        losses = []
        for i in range(0, n_pos, BPR_BATCH):
            pidx = pos_ep[i:i + BPR_BATCH]
            nidx = neg_ep[i:i + BPR_BATCH]
            Zpos = Ztr[pidx] if Ztr is not None else None
            Zneg = Ztr[nidx] if Ztr is not None else None
            losses.append(model.step_pairwise(Xtr[pidx], Zpos, Xtr[nidx], Zneg))
        va = evaluate(uva, yva, model.predict(Xva, Zva))
        log(f"[{tag_prefix}bpr] epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
            f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = model.state()
            best_metrics = va
        else:
            bad += 1
            if bad >= patience_ft:
                log(f"[{tag_prefix}bpr] early stop at epoch {ep}")
                break
    model.load_state(best_state)
    return model, best_metrics


def rank_avg_combine(score_list):
    """Rank-safe ensembling: monotonic rank-transform each seed's scores to [0,1] before
    averaging, so per-seed scale differences don't dominate the combination. Order-preserving
    per user since the transform is a global monotonic function of each seed's raw score."""
    n = len(score_list[0])
    acc = np.zeros(n, dtype=np.float64)
    for s in score_list:
        r = pd.Series(s).rank(method="average").values
        acc += r / n
    return (acc / len(score_list)).astype(np.float32)


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

    Xtr, ytr, users_tr = enc["train"]
    Xva, yva, uva = enc["valid"]

    # ---- build within-user pos/neg pair pools ONCE for the BPR fine-tune stage (train split only) ----
    pos_all, neg_flat, neg_offset_per_pos, neg_len_per_pos = build_pair_pools(Xtr, ytr, users_tr)
    pair_count = len(pos_all)
    assert pair_count > 0, "no eligible (pos,neg) pairs found in train"
    print(f"pair_count={pair_count} (eligible users with >=1 pos & >=1 neg train row)")

    Xtarget, _, _ = enc[split]
    Ztarget = Zdict[split]

    # ---- FULL bundle: pretrain(pointwise BCE) -> warm-started within-user BPR fine-tune, averaged over 3 seeds ----
    seed_metrics = {}
    val_preds_per_seed = []
    target_preds_per_seed = []
    for sd in FULL_SEEDS:
        elapsed = time.time() - t0
        if elapsed > 0.4 * budget and len(val_preds_per_seed) > 0:
            print(f"seed {sd} skipped: full bundle already inside 40% time budget checkpoint ({elapsed:.0f}s)")
            break
        model_bpr, va_bpr = run_pretrain_and_bpr(
            Xtr, ytr, uva, Xva, yva, dim, Zdict["train"], Zdict["valid"],
            pos_all, neg_flat, neg_offset_per_pos, neg_len_per_pos,
            seed=sd, tag_prefix=f"s{sd}_",
        )
        seed_metrics[sd] = va_bpr
        val_preds_per_seed.append(model_bpr.predict(Xva, Zdict["valid"]))
        target_preds_per_seed.append(model_bpr.predict(Xtarget, Ztarget))
        print(f"seed {sd} done at {time.time() - t0:.0f}s (budget {budget:.0f}s)")

    scores = rank_avg_combine(target_preds_per_seed)
    assert np.all(np.isfinite(scores)), "non-finite scores"
    write_preds(a.out, splits[split], scores)
    print(f"wrote {a.out}: {len(splits[split])} rows for split={split} in {time.time() - t0:.0f}s")

    va_full = evaluate(uva, yva, rank_avg_combine(val_preds_per_seed))
    print(f"ABLATION full primary={va_full['primary']:.4f} gauc={va_full['GAUC']:.4f} ndcg5={va_full['nDCG@5']:.4f}")

    if 0 in seed_metrics:
        va1 = seed_metrics[0]
        print(f"ABLATION bpr_finetune_1seed primary={va1['primary']:.4f} gauc={va1['GAUC']:.4f} ndcg5={va1['nDCG@5']:.4f}")
    else:
        print("ABLATION bpr_finetune_1seed skipped: out of time budget")

    if fast:
        print("KUAIRAND_FAST=1: skipping in-run ablations.")
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

    # (a) champion_equiv: original 5 fields, no MLP, no numeric priors at all, pointwise-only (no BPR stage).
    elapsed = time.time() - t0
    if elapsed < 0.75 * budget:
        model_ce, va_ce = train_deepfm(
            Xtr_sub[:, :5], ytr_sub, uva, Xva[:, :5], yva, dim, use_mlp=False,
            Ztr=None, Zva=None,
            epochs=ABLATION_EPOCHS, patience=ABLATION_PATIENCE, seed=SEED, tag="champion_equiv",
            m=M_NUM, n_fields=5,
        )
        print(f"ABLATION champion_equiv primary={va_ce['primary']:.4f} gauc={va_ce['GAUC']:.4f} ndcg5={va_ce['nDCG@5']:.4f}")
    else:
        print("ABLATION champion_equiv skipped: out of time budget")

    # (b) no_confidence_counts: 7 fields (with hour/session) + only the original 7 numeric rate features.
    elapsed = time.time() - t0
    if elapsed < 0.75 * budget:
        model_ncc, va_ncc = train_deepfm(
            Xtr_sub, ytr_sub, uva, Xva, yva, dim, use_mlp=True,
            Ztr=Ztr_sub[:, :7], Zva=Zdict["valid"][:, :7],
            epochs=ABLATION_EPOCHS, patience=ABLATION_PATIENCE, seed=SEED, tag="no_confidence_counts",
            m=7, n_fields=len(FIELDS),
        )
        print(f"ABLATION no_confidence_counts primary={va_ncc['primary']:.4f} gauc={va_ncc['GAUC']:.4f} ndcg5={va_ncc['nDCG@5']:.4f}")
    else:
        print("ABLATION no_confidence_counts skipped: out of time budget")

    # (c) no_session_fields: 5 fields (no hour/session) + all 11 numeric (with counts).
    elapsed = time.time() - t0
    if elapsed < 0.75 * budget:
        model_nsf, va_nsf = train_deepfm(
            Xtr_sub[:, :5], ytr_sub, uva, Xva[:, :5], yva, dim, use_mlp=True,
            Ztr=Ztr_sub, Zva=Zdict["valid"],
            epochs=ABLATION_EPOCHS, patience=ABLATION_PATIENCE, seed=SEED, tag="no_session_fields",
            m=M_NUM, n_fields=5,
        )
        print(f"ABLATION no_session_fields primary={va_nsf['primary']:.4f} gauc={va_nsf['GAUC']:.4f} ndcg5={va_nsf['nDCG@5']:.4f}")
    else:
        print("ABLATION no_session_fields skipped: out of time budget")


if __name__ == "__main__":
    main()
```
=== END FILE ===
