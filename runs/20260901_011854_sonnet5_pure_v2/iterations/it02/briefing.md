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