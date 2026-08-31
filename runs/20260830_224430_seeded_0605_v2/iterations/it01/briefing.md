# STATE BLOCK
CURRENT BEST: it00 | val primary 0.6050 (GAUC 0.6718 / nDCG5 0.5383) | baseline 0.6016 | margin +0.0034
BUDGET: iteration 1 of 20 | 0:01 of 6:00 elapsed | tokens so far 0
CONVERGENCE: streak 0 of 3 flat (EPSILON=0.002)
BLOCKED: none
ACTIVE THEMES: winning: none; losing/flat: none; untried: feature, model, training, multitask, other


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
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket", "hour", "sess_depth"]   # 7 categorical fields
K = 16            # embedding dim
LR = 0.001
L2 = 1e-6
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
    """Rows as (date, user_id, video_id, author_id, tab, duration_ms, label, hour, sess_depth);
    file order preserved (no global time sort)."""
    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    # read rows in original file order
    rows_orig = []   # will keep original order
    for f in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows_orig.append((int(r["date"]), r["user_id"], r["video_id"], vid2author.get(r["video_id"], "UNK"), r["tab"],
                                 float(r["duration_ms"]), 1 if r[LABEL] != "0" else 0, int(r["time_ms"]), int(r["hourmin"]) // 100))
    N = len(rows_orig)
    
    # indices sorted by time for historical stats
    time_order = sorted(range(N), key=lambda i: rows_orig[i][7])
    
    v_stats = collections.defaultdict(lambda: [0, 0])
    a_stats = collections.defaultdict(lambda: [0, 0])
    num_features = {}
    
    for i in time_order:
        x = rows_orig[i]
        date, _, vid, aid, _, _, label, _, _ = x
        v_imp, v_pos = v_stats[vid]
        a_imp, a_pos = a_stats[aid]
        
        v_rate = v_pos / v_imp if v_imp > 0 else 0.0
        a_rate = a_pos / a_imp if a_imp > 0 else 0.0
        
        num_features[i] = [np.log1p(v_imp), v_rate, np.log1p(a_imp), a_rate]
        
        if 20220408 <= date <= 20220421:
            v_stats[vid][0] += 1
            a_stats[aid][0] += 1
            if label == 1:
                v_stats[vid][1] += 1
                a_stats[aid][1] += 1

    # Compute session depth and time-gap from time-ordered sequence per (user, date)
    # Use a separate sort: order by user, date, time
    group_order = sorted(range(N), key=lambda i: (rows_orig[i][1], rows_orig[i][0], rows_orig[i][7]))
    user_date_counts = collections.defaultdict(int)
    user_date_last_time = {}
    depth_map = {}
    
    if len(group_order) == 0:
        depth_map = {}
    else:
        for idx in group_order:
            x = rows_orig[idx]
            key = (x[1], x[0])               # (user_id, date)
            curr_time = x[7]
            c = user_date_counts[key]
            user_date_counts[key] += 1
            depth = str(c) if c <= 4 else ("5-9" if c <= 9 else "10+")
            depth_map[idx] = depth
            
            time_gap = curr_time - user_date_last_time.get(key, curr_time)
            user_date_last_time[key] = curr_time
            num_features[idx].append(np.log1p(time_gap))
    
    # Build new rows in original order
    new_rows = []
    for i, x in enumerate(rows_orig):
        new_rows.append(x[:7] + (str(x[8]), depth_map[i], num_features[i]))
    rows = new_rows
    return {name: [x for x in rows if lo <= x[0] <= hi] for name, (lo, hi) in SPLITS.items()}


# ----------------------------------------------------------------------------- [3] feature encoding (= starter_kit/data.py)
def _bucket_edges(durations, n=N_DUR_BUCKETS):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


def encode(splits):
    """Categorical ids -> contiguous ints; unseen values fall into a per-field UNK slot.
    Returns ({split: (X int32 (N,F), X_num float32 (N,5), y float32, users)}, total_dim)."""
    tr = splits["train"]
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5]))), x[7], x[8]]

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
        X_num = np.empty((len(rws), 5), dtype=np.float32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            X_num[n] = x[9]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, X_num, y, users)
    return enc, int(sum(field_dims))


# ----------------------------------------------------------------------------- [4] model (= starter_kit/baseline.py FM)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Second-order factorization machine, pointwise logloss or BPR, Adam."""

    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.W_num = np.zeros(5, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mW_num = np.zeros_like(self.W_num); self.vW_num = np.zeros_like(self.W_num)
        self.t = 0

    def logits(self, X, X_num):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter, E, S

    def step(self, X, X_num, y):
        B = len(y)
        z, E, S = self.logits(X, X_num)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gW_num = (g[:, None] * X_num).sum(0)
        
        gV += self.l2 * self.V; gW += self.l2 * self.W
        gW_num += self.l2 * self.W_num
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW), (self.W_num, gW_num, self.mW_num, self.vW_num)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def step_bpr(self, X_pos, X_num_pos, X_neg, X_num_neg):
        B = len(X_pos)
        z_pos, E_pos, S_pos = self.logits(X_pos, X_num_pos)
        z_neg, E_neg, S_neg = self.logits(X_neg, X_num_neg)
        z_diff = z_pos - z_neg
        g = ((sigmoid(z_diff) - 1.0) / B).astype(np.float32)
        
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        
        np.add.at(gW, X_pos, g[:, None])
        np.add.at(gW, X_neg, -g[:, None])
        
        np.add.at(gV, X_pos, g[:, None, None] * (S_pos[:, None, :] - E_pos))
        np.add.at(gV, X_neg, -g[:, None, None] * (S_neg[:, None, :] - E_neg))
        
        gW_num = (g[:, None] * (X_num_pos - X_num_neg)).sum(0)
        
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        gW_num += self.l2 * self.W_num
        
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW), (self.W_num, gW_num, self.mW_num, self.vW_num)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
            
        return float(-np.mean(np.log(sigmoid(z_diff) + 1e-9)))

    def predict(self, X, X_num, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs], X_num[i:i + bs])[0] for i in range(0, len(X), bs)])


# ----------------------------------------------------------------------------- [5] training
def train(enc, dim, mode="pure_bpr", max_epochs=EPOCHS, seed=SEED, log=print):
    """Train on train, early-stop on validation primary. Returns the best model and its metrics."""
    Xtr, Xnum_tr, ytr, utr = enc["train"]
    Xva, Xnum_va, yva, uva = enc["valid"]
    
    if mode == "champion_equiv":
        Xnum_tr = np.zeros_like(Xnum_tr)
        Xnum_va = np.zeros_like(Xnum_va)
    
    if mode in ("pure_bpr", "champion_equiv"):
        user2pos = collections.defaultdict(list)
        user2neg = collections.defaultdict(list)
        for i, (u, y) in enumerate(zip(utr, ytr)):
            if y == 1.0:
                user2pos[u].append(i)
            else:
                user2neg[u].append(i)
                
        pos_indices = []
        neg_indices = []
        for u, pos_list in user2pos.items():
            neg_list = user2neg.get(u, [])
            if len(neg_list) > 0:
                for p in pos_list:
                    pos_indices.append(p)
                    neg_indices.append(neg_list)
                    
        pos_indices = np.array(pos_indices, dtype=np.int32)
        lens = np.array([len(n) for n in neg_indices], dtype=np.int32)
        
        if len(neg_indices) > 0:
            flat_negs = np.concatenate(neg_indices).astype(np.int32)
            offsets = np.cumsum([0] + list(lens[:-1]), dtype=np.int32)
        else:
            flat_negs = np.array([], dtype=np.int32)
            offsets = np.array([], dtype=np.int32)
            
        num_pairs = len(pos_indices)
        log(f"[{mode}] Total within-user pairs: {num_pairs}")

    m = FM(dim, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, best_metrics, bad = -1.0, None, None, 0
    
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        losses = []
        
        if mode in ("pure_bpr", "champion_equiv"):
            if num_pairs == 0:
                break
            sampled_neg = flat_negs[offsets + (rng.random(num_pairs) * lens).astype(np.int32)]
            idx = rng.permutation(num_pairs)
            for i in range(0, num_pairs, BATCH):
                b_idx = idx[i:i + BATCH]
                p_idx = pos_indices[b_idx]
                n_idx = sampled_neg[b_idx]
                losses.append(m.step_bpr(Xtr[p_idx], Xnum_tr[p_idx], Xtr[n_idx], Xnum_tr[n_idx]))
        else:
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), BATCH):
                b_idx = idx[i:i + BATCH]
                losses.append(m.step(Xtr[b_idx], Xnum_tr[b_idx], ytr[b_idx]))
                
        va = evaluate(uva, yva, m.predict(Xva, Xnum_va))
        loss_val = np.mean(losses) if losses else 0.0
        log(f"[{mode}] epoch {ep:2d} | loss {loss_val:.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
            f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
            
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), m.W_num.copy(), np.float32(m.b))
            best_metrics = va.copy()
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"[{mode}] early stop at epoch {ep}")
                break
                
    if best_state is not None:
        m.V, m.W, m.W_num, m.b = best_state
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
    t_start = time.time()
    budget = float(os.environ.get("KUAIRAND_TIME_BUDGET_S", 1500))
    fast = os.environ.get("KUAIRAND_FAST", "0") == "1"
    
    splits = load(a.data)
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}", f"load {time.time() - t_start:.1f}s")
    enc, dim = encode(splits)
    
    # 1. Full fit on champion (pure_bpr) and write predictions
    X, X_num, _, _ = enc[split]
    Xva, Xnum_va, yva, uva = enc["valid"]
    
    all_scores = []
    all_scores_va = []
    last_single_metrics = None
    
    seeds = [42] if fast else [42, 43, 44, 45, 46]
    for s in seeds:
        model_bpr, metrics_bpr = train(enc, dim, mode="pure_bpr", seed=s)
        all_scores.append(model_bpr.predict(X, X_num))
        if not fast:
            all_scores_va.append(model_bpr.predict(Xva, Xnum_va))
        last_single_metrics = metrics_bpr
    
    mean_scores = np.mean(all_scores, axis=0)
    assert np.all(np.isfinite(mean_scores)), "non-finite scores"
    write_preds(a.out, splits[split], mean_scores)
    print(f"wrote {a.out}: {len(splits[split])} rows for split={split} in {time.time() - t_start:.0f}s")
    
    if last_single_metrics is not None:
        print(f"ABLATION pure_bpr_single primary={last_single_metrics['primary']:.4f} gauc={last_single_metrics['GAUC']:.4f} ndcg5={last_single_metrics['nDCG@5']:.4f}")
    
    if not fast:
        mean_scores_va = np.mean(all_scores_va, axis=0)
        ens_metrics = evaluate(uva, yva, mean_scores_va)
        print(f"ABLATION pure_bpr_ensemble primary={ens_metrics['primary']:.4f} gauc={ens_metrics['GAUC']:.4f} ndcg5={ens_metrics['nDCG@5']:.4f}")
    
    if fast:
        return
        
    # 2. Ablations (champion_equiv)
    rem_budget = budget - (time.time() - t_start)
    if rem_budget >= 0.25 * budget:
        model_ce, metrics_ce = train(enc, dim, mode="champion_equiv", seed=42, max_epochs=EPOCHS)
        if metrics_ce is not None:
            print(f"ABLATION champion_equiv primary={metrics_ce['primary']:.4f} gauc={metrics_ce['GAUC']:.4f} ndcg5={metrics_ce['nDCG@5']:.4f}")
    else:
        print("ABLATION champion_equiv skipped: out of time budget")

if __name__ == "__main__":
    main()


# LEDGER (full history, oldest first)
# Ledger (tier-1 memory, append-only; one line per iteration, harness-written except LESSON)
# it00 champion installed from runs/20260830_165325_seeded_0605/best/code: val primary 0.6050 (GAUC 0.6718 / nDCG5 0.5383); published baseline 0.6016; rungs random 0.4827 pop 0.5807


# PRIOR RUNS — every experiment this agent has already measured (harness-recorded, earlier runs only)
These are YOUR OWN sealed measurements from previous runs of this same task, not advice. Do not spend an
iteration re-measuring something below unless you state what is different about your version. The deltas are
against the champion at that iteration's start, so a small delta on top of a strong champion is not the same
as a small delta on top of the baseline.

Best score ever recorded across all runs: **0.6050** (20260830_165325_seeded_0605 it03) — Stacking past-only numerical features (video/author historical rates and impression counts) and 5-seed ensembling (both validated riders) alongside a…

## WHAT WORKED — measured gains, largest first (13 of them)
| Δ vs then-champion | direction | what was tried | result |
|---|---|---|---|
| +0.0029 | training | Training with a within-user pairwise BPR loss directly aligns the objective with the evaluation metric (GAUC, nDCG@5) by optimizing relative ranking rather than absolute pointwise… | 0.6043 promoted |
| +0.0023 | training | Training with BPR loss on within-user positive-negative pairs directly aligns the objective with the ranking metric and eliminates user-bias confounding, increasing the primary me… | 0.6038 promoted |
| +0.0023 | training | Changing the pointwise logloss to a pairwise BPR loss aligned with the within-user ranking metric will directly optimise for the primary evaluation criteria and yield a structural… | 0.6038 promoted |
| +0.0021 | feature | Adding the user's daily session depth and hour-of-day as contextual categorical features will capture position bias and time context, and combining this with a 3-seed ensemble wil… | 0.6048 promoted |
| +0.0018 | feature | Adding the user's daily session depth and hour-of-day as categorical features will capture position bias and time context, and ensembling 3 seeds will reduce variance, jointly yie… | 0.6032 promoted |
| +0.0012 | training | Training with BPR loss on within-user positive-negative pairs directly aligns the objective with the ranking metric, raising primary. | 0.6027 promoted |
| +0.0010 | feature | Adding the user's daily session depth and hour-of-day as categorical context features models time and position bias, while a 3-seed ensemble reduces variance, together yielding a… | 0.6025 promoted |
| +0.0007 | feature | Adding daily session depth and hour-of-day as categorical context features, combined with a 3-seed ensemble, will capture position bias and time context to raise the primary metri… | 0.6022 promoted |
| +0.0005 | feature | Adding the user's daily session depth and hour-of-day as contextual categorical features captures position bias and time context, yielding new ranking signal. | 0.6048 promoted |
| +0.0005 | feature | Adding strictly past-only video and author historical long_view rates and impression counts as numerical features will provide strong item-quality signals, improving the BPR model… | 0.6042 kept_champion |
| +0.0004 | model | An ensemble of 3 BPR FMs trained with different random seeds will reliably reduce variance and boost the ranking metric by aggregating decorrelated predictions. | 0.6042 kept_champion |
| +0.0003 | feature | Stacking past-only numerical features (video/author historical rates and impression counts) and 5-seed ensembling (both validated riders) alongside a genuinely new numerical signa… | 0.6050 promoted |
| +0.0002 | feature | Stacking past-only item/author statistics as numerical features (a validated rider) along with session time-gap (a new signal) and a 5-seed ensemble will push the champion past th… | 0.6034 promoted |

## WHAT DID NOT WORK — measured losses or no movement (11 of them)
| Δ vs then-champion | direction | what was tried | result |
|---|---|---|---|
| -0.0115 | training | Replacing pointwise logloss with within-user pairwise BPR loss — which directly optimizes the same within-user ranking that GAUC and nDCG@5 measure — should raise primary because… | 0.5900 kept_champion |
| -0.0089 | training | Training with a within-user pairwise BPR loss directly aligns the objective with the primary ranking metrics (GAUC, nDCG@5), eliminating user-bias confounding and raising primary. | 0.5925 kept_champion |
| -0.0064 | feature | Adding the user's most recently interacted video IDs as past-only categorical fields will explicitly model sequential item-to-item transitions (Markov chains) and short-term inter… | 0.5984 kept_champion |
| -0.0046 | multitask | Adding an auxiliary MSE regression task on play_progress (play_time_ms / duration_ms) will provide a dense, continuous preference signal to the shared embeddings, improving the pr… | 0.6002 kept_champion |
| -0.0028 | feature | Adding the user's last 3 positively interacted videos mapped directly to the shared video_id embedding space will enable Factorized Personalized Markov Chains (FPMC) item-to-item… | 0.6019 kept_champion |
| -0.0010 | multitask | Adding an auxiliary pointwise logloss for is_click with shared embeddings and a weight of 0.5 will improve the representation of items and users, raising the primary long_view ran… | 0.6028 kept_champion |
| -0.0008 | feature | Adding strictly past-only historical long_view rates and impression counts for videos and authors as bucketed categorical fields will provide a dense item-quality signal that shar… | 0.6039 kept_champion |
| -0.0006 | feature | Ensembling 5 seeds, adding past-only global item/author rates (a validated rider), and injecting past-only user-author interaction rates (a new personalization signal) as numerica… | 0.6042 kept_champion |
| -0.0005 | training | Training with a hybrid pointwise logloss and within-user pairwise BPR loss will directly optimize the relative ordering of items for mixed users while maintaining calibration for… | 0.6027 kept_champion |
| -0.0003 | training | Replacing the pairwise BPR loss with a within-user sampled softmax loss over a list of 1 positive and 7 negatives will provide stronger gradients and implicitly mine hard negative… | 0.6045 kept_champion |
| -0.0001 | model | Implementing a DIN-style target attention over the user's past clicks provides strong explicit interest modeling, yielding significant new ranking signal that static FMs cannot ca… | 0.6048 kept_champion |

## WHAT BROKE — 3 iterations never produced a score (an implementation failure costs the same as a bad idea)
- other: (no valid plan: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction under consideration and either w…) — failed: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction un…
- other: (no valid plan: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction under consideration and either w…) — failed: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction un…
- other: (no valid plan: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction under consideration and either w…) — failed: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction un…

Attempts per direction across all prior runs: feature 12 (8 positive), model 2 (1 positive), multitask 2 (0 positive), training 8 (4 positive).

# SIZING DIRECTIVE (harness policy: flat streak 0 of 3 — 3 more miss(es) end the run)
The convergence rule is per iteration: only a gain > +0.002 over the best-so-far (0.6050) resets the streak. A
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