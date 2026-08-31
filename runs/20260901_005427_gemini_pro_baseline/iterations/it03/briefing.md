# STATE BLOCK
CURRENT BEST: it02 | val primary 0.6032 (GAUC 0.6700 / nDCG5 0.5364) | baseline 0.6016 | margin +0.0016
BUDGET: iteration 3 of 50 | 0:05 of 6:00 elapsed | tokens so far 86481
CONVERGENCE: streak 2 of 3 flat (EPSILON=0.002)
BLOCKED: it01: Replacing the numpy FM with a PyTorch DeepFM and concatenating strict… [abandoned by debugger]
ACTIVE THEMES: winning: training[1 promoted/0 flat/0 failed]; losing/flat: model[0 promoted/0 flat/1 failed]; untried: feature, multitask, other


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
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]   # 5 categorical fields of the baseline
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

    def step_pointwise(self, X, y):
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

    def step_pairwise(self, X_pos, X_neg):
        B = len(X_pos)
        z_pos, E_p, S_p = self.logits(X_pos)
        z_neg, E_n, S_n = self.logits(X_neg)
        diff = z_pos - z_neg
        sig = sigmoid(diff)
        g = (-(1.0 - sig) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        
        np.add.at(gW, X_pos, g[:, None])
        np.add.at(gW, X_neg, -g[:, None])
        
        np.add.at(gV, X_pos, g[:, None, None] * (S_p[:, None, :] - E_p))
        np.add.at(gV, X_neg, -g[:, None, None] * (S_n[:, None, :] - E_n))
        
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        return float(-np.mean(np.log(sig + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


# ----------------------------------------------------------------------------- [5] training
def train(enc, dim, log=print, loss_type="pairwise"):
    """Train on train, early-stop on validation primary (official recipe). Returns the best model."""
    Xtr, ytr, utr_list = enc["train"]
    Xva, yva, uva = enc["valid"]
    m = FM(dim)
    rng = np.random.default_rng(SEED)
    best, best_state, bad = -1.0, None, 0
    
    utr = np.array(utr_list)
    
    if loss_type == "pairwise":
        pos_mask = (ytr > 0)
        neg_mask = (ytr == 0)
        neg_idx = np.where(neg_mask)[0]
        
        neg_users = utr[neg_idx]
        neg_sort_order = np.argsort(neg_users)
        neg_idx_sorted = neg_idx[neg_sort_order]
        sorted_neg_users = neg_users[neg_sort_order]
        
        unique_neg_u, neg_starts, neg_counts = np.unique(sorted_neg_users, return_index=True, return_counts=True)
        neg_user_map = {u: (s, c) for u, s, c in zip(unique_neg_u, neg_starts, neg_counts)}
        
        all_pos_idx = np.where(pos_mask)[0]
        pos_idx_list = []
        pos_neg_starts_list = []
        pos_neg_counts_list = []
        
        for p_idx in all_pos_idx:
            u = utr[p_idx]
            if u in neg_user_map:
                s, c = neg_user_map[u]
                pos_idx_list.append(p_idx)
                pos_neg_starts_list.append(s)
                pos_neg_counts_list.append(c)
                
        pos_idx = np.array(pos_idx_list, dtype=np.int32)
        pos_neg_starts = np.array(pos_neg_starts_list, dtype=np.int32)
        pos_neg_counts = np.array(pos_neg_counts_list, dtype=np.int32)

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        
        if loss_type == "pairwise":
            rand_offsets = (rng.random(len(pos_neg_counts)) * pos_neg_counts).astype(np.int32)
            sampled_neg_idx = neg_idx_sorted[pos_neg_starts + rand_offsets]
            
            shuffle_idx = rng.permutation(len(pos_idx))
            p_idx_shuf = pos_idx[shuffle_idx]
            n_idx_shuf = sampled_neg_idx[shuffle_idx]
            
            losses = []
            for i in range(0, len(p_idx_shuf), BATCH):
                p_batch = p_idx_shuf[i:i + BATCH]
                n_batch = n_idx_shuf[i:i + BATCH]
                losses.append(m.step_pairwise(Xtr[p_batch], Xtr[n_batch]))
            n_pairs = len(pos_idx)
            pair_info = f"pairs {n_pairs} | "
        else:
            idx = rng.permutation(len(ytr))
            losses = [m.step_pointwise(Xtr[idx[i:i + BATCH]], ytr[idx[i:i + BATCH]]) for i in range(0, len(idx), BATCH)]
            pair_info = ""

        va = evaluate(uva, yva, m.predict(Xva))
        if ep == 1 and loss_type == "pairwise":
            assert va['GAUC'] > 0.5, "GAUC <= 0.5 after epoch 1"
            
        log(f"epoch {ep:2d} | {pair_info}loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
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
    
    t_start = time.time()
    time_budget = float(os.environ.get("KUAIRAND_TIME_BUDGET_S", 1500))
    fast_mode = os.environ.get("KUAIRAND_FAST", "0") == "1"

    t0 = time.time()
    splits = load(a.data)
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}", f"load {time.time() - t0:.1f}s")
    enc, dim = encode(splits)
    model = train(enc, dim, loss_type="pairwise")
    X, _, _ = enc[split]
    scores = model.predict(X)
    assert np.all(np.isfinite(scores)), "non-finite scores"
    write_preds(a.out, splits[split], scores)
    print(f"wrote {a.out}: {len(splits[split])} rows for split={split} in {time.time() - t_start:.0f}s")

    if fast_mode:
        return
        
    remains = time_budget - (time.time() - t_start)
    if remains < time_budget * 0.25:
        print("ABLATION champion_equiv skipped: out of time budget")
    else:
        m_pt = train(enc, dim, log=lambda x: None, loss_type="pointwise")
        Xva, yva, uva = enc["valid"]
        va = evaluate(uva, yva, m_pt.predict(Xva))
        print(f"ABLATION champion_equiv primary={va['primary']:.6f} gauc={va['GAUC']:.6f} ndcg5={va['nDCG@5']:.6f}")


if __name__ == "__main__":
    main()


# LEDGER (full history, oldest first)
# Ledger (tier-1 memory, append-only; one line per iteration, harness-written except LESSON)
# it00 champion installed from baseline_repro: val primary 0.6015 (GAUC 0.6671 / nDCG5 0.5358); published baseline 0.6016; rungs random 0.4827 pop 0.5807
[it01] HYP: Replacing the numpy FM with a PyTorch DeepFM and concatenating strictly past-only user/video historical rates as numeri… | CHANGE: pipeline.py (+187/-23) | RESULT: FAILED(exit code 1 Traceback (most recent call last): File "/home/…) (best 0.6015) -> FAILED | LESSON: PyTorch DeepFM failed due to missing torch module; no score.
[it02] HYP: Training the numpy FM with a within-user pairwise BPR loss directly aligns the optimization objective with the evaluati… | CHANGE: pipeline.py (+105/-8) | RESULT: 0.6032 (best 0.6032) -> PROMOTED | LESSON: Pairwise BPR: 0.601470 vs 0.6032277176425208, promoted; early-stopped at epoch 9.


# PRIOR RUNS — every experiment this agent has already measured (harness-recorded, earlier runs only)
These are YOUR OWN sealed measurements from previous runs of this same task, not advice. Do not spend an
iteration re-measuring something below unless you state what is different about your version. The deltas are
against the champion at that iteration's start, so a small delta on top of a strong champion is not the same
as a small delta on top of the baseline.

Best score ever recorded across all runs: **0.6563** (20260831_145457_1k_bonus_test it05) — Adding past-only user-tab specific historical impression and positive rates as numerical features will give the DeepFM MLP a highly personalized, con…

## WHAT WORKED — measured gains, largest first (11 of them)
| Δ vs then-champion | direction | what was tried | result |
|---|---|---|---|
| +0.0078 | model | Extending the FM to a DeepFM by adding a 1-layer MLP over the concatenated embeddings and numerical features will allow the model to learn arbitrary high-order feature interaction… | 0.6489 promoted |
| +0.0037 | feature | Standardizing past-only numerical features will stabilize DeepFM's gradients against scale imbalances, adding missing user click/like rates will complete the behavioral priors, an… | 0.6528 promoted |
| +0.0035 | feature | Adding past-only user-tab specific historical impression and positive rates as numerical features will give the DeepFM MLP a highly personalized, context-aware baseline for each u… | 0.6563 promoted |
| +0.0029 | training | Training with a within-user pairwise BPR loss directly aligns the objective with the evaluation metric (GAUC, nDCG@5) by optimizing relative ranking rather than absolute pointwise… | 0.6043 promoted |
| +0.0024 | model | Upgrading the FM to a DeepFM (1-hidden-layer MLP over the concatenated field embeddings) and adding standardized past-only numerical priors (user/video/author long_view & click ra… | 0.6039 promoted |
| +0.0021 | feature | Adding the user's daily session depth and hour-of-day as contextual categorical features will capture position bias and time context, and combining this with a 3-seed ensemble wil… | 0.6048 promoted |
| +0.0012 | training | Training with BPR loss on within-user positive-negative pairs directly aligns the objective with the ranking metric, raising primary. | 0.6027 promoted |
| +0.0005 | feature | Adding the user's daily session depth and hour-of-day as contextual categorical features captures position bias and time context, yielding new ranking signal. | 0.6048 promoted |
| +0.0003 | feature | Adding user historical long_view rates and item/author auxiliary feedback rates (click, like) as past-only numerical features will provide DeepFM's MLP with rich interaction surfa… | 0.6492 promoted |
| +0.0003 | feature | Stacking past-only numerical features (video/author historical rates and impression counts) and 5-seed ensembling (both validated riders) alongside a genuinely new numerical signa… | 0.6050 promoted |
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

Attempts per direction across all prior runs: feature 11 (7 positive), model 6 (2 positive), multitask 1 (0 positive), training 5 (2 positive).

# RESEARCH DIGEST — every iteration so far, grouped by direction (harness-measured facts)
| it | direction | what changed | predicted Δ | measured Δ vs then-champion | decision | status | in-run ablations (pipeline-reported, unsealed) | lesson |
|---|---|---|---|---|---|---|---|---|
| it01 | model | Replacing the numpy FM with a PyTorch DeepFM and concatenating strictly past-only user/video historical rates as numerical features into the MLP will allow the model to learn high-order interactions and behavioral prior… | +0.0050 | n/a | failed | failed: [debugger abandoned: PyTorch (torch) is not installed in th… | — | PyTorch DeepFM failed due to missing torch module; no score. |
| it02 | training | Training the numpy FM with a within-user pairwise BPR loss directly aligns the optimization objective with the evaluation metrics (GAUC, nDCG@5), providing a stronger ranking signal than pointwise logloss. | +0.0029 | +0.0018 | promoted | scored | champion_equiv 0.6015 (-0.0018 vs the full run) | Pairwise BPR: 0.601470 vs 0.6032277176425208, promoted; early-stopped at epoch 9. |

Calibration: over 1 scored iterations your predicted gain exceeded the measured one by +0.0011 on average (predicted − measured); size the next prediction accordingly.
Totals: 2 iterations; promoted 1 (it02); attempts per direction: model 1, training 1; never attempted: feature, multitask, other.

# RESEARCH SYNTHESIS (written by the Scribe from the digest above — interpretive; verify any claim against the table)
The only model change attempted was replacing the numpy FM with a PyTorch DeepFM (it01), which failed due to missing PyTorch. The only training change attempted was switching to pairwise BPR loss (it02), which succeeded with a measured gain of +0.0018 over the champion. No feature, multitask, or other directions have been attempted. Pairwise BPR was promoted, while PyTorch DeepFM failed. Calibration shows predicted gains overstated by +0.0011 on average.

# RECENT ITERATION DETAILS (harness-measured facts + what was actually changed)
Use these to decide whether to CONTINUE an idea: when a bundled change moved little, the diff shows which
components were in it, so you can keep the part that plausibly worked and drop the rest. State which
component you are keeping or dropping, and why, in `rationale`.

## it01 [model] — failed (failed), n/a
HYPOTHESIS: Replacing the numpy FM with a PyTorch DeepFM and concatenating strictly past-only user/video historical rates as numerical features into the MLP will allow the model to learn high-order interactions and behavioral priors, significantly raising primary.
YOUR PREDICTED GAIN: +0.0050 — evidence given: Ledger prior runs show that upgrading to a DeepFM (1-hidden-layer MLP over concatenated embeddings) and adding past-only numerical priors (user/video long_view rates) yielded +0.0078 and +0.0024 in similar KuaiRand settings. DeepFM leverages both low-order interactions and high-order dense represen…
RATIONALE (yours, at the time): The current champion is a linear + second-order FM in numpy which lacks the capacity to model high-order interactions and cannot natively consume dense numerical features. DeepFM (Guo et al., 2017) explicitly models both low- and high-order interactions by sharing embeddings between an FM and an MLP. Adding past-only historical rates provides explicit behavioral priors without leakage, heavily favored by tabular GBDTs and neural models alike.
CHANGE SPEC you gave the Engineer:
1. Replace numpy FM with PyTorch DeepFM. Use `torch` and `torch.nn`.
2. In `encode()`, process splits in order ('train', 'valid', 'test'). Maintain `user_stats` and `vid_stats` dicts mapping ID to `[impressions, long_views]`. For each row, BEFORE updating stats, extract 4 numerical features: `np.log1p(u_imp)`, `u_lv / u_imp if u_imp > 0 else 0.0`, `np.log1p(v_imp)`, `v_lv / v_imp if v_imp > 0 else 0.0`. Then update the stats with the row's label. Return `Xnum` along with `X`, `y`, `users`.
3. Standardize `Xnum` using mean and std computed ONLY from the 'train' split.
4. Create `class DeepFM(nn.Module)`: `V = nn.Embedding(dim, k)`, `W = nn.Embedding(dim, 1)`, `b = nn.Parameter(zeros(1))`. MLP is `nn.Sequential(nn.Linear(5*k + 4, 128), nn.ReLU(), nn.Linear(128, 1))`. Initialize Embeddings with `std=0.01`.
5. Forward pass: `E = V(X)`; FM part is `W(X).sum(1) + 0.5 * ((E.sum(1)**2) - (E**2).sum(1)).sum(1) + b`. MLP part is `mlp(cat([E.view(B, -1), Xnum], dim=1))`. Return `fm_part + mlp_part.squeeze(-1)`.
6. In `train()`, convert data to `torch.Tensor` (`X` as long, `Xnum` and `y` as float32). Use `torch.utils.data.DataLoader` with `batch_size=8192`, `shuffle=True` for train. Use `torch…
WHAT CHANGED: pipeline.py (+187/-23)
DIFF (champion -> attempt):
```diff
--- champion/pipeline.py
+++ attempt/pipeline.py
@@ -21,6 +21,9 @@
 import time
 
 import numpy as np
+import torch
+import torch.nn as nn
+from torch.utils.data import TensorDataset, DataLoader
 
 # ----------------------------------------------------------------------------- [1] config
 LABEL = "long_view"
@@ -80,7 +83,7 @@
 
 def encode(splits):
     """Categorical ids -> contiguous ints; unseen values fall into a per-field UNK slot.
-    Returns ({split: (X int32 (N,F), y float32, users)}, total_dim)."""
+    Returns ({split: (X int32 (N,F), Xnum float32 (N,4), y float32, users)}, total_dim)."""
     tr = splits["train"]
     edges = _bucket_edges([x[5] for x in tr])
 
@@ -96,27 +99,60 @@
     field_dims = [len(v) + 1 for v in vocabs]
     offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
     enc = {}
-    for name, rws in splits.items():
+    
+    user_stats = {}
+    vid_stats = {}
+    
+    for name in ["train", "valid", "test"]:
+        if name not in splits:
+            continue
+        rws = splits[name]
         X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
+        Xnum = np.empty((len(rws), 4), dtype=np.float32)
         y = np.empty(len(rws), dtype=np.float32)
         users = []
         for n, x in enumerate(rws):
+            uid, vid, label = x[1], x[2], x[6]
+            
+            u_imp, u_lv = user_stats.get(uid, [0, 0])
+            v_imp, v_lv = vid_stats.get(vid, [0, 0])
+            
+            Xnum[n, 0] = np.log1p(u_imp)
+            Xnum[n, 1] = u_lv / u_imp if u_imp > 0 else 0.0
+            Xnum[n, 2] = np.log1p(v_imp)
+            Xnum[n, 3] = v_lv / v_imp if v_imp > 0 else 0.0
+            
+            if uid not in user_stats: user_stats[uid] = [0, 0]
+            user_stats[uid][0] += 1
+            user_stats[uid][1] += label
+            
+            if vid not in vid_stats: vid_stats[vid] = [0, 0]
+            vid_stats[vid][0] += 1
+            vid_stats[vid][1] += label
+            
             for i, v in enumerate(raw(x)):
                 X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
-            y[n] = x[6]
-            users.append(x[1])
-        enc[name] = (X, y, users)
+            y[n] = label
+            users.append(uid)
+        enc[name] = (X, Xnum, y, users)
+
+    Xnum_tr = enc["train"][1]
+    mean = Xnum_tr.mean(axis=0)
+    std = Xnum_tr.std(axis=0)
+    std[std == 0] = 1.0
+    for name in enc:
+        enc[name] = (enc[name][0], (enc[name][1] - mean) / std, enc[n
… (diff truncated)
```
OUTCOME: failed — exit code 1 Traceback (most recent call last): File "/home/q3user/kuairand-research-agent/runs/20260901_005427_gemini_pro_baseline/iterations/it01/pipeline.py", line 24, in <module> import torch ModuleNotFoundError: No module named 'torch' [debugger abandoned: PyTorch (torch) is not installed in the environment; cannot implement the required DeepFM model. The experiment's hypothesis requires torc…
  debug attempt 1: ModuleNotFoundError: No module named 'torch' -> fix: ABANDONED: PyTorch (torch) is not installed in the environment; cannot implement the required DeepFM model. The experiment's hypothesis requires torch, which i… (abandoned)
LESSON: PyTorch DeepFM failed due to missing torch module; no score.

## it02 [training] — promoted (scored), +0.0018 vs the then-champion 0.6015
HYPOTHESIS: Training the numpy FM with a within-user pairwise BPR loss directly aligns the optimization objective with the evaluation metrics (GAUC, nDCG@5), providing a stronger ranking signal than pointwise logloss.
YOUR PREDICTED GAIN: +0.0029; measured +0.0018 — evidence given: Ledger prior runs demonstrate +0.0029 from replacing pointwise BCE with within-user pairwise BPR loss, explicitly optimizing the relative ordering required by GAUC.
RATIONALE (yours, at the time): The previous attempt (it01) failed due to a missing PyTorch dependency, invalidating the model upgrade path for now. Switching the training objective to BPR leverages pure numpy, bypasses environment limits, and directly optimizes the ranking metric. Vectorized sampling ensures the pair creation fits easily within the 900s runtime budget.
CHANGE SPEC you gave the Engineer:
1. In `FM` class, rename `step` to `step_pointwise`. Add `step_pairwise(self, X_pos, X_neg)`: Compute logits `z_pos, E_p, S_p` and `z_neg, E_n, S_n`. `diff = z_pos - z_neg`. `sig = sigmoid(diff)`. `g = -(1.0 - sig) / len(X_pos)`. Apply gradients `g` to `X_pos` and `-g` to `X_neg` using `np.add.at` for both V and W (similar to pointwise but updating pos and neg indices respectively). Apply Adam updates to V and W (skip `b`). Return `float(-np.mean(np.log(sig + 1e-9)))`.
2. In `train()`, before the epoch loop, implement fully vectorized sampling: separate train indices into positives (y>0) and negatives (y==0). Sort negatives by user, and use `np.unique(..., return_counts=True)` to build `neg_starts` and `neg_counts` arrays mapping each unique user to their negative pool block. Filter positive indices to only those users who have >= 1 negative. Create arrays `pos_neg_starts` and `pos_neg_counts` aligning with the valid positive indices.
3. In the epoch loop, vectorize sampling: `rand_offsets = (rng.random(len(pos_neg_counts)) * pos_neg_counts).astype(np.int32)`. `sampled_neg_idx = neg_idx_sorted[pos_neg_starts + rand_offsets]`. Shuffle `pos_idx` and `sampled_neg_idx` together.
4. Cal…
WHAT CHANGED: pipeline.py (+105/-8)
DIFF (champion -> attempt):
```diff
--- champion/pipeline.py
+++ attempt/pipeline.py
@@ -133,7 +133,7 @@
         inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
         return self.b + self.W[X].sum(1) + inter, E, S
 
-    def step(self, X, y):
+    def step_pointwise(self, X, y):
         B = len(y)
         z, E, S = self.logits(X)
         g = ((sigmoid(z) - y) / B).astype(np.float32)
@@ -150,25 +150,105 @@
         self.b -= self.lr * g.sum()
         return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))
 
+    def step_pairwise(self, X_pos, X_neg):
+        B = len(X_pos)
+        z_pos, E_p, S_p = self.logits(X_pos)
+        z_neg, E_n, S_n = self.logits(X_neg)
+        diff = z_pos - z_neg
+        sig = sigmoid(diff)
+        g = (-(1.0 - sig) / B).astype(np.float32)
+        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
+        
+        np.add.at(gW, X_pos, g[:, None])
+        np.add.at(gW, X_neg, -g[:, None])
+        
+        np.add.at(gV, X_pos, g[:, None, None] * (S_p[:, None, :] - E_p))
+        np.add.at(gV, X_neg, -g[:, None, None] * (S_n[:, None, :] - E_n))
+        
+        gV += self.l2 * self.V; gW += self.l2 * self.W
+        self.t += 1
+        b1, b2, eps = 0.9, 0.999, 1e-8
+        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
+            M *= b1; M += (1 - b1) * G
+            Vv *= b2; Vv += (1 - b2) * (G * G)
+            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
+        return float(-np.mean(np.log(sig + 1e-9)))
+
     def predict(self, X, bs=200_000):
         return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])
 
 
 # ----------------------------------------------------------------------------- [5] training
-def train(enc, dim, log=print):
+def train(enc, dim, log=print, loss_type="pairwise"):
     """Train on train, early-stop on validation primary (official recipe). Returns the best model."""
-    Xtr, ytr, _ = enc["train"]
+    Xtr, ytr, utr_list = enc["train"]
     Xva, yva, uva = enc["valid"]
     m = FM(dim)
     rng = np.random.default_rng(SEED)
     best, best_state, bad = -1.0, None, 0
+    
+    utr = np.array(utr_list)
+    
+    if loss_type == "pairwise":
+        pos_mask = (ytr > 0)
+        neg_mask = (ytr == 0)
+        neg_idx = np.where(neg_mask)[0]
+        
+        neg_users = utr[neg_idx]
+        neg_sort_order = np.argsort(neg_users)
+        neg_idx_sorted = neg
… (diff truncated)
```
MEASURED: primary 0.6032 (GAUC 0.6700 / nDCG@5 0.5364), runtime 28.5s
IN-RUN ABLATIONS (pipeline-reported on validation, unsealed — component attribution): champion_equiv 0.6015 (-0.0018 vs the full run)
  leak test: clean (flipped users scored 0.6038 on their true labels)
TRAINING CURVE (the experiment's own stdout):
  epoch  1 | pairs 382579 | loss 0.6749 | valid GAUC 0.6571 nDCG@5 0.5297 primary 0.5934 | 0.9s
  epoch  2 | pairs 382579 | loss 0.6171 | valid GAUC 0.6637 nDCG@5 0.5336 primary 0.5987 | 0.9s
  epoch  3 | pairs 382579 | loss 0.5745 | valid GAUC 0.6672 nDCG@5 0.5353 primary 0.6013 | 0.9s
  epoch  4 | pairs 382579 | loss 0.5571 | valid GAUC 0.6681 nDCG@5 0.5351 primary 0.6016 | 0.9s
  epoch  5 | pairs 382579 | loss 0.5500 | valid GAUC 0.6700 nDCG@5 0.5364 primary 0.6032 | 0.9s
  epoch  6 | pairs 382579 | loss 0.5441 | valid GAUC 0.6694 nDCG@5 0.5366 primary 0.6030 | 0.9s
  epoch  7 | pairs 382579 | loss 0.5397 | valid GAUC 0.6689 nDCG@5 0.5358 primary 0.6024 | 0.9s
  epoch  8 | pairs 382579 | loss 0.5359 | valid GAUC 0.6694 nDCG@5 0.5362 primary 0.6028 | 0.9s
  epoch  9 | pairs 382579 | loss 0.5318 | valid GAUC 0.6694 nDCG@5 0.5365 primary 0.6030 | 0.9s
  early stop at epoch 9
  wrote preds_val.csv: 124909 rows for split=valid in 13s
  ABLATION champion_equiv primary=0.601470 gauc=0.667133 ndcg5=0.535806
LESSON: Pairwise BPR: 0.601470 vs 0.6032277176425208, promoted; early-stopped at epoch 9.

# LAST-SHOT DIRECTIVE (harness policy: flat streak 2 of 3)
One more iteration without a gain > +0.002 over the best-so-far (0.6032) ENDS THE RUN. Choose the
highest-probability bundle: keep every component of the champion that produced its gain (its loss, its fields,
its seed averaging) exactly as is, add more seeds if the champion uses fewer than 5, stack EVERY validated rider
not yet in the champion, and add ONE genuinely new signal. Do NOT replace or remove a proven component, do NOT
re-try a lever kind whose last result was within ±0.0006 (noise), and state in `rationale` and `gain_evidence`
why this bundle should clear +0.002. Keep the `ablation_plan` minimal (champion-equivalent only).

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