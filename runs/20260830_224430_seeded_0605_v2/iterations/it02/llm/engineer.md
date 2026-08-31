# engineer — engineer (model google/gemini-3.1-pro-preview, 19517 tokens)

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
HYPOTHESIS: Generalizing the Factorization Machine to a Field-weighted FM (FwFM) will allow the model to learn the importance of different field-pair interactions, upweighting critical crosses like user-video while discounting noisy ones, raising primary.
CATEGORY: model
EXPECTED RISK: low
EXPECTED GAIN (Researcher's prediction): 0.003
ABLATION PLAN (variants to also score and print as ABLATION lines): no_fwfm: bundle without the FwFM capability (i.e. freeze_r=True, staying mathematically identical to the base FM).
CHANGE SPEC:
1. In `FM.__init__`, add `freeze_r=False` to the arguments. Add `self.freeze_r = freeze_r` and `self.num_fields = len(FIELDS)`. Initialize `self.R = np.ones((self.num_fields, self.num_fields), dtype=np.float32)`, alongside `self.mR = np.zeros_like(self.R)` and `self.vR = np.zeros_like(self.R)`.
2. In `FM.logits`, replace the `S` and `inter` calculations with a loop: initialize `inter = np.zeros(len(X), dtype=np.float32)` and `S_weighted = np.zeros_like(E)`. Loop `i` from 0 to `self.num_fields` and `j` from `i+1` to `self.num_fields`. Inside, compute `dot = (E[:, i, :] * E[:, j, :]).sum(axis=1)`, add `self.R[i, j] * dot` to `inter`, and accumulate gradients: `S_weighted[:, i, :] += self.R[i, j] * E[:, j, :]` and `S_weighted[:, j, :] += self.R[i, j] * E[:, i, :]`. Return `z, E, S_weighted`.
3. In `FM.step` and `FM.step_bpr`, update `gV` logic: remove the `- E` part, just use `S_pos` (which is now `S_weighted` from logits) directly in `g[:, None, None] * S_pos`. 
4. In `FM.step` and `FM.step_bpr`, compute `gR = np.zeros_like(self.R)` using the same `i`, `j` loops: `gR[i, j] = (g * dot).sum()` (or `(g * (dot_pos - dot_neg)).sum()` for BPR). Add `self.l2 * self.R` to `gR`.
5. In both step functions, replace the hardcoded Adam tuple with a list `params = [(self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW), (self.W_num, gW_num, self.mW_num, self.vW_num)]`. `if not self.freeze_r: params.append((self.R, gR, self.mR, self.vR))`. Loop over `params` for Adam updates.
6. In `train()`, add `m.R.copy()` to `best_state` and unpack it upon restore.
7. In `main()`, replace the `champion_equiv` ablation with `no_fwfm` by passing `freeze_r=(mode == 'no_fwfm')` to FM in `train()`, and print `ABLATION no_fwfm primary=...`.

# Current champion files
=== FILE: pipeline.py ===
```python
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

```python
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

    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED, freeze_r=False):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.W_num = np.zeros(5, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mW_num = np.zeros_like(self.W_num); self.vW_num = np.zeros_like(self.W_num)
        self.freeze_r = freeze_r
        self.num_fields = len(FIELDS)
        self.R = np.ones((self.num_fields, self.num_fields), dtype=np.float32)
        self.mR = np.zeros_like(self.R)
        self.vR = np.zeros_like(self.R)
        self.t = 0

    def logits(self, X, X_num):
        E = self.V[X]                                   # (B,F,k)
        inter = np.zeros(len(X), dtype=np.float32)
        S_weighted = np.zeros_like(E)
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                dot = (E[:, i, :] * E[:, j, :]).sum(axis=1)
                inter += self.R[i, j] * dot
                S_weighted[:, i, :] += self.R[i, j] * E[:, j, :]
                S_weighted[:, j, :] += self.R[i, j] * E[:, i, :]
        return self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter, E, S_weighted

    def step(self, X, X_num, y):
        B = len(y)
        z, E, S = self.logits(X, X_num)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * S)
        gW_num = (g[:, None] * X_num).sum(0)
        
        gR = np.zeros_like(self.R)
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                dot = (E[:, i, :] * E[:, j, :]).sum(axis=1)
                gR[i, j] = (g * dot).sum()
        
        gV += self.l2 * self.V; gW += self.l2 * self.W
        gW_num += self.l2 * self.W_num
        gR += self.l2 * self.R
        
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = [(self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW), (self.W_num, gW_num, self.mW_num, self.vW_num)]
        if not self.freeze_r:
            params.append((self.R, gR, self.mR, self.vR))
        for P, G, M, Vv in params:
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
        
        np.add.at(gV, X_pos, g[:, None, None] * S_pos)
        np.add.at(gV, X_neg, -g[:, None, None] * S_neg)
        
        gW_num = (g[:, None] * (X_num_pos - X_num_neg)).sum(0)
        
        gR = np.zeros_like(self.R)
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                dot_pos = (E_pos[:, i, :] * E_pos[:, j, :]).sum(axis=1)
                dot_neg = (E_neg[:, i, :] * E_neg[:, j, :]).sum(axis=1)
                gR[i, j] = (g * (dot_pos - dot_neg)).sum()
        
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        gW_num += self.l2 * self.W_num
        gR += self.l2 * self.R
        
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = [(self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW), (self.W_num, gW_num, self.mW_num, self.vW_num)]
        if not self.freeze_r:
            params.append((self.R, gR, self.mR, self.vR))
        for P, G, M, Vv in params:
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
    
    if mode in ("pure_bpr", "no_fwfm"):
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

    m = FM(dim, seed=seed, freeze_r=(mode == 'no_fwfm'))
    rng = np.random.default_rng(seed)
    best, best_state, best_metrics, bad = -1.0, None, None, 0
    
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        losses = []
        
        if mode in ("pure_bpr", "no_fwfm"):
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
            best_state = (m.V.copy(), m.W.copy(), m.W_num.copy(), np.float32(m.b), m.R.copy())
            best_metrics = va.copy()
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"[{mode}] early stop at epoch {ep}")
                break
                
    if best_state is not None:
        m.V, m.W, m.W_num, m.b, m.R = best_state
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
        
    # 2. Ablations (no_fwfm)
    rem_budget = budget - (time.time() - t_start)
    if rem_budget >= 0.25 * budget:
        model_ce, metrics_ce = train(enc, dim, mode="no_fwfm", seed=42, max_epochs=EPOCHS)
        if metrics_ce is not None:
            print(f"ABLATION no_fwfm primary={metrics_ce['primary']:.4f} gauc={metrics_ce['GAUC']:.4f} ndcg5={metrics_ce['nDCG@5']:.4f}")
    else:
        print("ABLATION no_fwfm skipped: out of time budget")

if __name__ == "__main__":
    main()
```
