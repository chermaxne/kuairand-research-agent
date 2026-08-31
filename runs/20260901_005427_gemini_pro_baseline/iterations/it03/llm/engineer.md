# engineer — engineer (model google/gemini-3.1-pro-preview, 16374 tokens)

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
HYPOTHESIS: Adding the user's daily session depth and hour-of-day as past-only categorical features captures position and time bias, and combining this with a 5-seed score average will reduce variance and safely raise primary over the threshold.
CATEGORY: feature
EXPECTED RISK: low
EXPECTED GAIN (Researcher's prediction): 0.0025
ABLATION PLAN (variants to also score and print as ABLATION lines): 1_seed: validation score of the new feature bundle using only seed 0, to separate the categorical feature gain from the ensembling gain.
CHANGE SPEC:
1. Update `FIELDS` to `["user_id", "video_id", "author_id", "tab", "dur_bucket", "hour", "sess_depth"]`.
2. In `load()`, parse `int(r["hourmin"])` and `int(r["time_ms"])` and include them in the appended tuple. Wrap each row tuple with a running integer `idx` to preserve original order. Sort the list of all rows by `time_ms` (index 9). Iterate through them: maintain a `user_date_counts` dict (key: `(uid, date)`), assign the current count as `depth`, then increment it. Map `depth` to a string ('10+' if >= 10, '6-9' if >= 6, else str(depth)) and `hourmin // 100` to a string `hour`. Append these two strings to each row. Re-sort the rows by the original `idx`, remove the `idx`, and return the list.
3. In `encode()`, update `raw(x)` to return the new fields (which will be at indices 9 and 10 of the tuple): `return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5]))), x[9], x[10]]`.
4. In `train()`, add `seed=SEED` to the signature. Pass `seed=seed` to `FM(dim, seed=seed)`, and initialize `rng = np.random.default_rng(seed)` instead of the global `SEED`.
5. In `main()`, initialize `scores_list = []`. Loop `seed` from 0 to 4: call `model = train(enc, dim, loss_type="pairwise", seed=seed)`, then append `model.predict(X)` to `scores_list`. Set `scores = np.mean(scores_list, axis=0)` and pass this to `write_preds`.
6. For the ablation block, simply evaluate the predictions of the first model (seed 0) on the validation split and print its metric as `ABLATION 1_seed` to isolate the ensemble gain.

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
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket", "hour", "sess_depth"]
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
    idx = 0
    for f in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((idx, int(r["date"]), r["user_id"], r["video_id"], vid2author.get(r["video_id"], "UNK"), r["tab"],
                             float(r["duration_ms"]), 1 if r[LABEL] != "0" else 0, int(r["hourmin"]), int(r["time_ms"])))
                idx += 1
                
    rows.sort(key=lambda x: x[9])
    
    user_date_counts = {}
    new_rows = []
    for r in rows:
        uid = r[2]
        date = r[1]
        k = (uid, date)
        depth = user_date_counts.get(k, 0)
        user_date_counts[k] = depth + 1
        
        if depth >= 10:
            depth_str = '10+'
        elif depth >= 6:
            depth_str = '6-9'
        else:
            depth_str = str(depth)
            
        hour_str = str(r[8] // 100)
        
        new_rows.append(r + (hour_str, depth_str))
        
    new_rows.sort(key=lambda x: x[0])
    final_rows = [x[1:] for x in new_rows]
    
    return {name: [x for x in final_rows if lo <= x[0] <= hi] for name, (lo, hi) in SPLITS.items()}


# ----------------------------------------------------------------------------- [3] feature encoding (= starter_kit/data.py)
def _bucket_edges(durations, n=N_DUR_BUCKETS):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


def encode(splits):
    """Categorical ids -> contiguous ints; unseen values fall into a per-field UNK slot.
    Returns ({split: (X int32 (N,F), y float32, users)}, total_dim)."""
    tr = splits["train"]
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5]))), x[9], x[10]]

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
def train(enc, dim, log=print, loss_type="pairwise", seed=SEED):
    """Train on train, early-stop on validation primary (official recipe). Returns the best model."""
    Xtr, ytr, utr_list = enc["train"]
    Xva, yva, uva = enc["valid"]
    m = FM(dim, seed=seed)
    rng = np.random.default_rng(seed)
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
    
    scores_list = []
    X, _, _ = enc[split]
    Xva, yva, uva = enc["valid"]
    va_preds = None

    num_seeds = 1 if fast_mode else 5
    for seed in range(num_seeds):
        model = train(enc, dim, loss_type="pairwise", seed=seed)
        scores_list.append(model.predict(X))
        if seed == 0:
            va_preds = model.predict(Xva)
            
    scores = np.mean(scores_list, axis=0)
    assert np.all(np.isfinite(scores)), "non-finite scores"
    write_preds(a.out, splits[split], scores)
    print(f"wrote {a.out}: {len(splits[split])} rows for split={split} in {time.time() - t_start:.0f}s")

    if fast_mode:
        return
        
    remains = time_budget - (time.time() - t_start)
    if remains < time_budget * 0.25:
        print("ABLATION 1_seed skipped: out of time budget")
    else:
        va = evaluate(uva, yva, va_preds)
        print(f"ABLATION 1_seed primary={va['primary']:.6f} gauc={va['GAUC']:.6f} ndcg5={va['nDCG@5']:.6f}")


if __name__ == "__main__":
    main()
```
