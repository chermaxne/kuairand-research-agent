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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
    Returns ({split: (X int32 (N,F), Xnum float32 (N,4), y float32, users)}, total_dim)."""
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
    
    user_stats = {}
    vid_stats = {}
    
    for name in ["train", "valid", "test"]:
        if name not in splits:
            continue
        rws = splits[name]
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        Xnum = np.empty((len(rws), 4), dtype=np.float32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            uid, vid, label = x[1], x[2], x[6]
            
            u_imp, u_lv = user_stats.get(uid, [0, 0])
            v_imp, v_lv = vid_stats.get(vid, [0, 0])
            
            Xnum[n, 0] = np.log1p(u_imp)
            Xnum[n, 1] = u_lv / u_imp if u_imp > 0 else 0.0
            Xnum[n, 2] = np.log1p(v_imp)
            Xnum[n, 3] = v_lv / v_imp if v_imp > 0 else 0.0
            
            if uid not in user_stats: user_stats[uid] = [0, 0]
            user_stats[uid][0] += 1
            user_stats[uid][1] += label
            
            if vid not in vid_stats: vid_stats[vid] = [0, 0]
            vid_stats[vid][0] += 1
            vid_stats[vid][1] += label
            
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = label
            users.append(uid)
        enc[name] = (X, Xnum, y, users)

    Xnum_tr = enc["train"][1]
    mean = Xnum_tr.mean(axis=0)
    std = Xnum_tr.std(axis=0)
    std[std == 0] = 1.0
    for name in enc:
        enc[name] = (enc[name][0], (enc[name][1] - mean) / std, enc[name][2], enc[name][3])
        
    return enc, int(sum(field_dims))


# ----------------------------------------------------------------------------- [4] model
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


class DeepFM(nn.Module):
    def __init__(self, dim, k, use_mlp=True, use_num=True):
        super().__init__()
        self.V = nn.Embedding(dim, k)
        self.W = nn.Embedding(dim, 1)
        self.b = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.V.weight, std=0.01)
        nn.init.normal_(self.W.weight, std=0.01)
        self.use_mlp = use_mlp
        self.use_num = use_num
        if self.use_mlp:
            mlp_in = 5 * k + (4 if self.use_num else 0)
            self.mlp = nn.Sequential(
                nn.Linear(mlp_in, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

    def forward(self, X, Xnum):
        E = self.V(X)
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm = self.W(X).sum(1).squeeze(-1) + inter + self.b
        
        if self.use_mlp:
            B = X.shape[0]
            parts = [E.view(B, -1)]
            if self.use_num:
                parts.append(Xnum)
            mlp_part = self.mlp(torch.cat(parts, dim=1)).squeeze(-1)
            return fm + mlp_part
        return fm


# ----------------------------------------------------------------------------- [5] training
def train_numpy_model(enc, dim, epochs=EPOCHS, subsample=False, log=lambda *a: None):
    Xtr, _, ytr, _ = enc["train"]
    Xva, _, yva, uva = enc["valid"]
    if subsample:
        n_sub = len(ytr) // 4
        Xtr, ytr = Xtr[:n_sub], ytr[:n_sub]
    m = FM(dim)
    rng = np.random.default_rng(SEED)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        losses = [m.step(Xtr[idx[i:i + BATCH]], ytr[idx[i:i + BATCH]]) for i in range(0, len(idx), BATCH)]
        preds = m.predict(Xva)
        va = evaluate(uva, yva, preds)
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        m.V, m.W, m.b = best_state
    return m, m.predict(Xva)


def train_pytorch(enc, dim, use_mlp=True, use_num=True, epochs=EPOCHS, subsample=False, log=print):
    torch.manual_seed(SEED)
    Xtr, Xnumtr, ytr, _ = enc["train"]
    Xva, Xnumva, yva, uva = enc["valid"]
    
    if subsample:
        n_sub = len(ytr) // 4
        Xtr, Xnumtr, ytr = Xtr[:n_sub], Xnumtr[:n_sub], ytr[:n_sub]
        
    Xtr_t = torch.tensor(Xtr, dtype=torch.long)
    Xnumtr_t = torch.tensor(Xnumtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    
    Xva_t = torch.tensor(Xva, dtype=torch.long)
    Xnumva_t = torch.tensor(Xnumva, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(Xtr_t, Xnumtr_t, ytr_t), batch_size=BATCH, shuffle=True)
    
    model = DeepFM(dim, K, use_mlp=use_mlp, use_num=use_num)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    criterion = nn.BCEWithLogitsLoss()
    
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for Xb, Xnumb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb, Xnumb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        model.eval()
        with torch.no_grad():
            preds_va = []
            for i in range(0, len(Xva_t), BATCH):
                preds_va.append(model(Xva_t[i:i+BATCH], Xnumva_t[i:i+BATCH]))
            preds_va = torch.cat(preds_va).numpy()
            
        va = evaluate(uva, yva, preds_va)
        log(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
            f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"early stop at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_pytorch(model, X, Xnum, bs=200_000):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.long)
    Xnumt = torch.tensor(Xnum, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            preds.append(model(Xt[i:i+bs], Xnumt[i:i+bs]))
    return torch.cat(preds).numpy()


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
    
    t_budget = float(os.environ.get("KUAIRAND_TIME_BUDGET_S", 1500))
    t0 = time.time()
    
    split = "valid" if a.split in ("val", "valid") else "test"
    splits = load(a.data)
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}", f"load {time.time() - t0:.1f}s")
    
    enc, dim = encode(splits)
    
    # Fit FULL bundle
    model = train_pytorch(enc, dim, use_mlp=True, use_num=True, epochs=EPOCHS, subsample=False)
    X, Xnum, _, _ = enc[split]
    scores = predict_pytorch(model, X, Xnum)
    assert np.all(np.isfinite(scores)), "non-finite scores"
    write_preds(a.out, splits[split], scores)
    print(f"wrote {a.out}: {len(splits[split])} rows for split={split} in {time.time() - t0:.0f}s")
    
    if os.environ.get("KUAIRAND_FAST", "0") == "1":
        return

    # Ablations
    def run_ab(name, fn):
        if time.time() - t0 > t_budget * 0.75:
            print(f"ABLATION {name} skipped: out of time budget")
            return
        _, preds = fn()
        va = evaluate(enc["valid"][3], enc["valid"][2], preds)
        print(f"ABLATION {name} primary={va['primary']:.4f} gauc={va['GAUC']:.4f} ndcg5={va['nDCG@5']:.4f}")

    run_ab("champion_equiv", lambda: train_numpy_model(enc, dim, epochs=5, subsample=True))
    
    def run_pt(use_mlp, use_num):
        m = train_pytorch(enc, dim, use_mlp, use_num, epochs=5, subsample=True, log=lambda *a: None)
        return m, predict_pytorch(m, enc["valid"][0], enc["valid"][1])

    run_ab("pytorch_fm_only", lambda: run_pt(use_mlp=False, use_num=False))
    run_ab("deepfm_no_num", lambda: run_pt(use_mlp=True, use_num=False))

if __name__ == "__main__":
    main()
