"""Probe 3: seed ensembling of the FM, recency weighting, tab-context tweaks."""
import numpy as np, pandas as pd, sys, os, time, json, importlib.util
ROOT = "/Users/ckwang/Documents/TechJam/kuairand-starter-kit"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/sealed")
from agent import tools
ev = tools.import_sealed_evaluate(f"{ROOT}/sealed")
spec = importlib.util.spec_from_file_location("pipe", f"{ROOT}/baseline_repro/pipeline.py"); pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)
t0 = time.time()
splits = pipe.load(f"{ROOT}/data_cache/loop_train_valid"); enc, dim = pipe.encode(splits)
Xtr, ytr, _ = enc["train"]; Xva, yva, uva = enc["valid"]
dates_tr = np.array([x[0] for x in splits["train"]])
res = {}
def score(name, s):
    r = ev(uva, yva, list(map(float, s))); res[name] = round(r["primary"], 4)
    print(f"{name:<48} primary={r['primary']:.4f} GAUC={r['GAUC']:.4f} nDCG5={r['nDCG@5']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
def train_fm(seed, weights=None, k=16, lr=0.001, epochs=40):
    m = pipe.FM(dim, k=k, lr=lr, seed=seed); rng = np.random.default_rng(seed)
    best, state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            bi = idx[i:i + 8192]
            if weights is None:
                m.step(Xtr[bi], ytr[bi])
            else:   # weighted logloss step (recency weighting): scale the per-row gradient
                X = Xtr[bi]; y = ytr[bi]; w = weights[bi]; B = len(bi)
                z, E, S = m.logits(X); g = ((pipe.sigmoid(z) - y) * w / w.mean() / B).astype(np.float32)
                gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
                np.add.at(gW, X, g[:, None]); np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
                gV += m.l2 * m.V; gW += m.l2 * m.W; m.t += 1; b1, b2, eps = 0.9, 0.999, 1e-8
                for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
                    M *= b1; M += (1 - b1) * G; Vv *= b2; Vv += (1 - b2) * (G * G)
                    P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
                m.b -= m.lr * g.sum()
        p = ev(uva, yva, m.predict(Xva))["primary"]
        if p > best + 1e-5: best, bad, state = p, 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 4: break
    m.V, m.W, m.b = state
    return m.predict(Xva)
def rank_in_user(s):
    return pd.Series(s).groupby(np.array(uva)).rank(pct=True).values
preds = []
for seed in range(5):
    p = train_fm(seed); preds.append(p); score(f"FM seed {seed}", p)
score("FM 3-seed rank-average", np.mean([rank_in_user(p) for p in preds[:3]], axis=0))
score("FM 5-seed rank-average", np.mean([rank_in_user(p) for p in preds], axis=0))
score("FM 5-seed logit-average", np.mean(preds, axis=0))
# recency weighting: weight = exp((date_index - last)/tau), tau in days
di = (pd.to_datetime(dates_tr.astype(str)) - pd.Timestamp("2022-04-21")).days.values.astype(np.float32)
for tau in (5.0, 10.0):
    score(f"FM recency-weighted tau={tau:.0f}d (seed 0)", train_fm(0, weights=np.exp(di / tau).astype(np.float32)))
print(json.dumps(res)); print("PROBE3_DONE")
