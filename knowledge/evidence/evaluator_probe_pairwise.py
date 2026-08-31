"""Evaluator probe 2 (E3 + seed check of FM6): within-user pairwise (BPR/logistic) fine-tuning of the pointwise FM,
sign-checked with train GAUC; FM6 (baseline + within-day position field) on two more seeds."""
import pandas as pd, numpy as np, time, sys, json, importlib.util, collections
ROOT = "."; D = f"{ROOT}/data_cache/loop_train_valid"
sys.path.insert(0, f"{ROOT}/sealed")
spec = importlib.util.spec_from_file_location("pipe", f"{ROOT}/baseline_repro/pipeline.py"); pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)
ev = pipe.evaluate; t0 = time.time()
a = pd.read_csv(f"{D}/log_standard_4_08_to_4_21_pure.csv"); b = pd.read_csv(f"{D}/log_standard_4_22_to_5_08_pure.csv"); assert b.date.max() <= 20220428
log = pd.concat([a, b], ignore_index=True); vb = pd.read_csv(f"{D}/video_features_basic_pure.csv")[["video_id", "author_id"]]
log = log.merge(vb, on="video_id", how="left"); log["author_id"] = log.author_id.fillna(-1).astype(int); log["is_tr"] = log.date <= 20220421
o = log.sort_values(["user_id", "time_ms"], kind="stable"); o["pos_day"] = o.groupby(["user_id", "date"]).cumcount(); log = o.sort_index()
log["pos_b"] = np.digitize(log.pos_day.values, [1, 2, 3, 4, 6, 10])
tr_edges = np.quantile(log.loc[log.is_tr, "duration_ms"], np.linspace(0, 1, 11)[1:-1]); log["dur_b"] = np.searchsorted(tr_edges, log.duration_ms.values)
is_tr = log.is_tr.values; va = log[~is_tr]; uva = va.user_id.astype(str).tolist(); yva = va.long_view.tolist(); y_lv = log.long_view.values.astype(np.float32)
res = {}
def score(name, s):
    r = ev(uva, yva, list(map(float, s))); res[name] = round(r["primary"], 4)
    print(f"{name:<60} primary={r['primary']:.4f} GAUC={r['GAUC']:.4f} nDCG5={r['nDCG@5']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
def encode(fields):
    X = np.empty((len(log), len(fields)), np.int32); off = 0
    for i, f in enumerate(fields):
        col = log[f].astype(str).values; vocab = {v: j for j, v in enumerate(pd.unique(col[is_tr]))}; unk = len(vocab)
        X[:, i] = np.array([vocab.get(v, unk) for v in col], np.int32) + off; off += unk + 1
    return X
def train_fm(X, seed=0, m=None):
    Xtr, ytr, Xva = X[is_tr], y_lv[is_tr], X[~is_tr]
    m = m or pipe.FM(X.max() + 1, seed=seed); rng = np.random.default_rng(seed); best, state, bad = -1, None, 0
    for ep in range(1, 41):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192): m.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]])
        p = ev(uva, yva, m.predict(Xva))["primary"]
        if p > best + 1e-5: best, bad, state = p, 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 4: break
    m.V, m.W, m.b = state; return m
base = ["user_id", "video_id", "author_id", "tab", "dur_b"]
X6 = encode(base + ["pos_b"])
for seed in (1, 2): score(f"FM6 (+within-day position field) seed {seed}", train_fm(X6, seed).predict(X6[~is_tr]))
# ---- within-user pairwise fine-tuning ----
X5 = encode(base); Xtr, ytr = X5[is_tr], y_lv[is_tr]; utr = log.user_id.values[is_tr]
# pair pools: positives and negatives of users with BOTH labels
df = pd.DataFrame({"u": utr, "y": ytr, "i": np.arange(len(ytr))}); s = df.groupby("u").y.agg(["sum", "size"]); mixed = s[(s["sum"] > 0) & (s["sum"] < s["size"])].index
df = df[df.u.isin(mixed)]; pos = df[df.y == 1].sort_values("u"); neg = df[df.y == 0].sort_values("u")
neg_start = neg.groupby("u").i.apply(lambda x: 0).index; ns = neg.groupby("u").size(); starts = np.cumsum(np.r_[0, ns.values[:-1]]); u2s = dict(zip(ns.index, starts)); u2n = dict(zip(ns.index, ns.values))
pos_i = pos.i.values; pos_s = pos.u.map(u2s).values; pos_n = pos.u.map(u2n).values; neg_i = neg.i.values
print(f"pairs/epoch {len(pos_i)} (positives of {len(mixed)} mixed train users; {len(mixed)/len(s):.2f} of train users are mixed)", flush=True)
def train_gauc(m, n=200000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.choice(len(ytr), n, replace=False); return ev(utr[idx].astype(str).tolist(), ytr[idx].tolist(), m.predict(Xtr[idx]))["GAUC"]
def pair_step(m, ip, ineg, lr):
    Xp, Xn = Xtr[ip], Xtr[ineg]; zp, Ep, Sp = m.logits(Xp); zn, En, Sn = m.logits(Xn)
    g = (pipe.sigmoid(zn - zp) / len(ip)).astype(np.float32)          # d softplus(zn - zp) / d(zn - zp)
    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    np.add.at(gW, Xn, g[:, None]); np.add.at(gW, Xp, -g[:, None])
    np.add.at(gV, Xn, g[:, None, None] * (Sn[:, None, :] - En)); np.add.at(gV, Xp, -g[:, None, None] * (Sp[:, None, :] - Ep))
    gV += m.l2 * m.V; gW += m.l2 * m.W; m.t += 1; b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G; Vv *= b2; Vv += (1 - b2) * (G * G); P -= lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    return float(np.mean(np.log1p(np.exp(zn - zp))))
def pairwise(m, lr, tag, epochs=15, seed=0, mix_pointwise=False):
    rng = np.random.default_rng(seed); best, state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(pos_i)); ip = pos_i[perm]; ineg = neg_i[pos_s[perm] + (rng.random(len(perm)) * pos_n[perm]).astype(int)]
        losses = [pair_step(m, ip[i:i + 8192], ineg[i:i + 8192], lr) for i in range(0, len(ip), 8192)]
        if mix_pointwise:
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), 8192): m.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]])
        r = ev(uva, yva, m.predict(X5[~is_tr])); p = r["primary"]
        if ep == 1: print(f"   [{tag}] epoch 1: pair loss {np.mean(losses):.4f} train GAUC {train_gauc(m):.4f} valid primary {p:.4f}", flush=True)
        if p > best + 1e-5: best, bad, state = p, 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 4: break
    m.V, m.W, m.b = state; return m
fm = train_fm(X5, 0); p_fm = fm.predict(X5[~is_tr]); score("FM5 pointwise (seed 0)", p_fm)
import copy
m1 = pairwise(copy.deepcopy(fm), 0.001, "warm pairwise lr1e-3"); score("FM5 -> within-user pairwise fine-tune (lr 1e-3)", m1.predict(X5[~is_tr]))
m2 = pairwise(copy.deepcopy(fm), 0.0003, "warm pairwise lr3e-4"); score("FM5 -> within-user pairwise fine-tune (lr 3e-4)", m2.predict(X5[~is_tr]))
m3 = pairwise(copy.deepcopy(fm), 0.001, "warm pair+pointwise", mix_pointwise=True); score("FM5 -> pairwise + pointwise epochs interleaved (lr 1e-3)", m3.predict(X5[~is_tr]))
m4 = pairwise(pipe.FM(X5.max() + 1, seed=0), 0.001, "scratch pairwise", epochs=25); score("pairwise from scratch (lr 1e-3)", m4.predict(X5[~is_tr]))
def rank_in_user(s): return pd.Series(np.asarray(s)).groupby(va.user_id.values).rank(pct=True).values
score("rank-avg FM5 pointwise + pairwise fine-tune (lr 3e-4)", 0.5 * rank_in_user(p_fm) + 0.5 * rank_in_user(m2.predict(X5[~is_tr])))
print(json.dumps(res, indent=1)); print(f"total {time.time()-t0:.0f}s"); print("EVAL_PROBE2_DONE")
