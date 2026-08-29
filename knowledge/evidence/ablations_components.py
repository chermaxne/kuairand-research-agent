"""One-component-at-a-time ablations on top of the proven pairwise+position champion.
Base = pairwise within-user loss from scratch, 5 baseline fields + within-day position field, 3-seed rank-average.
Each variant changes exactly ONE thing. All features label-free or past-only; scored by sealed evaluate on valid."""
import numpy as np, pandas as pd, sys, os, time, json, importlib.util, hashlib
import concurrent.futures as cf
ROOT = "/Users/ckwang/Documents/TechJam/kuairand-starter-kit"; sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/sealed")
from agent import tools
ev = tools.import_sealed_evaluate(f"{ROOT}/sealed")
spec = importlib.util.spec_from_file_location("pipe", f"{ROOT}/baseline_repro/pipeline.py"); pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)
D = f"{ROOT}/data_cache/loop_train_valid"
t0 = time.time()
a = pd.read_csv(f"{D}/log_standard_4_08_to_4_21_pure.csv"); b = pd.read_csv(f"{D}/log_standard_4_22_to_5_08_pure.csv")
log = pd.concat([a, b], ignore_index=True)
vb = pd.read_csv(f"{D}/video_features_basic_pure.csv")[["video_id", "author_id"]]
log = log.merge(vb, on="video_id", how="left"); log["author_id"] = log.author_id.fillna("UNK").astype(str)
is_tr = (log.date <= 20220421).values
log["dur_b"] = pd.qcut(log.duration_ms, 10, labels=False, duplicates="drop").astype(str)
o = log.sort_values(["user_id", "time_ms"], kind="stable")
o["pos_day"] = o.groupby(["user_id", "date"]).cumcount()
gap = o.groupby("user_id").time_ms.diff()
o["new_sess"] = (gap.fillna(1e9) / 60000.0 > 30).astype(int)
log["pos_b"] = np.digitize(o.pos_day.sort_index().values, [1, 2, 3, 4, 6, 10]).astype(str)
log["sess_b"] = o.new_sess.sort_index().values.astype(str)
log["hour_b"] = (log.hourmin // 100 // 3).astype(str)                      # 8 three-hour buckets
log["vtab"] = log.video_id.astype(str) + "|" + log.tab.astype(str)         # video x tab cross
gmean = log.long_view[is_tr].mean()
g = log[is_tr].groupby(["user_id", "author_id"]).long_view.agg(["sum", "size"])
ua = ((g["sum"] + 20 * gmean) / (g["size"] + 20)).rename("ua_r").reset_index()
m = log[["user_id", "author_id"]].merge(ua, on=["user_id", "author_id"], how="left").ua_r.fillna(gmean)
log["ua_b"] = pd.qcut(m, 10, labels=False, duplicates="drop").astype(str)  # past-only-ish (train-derived) user x author rate
y = log.long_view.values.astype(np.float32); users = log.user_id.astype(str).values
va_users, va_y = users[~is_tr], y[~is_tr]
BASE = ["user_id", "video_id", "author_id", "tab", "dur_b", "pos_b"]
def encode(fields):
    X = np.empty((len(log), len(fields)), np.int32); off = 0
    for i, f in enumerate(fields):
        col = log[f].astype(str).values
        vocab = {v: j for j, v in enumerate(pd.unique(col[is_tr]))}
        unk = len(vocab)
        X[:, i] = np.array([vocab.get(v, unk) for v in col], np.int32) + off
        off += unk + 1
    return X, off
utr, ytr = users[is_tr], y[is_tr]
df = pd.DataFrame({"u": utr, "y": ytr, "i": np.arange(len(ytr))})
sm = df.groupby("u").y.agg(["sum", "size"]); mixed = sm[(sm["sum"] > 0) & (sm["sum"] < sm["size"])].index
dfm = df[df.u.isin(mixed)]; pos = dfm[dfm.y == 1].sort_values("u"); neg = dfm[dfm.y == 0].sort_values("u")
ns = neg.groupby("u").size(); starts = np.cumsum(np.r_[0, ns.values[:-1]])
pos_i = pos.i.values; pos_s = pos.u.map(dict(zip(ns.index, starts))).values; pos_n = pos.u.map(dict(zip(ns.index, ns.values))).values; neg_i = neg.i.values
def pair_step(m_, Xtr, ip, ineg, lr):
    Xp, Xn = Xtr[ip], Xtr[ineg]; zp, Ep, Sp = m_.logits(Xp); zn, En, Sn = m_.logits(Xn)
    gg = (pipe.sigmoid(zn - zp) / len(ip)).astype(np.float32)
    gV = np.zeros_like(m_.V); gW = np.zeros_like(m_.W)
    np.add.at(gW, Xn, gg[:, None]); np.add.at(gW, Xp, -gg[:, None])
    np.add.at(gV, Xn, gg[:, None, None] * (Sn[:, None, :] - En)); np.add.at(gV, Xp, -gg[:, None, None] * (Sp[:, None, :] - Ep))
    gV += m_.l2 * m_.V; gW += m_.l2 * m_.W; m_.t += 1; b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((m_.V, gV, m_.mV, m_.vV), (m_.W, gW, m_.mW, m_.vW)):
        M *= b1; M += (1 - b1) * G; Vv *= b2; Vv += (1 - b2) * (G * G); P -= lr * (M / (1 - b1 ** m_.t)) / (np.sqrt(Vv / (1 - b2 ** m_.t)) + eps)
def run(fields, seeds=3, negs=1, patience=4, lr=0.001, epochs=30):
    X, dim = encode(fields); Xtr, Xva = X[is_tr], X[~is_tr]
    preds = []
    for seed in range(seeds):
        m_ = pipe.FM(dim, seed=seed); rng = np.random.default_rng(seed); best, state, bad = -1, None, 0
        for ep in range(1, epochs + 1):
            perm = rng.permutation(len(pos_i))
            ip = np.repeat(pos_i[perm], negs)
            base = np.repeat(pos_s[perm], negs); cnt = np.repeat(pos_n[perm], negs)
            ineg = neg_i[base + (rng.random(len(ip)) * cnt).astype(int)]
            for i in range(0, len(ip), 8192): pair_step(m_, Xtr, ip[i:i + 8192], ineg[i:i + 8192], lr)
            p = ev(va_users.tolist(), va_y.tolist(), m_.predict(Xva))["primary"]
            if p > best + 1e-5: best, bad, state = p, 0, (m_.V.copy(), m_.W.copy(), np.float32(m_.b))
            else:
                bad += 1
                if bad >= patience: break
        m_.V, m_.W, m_.b = state; preds.append(m_.predict(Xva))
    rank = lambda s: pd.Series(s).groupby(va_users).rank(pct=True).values
    avg = np.mean([rank(p) for p in preds], axis=0)
    return ev(va_users.tolist(), va_y.tolist(), avg.tolist())["primary"]
VARIANTS = [
    ("BASE: pairwise + pos field, 3 seeds", dict(fields=BASE)),
    ("+ 5 seeds (instead of 3)", dict(fields=BASE, seeds=5)),
    ("+ hour-of-day bucket field", dict(fields=BASE + ["hour_b"])),
    ("+ 2 negatives per positive", dict(fields=BASE, negs=2)),
    ("+ video x tab cross field", dict(fields=BASE + ["vtab"])),
    ("+ session-gap (new session) field", dict(fields=BASE + ["sess_b"])),
    ("+ user x author rate field (10 buckets)", dict(fields=BASE + ["ua_b"])),
    ("+ patience 6 (instead of 4)", dict(fields=BASE, patience=6)),
    ("+ lower lr 0.0005, patience 6", dict(fields=BASE, lr=0.0005, patience=6)),
]
def job(v):
    name, kw = v; t = time.time()
    try: return name, run(**kw), time.time() - t
    except Exception as e: return name, None, f"{type(e).__name__}: {e}"
res = {}
with cf.ThreadPoolExecutor(3) as ex:
    for name, p, info in ex.map(job, VARIANTS):
        res[name] = p
        print(f"{name:<42} {'FAILED '+str(info) if p is None else f'primary={p:.4f}'}" + (f"  ({info:.0f}s)" if p is not None else ""), flush=True)
base = res.get("BASE: pairwise + pos field, 3 seeds")
print("\n--- deltas vs BASE ---")
for k, v in res.items():
    if v is not None and base is not None and k != "BASE: pairwise + pos field, 3 seeds":
        print(f"{k:<42} {v-base:+.4f}")
json.dump(res, open("/private/tmp/claude-501/-Users-ckwang-Documents-TechJam-kuairand-starter-kit/afbb0d1b-da0b-4c92-b210-73859310e750/scratchpad/ablate/ablations.json", "w"), indent=1)
print(f"total {time.time()-t0:.0f}s"); print("ABLATE_DONE")
