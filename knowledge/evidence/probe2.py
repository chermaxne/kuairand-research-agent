"""Probe 2: within-user objectives and multi-task heads. All features train-only/past-only."""
import pandas as pd, numpy as np, time, sys, json, os
sys.path.insert(0, ".")
from agent import tools
ROOT = "."; D = f"{ROOT}/starter_kit/KuaiRand-Pure/data"
ev = tools.import_sealed_evaluate(f"{ROOT}/sealed")
t0 = time.time()
a = pd.read_csv(f"{D}/log_standard_4_08_to_4_21_pure.csv"); b = pd.read_csv(f"{D}/log_standard_4_22_to_5_08_pure.csv")
log = pd.concat([a, b], ignore_index=True); log = log[log.date <= 20220428].reset_index(drop=True)
vb = pd.read_csv(f"{D}/video_features_basic_pure.csv")[["video_id", "author_id", "tag", "music_id"]]
log = log.merge(vb, on="video_id", how="left")
log["is_tr"] = log.date <= 20220421
tr, va = log[log.is_tr], log[~log.is_tr]
res = {}
def score(name, s):
    r = ev(va.user_id.astype(str).tolist(), va.long_view.tolist(), list(map(float, s)))
    res[name] = round(r["primary"], 4); print(f"{name:<52} primary={r['primary']:.4f} GAUC={r['GAUC']:.4f} nDCG5={r['nDCG@5']:.4f}", flush=True)
gmean = tr.long_view.mean(); prior = 20
def tr_rate(keys, name):
    g = tr.groupby(keys).long_view.agg(["sum", "size"]); rate = (g["sum"] + prior * gmean) / (g["size"] + prior)
    return va[keys].merge(rate.rename(name + "_r").reset_index(), on=keys, how="left")[name + "_r"].fillna(gmean).values, \
           tr[keys].merge(rate.rename(name + "_r").reset_index(), on=keys, how="left")[name + "_r"].fillna(gmean).values
va_tab, tr_tab = tr_rate(["tab"], "tab_rate"); score("global tab rate (train)", va_tab)
va_ut, _ = tr_rate(["user_id", "tab"], "ut"); score("user x tab rate (train-only, not cumulative)", va_ut)
va_v, tr_v = tr_rate(["video_id"], "v"); va_au, tr_au = tr_rate(["author_id"], "au"); va_tag, tr_tag = tr_rate(["tag"], "tag")
va_ua, tr_ua = tr_rate(["user_id", "author_id"], "ua"); va_utag, tr_utag = tr_rate(["user_id", "tag"], "utag")
# how often does tab vary within a user's validation impressions?
nt = va.groupby("user_id").tab.nunique(); print("valid users with >1 tab:", round(float((nt > 1).mean()), 3), "| valid rows in tab-1:", round(float((va.tab == 1).mean()), 3))
# ---- LightGBM lambdarank grouped by user (within-user objective) on item-side + user x item features only ----
import lightgbm as lgb
def feats(df, v, au, tag, ua, utag, ut):
    return pd.DataFrame({"v_rate": v, "au_rate": au, "tag_rate": tag, "ua_rate": ua, "utag_rate": utag, "ut_rate": ut,
                         "tab": df.tab.values, "duration_ms": df.duration_ms.values, "hour": (df.hourmin // 100).values})
_, tr_ut = tr_rate(["user_id", "tab"], "ut")
# train-row features must be past-only: use leave-one-out train rates (subtract the row itself) to avoid self-label leakage
def loo(keys, name):
    g = tr.groupby(keys).long_view.agg(["sum", "size"]).rename(columns={"sum": "s", "size": "n"}).reset_index()
    m = tr[keys].merge(g, on=keys, how="left")
    return ((m.s.values - tr.long_view.values) + prior * gmean) / ((m.n.values - 1) + prior)
Xtr = feats(tr, loo(["video_id"], "v"), loo(["author_id"], "au"), loo(["tag"], "tag"), loo(["user_id", "author_id"], "ua"), loo(["user_id", "tag"], "utag"), loo(["user_id", "tab"], "ut"))
Xva = feats(va, va_v, va_au, va_tag, va_ua, va_utag, va_ut)
trs = tr.sort_values("user_id", kind="stable"); order = trs.index
Xtr_s, ytr_s = Xtr.loc[order], trs.long_view.values; grp = trs.groupby("user_id", sort=False).size().values
m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8)
m.fit(Xtr_s, ytr_s, group=grp)
score("LightGBM lambdarank (user groups), LOO item/user x item feats", m.predict(Xva))
mc = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8)
mc.fit(Xtr, tr.long_view.values); score("LightGBM logloss, same feats (no user-constant feats)", mc.predict_proba(Xva)[:, 1])
print("feature importance (ranker):", sorted(zip(m.feature_importances_, Xtr.columns), reverse=True))
# ---- multi-task FM: shared V; heads = long_view (main), is_click, log1p(play_time) censored-at-duration ----
sys.path.insert(0, f"{ROOT}/baseline_repro"); os.environ["PYTHONPATH"] = f"{ROOT}/sealed"
import importlib.util
spec = importlib.util.spec_from_file_location("pipe", f"{ROOT}/baseline_repro/pipeline.py"); pipe = importlib.util.module_from_spec(spec)
sys.path.insert(0, f"{ROOT}/sealed"); spec.loader.exec_module(pipe)
splits = pipe.load(f"{ROOT}/data_cache/loop_train_valid"); enc, dim = pipe.encode(splits)
Xtr_i, ytr_i, _ = enc["train"]; Xva_i, yva_i, uva = enc["valid"]
# aux targets aligned with the kit's row order (same files, same filtering -> identical order)
assert len(tr) == len(ytr_i)
clk = tr.is_click.values.astype(np.float32); pt = np.log1p(tr.play_time_ms.values).astype(np.float32); pt = pt / pt.max()
cens = (tr.play_time_ms.values >= tr.duration_ms.values)
def run_mt(w_click, w_wt, tag_):
    m = pipe.FM(dim); rng = np.random.default_rng(0)
    W2 = np.zeros(dim, np.float32); b2 = 0.0; W3 = np.zeros(dim, np.float32); b3 = 0.0; a3 = 1.0
    best, best_state, bad = -1, None, 0
    for ep in range(1, 41):
        idx = rng.permutation(len(ytr_i))
        for i in range(0, len(idx), 8192):
            bi = idx[i:i + 8192]; X = Xtr_i[bi]; y = ytr_i[bi]; B = len(bi)
            z, E, S = m.logits(X); inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
            g_main = (pipe.sigmoid(z) - y) / B
            zc = b2 + W2[X].sum(1) + inter; g_c = w_click * (pipe.sigmoid(zc) - clk[bi]) / B
            zw = b3 + W3[X].sum(1) + a3 * inter; r = zw - pt[bi]
            r = np.where(cens[bi] & (r > 0), 0.0, r)                       # censored: no penalty for predicting MORE than a complete play
            g_w = w_wt * 2 * r / B
            gV = np.zeros_like(m.V); gW = np.zeros_like(m.W); gW2 = np.zeros_like(W2); gW3 = np.zeros_like(W3)
            gtot = (g_main + g_c + a3 * g_w).astype(np.float32)
            np.add.at(gW, X, g_main[:, None].astype(np.float32)); np.add.at(gW2, X, g_c[:, None].astype(np.float32)); np.add.at(gW3, X, g_w[:, None].astype(np.float32))
            np.add.at(gV, X, gtot[:, None, None] * (S[:, None, :] - E))
            gV += m.l2 * m.V; gW += m.l2 * m.W
            m.t += 1; b1_, b2_, eps = 0.9, 0.999, 1e-8
            for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
                M *= b1_; M += (1 - b1_) * G; Vv *= b2_; Vv += (1 - b2_) * (G * G)
                P -= m.lr * (M / (1 - b1_ ** m.t)) / (np.sqrt(Vv / (1 - b2_ ** m.t)) + eps)
            m.b -= m.lr * g_main.sum(); W2 -= 0.01 * gW2; b2 -= 0.01 * g_c.sum(); W3 -= 0.01 * gW3; b3 -= 0.01 * g_w.sum()
        p = ev(uva, yva_i, m.predict(Xva_i))["primary"]
        if p > best + 1e-5: best, bad, best_state = p, 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 4: break
    m.V, m.W, m.b = best_state
    score(f"multi-task FM {tag_} (best ep {ep - bad})", m.predict(Xva_i))
run_mt(0.0, 0.0, "control: no aux heads")
run_mt(0.3, 0.0, "+ is_click head w=0.3")
run_mt(0.0, 0.5, "+ censored watch-time head w=0.5")
run_mt(0.3, 0.5, "+ click 0.3 + watch-time 0.5")
print(json.dumps(res)); print(f"total {time.time()-t0:.0f}s")
