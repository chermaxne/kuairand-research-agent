"""Evaluator probe (G1/G2/G5): (i) label-free session-position fields inside the FM, (ii) duration-aware buckets,
(iii) FM trained on is_click as a multi-task-by-ensembling check, (iv) LightGBM lambdarank grouped by user on PAST-ONLY
features + a time-split out-of-fold FM score. All scoring: sealed evaluate.py on the validation split (0422-0428).
Data: data_cache/loop_train_valid (test rows already removed)."""
import pandas as pd, numpy as np, time, sys, json, importlib.util
ROOT = "."; D = f"{ROOT}/data_cache/loop_train_valid"
sys.path.insert(0, f"{ROOT}/sealed")
spec = importlib.util.spec_from_file_location("pipe", f"{ROOT}/baseline_repro/pipeline.py"); pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)
ev = pipe.evaluate
t0 = time.time()
a = pd.read_csv(f"{D}/log_standard_4_08_to_4_21_pure.csv"); b = pd.read_csv(f"{D}/log_standard_4_22_to_5_08_pure.csv"); assert b.date.max() <= 20220428
log = pd.concat([a, b], ignore_index=True)
vb = pd.read_csv(f"{D}/video_features_basic_pure.csv")[["video_id", "author_id", "tag"]]
log = log.merge(vb, on="video_id", how="left"); log["author_id"] = log.author_id.fillna(-1).astype(int); log["tag"] = log.tag.fillna("UNK").astype(str)
log["is_tr"] = log.date <= 20220421
# ---- label-free, within-user-varying context features (from the log's non-label columns only) ----
o = log.sort_values(["user_id", "time_ms"], kind="stable")
o["pos_day"] = o.groupby(["user_id", "date"]).cumcount(); o["n_day"] = o.groupby(["user_id", "date"]).time_ms.transform("size")
gap = o.groupby("user_id").time_ms.diff(); o["gap_min"] = (gap / 60000.0).fillna(1e4)
o["new_sess"] = (o.gap_min > 30).astype(int); o["sess_id"] = o.groupby("user_id").new_sess.cumsum(); o["pos_sess"] = o.groupby(["user_id", "sess_id"]).cumcount()
log = o.sort_index()
log["hour"] = log.hourmin // 100
edges_d = [-1, 0, 7000, 10000, 15000, 18000, 25000, 40000, 60000, 120000, 300000, 1e9]
log["dur_fine"] = np.digitize(log.duration_ms.values, edges_d[1:], right=True)      # 0 = duration 0, 1 = <=7s, 2 = <=10s, ... 5 = <=18s, ...
log["pos_b"] = np.digitize(log.pos_day.values, [1, 2, 3, 4, 6, 10])                # 0,1,2,3,4-5,6-9,10+
log["sess_b"] = np.digitize(log.pos_sess.values, [1, 2, 3, 5, 10])
tr_edges = np.quantile(log.loc[log.is_tr, "duration_ms"], np.linspace(0, 1, 11)[1:-1]); log["dur_b"] = np.searchsorted(tr_edges, log.duration_ms.values)
# ---- past-only (strictly earlier dates) smoothed rates & counts; valid rows see all of train ----
prior = 20.0; gmean = log.loc[log.is_tr, "long_view"].mean()
def past_only(keys, name):
    g = log[log.is_tr].groupby(keys + ["date"]).agg(n=("long_view", "size"), p=("long_view", "sum")).reset_index().sort_values(keys + ["date"])
    g["cn"] = g.groupby(keys)["n"].cumsum() - g["n"]; g["cp"] = g.groupby(keys)["p"].cumsum() - g["p"]
    tot = g.groupby(keys).agg(tn=("n", "sum"), tp=("p", "sum")).reset_index()
    m = log[keys + ["date", "is_tr"]].merge(g[keys + ["date", "cn", "cp"]], on=keys + ["date"], how="left").merge(tot, on=keys, how="left")
    cn = np.where(m.is_tr, m.cn.fillna(0), m.tn.fillna(0)); cp = np.where(m.is_tr, m.cp.fillna(0), m.tp.fillna(0))
    log[f"{name}_n"] = cn; log[f"{name}_rate"] = (cp + prior * gmean) / (cn + prior)
for keys, name in [(["video_id"], "v"), (["author_id"], "au"), (["tag"], "tag"), (["user_id", "author_id"], "ua"), (["user_id", "tag"], "utag"), (["user_id", "tab"], "ut"), (["user_id", "dur_b"], "ud"), (["user_id"], "u")]:
    past_only(keys, name)
tr, va = log[log.is_tr], log[~log.is_tr]
print(f"features ready {time.time()-t0:.0f}s; train {len(tr)} valid {len(va)}", flush=True)
res = {}
def score(name, s):
    r = ev(va.user_id.astype(str).tolist(), va.long_view.tolist(), list(map(float, s))); res[name] = round(r["primary"], 4)
    print(f"{name:<62} primary={r['primary']:.4f} GAUC={r['GAUC']:.4f} nDCG5={r['nDCG@5']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
def rank_in_user(s):
    return pd.Series(np.asarray(s)).groupby(va.user_id.values).rank(pct=True).values
# ---- generic FM encoder over arbitrary categorical fields ----
def encode(fields, fit_mask):
    X = np.empty((len(log), len(fields)), np.int32); off = 0
    for i, f in enumerate(fields):
        col = log[f].astype(str).values; vocab = {v: j for j, v in enumerate(pd.unique(col[fit_mask]))}; unk = len(vocab)
        X[:, i] = np.array([vocab.get(v, unk) for v in col], np.int32) + off; off += unk + 1
    return X, off
def train_fm(X, y, fit_mask, seed=0, tag=""):
    Xtr, ytr = X[fit_mask], y[fit_mask].astype(np.float32); Xva = X[(~log.is_tr).values]; uva = va.user_id.astype(str).tolist(); yva = va.long_view.tolist()
    m = pipe.FM(X.max() + 1, seed=seed); rng = np.random.default_rng(seed); best, state, bad = -1, None, 0
    for ep in range(1, 41):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192): m.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]])
        p = ev(uva, yva, m.predict(Xva))["primary"]
        if p > best + 1e-5: best, bad, state = p, 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 4: break
    m.V, m.W, m.b = state; return m
base = ["user_id", "video_id", "author_id", "tab", "dur_b"]; is_tr = log.is_tr.values; y_lv = log.long_view.values
X5, _ = encode(base, is_tr); fm5 = train_fm(X5, y_lv, is_tr); p_fm = fm5.predict(X5[~is_tr]); score("FM5 baseline fields (seed 0)", p_fm)
X6, _ = encode(base + ["pos_b"], is_tr); p6 = train_fm(X6, y_lv, is_tr).predict(X6[~is_tr]); score("FM6 = FM5 + within-day position bucket", p6)
X7, _ = encode(base + ["pos_b", "sess_b", "dur_fine"], is_tr); p7 = train_fm(X7, y_lv, is_tr).predict(X7[~is_tr]); score("FM7 = FM5 + day-pos + session-pos + fine duration (0/7/18s)", p7)
X8, _ = encode(base + ["dur_fine"], is_tr); p8 = train_fm(X8, y_lv, is_tr).predict(X8[~is_tr]); score("FM5 + fine duration bucket only", p8)
p_clk = train_fm(X5, log.is_click.values, is_tr).predict(X5[~is_tr]); score("FM5 trained on is_click (scored on long_view)", p_clk)
score("rank-avg FM5(long_view) + FM5(is_click)", 0.5 * rank_in_user(p_fm) + 0.5 * rank_in_user(p_clk))
score("rank-avg FM5 + FM7", 0.5 * rank_in_user(p_fm) + 0.5 * rank_in_user(p7))
# ---- G2: LightGBM lambdarank grouped by user, PAST-ONLY features, + time-split OOF FM score ----
import lightgbm as lgb
wk1 = is_tr & (log.date.values <= 20220414); wk2 = is_tr & (log.date.values >= 20220415)
fm_w1 = train_fm(X5, y_lv, wk1)                       # trained on week 1 only -> honest score for week-2 train rows
log["fm_oof"] = np.nan; log.loc[wk2, "fm_oof"] = fm_w1.predict(X5[wk2]); log.loc[~is_tr, "fm_oof"] = p_fm
score("FM5 trained on week 1 only", fm_w1.predict(X5[~is_tr]))
feats = [c for c in log.columns if c.endswith("_rate") or c.endswith("_n")] + ["tab", "duration_ms", "dur_fine", "hour", "pos_day", "n_day", "gap_min", "pos_sess"]
feats_noU = [f for f in feats if f not in ("u_rate", "u_n")]
trw = log[wk2].sort_values("user_id", kind="stable"); grp = trw.groupby("user_id", sort=False).size().values
vaf = log[~is_tr]
def ranker(cols, name):
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8)
    m.fit(trw[cols].astype(float), trw.long_view.values, group=grp); p = m.predict(vaf[cols].astype(float)); score(name, p)
    print("   top feats:", sorted(zip(m.feature_importances_, cols), reverse=True)[:8], flush=True); return p
p_r1 = ranker(feats_noU, "LGBM lambdarank(user) past-only feats + session feats, no FM")
score("   rank-avg FM5 + ranker(no FM)", 0.5 * rank_in_user(p_fm) + 0.5 * rank_in_user(p_r1))
p_r2 = ranker(feats_noU + ["fm_oof"], "LGBM lambdarank(user) same + OOF FM score")
score("   rank-avg FM5 + ranker(with FM)", 0.5 * rank_in_user(p_fm) + 0.5 * rank_in_user(p_r2))
mc = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8)
mc.fit(trw[feats_noU + ["fm_oof"]].astype(float), trw.long_view.values); score("LGBM logloss same feats + OOF FM score", mc.predict_proba(vaf[feats_noU + ["fm_oof"]].astype(float))[:, 1])
print(json.dumps(res, indent=1)); print(f"total {time.time()-t0:.0f}s"); print("EVAL_PROBE_DONE")
