"""Empirical probes on validation (train-only / past-only features): GBDT stacking, user x author rates, ensembles."""
import pandas as pd, numpy as np, time, sys, json, os
sys.path.insert(0, "/Users/ckwang/Documents/TechJam/kuairand-starter-kit")
from agent import tools
ROOT = "/Users/ckwang/Documents/TechJam/kuairand-starter-kit"
D = f"{ROOT}/starter_kit/KuaiRand-Pure/data"
ev = tools.import_sealed_evaluate(f"{ROOT}/sealed")
t0 = time.time()
a = pd.read_csv(f"{D}/log_standard_4_08_to_4_21_pure.csv"); b = pd.read_csv(f"{D}/log_standard_4_22_to_5_08_pure.csv")
log = pd.concat([a, b], ignore_index=True); log = log[log.date <= 20220428].reset_index(drop=True)   # loop data only
vb = pd.read_csv(f"{D}/video_features_basic_pure.csv")[["video_id", "author_id", "upload_dt", "video_duration", "music_id", "tag", "video_type"]]
log = log.merge(vb, on="video_id", how="left")
log["is_tr"] = log.date <= 20220421
log["dur_b"] = pd.qcut(log.duration_ms, 10, labels=False, duplicates="drop")
log["upload_days"] = (pd.to_datetime(log.date.astype(str)) - pd.to_datetime(log.upload_dt, errors="coerce")).dt.days
dates = sorted(log.date.unique())
prior = 20.0
gmean = log.loc[log.is_tr, "long_view"].mean()

def past_only(keys, name):
    """rate & count for each row from strictly earlier dates (train rows) or from all train (valid rows)."""
    g = log.groupby(keys + ["date"]).agg(n=("long_view", "size"), p=("long_view", "sum")).reset_index()
    g = g.sort_values(keys + ["date"])
    g["cn"] = g.groupby(keys)["n"].cumsum() - g["n"]; g["cp"] = g.groupby(keys)["p"].cumsum() - g["p"]   # strictly before this date
    m = log[keys + ["date"]].merge(g[keys + ["date", "cn", "cp"]], on=keys + ["date"], how="left")
    cn, cp = m.cn.fillna(0).values, m.cp.fillna(0).values
    # valid rows: everything from train (all dates <= 0421), which the cumulative already gives since valid dates are later,
    # except rows on a date with no earlier record of that key -> 0 counts (correct: unseen)
    log[f"{name}_n"] = cn
    log[f"{name}_rate"] = (cp + prior * gmean) / (cn + prior)

for keys, name in [(["user_id"], "u"), (["video_id"], "v"), (["author_id"], "au"), (["user_id", "author_id"], "ua"),
                   (["user_id", "tab"], "ut"), (["user_id", "dur_b"], "ud"), (["tab"], "t"), (["user_id", "tag"], "utag")]:
    past_only(keys, name)
# recency: days since the user's previous impression / since the video's first impression (past-only by construction)
first_v = log.groupby("video_id").date.transform("min")
log["v_age_days"] = (pd.to_datetime(log.date.astype(str)) - pd.to_datetime(first_v.astype(str))).dt.days
feats = [c for c in log.columns if c.endswith("_n") or c.endswith("_rate")] + ["v_age_days", "upload_days", "duration_ms", "tab", "dur_b", "hourmin"]
print("features:", feats, f"({time.time()-t0:.0f}s)")
tr, va = log[log.is_tr], log[~log.is_tr]
# the champion FM validation predictions from the last run's it00 (baseline) for stacking/ensembling
fm_path = sorted([p for p in [f"{ROOT}/runs/20260828_222721_ten/phase0/champion_check/preds_val.csv"] if os.path.exists(p)])
fm = pd.read_csv(fm_path[0]).score.values if fm_path else None
res = {}
def score(name, s):
    r = ev(va.user_id.astype(str).tolist(), va.long_view.tolist(), list(map(float, s)))
    res[name] = round(r["primary"], 4); print(f"{name:<40} primary={r['primary']:.4f} GAUC={r['GAUC']:.4f} nDCG5={r['nDCG@5']:.4f}", flush=True)
if fm is not None: score("FM champion (it00)", fm)
score("video past-only rate", va.v_rate.values)
score("user x author rate", va.ua_rate.values)
score("user x tab rate", va.ut_rate.values)
import lightgbm as lgb
X, y = tr[feats].astype(float), tr.long_view.values
Xv = va[feats].astype(float)
# train rows on the first days have empty histories: drop the first 3 train days (cold history) from the GBDT fit
mask = tr.date >= 20220411
m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, verbose=-1, n_jobs=8)
m.fit(X[mask], y[mask], eval_set=[(Xv, va.long_view.values)], callbacks=[lgb.early_stopping(50, verbose=False)])
p_gbdt = m.predict_proba(Xv)[:, 1]
score(f"LightGBM past-only feats (it={m.best_iteration_})", p_gbdt)
imp = sorted(zip(m.feature_importances_, feats), reverse=True)[:10]; print("top features:", imp)
if fm is not None:
    def rank_in_user(s):
        return pd.Series(s).groupby(va.user_id.values).rank(pct=True).values
    score("rank-avg ensemble FM + LightGBM", 0.5 * rank_in_user(fm) + 0.5 * rank_in_user(p_gbdt))
    X2, Xv2 = X.copy(), Xv.copy()
    # FM score as a feature for VALID only is not available for train rows without OOF; skip stacking-with-FM here
print(json.dumps(res)); print(f"total {time.time()-t0:.0f}s")
