import pandas as pd, numpy as np, json, time
D = "/Users/ckwang/Documents/TechJam/kuairand-starter-kit/starter_kit/KuaiRand-Pure/data"
t0 = time.time()
a = pd.read_csv(f"{D}/log_standard_4_08_to_4_21_pure.csv")
b = pd.read_csv(f"{D}/log_standard_4_22_to_5_08_pure.csv")
log = pd.concat([a, b], ignore_index=True)
log["split"] = np.where(log.date <= 20220421, "train", np.where(log.date <= 20220428, "valid", "test"))
tr, va = log[log.split == "train"], log[log.split == "valid"]
out = {}
out["rows"] = log.split.value_counts().to_dict()
fb = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate", "long_view", "is_profile_enter"]
out["train_rates"] = {c: round(float(tr[c].mean()), 4) for c in fb}
out["valid_rates"] = {c: round(float(va[c].mean()), 4) for c in fb}
# ---- label definition: long_view vs play ratio ----
tr2 = tr[tr.duration_ms > 0].copy()
tr2["ratio"] = tr2.play_time_ms / tr2.duration_ms
pos, neg = tr2[tr2.long_view == 1], tr2[tr2.long_view == 0]
out["long_view_vs_play"] = {
    "pos_ratio_min": round(float(pos.ratio.min()), 4), "pos_play_ms_min": int(pos.play_time_ms.min()),
    "neg_ratio_max": round(float(neg.ratio.max()), 4), "neg_play_ms_max": int(neg.play_time_ms.max()),
    "pos_ratio_p01_p50": [round(float(pos.ratio.quantile(q)), 3) for q in (0.01, 0.5)],
    "neg_ratio_p50_p99": [round(float(neg.ratio.quantile(q)), 3) for q in (0.5, 0.99)],
}
# test simple rules
for name, rule in {"ratio>=0.5": tr2.ratio >= 0.5, "play>=18s": tr2.play_time_ms >= 18000,
                   "play>=18s_or_ratio>=0.5": (tr2.play_time_ms >= 18000) | (tr2.ratio >= 0.5),
                   "ratio>=0.5_short_or_18s_long": np.where(tr2.duration_ms <= 18000*2, tr2.ratio >= 0.5, tr2.play_time_ms >= 18000)}.items():
    out["long_view_vs_play"][f"rule_{name}_agree"] = round(float((rule.astype(int) == tr2.long_view).mean()), 4)
# ---- play time / duration ----
out["duration_ms_quantiles"] = {q: int(tr.duration_ms.quantile(q)) for q in (0.1, 0.5, 0.9, 0.99)}
out["play_time_ms_quantiles"] = {q: int(tr.play_time_ms.quantile(q)) for q in (0.1, 0.5, 0.9, 0.99)}
out["play_ratio_censored_share_train"] = round(float((tr2.play_time_ms >= tr2.duration_ms).mean()), 4)
# long_view by duration bucket
tr2["dur_b"] = pd.qcut(tr2.duration_ms, 5, duplicates="drop")
out["long_view_rate_by_duration_quintile"] = {str(k): round(float(v), 3) for k, v in tr2.groupby("dur_b", observed=True).long_view.mean().items()}
# ---- users / items / cold start ----
out["users"] = {"train": int(tr.user_id.nunique()), "valid": int(va.user_id.nunique()), "valid_not_in_train": int((~va.user_id.isin(tr.user_id.unique())).sum())}
out["videos"] = {"train": int(tr.video_id.nunique()), "valid": int(va.video_id.nunique()),
                 "valid_rows_with_video_unseen_in_train": round(float((~va.video_id.isin(tr.video_id.unique())).mean()), 4)}
imp = tr.groupby("user_id").size()
out["train_impressions_per_user"] = {"median": int(imp.median()), "p10": int(imp.quantile(0.1)), "p90": int(imp.quantile(0.9)), "max": int(imp.max())}
vimp = tr.groupby("video_id").size()
out["train_impressions_per_video"] = {"median": int(vimp.median()), "p10": int(vimp.quantile(0.1)), "p90": int(vimp.quantile(0.9)), "max": int(vimp.max())}
vu = va.groupby("user_id").size()
out["valid_impressions_per_user"] = {"median": int(vu.median()), "p10": int(vu.quantile(0.1)), "p90": int(vu.quantile(0.9))}
ur = va.groupby("user_id").long_view.mean()
out["valid_user_composition"] = {"all_negative": round(float((ur == 0).mean()), 3), "all_positive": round(float((ur == 1).mean()), 3), "mixed": round(float(((ur > 0) & (ur < 1)).mean()), 3)}
# ---- user-level consistency: does a user's train rate predict their valid rate? (calibration signal, not ranking)
utr = tr.groupby("user_id").long_view.mean()
common = ur.index.intersection(utr.index)
out["corr_user_train_rate_vs_valid_rate"] = round(float(np.corrcoef(utr[common], ur[common])[0, 1]), 3)
# ---- item-level: does a video's train rate predict its valid rate?
vtr = tr.groupby("video_id").long_view.agg(["mean", "size"]); vva = va.groupby("video_id").long_view.agg(["mean", "size"])
c = vtr.join(vva, lsuffix="_tr", rsuffix="_va", how="inner"); c = c[(c.size_tr >= 20) & (c.size_va >= 20)]
out["corr_video_train_rate_vs_valid_rate_(n>=20)"] = round(float(np.corrcoef(c.mean_tr, c.mean_va)[0, 1]), 3)
# ---- context signals ----
out["tab_share_train"] = {str(k): round(float(v), 3) for k, v in tr.tab.value_counts(normalize=True).head(6).items()}
out["long_view_rate_by_tab"] = {str(k): round(float(v), 3) for k, v in tr.groupby("tab").long_view.mean().head(6).items()}
tr3 = tr.copy(); tr3["hour"] = tr3.hourmin // 100
out["long_view_rate_by_hour_minmax"] = [round(float(tr3.groupby("hour").long_view.mean().min()), 3), round(float(tr3.groupby("hour").long_view.mean().max()), 3)]
out["long_view_rate_by_date"] = {str(k): round(float(v), 3) for k, v in log.groupby("date").long_view.mean().items()}
out["is_rand_share_standard_log"] = round(float(log.is_rand.mean()), 4)
# ---- feedback relationships ----
out["p_long_view_given_click"] = round(float(tr[tr.is_click == 1].long_view.mean()), 3)
out["p_long_view_given_noclick"] = round(float(tr[tr.is_click == 0].long_view.mean()), 3)
out["p_long_view_given_like"] = round(float(tr[tr.is_like == 1].long_view.mean()), 3)
out["corr_click_longview"] = round(float(tr[["is_click", "long_view"]].corr().iloc[0, 1]), 3)
# ---- repeats: same (user, video) pairs across train/valid
pairs_tr = set(zip(tr.user_id, tr.video_id))
out["valid_rows_with_same_user_video_pair_in_train"] = round(float(np.mean([(u, v) in pairs_tr for u, v in zip(va.user_id, va.video_id)])), 4)
# ---- side tables ----
vs = pd.read_csv(f"{D}/video_features_statistic_pure.csv")
vb = pd.read_csv(f"{D}/video_features_basic_pure.csv")
uf = pd.read_csv(f"{D}/user_features_pure.csv")
out["video_basic_cols"] = list(vb.columns)
out["video_stat_cols_head"] = list(vs.columns[:12])
out["video_stat_show_cnt_vs_train_impressions_corr"] = round(float(np.corrcoef(vs.set_index("video_id").loc[vimp.index.intersection(vs.video_id)].show_cnt, vimp[vimp.index.intersection(vs.video_id)])[0, 1]), 3)
out["video_stat_counts_days_median"] = int(vs.counts.median())
out["video_upload_dt_range"] = [str(vb.upload_dt.min()), str(vb.upload_dt.max())]
out["user_feature_cols"] = list(uf.columns)[:14]
out["videos_total"] = int(len(vb)); out["authors"] = int(vb.author_id.nunique())
r = pd.read_csv(f"{D}/log_random_4_22_to_5_08_pure.csv")
rv = r[(r.date >= 20220422) & (r.date <= 20220428)]
out["random_log_valid_period"] = {"rows": int(len(rv)), "users": int(rv.user_id.nunique()), "long_view_rate": round(float(rv.long_view.mean()), 4)}
out["elapsed_s"] = round(time.time() - t0, 1)
json.dump(out, open("stats.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
