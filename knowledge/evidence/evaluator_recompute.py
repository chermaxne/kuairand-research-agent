"""Independent recomputation of data facts A1-A12/B1-B3 + extra checks. Uses ONLY the masked loop data (no test rows)."""
import pandas as pd, numpy as np, time, sys, importlib.util
ROOT = "."; D = f"{ROOT}/data_cache/loop_train_valid"
spec = importlib.util.spec_from_file_location("ev", f"{ROOT}/sealed/evaluate.py"); evm = importlib.util.module_from_spec(spec); spec.loader.exec_module(evm); ev = evm.evaluate
t0 = time.time()
a = pd.read_csv(f"{D}/log_standard_4_08_to_4_21_pure.csv"); b = pd.read_csv(f"{D}/log_standard_4_22_to_5_08_pure.csv")
assert b.date.max() <= 20220428
vb = pd.read_csv(f"{D}/video_features_basic_pure.csv"); vs = pd.read_csv(f"{D}/video_features_statistic_pure.csv"); rnd = pd.read_csv(f"{D}/log_random_4_22_to_5_08_pure.csv")
tr, va = a, b
print(f"load {time.time()-t0:.1f}s  train rows {len(tr)}  valid rows {len(va)}")
# A1
print("A1 videos(basic file)", vb.video_id.nunique(), "authors", vb.author_id.nunique(), "train users", tr.user_id.nunique(), "train videos", tr.video_id.nunique())
c = tr.video_id.value_counts(); print("A1 train imp/video median,p90,max", c.median(), c.quantile(.9), c.max())
c = tr.user_id.value_counts(); print("A1 train imp/user median,p90,max", c.median(), c.quantile(.9), c.max())
c = va.user_id.value_counts(); print("A1 valid imp/user median,p10,p90,max", c.median(), c.quantile(.1), c.quantile(.9), c.max(), "| users with 1 impression:", round((c==1).mean(),3), "| valid users", va.user_id.nunique())
# A2
tu = set(tr.user_id); vu = va.user_id.unique(); cold = [u for u in vu if u not in tu]
print("A2 valid users unseen in train", len(cold), "/", len(vu), round(len(cold)/len(vu),3), "| rows of cold users", round(va.user_id.isin(cold).mean(),3))
print("A2 valid rows with unseen video", round((~va.video_id.isin(set(tr.video_id))).mean(),4))
pairs_tr = set(zip(tr.user_id, tr.video_id)); print("A2 valid rows whose (user,video) pair is in train", round(np.mean([p in pairs_tr for p in zip(va.user_id, va.video_id)]),4))
# A3
print("A3 upload_dt range", vb.upload_dt.min(), vb.upload_dt.max(), "n distinct", vb.upload_dt.nunique())
# A4
cols = ["is_click","long_view","is_like","is_profile_enter","is_comment","is_follow","is_forward","is_hate"]
print("A4 train rates", tr[cols].mean().round(4).to_dict()); print("A4 valid rates", va[cols].mean().round(4).to_dict())
print("A4 daily lv rate", tr.groupby("date").long_view.mean().round(3).to_dict(), va.groupby("date").long_view.mean().round(3).to_dict())
# A5
g = tr.groupby("video_id").long_view.agg(["mean","size"]); h = va.groupby("video_id").long_view.agg(["mean","size"]); j = g.join(h, lsuffix="_tr", rsuffix="_va", how="inner"); j = j[(j.size_tr>=20)&(j.size_va>=20)]
print("A5 corr video train rate vs valid rate (n>=20 both)", round(j.mean_tr.corr(j.mean_va),3), "n videos", len(j))
j = g.join(h, lsuffix="_tr", rsuffix="_va", how="inner"); j = j[(j.size_tr>=20)]; print("A5 corr (n_tr>=20 only)", round(j.mean_tr.corr(j.mean_va),3))
g = tr.groupby("user_id").long_view.agg(["mean","size"]); h = va.groupby("user_id").long_view.agg(["mean","size"]); j = g.join(h, lsuffix="_tr", rsuffix="_va", how="inner")
print("A5 corr user train rate vs valid rate (all)", round(j.mean_tr.corr(j.mean_va),3), "| n_tr>=20 & n_va>=5:", round(j[(j.size_tr>=20)&(j.size_va>=5)].mean_tr.corr(j[(j.size_tr>=20)&(j.size_va>=5)].mean_va),3))
# A6
s = va.groupby("user_id").long_view.agg(["sum","size"]); print("A6 valid users all-neg", round((s["sum"]==0).mean(),3), "all-pos", round((s["sum"]==s["size"]).mean(),3), "mixed", round(((s["sum"]>0)&(s["sum"]<s["size"])).mean(),3))
print("A6 rows in mixed users", round(va.user_id.isin(s[(s["sum"]>0)&(s["sum"]<s["size"])].index).mean(),3), "| oracle primary = (1 + (1-all_neg))/2 =", round((1 + 1 - (s['sum']==0).mean())/2, 4))
# A7
print("A7 tab share train", tr.tab.value_counts(normalize=True).round(3).to_dict()); print("A7 lv rate by tab train", tr.groupby("tab").long_view.mean().round(3).to_dict()); print("A7 lv rate by tab valid", va.groupby("tab").long_view.mean().round(3).to_dict())
print("A7 valid users with >1 tab", round((va.groupby("user_id").tab.nunique()>1).mean(),3), "| valid users with >1 tab among MIXED users", round((va[va.user_id.isin(s[(s['sum']>0)&(s['sum']<s['size'])].index)].groupby("user_id").tab.nunique()>1).mean(),3))
# A8
print("A8 duration quantiles p10/p50/p90", tr.duration_ms.quantile([.1,.5,.9]).round(0).to_dict()); q = pd.qcut(tr.duration_ms, 5, labels=False); print("A8 lv by duration quintile", tr.groupby(q).long_view.mean().round(3).tolist())
print("A8 complete plays train", round((tr.play_time_ms>=tr.duration_ms).mean(),4), "| rows with duration<=18s", round((tr.duration_ms<=18000).mean(),4), "lv rate there", round(tr[tr.duration_ms<=18000].long_view.mean(),3), "| lv rate duration>18s", round(tr[tr.duration_ms>18000].long_view.mean(),3))
print("A8 lv rate by finer duration bins", tr.groupby(pd.cut(tr.duration_ms/1000, [0,7,10,15,18,25,40,60,120,300,1e5])).long_view.mean().round(3).to_dict())
# A9
print("A9 lv by hour min/max", tr.groupby(tr.hourmin//100).long_view.mean().agg(["min","max"]).round(3).tolist())
# A10
m = vs.merge(tr.video_id.value_counts().rename("n_tr"), left_on="video_id", right_index=True, how="left").fillna({"n_tr":0}); print("A10 corr(show_cnt, train impressions)", round(m.show_cnt.corr(m.n_tr),3), "| counts median", vs.counts.median(), "counts range", vs.counts.min(), vs.counts.max())
# A11
print("A11 random log rows", len(rnd), "users", rnd.user_id.nunique(), "lv rate", round(rnd.long_view.mean(),4), "dates", rnd.date.min(), rnd.date.max(), "| is_rand share", rnd.is_rand.mean(), "| random users also in valid std log", round(rnd.user_id.isin(set(va.user_id)).mean(),3))
# A12
print("A12 is_rand share standard log", tr.is_rand.mean(), va.is_rand.mean())
# B1: label rule
rule = np.where(tr.duration_ms<=18000, tr.play_time_ms>=tr.duration_ms, tr.play_time_ms>=18000).astype(int)
print("B1 rule(log duration_ms) agreement", round((rule==tr.long_view).mean(),4), "| max play_time among negatives", tr[tr.long_view==0].play_time_ms.max(), "| min play among positives", tr[tr.long_view==1].play_time_ms.min())
t2 = tr.merge(vb[["video_id","video_duration"]], on="video_id", how="left")
rule2 = np.where(t2.video_duration<=18000, t2.play_time_ms>=t2.video_duration, t2.play_time_ms>=18000).astype(int)
print("B1 rule(feature-file video_duration) agreement", round((rule2==t2.long_view).mean(),4), "| duration_ms==video_duration share", round((t2.duration_ms==t2.video_duration).mean(),4))
dis = t2[rule!=t2.long_view]; print("B1 disagreements: n", len(dis), "label dist", dis.long_view.value_counts().to_dict(), "| duration<=18s share among disagreements", round((dis.duration_ms<=18000).mean(),3), "| play/duration ratio quantiles", (dis.play_time_ms/dis.duration_ms).quantile([.1,.5,.9]).round(3).tolist())
# B2: click rule
crule = np.where(tr.duration_ms<=7000, tr.play_time_ms>=tr.duration_ms, tr.play_time_ms>7000).astype(int)
print("B2 click rule agreement", round((crule==tr.is_click).mean(),4), "| P(lv|click)", round(tr[tr.is_click==1].long_view.mean(),3), "P(lv|noclick)", round(tr[tr.is_click==0].long_view.mean(),4), "corr", round(tr.is_click.corr(tr.long_view),3))
print("B2 P(lv|like)", round(tr[tr.is_like==1].long_view.mean(),3), "P(lv|profile_enter)", round(tr[tr.is_profile_enter==1].long_view.mean(),3), "P(like|lv)", round(tr[tr.long_view==1].is_like.mean(),4))
# Extra: order/tie-break and time-order effects on validation
def sc(name, s):
    r = ev(va.user_id.astype(str).tolist(), va.long_view.tolist(), list(map(float, s))); print(f"X {name:<45} primary={r['primary']:.4f} GAUC={r['GAUC']:.4f} nDCG5={r['nDCG@5']:.4f}")
rng = np.random.default_rng(0); sc("random scores", rng.random(len(va))); sc("constant score (file order tie-break)", np.zeros(len(va)))
sc("earlier-in-time first (-time_ms)", -va.time_ms.values.astype(float)); sc("later-in-time first (+time_ms)", va.time_ms.values.astype(float))
gm = tr.long_view.mean(); vr = tr.groupby("video_id").long_view.agg(["sum","size"]); vr = (vr["sum"]+20*gm)/(vr["size"]+20)
sc("video train rate (prior 20)", va.video_id.map(vr).fillna(gm).values)
sc("video TRAIN impression count", va.video_id.map(tr.video_id.value_counts()).fillna(0).values)
sc("video VALID-period impression count (transductive)", va.video_id.map(va.video_id.value_counts()).values)
sc("-duration_ms", -va.duration_ms.values.astype(float)); sc("+duration_ms", va.duration_ms.values.astype(float))
sc("tab==1 indicator", (va.tab==1).astype(float).values)
# position within user's day / session
va2 = va.sort_values(["user_id","time_ms"]); va2["pos_day"] = va2.groupby(["user_id","date"]).cumcount(); va2["n_day"] = va2.groupby(["user_id","date"]).time_ms.transform("size")
print("X lv rate by within-day position (valid):", va2.groupby(va2.pos_day.clip(upper=9)).long_view.mean().round(3).tolist())
print("X lv rate by #impressions that user-day (valid):", va2.groupby(va2.n_day.clip(upper=10)).long_view.mean().round(3).tolist())
sc("-pos_in_day (first impressions first)", -va2.sort_index().pos_day.values.astype(float))
print(f"done {time.time()-t0:.0f}s")
