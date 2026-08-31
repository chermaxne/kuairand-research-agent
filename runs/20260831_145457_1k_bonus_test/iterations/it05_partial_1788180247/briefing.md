# STATE BLOCK
CURRENT BEST: it04 | val primary 0.6528 (GAUC 0.6783 / nDCG5 0.6274) | baseline 0.6428 | margin +0.0100
BUDGET: iteration 5 of 5 | 5:43 of 6:00 elapsed | tokens so far 426607
CONVERGENCE: streak 0 of 3 flat (EPSILON=0.002)
BLOCKED: none
ACTIVE THEMES: winning: model[1 promoted/1 flat/0 failed], feature[2 promoted/0 flat/0 failed]; losing/flat: none; untried: training, multitask, other


## Data profile (measured by the harness)
data dir: `/home/q3user/kuairand-research-agent/data_cache/loop_train_valid_1k`

- train: 5,055,984 rows | 983 users | 2,119,510 videos | long_view rate 0.2635 | dates 20220408–20220421
- valid: 2,524,980 rows | 978 users | 1,159,803 videos | long_view rate 0.2645 | dates 20220422–20220428
- test: 0 rows (masked during the loop)
- train impressions per user: median 3489, p90 11648, max 49242


# CHAMPION CODE (current best pipeline; every experiment builds on it)
--- pipeline.py ---
"""Memory-light rewrite of champion_1k.py for the KuaiRand-1K variant, whose interaction logs
(~11.7M rows across two files) are far larger than KuaiRand-Pure's sample despite the "1K" (user-count)
name. The original load()/encode() build several full-size parallel Python object structures (tuples of
strings, dicts of lists) at once and OOMs past ~60GB RSS on an 11.7M-row dataset.

Same features, same model (FM, pure_bpr / champion_equiv), same splits, same CLI contract as
champion_1k.py -- only load()+encode() are rewritten to use array.array/numpy int-coded columns
instead of Python tuples-of-strings, and vectorized numpy instead of per-row dict/list bookkeeping.
Vocab *index assignment order* differs from the original (np.unique's sorted order vs. first-appearance
order) -- this does not change model behaviour: embedding indices are exchangeable, only the trained
values differ, not what they represent or how well they generalize.

    python champion_1k_lowmem.py --data <data_dir> --split val|test --out preds.csv
"""
import argparse
import array
import csv
import importlib.util
import os
import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- [1] config (= champion_1k.py)
LABEL = "long_view"
SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428), "test": (20220429, 20220508)}
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket", "hour", "sess_depth"]
K = 16
LR = 0.001
L2 = 1e-6
EPOCHS = 40
BATCH = 8192
PATIENCE = 4
SEED = 0
N_DUR_BUCKETS = 10


def _import_evaluate():
    try:
        from evaluate import evaluate
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


# ----------------------------------------------------------------------------- [2]+[3] data loading + encoding, fused
class VocabCoder:
    """str -> dense int32 code, order of first appearance (only used for the raw parse pass; index
    order here is irrelevant to model behaviour -- see module docstring)."""

    def __init__(self):
        self.d = {}

    def code(self, s):
        c = self.d.get(s)
        if c is None:
            c = len(self.d)
            self.d[s] = c
        return c

    def __len__(self):
        return len(self.d)


def _train_vocab_map(field_arr, train_mask):
    """Build a train-only vocab (unseen-in-train -> UNK) and apply it to the WHOLE field array,
    fully vectorized (np.searchsorted on the sorted unique-train-values array) instead of per-row
    dict lookups. Mirrors champion_1k.py's encode(): vocabs built from `tr` only, `unk[i] = len(vocab)`."""
    uniq = np.unique(field_arr[train_mask])
    unk_idx = len(uniq)
    pos = np.searchsorted(uniq, field_arr)
    pos_clip = np.clip(pos, 0, len(uniq) - 1)
    known = (pos < len(uniq)) & (uniq[pos_clip] == field_arr)
    mapped = np.where(known, pos_clip, unk_idx).astype(np.int32)
    return mapped, len(uniq) + 1  # field_dim = train vocab size + 1 UNK slot, exactly as champion_1k.py


def load_and_encode(data_dir):
    """Returns ({split: (X int32 (N,F), X_num float32 (N,5), y float32, u int32, uid_str list, vid_str list)},
    total_dim). uid_str/vid_str are only materialized for OUTPUT (write_preds needs real ids)."""
    t0 = time.time()
    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_1k.csv")) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    print(f"  video_features loaded ({len(vid2author)} videos) {time.time() - t0:.1f}s")

    date_a, user_a, video_a, author_a, tab_a = (array.array("i") for _ in range(5))
    dur_a = array.array("f")
    label_a = array.array("b")
    click_a = array.array("b")
    like_a = array.array("b")
    time_a = array.array("q")
    hour_a = array.array("i")

    user_vc, video_vc, author_vc, tab_vc = VocabCoder(), VocabCoder(), VocabCoder(), VocabCoder()
    author_vc.code("UNK")  # matches vid2author.get(vid, "UNK") in the original -- reserve the token up front

    for fname in ("log_standard_4_08_to_4_21_1k.csv", "log_standard_4_22_to_5_08_1k.csv"):
        with open(os.path.join(data_dir, fname)) as fh:
            for r in csv.DictReader(fh):
                vid_s = r["video_id"]
                date_a.append(int(r["date"]))
                user_a.append(user_vc.code(r["user_id"]))
                video_a.append(video_vc.code(vid_s))
                author_a.append(author_vc.code(vid2author.get(vid_s, "UNK")))
                tab_a.append(tab_vc.code(r["tab"]))
                dur_a.append(float(r["duration_ms"]))
                label_a.append(1 if r[LABEL] != "0" else 0)
                click_a.append(1 if r["is_click"] != "0" else 0)
                like_a.append(1 if r["is_like"] != "0" else 0)
                time_a.append(int(r["time_ms"]))
                hour_a.append(int(r["hourmin"]) // 100)
        print(f"  {fname} parsed, running total {len(date_a)} rows, {time.time() - t0:.1f}s")

    N = len(date_a)
    date_arr = np.frombuffer(date_a, dtype=np.int32)
    user_arr = np.frombuffer(user_a, dtype=np.int32)
    video_arr = np.frombuffer(video_a, dtype=np.int32)
    author_arr = np.frombuffer(author_a, dtype=np.int32)
    tab_arr = np.frombuffer(tab_a, dtype=np.int32)
    dur_arr = np.frombuffer(dur_a, dtype=np.float32)
    label_arr = np.frombuffer(label_a, dtype=np.int8)
    click_arr = np.frombuffer(click_a, dtype=np.int8)
    like_arr = np.frombuffer(like_a, dtype=np.int8)
    time_arr = np.frombuffer(time_a, dtype=np.int64)
    hour_arr = np.frombuffer(hour_a, dtype=np.int32)
    del date_a, user_a, video_a, author_a, tab_a, dur_a, label_a, click_a, like_a, time_a, hour_a
    print(f"  {N} rows -> typed arrays, {time.time() - t0:.1f}s")

    n_users = len(user_vc)
    n_videos, n_authors = len(video_vc), len(author_vc)
    train_mask = (date_arr >= SPLITS["train"][0]) & (date_arr <= SPLITS["train"][1])

    # --- video/author historical-rate features, in chronological (time_ms) order; only train rows update
    #     the running counters -- identical semantics to champion_1k.py's load(), just array-backed. ---
    num_features = np.empty((N, 13), dtype=np.float32)
    v_imp = [0] * n_videos; v_pos = [0] * n_videos
    a_imp = [0] * n_authors; a_pos = [0] * n_authors
    u_imp = [0] * n_users; u_pos = [0] * n_users
    v_click = [0] * n_videos; v_like = [0] * n_videos
    a_click = [0] * n_authors; a_like = [0] * n_authors
    u_click = [0] * n_users; u_like = [0] * n_users
    time_order = np.argsort(time_arr, kind="stable").tolist()
    video_l, author_l, user_l, label_l, click_l, like_l, train_l = video_arr.tolist(), author_arr.tolist(), user_arr.tolist(), label_arr.tolist(), click_arr.tolist(), like_arr.tolist(), train_mask.tolist()
    log1p = np.log1p
    for i in time_order:
        vid, aid, uid = video_l[i], author_l[i], user_l[i]
        vi, vp, ai, ap = v_imp[vid], v_pos[vid], a_imp[aid], a_pos[aid]
        ui, up = u_imp[uid], u_pos[uid]
        vc, vl = v_click[vid], v_like[vid]
        ac, al = a_click[aid], a_like[aid]
        uc, ul = u_click[uid], u_like[uid]
        
        num_features[i, 0] = log1p(vi)
        num_features[i, 1] = vp / vi if vi > 0 else 0.0
        num_features[i, 2] = log1p(ai)
        num_features[i, 3] = ap / ai if ai > 0 else 0.0
        num_features[i, 5] = log1p(ui)
        num_features[i, 6] = up / ui if ui > 0 else 0.0
        num_features[i, 7] = vc / vi if vi > 0 else 0.0
        num_features[i, 8] = ac / ai if ai > 0 else 0.0
        num_features[i, 9] = vl / vi if vi > 0 else 0.0
        num_features[i, 10] = al / ai if ai > 0 else 0.0
        num_features[i, 11] = uc / ui if ui > 0 else 0.0
        num_features[i, 12] = ul / ui if ui > 0 else 0.0
        
        if train_l[i]:
            v_imp[vid] = vi + 1
            a_imp[aid] = ai + 1
            u_imp[uid] = ui + 1
            if label_l[i] == 1:
                v_pos[vid] = vp + 1
                a_pos[aid] = ap + 1
                u_pos[uid] = up + 1
            if click_l[i] == 1:
                v_click[vid] = vc + 1
                a_click[aid] = ac + 1
                u_click[uid] = uc + 1
            if like_l[i] == 1:
                v_like[vid] = vl + 1
                a_like[aid] = al + 1
                u_like[uid] = ul + 1
    del v_imp, v_pos, a_imp, a_pos, u_imp, u_pos, v_click, v_like, a_click, a_like, u_click, u_like, time_order, video_l, author_l, user_l, label_l, click_l, like_l, train_l
    print(f"  historical-rate features done, {time.time() - t0:.1f}s")

    # --- session depth + time-gap-since-last-event, per (user, date) group ordered by time_ms ---
    depth_code = np.empty(N, dtype=np.int8)
    group_order = np.lexsort((time_arr, date_arr, user_arr)).tolist()  # last key = primary: user, then date, then time
    user_l, date_l, time_l = user_arr.tolist(), date_arr.tolist(), time_arr.tolist()
    counts, last_time = {}, {}
    for idx in group_order:
        key = (user_l[idx], date_l[idx])
        c = counts.get(key, 0)
        counts[key] = c + 1
        depth_code[idx] = c if c <= 4 else (5 if c <= 9 else 6)  # 0..4 raw counts, 5="5-9", 6="10+" (relabeled ints)
        t = time_l[idx]
        num_features[idx, 4] = log1p(t - last_time.get(key, t))
        last_time[key] = t
    del group_order, user_l, date_l, time_l, counts, last_time
    print(f"  session-depth features done, {time.time() - t0:.1f}s")

    # --- dur_bucket: quantile edges fit on TRAIN durations only, applied to every row (vectorized) ---
    edges = np.quantile(dur_arr[train_mask], np.linspace(0, 1, N_DUR_BUCKETS + 1)[1:-1])
    dur_bucket_arr = np.searchsorted(edges, dur_arr).astype(np.int32)

    # --- FIELDS = [user_id, video_id, author_id, tab, dur_bucket, hour, sess_depth]; train-only vocab + UNK,
    #     exactly matching champion_1k.py's encode() (unseen-in-train falls into the field's UNK slot). ---
    raw_fields = [user_arr, video_arr, author_arr, tab_arr, dur_bucket_arr, hour_arr, depth_code.astype(np.int32)]
    mapped_fields, field_dims = [], []
    for f in raw_fields:
        m, d = _train_vocab_map(f, train_mask)
        mapped_fields.append(m)
        field_dims.append(d)
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    X_full = np.stack([m + o for m, o in zip(mapped_fields, offsets)], axis=1).astype(np.int32)
    print(f"  categorical encoding done (dim={sum(field_dims)}), {time.time() - t0:.1f}s")

    nf_mean = num_features[train_mask].mean(axis=0)
    nf_std = num_features[train_mask].std(axis=0) + 1e-8
    num_features = (num_features - nf_mean) / nf_std
    print(f'TRAIN nf mean: {num_features[train_mask].mean():.4f}, std: {num_features[train_mask].std():.4f}')

    # reverse maps for OUTPUT only (real ids the pipeline contract requires in the predictions CSV)
    uid_by_code = [None] * len(user_vc)
    for s, c in user_vc.d.items():
        uid_by_code[c] = s
    vid_by_code = [None] * len(video_vc)
    for s, c in video_vc.d.items():
        vid_by_code[c] = s

    enc = {}
    for name, (lo, hi) in SPLITS.items():
        idx = np.flatnonzero((date_arr >= lo) & (date_arr <= hi))  # ascending -> preserves original file order
        uid_list = [uid_by_code[c] for c in user_arr[idx]]
        vid_list = [vid_by_code[c] for c in video_arr[idx]]
        enc[name] = (X_full[idx], num_features[idx], label_arr[idx].astype(np.float32), user_arr[idx], uid_list, vid_list)
    print(f"  splits sliced, total load+encode {time.time() - t0:.1f}s")
    return enc, int(sum(field_dims))


# ----------------------------------------------------------------------------- [4] model (= champion_1k.py, unchanged)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED, use_mlp=True):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.W_num = np.zeros(13, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mW_num = np.zeros_like(self.W_num); self.vW_num = np.zeros_like(self.W_num)
        self.t = 0
        
        self.use_mlp = use_mlp
        self.k = k
        if self.use_mlp:
            self.mlp_in_dim = 7 * k + 13
            self.mlp_h = 64
            self.mlp_w1 = rng.normal(0, 0.05, (self.mlp_in_dim, self.mlp_h)).astype(np.float32)
            self.mlp_b1 = np.zeros(self.mlp_h, dtype=np.float32)
            self.mlp_w2 = rng.normal(0, 0.05, (self.mlp_h, 1)).astype(np.float32)
            self.mlp_b2 = np.zeros(1, dtype=np.float32)
            
            self.mM_w1 = np.zeros_like(self.mlp_w1)
            self.vM_w1 = np.zeros_like(self.mlp_w1)
            self.mM_b1 = np.zeros_like(self.mlp_b1)
            self.vM_b1 = np.zeros_like(self.mlp_b1)
            self.mM_w2 = np.zeros_like(self.mlp_w2)
            self.vM_w2 = np.zeros_like(self.mlp_w2)
            self.mM_b2 = np.zeros_like(self.mlp_b2)
            self.vM_b2 = np.zeros_like(self.mlp_b2)

    def logits(self, X, X_num):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_out = self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter
        
        if self.use_mlp:
            E_flat = E.reshape(len(X), -1)
            mlp_in = np.concatenate([E_flat, X_num], axis=1)
            h1 = np.maximum(0, mlp_in.dot(self.mlp_w1) + self.mlp_b1)
            mlp_out = (h1.dot(self.mlp_w2) + self.mlp_b2).squeeze()
            return fm_out + mlp_out, E, S, h1, mlp_in
        else:
            return fm_out, E, S, None, None

    def step_bpr(self, X_pos, X_num_pos, X_neg, X_num_neg):
        B = len(X_pos)
        z_pos, E_pos, S_pos, h1_pos, mlp_in_pos = self.logits(X_pos, X_num_pos)
        z_neg, E_neg, S_neg, h1_neg, mlp_in_neg = self.logits(X_neg, X_num_neg)
        z_diff = z_pos - z_neg
        g = ((sigmoid(z_diff) - 1.0) / B).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X_pos, g[:, None])
        np.add.at(gW, X_neg, -g[:, None])
        
        gE_pos_fm = g[:, None, None] * (S_pos[:, None, :] - E_pos)
        gE_neg_fm = -g[:, None, None] * (S_neg[:, None, :] - E_neg)

        if self.use_mlp:
            g_out_pos = g[:, None]
            g_out_neg = -g[:, None]
            
            gh1_pos = g_out_pos.dot(self.mlp_w2.T) * (h1_pos > 0)
            g_mlp_w2 = h1_pos.T.dot(g_out_pos)
            g_mlp_b2 = g_out_pos.sum(0)
            g_mlp_w1 = mlp_in_pos.T.dot(gh1_pos)
            g_mlp_b1 = gh1_pos.sum(0)
            g_mlp_in_pos = gh1_pos.dot(self.mlp_w1.T)
            
            gh1_neg = g_out_neg.dot(self.mlp_w2.T) * (h1_neg > 0)
            g_mlp_w2 += h1_neg.T.dot(g_out_neg)
            g_mlp_b2 += g_out_neg.sum(0)
            g_mlp_w1 += mlp_in_neg.T.dot(gh1_neg)
            g_mlp_b1 += gh1_neg.sum(0)
            g_mlp_in_neg = gh1_neg.dot(self.mlp_w1.T)
            
            gE_pos_fm += g_mlp_in_pos[:, :7*self.k].reshape(-1, 7, self.k)
            gE_neg_fm += g_mlp_in_neg[:, :7*self.k].reshape(-1, 7, self.k)
            
            gW_num = (g[:, None] * (X_num_pos - X_num_neg)).sum(0)
            gW_num += g_mlp_in_pos[:, 7*self.k:].sum(0) + g_mlp_in_neg[:, 7*self.k:].sum(0)
            
            g_mlp_w2 += self.l2 * self.mlp_w2
            g_mlp_w1 += self.l2 * self.mlp_w1
        else:
            gW_num = (g[:, None] * (X_num_pos - X_num_neg)).sum(0)
            
        np.add.at(gV, X_pos, gE_pos_fm)
        np.add.at(gV, X_neg, gE_neg_fm)

        gV += self.l2 * self.V
        gW += self.l2 * self.W
        gW_num += self.l2 * self.W_num

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        
        params = [
            (self.V, gV, self.mV, self.vV), 
            (self.W, gW, self.mW, self.vW), 
            (self.W_num, gW_num, self.mW_num, self.vW_num)
        ]
        
        if self.use_mlp:
            params.extend([
                (self.mlp_w1, g_mlp_w1, self.mM_w1, self.vM_w1),
                (self.mlp_b1, g_mlp_b1, self.mM_b1, self.vM_b1),
                (self.mlp_w2, g_mlp_w2, self.mM_w2, self.vM_w2),
                (self.mlp_b2, g_mlp_b2, self.mM_b2, self.vM_b2)
            ])
            
        for P, G, M, Vv in params:
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

        return float(-np.mean(np.log(sigmoid(z_diff) + 1e-9)))

    def predict(self, X, X_num, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs], X_num[i:i + bs])[0] for i in range(0, len(X), bs)])


# ----------------------------------------------------------------------------- [5] training (= champion_1k.py, unchanged)
def train(enc, dim, mode="pure_bpr", max_epochs=EPOCHS, seed=SEED, log=print):
    Xtr, Xnum_tr, ytr, utr = enc["train"][0], enc["train"][1], enc["train"][2], enc["train"][3]
    Xva, Xnum_va, yva, uva = enc["valid"][0], enc["valid"][1], enc["valid"][2], enc["valid"][3]

    if mode == "champion_equiv":
        Xnum_tr = np.zeros_like(Xnum_tr)
        Xnum_va = np.zeros_like(Xnum_va)

    user2pos, user2neg = {}, {}
    for i, (u, y) in enumerate(zip(utr.tolist(), ytr.tolist())):
        (user2pos if y == 1.0 else user2neg).setdefault(u, []).append(i)

    pos_indices, owner = [], []       # owner: per-positive index into neg_lists/lens/offsets (one slot per
    neg_lists = []                    # USER, not one per positive -- avoids a pos_u*neg_u blowup below)
    user2neg_slot = {}
    for u, pos_list in user2pos.items():
        neg_list = user2neg.get(u, [])
        if neg_list:
            slot = user2neg_slot.get(u)
            if slot is None:
                slot = len(neg_lists)
                neg_lists.append(neg_list)
                user2neg_slot[u] = slot
            for p in pos_list:
                pos_indices.append(p)
                owner.append(slot)

    pos_indices = np.array(pos_indices, dtype=np.int32)
    owner = np.array(owner, dtype=np.int32)
    lens = np.array([len(n) for n in neg_lists], dtype=np.int32)
    if len(neg_lists) > 0:
        flat_negs = np.concatenate(neg_lists).astype(np.int32)
        offsets = np.cumsum([0] + list(lens[:-1]), dtype=np.int32)
    else:
        flat_negs = np.array([], dtype=np.int32)
        offsets = np.array([], dtype=np.int32)
    num_pairs = len(pos_indices)
    log(f"[{mode}] Total within-user pairs: {num_pairs}")

    use_mlp = (mode != "champion_equiv")
    m = FM(dim, seed=seed, use_mlp=use_mlp)
    rng = np.random.default_rng(seed)
    best, best_state, best_metrics, bad = -1.0, None, None, 0

    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        losses = []
        if num_pairs == 0:
            break
        sampled_neg = flat_negs[offsets[owner] + (rng.random(num_pairs) * lens[owner]).astype(np.int32)]
        idx = rng.permutation(num_pairs)
        for i in range(0, num_pairs, BATCH):
            b_idx = idx[i:i + BATCH]
            p_idx = pos_indices[b_idx]
            n_idx = sampled_neg[b_idx]
            losses.append(m.step_bpr(Xtr[p_idx], Xnum_tr[p_idx], Xtr[n_idx], Xnum_tr[n_idx]))

        va = evaluate(uva.tolist(), yva.tolist(), m.predict(Xva, Xnum_va))
        loss_val = np.mean(losses) if losses else 0.0
        log(f"[{mode}] epoch {ep:2d} | loss {loss_val:.4f} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
            f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")

        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            if m.use_mlp:
                best_state = (m.V.copy(), m.W.copy(), m.W_num.copy(), np.float32(m.b), m.mlp_w1.copy(), m.mlp_b1.copy(), m.mlp_w2.copy(), m.mlp_b2.copy())
            else:
                best_state = (m.V.copy(), m.W.copy(), m.W_num.copy(), np.float32(m.b))
            best_metrics = va.copy()
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"[{mode}] early stop at epoch {ep}")
                break

    if best_state is not None:
        if m.use_mlp:
            m.V, m.W, m.W_num, m.b, m.mlp_w1, m.mlp_b1, m.mlp_w2, m.mlp_b2 = best_state
        else:
            m.V, m.W, m.W_num, m.b = best_state
    return m, best_metrics


# ----------------------------------------------------------------------------- [6] CLI (= champion_1k.py contract)
def write_preds(path, uid_list, vid_list, scores):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v, s) in enumerate(zip(uid_list, vid_list, scores)):
            w.writerow([i, u, v, f"{float(s):.6g}"])


def within_user_rank(users, scores):
    return pd.Series(scores).groupby(users, sort=False).rank(pct=True).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["val", "valid", "test"])
    ap.add_argument("--out", default="preds_val.csv")
    a = ap.parse_args()

    split = "valid" if a.split in ("val", "valid") else "test"
    t_start = time.time()
    budget = float(os.environ.get("KUAIRAND_TIME_BUDGET_S", 3000))
    fast = os.environ.get("KUAIRAND_FAST", "0") == "1"

    enc, dim = load_and_encode(a.data)
    print({k: len(v[2]) for k, v in enc.items()}, f"fields={FIELDS}", f"load+encode {time.time() - t_start:.1f}s")

    X, X_num, _, _, uid_list, vid_list = enc[split]
    Xva, Xnum_va, yva, uva = enc["valid"][0], enc["valid"][1], enc["valid"][2], enc["valid"][3]

    all_scores, all_scores_va, last_single_metrics = [], [], None
    seeds = [42] if fast else [42, 43, 44, 45, 46]
    for s in seeds:
        model_bpr, metrics_bpr = train(enc, dim, mode="pure_bpr", seed=s)
        all_scores.append(within_user_rank(enc[split][3], model_bpr.predict(X, X_num)))
        if not fast:
            all_scores_va.append(within_user_rank(uva, model_bpr.predict(Xva, Xnum_va)))
        if last_single_metrics is None:
            last_single_metrics = metrics_bpr

    mean_scores = np.mean(all_scores, axis=0)
    assert np.all(np.isfinite(mean_scores)), "non-finite scores"
    write_preds(a.out, uid_list, vid_list, mean_scores)
    print(f"wrote {a.out}: {len(uid_list)} rows for split={split} in {time.time() - t_start:.0f}s")

    if last_single_metrics is not None:
        print(f"ABLATION pure_bpr_single primary={last_single_metrics['primary']:.4f} gauc={last_single_metrics['GAUC']:.4f} ndcg5={last_single_metrics['nDCG@5']:.4f}")

    if not fast:
        mean_scores_va = np.mean(all_scores_va, axis=0)
        ens_metrics = evaluate(uva.tolist(), yva.tolist(), mean_scores_va)
        print(f"ABLATION pure_bpr_ensemble primary={ens_metrics['primary']:.4f} gauc={ens_metrics['GAUC']:.4f} ndcg5={ens_metrics['nDCG@5']:.4f}")

    if not fast:
        if (time.time() - t_start) < 0.75 * budget:
            _, eq_metrics = train(enc, dim, mode="champion_equiv", seed=42)
            print(f"ABLATION champion_equiv primary={eq_metrics['primary']:.4f} gauc={eq_metrics['GAUC']:.4f} ndcg5={eq_metrics['nDCG@5']:.4f}")
        else:
            print("ABLATION champion_equiv skipped: out of time budget")


if __name__ == "__main__":
    main()


# LEDGER (full history, oldest first)
# Ledger (tier-1 memory, append-only; one line per iteration, harness-written except LESSON)
# it00 champion installed from runs/manual_1k_test/seed_champion_1k: val primary 0.6411 (GAUC 0.6729 / nDCG5 0.6092); published baseline 0.6428; rungs random 0.4334 pop 0.5427
[it01] HYP: Projecting the 5 numerical features (past-only historical rates and session time gaps) into the FM's embedding space to… | CHANGE: pipeline.py (+28/-9) | RESULT: 0.6406 (best 0.6411) -> kept | LESSON: FM with projected numerical features: 0.6406 vs 0.6411, kept; early-stopped at epoch 5.
[it02] HYP: Extending the FM to a DeepFM by adding a 1-layer MLP over the concatenated embeddings and numerical features will allow… | CHANGE: pipeline.py (+103/-12) | RESULT: 0.6489 (best 0.6489) -> PROMOTED | LESSON: DeepFM primary=0.6489 gauc=0.6762 ndcg5=0.6215, promoted.
[it03] HYP: Adding user historical long_view rates and item/author auxiliary feedback rates (click, like) as past-only numerical fe… | CHANGE: pipeline.py (+36/-7) | RESULT: 0.6492 (best 0.6492) -> PROMOTED | LESSON: Adding user historical long_view rates and item/author auxiliary feedback rates as past-only numerical features promoted the ranking metric to 0.6492.
[it04] HYP: Standardizing past-only numerical features will stabilize DeepFM's gradients against scale imbalances, adding missing u… | CHANGE: pipeline.py (+22/-6) | RESULT: 0.6528 (best 0.6528) -> PROMOTED | LESSON: DeepFM with standardized features and within-user rank ensembling promoted to new champion with primary score 0.6528.


# PRIOR RUNS — every experiment this agent has already measured (harness-recorded, earlier runs only)
These are YOUR OWN sealed measurements from previous runs of this same task, not advice. Do not spend an
iteration re-measuring something below unless you state what is different about your version. The deltas are
against the champion at that iteration's start, so a small delta on top of a strong champion is not the same
as a small delta on top of the baseline.

Best score ever recorded across all runs: **0.6051** (20260830_224430_seeded_0605_v2 it01) — Providing the model with strictly past-only video and author historical click (valid play) and like rates as numerical features will inject granular…

## WHAT WORKED — measured gains, largest first (14 of them)
| Δ vs then-champion | direction | what was tried | result |
|---|---|---|---|
| +0.0029 | training | Training with a within-user pairwise BPR loss directly aligns the objective with the evaluation metric (GAUC, nDCG@5) by optimizing relative ranking rather than absolute pointwise… | 0.6043 promoted |
| +0.0023 | training | Training with BPR loss on within-user positive-negative pairs directly aligns the objective with the ranking metric and eliminates user-bias confounding, increasing the primary me… | 0.6038 promoted |
| +0.0023 | training | Changing the pointwise logloss to a pairwise BPR loss aligned with the within-user ranking metric will directly optimise for the primary evaluation criteria and yield a structural… | 0.6038 promoted |
| +0.0021 | feature | Adding the user's daily session depth and hour-of-day as contextual categorical features will capture position bias and time context, and combining this with a 3-seed ensemble wil… | 0.6048 promoted |
| +0.0018 | feature | Adding the user's daily session depth and hour-of-day as categorical features will capture position bias and time context, and ensembling 3 seeds will reduce variance, jointly yie… | 0.6032 promoted |
| +0.0012 | training | Training with BPR loss on within-user positive-negative pairs directly aligns the objective with the ranking metric, raising primary. | 0.6027 promoted |
| +0.0010 | feature | Adding the user's daily session depth and hour-of-day as categorical context features models time and position bias, while a 3-seed ensemble reduces variance, together yielding a… | 0.6025 promoted |
| +0.0007 | feature | Adding daily session depth and hour-of-day as categorical context features, combined with a 3-seed ensemble, will capture position bias and time context to raise the primary metri… | 0.6022 promoted |
| +0.0005 | feature | Adding the user's daily session depth and hour-of-day as contextual categorical features captures position bias and time context, yielding new ranking signal. | 0.6048 promoted |
| +0.0005 | feature | Adding strictly past-only video and author historical long_view rates and impression counts as numerical features will provide strong item-quality signals, improving the BPR model… | 0.6042 kept_champion |
| +0.0004 | model | An ensemble of 3 BPR FMs trained with different random seeds will reliably reduce variance and boost the ranking metric by aggregating decorrelated predictions. | 0.6042 kept_champion |
| +0.0003 | feature | Stacking past-only numerical features (video/author historical rates and impression counts) and 5-seed ensembling (both validated riders) alongside a genuinely new numerical signa… | 0.6050 promoted |
| +0.0002 | feature | Stacking past-only item/author statistics as numerical features (a validated rider) along with session time-gap (a new signal) and a 5-seed ensemble will push the champion past th… | 0.6034 promoted |
| +0.0001 | feature | Providing the model with strictly past-only video and author historical click (valid play) and like rates as numerical features will inject granular item-engagement priors that di… | 0.6051 kept_champion |

## WHAT DID NOT WORK — measured losses or no movement (13 of them)
| Δ vs then-champion | direction | what was tried | result |
|---|---|---|---|
| -0.0115 | training | Replacing pointwise logloss with within-user pairwise BPR loss — which directly optimizes the same within-user ranking that GAUC and nDCG@5 measure — should raise primary because… | 0.5900 kept_champion |
| -0.0089 | training | Training with a within-user pairwise BPR loss directly aligns the objective with the primary ranking metrics (GAUC, nDCG@5), eliminating user-bias confounding and raising primary. | 0.5925 kept_champion |
| -0.0080 | training | Treating click and long_view as ordinal feedback levels and training BPR on all valid pairs (long_view > no_click, long_view > click_only, click_only > no_click) will provide gran… | 0.5970 kept_champion |
| -0.0064 | feature | Adding the user's most recently interacted video IDs as past-only categorical fields will explicitly model sequential item-to-item transitions (Markov chains) and short-term inter… | 0.5984 kept_champion |
| -0.0046 | multitask | Adding an auxiliary MSE regression task on play_progress (play_time_ms / duration_ms) will provide a dense, continuous preference signal to the shared embeddings, improving the pr… | 0.6002 kept_champion |
| -0.0028 | feature | Adding the user's last 3 positively interacted videos mapped directly to the shared video_id embedding space will enable Factorized Personalized Markov Chains (FPMC) item-to-item… | 0.6019 kept_champion |
| -0.0010 | multitask | Adding an auxiliary pointwise logloss for is_click with shared embeddings and a weight of 0.5 will improve the representation of items and users, raising the primary long_view ran… | 0.6028 kept_champion |
| -0.0008 | feature | Adding strictly past-only historical long_view rates and impression counts for videos and authors as bucketed categorical fields will provide a dense item-quality signal that shar… | 0.6039 kept_champion |
| -0.0006 | feature | Ensembling 5 seeds, adding past-only global item/author rates (a validated rider), and injecting past-only user-author interaction rates (a new personalization signal) as numerica… | 0.6042 kept_champion |
| -0.0005 | training | Training with a hybrid pointwise logloss and within-user pairwise BPR loss will directly optimize the relative ordering of items for mixed users while maintaining calibration for… | 0.6027 kept_champion |
| -0.0003 | training | Replacing the pairwise BPR loss with a within-user sampled softmax loss over a list of 1 positive and 7 negatives will provide stronger gradients and implicitly mine hard negative… | 0.6045 kept_champion |
| -0.0002 | model | Generalizing the Factorization Machine to a Field-weighted FM (FwFM) will allow the model to learn the importance of different field-pair interactions, upweighting critical crosse… | 0.6049 kept_champion |
| -0.0001 | model | Implementing a DIN-style target attention over the user's past clicks provides strong explicit interest modeling, yielding significant new ranking signal that static FMs cannot ca… | 0.6048 kept_champion |

## WHAT BROKE — 3 iterations never produced a score (an implementation failure costs the same as a bad idea)
- other: (no valid plan: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction under consideration and either w…) — failed: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction un…
- other: (no valid plan: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction under consideration and either w…) — failed: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction un…
- other: (no valid plan: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction under consideration and either w…) — failed: researcher_malformed: SEARCH_CHECK is required: state the specific technique/direction un…

Attempts per direction across all prior runs: feature 13 (9 positive), model 3 (1 positive), multitask 2 (0 positive), training 9 (4 positive).

# RESEARCH DIGEST — every iteration so far, grouped by direction (harness-measured facts)
| it | direction | what changed | predicted Δ | measured Δ vs then-champion | decision | status | in-run ablations (pipeline-reported, unsealed) | lesson |
|---|---|---|---|---|---|---|---|---|
| it03 | feature | Adding user historical long_view rates and item/author auxiliary feedback rates (click, like) as past-only numerical features will provide DeepFM's MLP with rich interaction surfaces, allowing it to learn non-linear per… | +0.0030 | +0.0003 | promoted | scored | pure_bpr_single 0.6353 (-0.0139 vs the full run); pure_bpr_ensemble 0.6492 (+0.0000 vs the full run); champion_equiv 0.6395 (-0.0097 vs the full run) | Adding user historical long_view rates and item/author auxiliary feedback rates as past-only numerical features promote… |
| it04 | feature | Standardizing past-only numerical features will stabilize DeepFM's gradients against scale imbalances, adding missing user click/like rates will complete the behavioral priors, and within-user rank ensembling will optim… | +0.0025 | +0.0037 | promoted | scored | pure_bpr_single 0.6472 (-0.0056 vs the full run); pure_bpr_ensemble 0.6528 (-0.0000 vs the full run); champion_equiv 0.6395 (-0.0133 vs the full run) | DeepFM with standardized features and within-user rank ensembling promoted to new champion with primary score 0.6528. |
| it01 | model | Projecting the 5 numerical features (past-only historical rates and session time gaps) into the FM's embedding space to compute pairwise interactions with the categorical IDs will allow the model to learn personalized a… | +0.0030 | -0.0005 | kept_champion | scored | pure_bpr_single 0.6388 (-0.0018 vs the full run); pure_bpr_ensemble 0.6407 (+0.0001 vs the full run); champion_equiv 0.6395 (-0.0011 vs the full run) | FM with projected numerical features: 0.6406 vs 0.6411, kept; early-stopped at epoch 5. |
| it02 | model | Extending the FM to a DeepFM by adding a 1-layer MLP over the concatenated embeddings and numerical features will allow the model to learn arbitrary high-order feature interactions, providing a stronger personalization… | +0.0025 | +0.0078 | promoted | scored | pure_bpr_single 0.6347 (-0.0142 vs the full run); pure_bpr_ensemble 0.6489 (+0.0000 vs the full run); champion_equiv 0.6395 (-0.0094 vs the full run) | DeepFM primary=0.6489 gauc=0.6762 ndcg5=0.6215, promoted. |

Calibration: over 4 scored iterations your predicted gain exceeded the measured one by -0.0001 on average (predicted − measured); size the next prediction accordingly.
Totals: 4 iterations; promoted 3 (it02, it03, it04); attempts per direction: feature 2, model 2; never attempted: training, multitask, other.

# RESEARCH SYNTHESIS (written by the Scribe from the digest above — interpretive; verify any claim against the table)
The model direction has been explored twice: it01 projected numerical features into the FM's embedding space, yielding a 0.6406 GAUC, while it02 extended the FM to a DeepFM, achieving a 0.6489 GAUC. The feature direction has been tried once: it03 added user historical long_view rates and item/author auxiliary feedback rates, resulting in a 0.6492 GAUC. No attempts have been made in the training, multitask, or other directions. The DeepFM (it02) and the feature addition (it03) were promoted, while the FM with projected numerical features (it01) was kept as the champion. The pure_bpr_ensemble ablation was consistently at 0.6492 GAUC, while the pure_bpr_single and champion_equiv ablations varied more. The calibration shows a slight overestimation of gains, with an average difference of +0.0003 between predicted and measured gains.

# RECENT ITERATION DETAILS (harness-measured facts + what was actually changed)
Use these to decide whether to CONTINUE an idea: when a bundled change moved little, the diff shows which
components were in it, so you can keep the part that plausibly worked and drop the rest. State which
component you are keeping or dropping, and why, in `rationale`.

## it01 [model] — kept_champion (scored), -0.0005 vs the then-champion 0.6411
HYPOTHESIS: Projecting the 5 numerical features (past-only historical rates and session time gaps) into the FM's embedding space to compute pairwise interactions with the categorical IDs will allow the model to learn personalized affinities for items with high historical rates, raising the primary ranking metric.
YOUR PREDICTED GAIN: +0.0030; measured -0.0005 — evidence given: Previous runs showed that injecting numerical historical rates yielded small linear gains (+0.0005). The current champion limits these powerful dynamic signals to just linear weights, completely ignoring how different users might value 'author hit rate' differently. Permitting full inner-product in…
RATIONALE (yours, at the time): Our current Numpy FM limits the 5 continuous features (historical click rates and time gaps) to a linear `X_num.dot(W_num)` term, meaning they cannot interact with `user_id` or `video_id`. By projecting them via a 5xK matrix `V_num` into the shared embedding space, we naturally extend the FM formula to model second-order interactions like (user_id x video_rate), directly answering the need for stronger personalization without expanding capacity for static fields.
CHANGE SPEC you gave the Engineer:
1. In `FM.__init__`, add a new parameter matrix `self.V_num = rng.normal(0, 0.01, (5, k)).astype(np.float32)` and its Adam momentum variables `self.mV_num` and `self.vV_num`.
2. In `FM.logits`, compute `E_cat = self.V[X]` and `E_num = X_num[:, :, None] * self.V_num[None, :, :]`. Concatenate them into a single tensor `E = np.concatenate([E_cat, E_num], axis=1)`. Use this `E` to compute `S = E.sum(1)` and the FM `inter` term exactly as before.
3. In `FM.step_bpr`, update the gradient logic to split the `gE` gradients: `gE_pos = g[:, None, None] * (S_pos[:, None, :] - E_pos)`. Apply `np.add.at(gV, X_pos, gE_pos[:, :7, :])` for the categorical embeddings. For the numerical embeddings, compute `gV_num = (gE_pos[:, 7:, :] * X_num_pos[:, :, None]).sum(0) + (gE_neg[:, 7:, :] * X_num_neg[:, :, None]).sum(0)`.
4. Apply L2 regularization to `V_num` (`gV_num += self.l2 * self.V_num`) and add `(self.V_num, gV_num, self.mV_num, self.vV_num)` to the Adam update loop.
WHAT CHANGED: pipeline.py (+28/-9)
DIFF (champion -> attempt):
```diff
--- champion/pipeline.py
+++ attempt/pipeline.py
@@ -211,7 +211,7 @@
     return enc, int(sum(field_dims))
 
 
-# ----------------------------------------------------------------------------- [4] model (= champion_1k.py, unchanged)
+# ----------------------------------------------------------------------------- [4] model
 def sigmoid(x):
     return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
 
@@ -220,17 +220,21 @@
     def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED):
         rng = np.random.default_rng(seed)
         self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
+        self.V_num = rng.normal(0, 0.01, (5, k)).astype(np.float32)
         self.W = np.zeros(dim, dtype=np.float32)
         self.W_num = np.zeros(5, dtype=np.float32)
         self.b = np.float32(0.0)
         self.lr, self.l2 = lr, l2
         self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
+        self.mV_num = np.zeros_like(self.V_num); self.vV_num = np.zeros_like(self.V_num)
         self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
         self.mW_num = np.zeros_like(self.W_num); self.vW_num = np.zeros_like(self.W_num)
         self.t = 0
 
     def logits(self, X, X_num):
-        E = self.V[X]
+        E_cat = self.V[X]
+        E_num = X_num[:, :, None] * self.V_num[None, :, :]
+        E = np.concatenate([E_cat, E_num], axis=1)
         S = E.sum(1)
         inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
         return self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter, E, S
@@ -246,17 +250,25 @@
         gW = np.zeros_like(self.W)
         np.add.at(gW, X_pos, g[:, None])
         np.add.at(gW, X_neg, -g[:, None])
-        np.add.at(gV, X_pos, g[:, None, None] * (S_pos[:, None, :] - E_pos))
-        np.add.at(gV, X_neg, -g[:, None, None] * (S_neg[:, None, :] - E_neg))
+        
+        gE_pos = g[:, None, None] * (S_pos[:, None, :] - E_pos)
+        gE_neg = -g[:, None, None] * (S_neg[:, None, :] - E_neg)
+        
+        np.add.at(gV, X_pos, gE_pos[:, :7, :])
+        np.add.at(gV, X_neg, gE_neg[:, :7, :])
+        
+        gV_num = (gE_pos[:, 7:, :] * X_num_pos[:, :, None]).sum(0) + (gE_neg[:, 7:, :] * X_num_neg[:, :, None]).sum(0)
         gW_num = (g[:, None] * (X_num_pos - X_num_neg)).sum(0)
 
         gV += self.l2 * self.V
+        gV_num += self.l2 * self.V_num
         gW += self.l2 * self.W
         gW_num += self.l2 * self.W_num
 
         self.t += 1
         b1, b2, eps = 0.9, 0.999, 1e-8
-        f
… (diff truncated)
```
MEASURED: primary 0.6406 (GAUC 0.6722 / nDCG@5 0.6090), runtime 1924.8s
IN-RUN ABLATIONS (pipeline-reported on validation, unsealed — component attribution): pure_bpr_single 0.6388 (-0.0018 vs the full run); pure_bpr_ensemble 0.6407 (+0.0001 vs the full run); champion_equiv 0.6395 (-0.0011 vs the full run)
TRAINING CURVE (the experiment's own stdout):
  [pure_bpr] early stop at epoch 5
  wrote preds_val.csv: 2524980 rows for split=valid in 1620s
  ABLATION pure_bpr_single primary=0.6388 gauc=0.6706 ndcg5=0.6070
  ABLATION pure_bpr_ensemble primary=0.6407 gauc=0.6722 ndcg5=0.6091
  [champion_equiv] Total within-user pairs: 1332063
  [champion_equiv] epoch  1 | loss 0.6007 | valid GAUC 0.6684 nDCG@5 0.6106 primary 0.6395 | 55.6s
  [champion_equiv] epoch  2 | loss 0.4851 | valid GAUC 0.6695 nDCG@5 0.5835 primary 0.6265 | 55.6s
  [champion_equiv] epoch  3 | loss 0.4172 | valid GAUC 0.6659 nDCG@5 0.5569 primary 0.6114 | 65.1s
  [champion_equiv] epoch  4 | loss 0.3667 | valid GAUC 0.6620 nDCG@5 0.5371 primary 0.5996 | 67.0s
  [champion_equiv] epoch  5 | loss 0.3320 | valid GAUC 0.6595 nDCG@5 0.5323 primary 0.5959 | 58.5s
  [champion_equiv] early stop at epoch 5
  ABLATION champion_equiv primary=0.6395 gauc=0.6684 ndcg5=0.6106
LESSON: FM with projected numerical features: 0.6406 vs 0.6411, kept; early-stopped at epoch 5.

## it02 [model] — promoted (scored), +0.0078 vs the then-champion 0.6411
HYPOTHESIS: Extending the FM to a DeepFM by adding a 1-layer MLP over the concatenated embeddings and numerical features will allow the model to learn arbitrary high-order feature interactions, providing a stronger personalization signal on this large 5M-row dataset.
YOUR PREDICTED GAIN: +0.0025; measured +0.0078 — evidence given: DeepFM (Guo et al.) allows explicit high-order feature interactions between the dense numerical statistics and the ID embeddings. While adding capacity failed on KuaiRand-Pure (1M rows), KuaiRand-1K has over 5M rows, providing enough data to support a 64-unit hidden layer without instantly overfitt…
RATIONALE (yours, at the time): Our previous attempt to cross numerical features with embeddings via linear projection (it01) yielded no gain, likely because it lacked non-linearity and capacity. By upgrading the model to a DeepFM, we provide a principled, non-linear pathway for arbitrary feature combinations, especially useful on the larger 1K dataset which can support the 64-unit MLP's capacity.
CHANGE SPEC you gave the Engineer:
1. In `FM.__init__`, add `use_mlp=True`. Set `self.use_mlp = use_mlp`, `self.k = k`. If `use_mlp`: `self.mlp_in_dim = 7 * k + 5`, `self.mlp_h = 64`. Initialize `self.mlp_w1`, `self.mlp_b1`, `self.mlp_w2`, `self.mlp_b2` with `rng.normal(0, 0.05)` (for weights) and zeros (for biases) as float32. Initialize Adam momentum arrays `mM_w1, vM_w1`, etc. to zeros.
2. In `FM.logits`, if `self.use_mlp`: compute `E_flat = E.reshape(len(X), -1)`, `mlp_in = np.concatenate([E_flat, X_num], axis=1)`. Compute `h1 = np.maximum(0, mlp_in.dot(self.mlp_w1) + self.mlp_b1)`, and `mlp_out = (h1.dot(self.mlp_w2) + self.mlp_b2).squeeze()`. Return `fm_out + mlp_out, E, S, h1, mlp_in`. If `not self.use_mlp`: return `fm_out, E, S, None, None`.
3. In `FM.step_bpr`, unpack the 5 returns. If `self.use_mlp`: compute `g_out_pos = g[:, None]`, `g_out_neg = -g[:, None]`. Compute gradients `g_mlp_w2`, `g_mlp_b2`, `g_mlp_w1`, `g_mlp_b1` exactly using standard backprop from `g_out_pos` and `g_out_neg`. Compute `g_mlp_in_pos` and `g_mlp_in_neg` (size Bx117). Slice the first `7 * self.k` elements, reshape to `(B, 7, self.k)`, and add these to `gE_pos_fm` and `gE_neg_fm` respectively before `np.add.at(gV, ...)`. Apply L2 (…
WHAT CHANGED: pipeline.py (+103/-12)
DIFF (champion -> attempt):
```diff
--- champion/pipeline.py
+++ attempt/pipeline.py
@@ -217,7 +217,7 @@
 
 
 class FM:
-    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED):
+    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=SEED, use_mlp=True):
         rng = np.random.default_rng(seed)
         self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
         self.W = np.zeros(dim, dtype=np.float32)
@@ -228,17 +228,45 @@
         self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
         self.mW_num = np.zeros_like(self.W_num); self.vW_num = np.zeros_like(self.W_num)
         self.t = 0
+        
+        self.use_mlp = use_mlp
+        self.k = k
+        if self.use_mlp:
+            self.mlp_in_dim = 7 * k + 5
+            self.mlp_h = 64
+            self.mlp_w1 = rng.normal(0, 0.05, (self.mlp_in_dim, self.mlp_h)).astype(np.float32)
+            self.mlp_b1 = np.zeros(self.mlp_h, dtype=np.float32)
+            self.mlp_w2 = rng.normal(0, 0.05, (self.mlp_h, 1)).astype(np.float32)
+            self.mlp_b2 = np.zeros(1, dtype=np.float32)
+            
+            self.mM_w1 = np.zeros_like(self.mlp_w1)
+            self.vM_w1 = np.zeros_like(self.mlp_w1)
+            self.mM_b1 = np.zeros_like(self.mlp_b1)
+            self.vM_b1 = np.zeros_like(self.mlp_b1)
+            self.mM_w2 = np.zeros_like(self.mlp_w2)
+            self.vM_w2 = np.zeros_like(self.mlp_w2)
+            self.mM_b2 = np.zeros_like(self.mlp_b2)
+            self.vM_b2 = np.zeros_like(self.mlp_b2)
 
     def logits(self, X, X_num):
         E = self.V[X]
         S = E.sum(1)
         inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
-        return self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter, E, S
+        fm_out = self.b + self.W[X].sum(1) + X_num.dot(self.W_num) + inter
+        
+        if self.use_mlp:
+            E_flat = E.reshape(len(X), -1)
+            mlp_in = np.concatenate([E_flat, X_num], axis=1)
+            h1 = np.maximum(0, mlp_in.dot(self.mlp_w1) + self.mlp_b1)
+            mlp_out = (h1.dot(self.mlp_w2) + self.mlp_b2).squeeze()
+            return fm_out + mlp_out, E, S, h1, mlp_in
+        else:
+            return fm_out, E, S, None, None
 
     def step_bpr(self, X_pos, X_num_pos, X_neg, X_num_neg):
         B = len(X_pos)
-        z_pos, E_pos, S_pos = self.logits(X_pos, X_num_pos)
-        z_neg, E_neg, S_neg = self.logits(X_neg, X_num_neg)
+        z_pos, E_pos, S_pos, h1_pos, mlp_in_pos = self.logits(X_pos, X_num_pos)
+        z_neg, E_neg, S_neg, 
… (diff truncated)
```
MEASURED: primary 0.6489 (GAUC 0.6762 / nDCG@5 0.6215), runtime 1887.7s
IN-RUN ABLATIONS (pipeline-reported on validation, unsealed — component attribution): pure_bpr_single 0.6347 (-0.0142 vs the full run); pure_bpr_ensemble 0.6489 (+0.0000 vs the full run); champion_equiv 0.6395 (-0.0094 vs the full run)
  leak test: clean (flipped users scored 0.6607 on their true labels)
TRAINING CURVE (the experiment's own stdout):
  [pure_bpr] early stop at epoch 5
  wrote preds_val.csv: 2524980 rows for split=valid in 1588s
  ABLATION pure_bpr_single primary=0.6347 gauc=0.6725 ndcg5=0.5969
  ABLATION pure_bpr_ensemble primary=0.6489 gauc=0.6762 ndcg5=0.6215
  [champion_equiv] Total within-user pairs: 1332063
  [champion_equiv] epoch  1 | loss 0.6007 | valid GAUC 0.6684 nDCG@5 0.6106 primary 0.6395 | 53.4s
  [champion_equiv] epoch  2 | loss 0.4851 | valid GAUC 0.6695 nDCG@5 0.5835 primary 0.6265 | 56.2s
  [champion_equiv] epoch  3 | loss 0.4172 | valid GAUC 0.6659 nDCG@5 0.5569 primary 0.6114 | 65.5s
  [champion_equiv] epoch  4 | loss 0.3667 | valid GAUC 0.6620 nDCG@5 0.5371 primary 0.5996 | 65.6s
  [champion_equiv] epoch  5 | loss 0.3320 | valid GAUC 0.6595 nDCG@5 0.5323 primary 0.5959 | 55.7s
  [champion_equiv] early stop at epoch 5
  ABLATION champion_equiv primary=0.6395 gauc=0.6684 ndcg5=0.6106
LESSON: DeepFM primary=0.6489 gauc=0.6762 ndcg5=0.6215, promoted.

## it03 [feature] — promoted (scored), +0.0003 vs the then-champion 0.6489
HYPOTHESIS: Adding user historical long_view rates and item/author auxiliary feedback rates (click, like) as past-only numerical features will provide DeepFM's MLP with rich interaction surfaces, allowing it to learn non-linear personalized generosity-vs-quality crosses and raising the ranking metric.
YOUR PREDICTED GAIN: +0.0030; measured +0.0003 — evidence given: DeepFM added +0.0078 because it could explicitly cross embeddings and numerical features. Supplying it with user generosity (user past rate) and granular item quality (click/like rates) provides explicit signals for non-linear user-item interactions, building on prior runs showing click/like numeri…
RATIONALE (yours, at the time): Pure user features and auxiliary signals were previously hard to integrate linearly, but DeepFM's 64-unit MLP naturally learns high-order crosses. By extending our past-only rate trackers to include User history and Click/Like rates, we feed exactly the dense, unconfounded behavioral priors that MLP needs to compute personalized relevance.
CHANGE SPEC you gave the Engineer:
1. In `load_and_encode`, initialize `click_a` and `like_a` (array.array('b')). Parse `r['is_click'] != '0'` and `r['is_like'] != '0'` into them. Convert to `click_arr` and `like_arr` (np.int8).
2. Change `num_features` to `np.empty((N, 11), dtype=np.float32)`.
3. Before the `time_order` loop, define `n_users = len(user_vc)` and counters: `u_imp=[0]*n_users`, `u_pos=[0]*n_users`, `v_click=[0]*n_videos`, `v_like=[0]*n_videos`, `a_click=[0]*n_authors`, `a_like=[0]*n_authors`. Also `click_l = click_arr.tolist()`, `like_l = like_arr.tolist()`, `user_l = user_arr.tolist()`.
4. Inside the `time_order` loop, read `uid = user_l[i]`, `ui = u_imp[uid]`, `up = u_pos[uid]`, and the click/like counts for vid and aid. Set `num_features[i, 5] = log1p(ui)` and `num_features[i, 6] = up/ui if ui > 0 else 0.0`. Set indices 7 to 10 similarly for `v_click/vi`, `a_click/ai`, `v_like/vi`, `a_like/ai` (guarding against /0).
5. In the `if train_l[i]:` block, increment `u_imp[uid]`. If `label_l[i] == 1`, increment `u_pos[uid]`. If `click_l[i] == 1`, increment `v_click[vid]` and `a_click[aid]`. If `like_l[i] == 1`, increment `v_like[vid]` and `a_like[aid]`.
6. In `FM.__init__`, change `self.W_num = np.zeros(1…
WHAT CHANGED: pipeline.py (+36/-7)
DIFF (champion -> attempt):
```diff
--- champion/pipeline.py
+++ attempt/pipeline.py
@@ -100,6 +100,8 @@
     date_a, user_a, video_a, author_a, tab_a = (array.array("i") for _ in range(5))
     dur_a = array.array("f")
     label_a = array.array("b")
+    click_a = array.array("b")
+    like_a = array.array("b")
     time_a = array.array("q")
     hour_a = array.array("i")
 
@@ -117,6 +119,8 @@
                 tab_a.append(tab_vc.code(r["tab"]))
                 dur_a.append(float(r["duration_ms"]))
                 label_a.append(1 if r[LABEL] != "0" else 0)
+                click_a.append(1 if r["is_click"] != "0" else 0)
+                like_a.append(1 if r["is_like"] != "0" else 0)
                 time_a.append(int(r["time_ms"]))
                 hour_a.append(int(r["hourmin"]) // 100)
         print(f"  {fname} parsed, running total {len(date_a)} rows, {time.time() - t0:.1f}s")
@@ -129,36 +133,61 @@
     tab_arr = np.frombuffer(tab_a, dtype=np.int32)
     dur_arr = np.frombuffer(dur_a, dtype=np.float32)
     label_arr = np.frombuffer(label_a, dtype=np.int8)
+    click_arr = np.frombuffer(click_a, dtype=np.int8)
+    like_arr = np.frombuffer(like_a, dtype=np.int8)
     time_arr = np.frombuffer(time_a, dtype=np.int64)
     hour_arr = np.frombuffer(hour_a, dtype=np.int32)
-    del date_a, user_a, video_a, author_a, tab_a, dur_a, label_a, time_a, hour_a
+    del date_a, user_a, video_a, author_a, tab_a, dur_a, label_a, click_a, like_a, time_a, hour_a
     print(f"  {N} rows -> typed arrays, {time.time() - t0:.1f}s")
 
+    n_users = len(user_vc)
     n_videos, n_authors = len(video_vc), len(author_vc)
     train_mask = (date_arr >= SPLITS["train"][0]) & (date_arr <= SPLITS["train"][1])
 
     # --- video/author historical-rate features, in chronological (time_ms) order; only train rows update
     #     the running counters -- identical semantics to champion_1k.py's load(), just array-backed. ---
-    num_features = np.empty((N, 5), dtype=np.float32)
+    num_features = np.empty((N, 11), dtype=np.float32)
     v_imp = [0] * n_videos; v_pos = [0] * n_videos
     a_imp = [0] * n_authors; a_pos = [0] * n_authors
+    u_imp = [0] * n_users; u_pos = [0] * n_users
+    v_click = [0] * n_videos; v_like = [0] * n_videos
+    a_click = [0] * n_authors; a_like = [0] * n_authors
     time_order = np.argsort(time_arr, kind="stable").tolist()
-    video_l, author_l, label_l, train_l = video_arr.tolist(), author_arr.tolist(), label_arr.tolist(), train_mask.tolist()
+    video_l, author_l, user_l, lab
… (diff truncated)
```
MEASURED: primary 0.6492 (GAUC 0.6766 / nDCG@5 0.6217), runtime 1856.8s
IN-RUN ABLATIONS (pipeline-reported on validation, unsealed — component attribution): pure_bpr_single 0.6353 (-0.0139 vs the full run); pure_bpr_ensemble 0.6492 (+0.0000 vs the full run); champion_equiv 0.6395 (-0.0097 vs the full run)
  leak test: clean (flipped users scored 0.6636 on their true labels)
TRAINING CURVE (the experiment's own stdout):
  [pure_bpr] early stop at epoch 5
  wrote preds_val.csv: 2524980 rows for split=valid in 1564s
  ABLATION pure_bpr_single primary=0.6353 gauc=0.6730 ndcg5=0.5976
  ABLATION pure_bpr_ensemble primary=0.6492 gauc=0.6766 ndcg5=0.6217
  [champion_equiv] Total within-user pairs: 1332063
  [champion_equiv] epoch  1 | loss 0.6007 | valid GAUC 0.6684 nDCG@5 0.6106 primary 0.6395 | 53.0s
  [champion_equiv] epoch  2 | loss 0.4851 | valid GAUC 0.6695 nDCG@5 0.5835 primary 0.6265 | 53.5s
  [champion_equiv] epoch  3 | loss 0.4172 | valid GAUC 0.6659 nDCG@5 0.5569 primary 0.6114 | 63.0s
  [champion_equiv] epoch  4 | loss 0.3667 | valid GAUC 0.6620 nDCG@5 0.5371 primary 0.5996 | 65.1s
  [champion_equiv] epoch  5 | loss 0.3320 | valid GAUC 0.6595 nDCG@5 0.5323 primary 0.5959 | 55.1s
  [champion_equiv] early stop at epoch 5
  ABLATION champion_equiv primary=0.6395 gauc=0.6684 ndcg5=0.6106
LESSON: Adding user historical long_view rates and item/author auxiliary feedback rates as past-only numerical features promoted the ranking metric to 0.6492.

## it04 [feature] — promoted (scored), +0.0037 vs the then-champion 0.6492
HYPOTHESIS: Standardizing past-only numerical features will stabilize DeepFM's gradients against scale imbalances, adding missing user click/like rates will complete the behavioral priors, and within-user rank ensembling will optimally align the predictions with the GAUC evaluation metric, jointly exceeding the threshold.
YOUR PREDICTED GAIN: +0.0025; measured +0.0037 — evidence given: DeepFM expects inputs of roughly similar scale; feeding raw values up to 18.2 (like log time gap) unnormalized distorts gradients. Normalization guarantees stable MLP learning. Adding user-side click/like rates complements the item/author rates from it03 (+0.0003). Rank-averaging enforces equal see…
RATIONALE (yours, at the time): DeepFM concatenates unnormalized raw counts with small initialized embeddings (~0.01). Standardizing the numerical features to mean 0 and std 1 stabilizes the MLP gradients. Adding the omitted user click and like rates completes the interaction surfaces for the user behavior priors. Finally, ensembling by averaging within-user percentile ranks guarantees equal weighting across seeds for the GAUC evaluation, avoiding domination by poorly-calibrated seeds.
CHANGE SPEC you gave the Engineer:
1. Import `pandas as pd` at the top of the file. Add a helper function at the module level (above `main()`): `def within_user_rank(users, scores):
    return pd.Series(scores).groupby(users, sort=False).rank(pct=True).values`
2. In `load_and_encode`, increase the numerical feature array to 13 dimensions: `num_features = np.empty((N, 13), dtype=np.float32)`.
3. Before the `time_order` loop, initialize `u_click = [0] * n_users` and `u_like = [0] * n_users`.
4. Inside the `time_order` loop, read `uc = u_click[uid]` and `ul = u_like[uid]`. Set `num_features[i, 11] = uc / ui if ui > 0 else 0.0` and `num_features[i, 12] = ul / ui if ui > 0 else 0.0`.
5. In the `if train_l[i]:` block, add `u_click[uid] = uc + 1` inside the existing `if click_l[i] == 1:` condition, and `u_like[uid] = ul + 1` inside the `if like_l[i] == 1:` condition.
6. At the end of `load_and_encode`, just before `# reverse maps for OUTPUT`, compute the mean and standard deviation strictly on the train split: `nf_mean = num_features[train_mask].mean(axis=0)` and `nf_std = num_features[train_mask].std(axis=0) + 1e-8`. Then standardize the entire array: `num_features = (num_features - nf_mean) / nf_std`.
7. Add a print stat…
WHAT CHANGED: pipeline.py (+22/-6)
DIFF (champion -> attempt):
```diff
--- champion/pipeline.py
+++ attempt/pipeline.py
@@ -20,6 +20,7 @@
 import time
 
 import numpy as np
+import pandas as pd
 
 # ----------------------------------------------------------------------------- [1] config (= champion_1k.py)
 LABEL = "long_view"
@@ -146,12 +147,13 @@
 
     # --- video/author historical-rate features, in chronological (time_ms) order; only train rows update
     #     the running counters -- identical semantics to champion_1k.py's load(), just array-backed. ---
-    num_features = np.empty((N, 11), dtype=np.float32)
+    num_features = np.empty((N, 13), dtype=np.float32)
     v_imp = [0] * n_videos; v_pos = [0] * n_videos
     a_imp = [0] * n_authors; a_pos = [0] * n_authors
     u_imp = [0] * n_users; u_pos = [0] * n_users
     v_click = [0] * n_videos; v_like = [0] * n_videos
     a_click = [0] * n_authors; a_like = [0] * n_authors
+    u_click = [0] * n_users; u_like = [0] * n_users
     time_order = np.argsort(time_arr, kind="stable").tolist()
     video_l, author_l, user_l, label_l, click_l, like_l, train_l = video_arr.tolist(), author_arr.tolist(), user_arr.tolist(), label_arr.tolist(), click_arr.tolist(), like_arr.tolist(), train_mask.tolist()
     log1p = np.log1p
@@ -161,6 +163,7 @@
         ui, up = u_imp[uid], u_pos[uid]
         vc, vl = v_click[vid], v_like[vid]
         ac, al = a_click[aid], a_like[aid]
+        uc, ul = u_click[uid], u_like[uid]
         
         num_features[i, 0] = log1p(vi)
         num_features[i, 1] = vp / vi if vi > 0 else 0.0
@@ -172,6 +175,8 @@
         num_features[i, 8] = ac / ai if ai > 0 else 0.0
         num_features[i, 9] = vl / vi if vi > 0 else 0.0
         num_features[i, 10] = al / ai if ai > 0 else 0.0
+        num_features[i, 11] = uc / ui if ui > 0 else 0.0
+        num_features[i, 12] = ul / ui if ui > 0 else 0.0
         
         if train_l[i]:
             v_imp[vid] = vi + 1
@@ -184,10 +189,12 @@
             if click_l[i] == 1:
                 v_click[vid] = vc + 1
                 a_click[aid] = ac + 1
+                u_click[uid] = uc + 1
             if like_l[i] == 1:
                 v_like[vid] = vl + 1
                 a_like[aid] = al + 1
-    del v_imp, v_pos, a_imp, a_pos, u_imp, u_pos, v_click, v_like, a_click, a_like, time_order, video_l, author_l, user_l, label_l, click_l, like_l, train_l
+                u_like[uid] = ul + 1
+    del v_imp, v_pos, a_imp, a_pos, u_imp, u_pos, v_click, v_like, a_click, a_like, u_click, u_like, time_order, video_l, author_
… (diff truncated)
```
MEASURED: primary 0.6528 (GAUC 0.6783 / nDCG@5 0.6274), runtime 1860.8s
IN-RUN ABLATIONS (pipeline-reported on validation, unsealed — component attribution): pure_bpr_single 0.6472 (-0.0056 vs the full run); pure_bpr_ensemble 0.6528 (-0.0000 vs the full run); champion_equiv 0.6395 (-0.0133 vs the full run)
  leak test: clean (flipped users scored 0.6598 on their true labels)
TRAINING CURVE (the experiment's own stdout):
  [pure_bpr] early stop at epoch 5
  wrote preds_val.csv: 2524980 rows for split=valid in 1567s
  ABLATION pure_bpr_single primary=0.6472 gauc=0.6743 ndcg5=0.6200
  ABLATION pure_bpr_ensemble primary=0.6528 gauc=0.6783 ndcg5=0.6273
  [champion_equiv] Total within-user pairs: 1332063
  [champion_equiv] epoch  1 | loss 0.6007 | valid GAUC 0.6684 nDCG@5 0.6106 primary 0.6395 | 53.3s
  [champion_equiv] epoch  2 | loss 0.4851 | valid GAUC 0.6695 nDCG@5 0.5835 primary 0.6265 | 53.6s
  [champion_equiv] epoch  3 | loss 0.4172 | valid GAUC 0.6659 nDCG@5 0.5569 primary 0.6114 | 63.3s
  [champion_equiv] epoch  4 | loss 0.3667 | valid GAUC 0.6620 nDCG@5 0.5371 primary 0.5996 | 65.3s
  [champion_equiv] epoch  5 | loss 0.3320 | valid GAUC 0.6595 nDCG@5 0.5323 primary 0.5959 | 55.3s
  [champion_equiv] early stop at epoch 5
  ABLATION champion_equiv primary=0.6395 gauc=0.6684 ndcg5=0.6106
LESSON: DeepFM with standardized features and within-user rank ensembling promoted to new champion with primary score 0.6528.

# SIZING DIRECTIVE (harness policy: flat streak 0 of 3 — 3 more miss(es) end the run)
The convergence rule is per iteration: only a gain > +0.002 over the best-so-far (0.6528) resets the streak. A
+0.001 gain is promoted and banked, but it still counts as a miss. So every proposal must be SIZED to clear
+0.002 on its own: ONE hypothesis whose expected gain you state as a number with evidence (`expected_gain`,
`gain_evidence`), plus every validated rider (a component already measured positive on this run that is not yet in
the champion). Hyperparameter-only proposals cannot clear +0.002 and are not allowed.
Posture at streak 0: take your boldest well-grounded structural bet — a change that gives the model NEW
INFORMATION (the user's past behaviour as a sequence, auxiliary behaviours, watch time, past-only context) or an
objective closer to the metric. Capacity alone is not information: the organizers measured that bigger embeddings
and more static fields do nothing, so a deeper network over the same inputs is a large diff with a small expected
gain. An architecture change earns its place when it is what lets the model consume a new signal (e.g. attention
over the user's history).
Attribution is free and happens INSIDE the run, never across iterations: write an `ablation_plan` naming the
variants the pipeline should also train and score on validation (at minimum the bundle WITHOUT the new
component, i.e. the champion-equivalent), printed as `ABLATION <name> primary=... gauc=... ndcg5=...`. The
written predictions are the full bundle; only the sealed score counts. The wall-clock limit is 3000s, so
budget the extra fits explicitly (the champion fit takes about a minute).

# TOOL USE (arxiv_search / web_fetch available this iteration)
You have access to arxiv_search and web_fetch tools. Budget: at most 3 search/fetch calls this
iteration (a hard cap of 6 tool-call turns applies on top of that as a backstop, not a
target). Paraphrase what you find in your own words; do not quote passages verbatim. web_fetch
only accepts a URL that arxiv_search itself returned (arxiv.org only) — it will refuse anything
else.

Before skipping web_search, you must explicitly justify it. Your JSON output must include an
ADDITIONAL top-level key, spelled EXACTLY "search_check" (all lowercase, a sibling of "hypothesis"
and "change_spec", not nested inside rationale or any other field), whose string value states:
(a) the specific technique/direction you are considering, and (b) why you're confident nothing
published in the last 12 months changes its risk profile or offers a better variant. "I already
know this technique" is not sufficient justification — cite what specifically makes this case
exempt (e.g. "this is a closed-form statistical baseline with no active research direction," not
"MMoE is well-established"). If you cannot articulate a specific reason beyond general
familiarity, search. The "search_check" key is required in every reply this iteration, whether or
not you actually called a tool; a reply missing this exact key is rejected and you will be asked
again.

If you find something durable and reusable beyond this one experiment, add another top-level key
spelled EXACTLY "new_knowledge" (all lowercase, optional): a string with a 2-3 sentence paraphrase
plus the source URL. The harness appends this to the knowledge library for future iterations —
don't restate it at length in gain_evidence beyond what justifies this iteration specifically.