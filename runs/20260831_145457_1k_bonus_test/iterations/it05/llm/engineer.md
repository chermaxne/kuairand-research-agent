# engineer — engineer (model google/gemini-3.1-pro-preview, 23364 tokens)

## system block 1

# ROLE: Engineer

You implement one experiment on the champion pipeline of an autonomous ML research agent
(KuaiRand-Pure ranking, label `long_view`, metric GAUC/nDCG@5 within user). You receive the
Researcher's change specification and the exact champion file(s). You output the complete modified
file(s). A deterministic harness then runs `python pipeline.py --data <dir> --split val --out
preds_val.csv` in a sandbox and scores the predictions with the organizers' sealed evaluator.

## Hard rules
1. Implement the change specification faithfully and MINIMALLY. Do not refactor, rename, reformat or
   "improve" unrelated code — the harness diffs champion vs. attempt and judges read that diff.
2. Keep the pipeline contract intact: CLI flags `--data`, `--split val|valid|test`, `--out`; write EVERY
   row of the requested split in data order as `row_id,user_id,video_id,score` (ids echoed exactly as
   read from the CSV, finite float scores); exit 0 on success. `--split test` must keep working.
3. Fit ONLY on the train split (dates 20220408–20220421). Validation rows may be used for early stopping
   / model selection only. Never read labels of the split you are predicting for anything else.
4. No leakage: never use same-row feedback columns (`is_click`, `is_like`, `play_time_ms`, `long_view`, …) as
   input features of the row being scored; any aggregate feature must be computed from strictly earlier
   dates than the row it describes (past-only / rolling), and for validation/test rows only from train.
   When you extend the row tuple or the field list, re-check every index: the label element must never end up
   inside `raw(x)` / the encoded fields. The harness re-runs every would-be promotion on a copy where ~10% of the
   validation users have corrupted labels and checks that those users are still ranked well — a leaked score is
   worth nothing. Consequence for self-checks: print validation sanity numbers, but do NOT hard-assert on the
   validation metric (an `assert primary > 0.55` crashed that re-run once); assert on train-side quantities
   (train GAUC after epoch 1, pair count, vocabulary sizes) instead.
5. Sandbox: no network, no package installs, no subprocesses, no writes outside the working directory.
   Only numpy, pandas, scikit-learn, lightgbm, torch (CPU) and the standard library are available.
   Keep memory moderate (16 GB box) and respect the runtime limit stated in the contract.
6. Never print fake metrics or mock results. Never skip work with hardcoded outputs.
7. Keep determinism: fixed seeds, no time-dependent randomness.
8. Runtime is a hard budget (the kill limit is stated in the pipeline contract; aim for well under half of it). Vectorise every per-user operation with pandas
   `groupby(...).rank/transform` or numpy `np.unique(..., return_inverse)` + `np.argsort` / `np.add.at`. NEVER loop
   over users and mask the rows inside the loop (quadratic — it took a previous run past the kill limit). Build
   pair pools / index structures once and resample by array indexing per epoch. Print per-epoch timing.
9. In-run attribution, under an explicit time budget. `KUAIRAND_TIME_BUDGET_S` is in the environment and the process
   is killed at it. Structure every pipeline as: (a) fit the FULL bundle and write its predictions — this must finish
   inside 40% of the budget; (b) then, for each variant in the ABLATION PLAN, check the clock and start it only if at
   least 25% of the budget remains, otherwise print `ABLATION <name> skipped: out of time budget`; (c) score each
   variant you did run with the official `evaluate()` and print exactly one line
   `ABLATION <name> primary=<float> gauc=<float> ndcg5=<float>` — real numbers from real fits, never estimated.
   Ablation variants are DIAGNOSTICS: make them cheap (subsample the training rows, cap the rounds/epochs, one seed).
   A variant must never cost as much as the full fit. The written predictions are always the full bundle.
9b. Do not score the validation metric on a tight loop. `evaluate()` runs over ~125k rows; call it for early stopping
   at most every ~50 boosting rounds or once per epoch. Scoring every 10 rounds turned a 30-second LightGBM fit into
   6 minutes and blew the whole budget on one iteration (run ten16). Print per-fit timing so the cost is visible.
10. Fast path: when the environment variable `KUAIRAND_FAST=1` is set (the harness sets it for its flipped-label
   re-run), skip in-run ablations, sweeps and extra seeds (one seed, the full bundle only) — but NEVER change which
   features or labels are used; the feature/label code path must be identical with and without the flag.
11. If a detail of the spec is impossible under these rules, implement the closest faithful version and
   say what you changed in a single `NOTE:` line BEFORE the file blocks.

## Output format (strict)
For every file you change or add, output the COMPLETE file (not a diff, not a snippet):

=== FILE: pipeline.py ===
```python
<entire file content>
```
=== END FILE ===

Nothing else except an optional leading `NOTE:` line. The harness discards anything it cannot parse.

## user

# Change specification (from the Researcher)
HYPOTHESIS: Adding past-only user-tab specific historical impression and positive rates as numerical features will give the DeepFM MLP a highly personalized, context-aware baseline for each user's generosity across different UI tabs, improving within-user ranking.
CATEGORY: feature
EXPECTED RISK: low
EXPECTED GAIN (Researcher's prediction): 0.0025
ABLATION PLAN (variants to also score and print as ABLATION lines): pure_bpr_single: single-seed version of this full bundle; champion_equiv: bundle without MLP or numerical features to isolate baseline drift.
CHANGE SPEC:
1. In `load_and_encode`, allocate 15 numerical features: `num_features = np.empty((N, 15), dtype=np.float32)`.
2. Before the `time_order` loop, define `n_tabs = len(tab_vc)` and initialize flat lists for speed: `u_tab_imp = [0] * (n_users * n_tabs)` and `u_tab_pos = [0] * (n_users * n_tabs)`.
3. When creating lists for the loop, add `tab_l`: `video_l, author_l, user_l, tab_l, label_l, click_l, like_l, train_l = video_arr.tolist(), author_arr.tolist(), user_arr.tolist(), tab_arr.tolist(), label_arr.tolist(), click_arr.tolist(), like_arr.tolist(), train_mask.tolist()`.
4. Inside the `time_order` loop, change the unpack to `vid, aid, uid, tid = video_l[i], author_l[i], user_l[i], tab_l[i]`. Compute `ut_idx = uid * n_tabs + tid` and read `uti = u_tab_imp[ut_idx]`, `utp = u_tab_pos[ut_idx]`.
5. Assign indices 13 and 14: `num_features[i, 13] = log1p(uti)` and `num_features[i, 14] = utp / uti if uti > 0 else 0.0`.
6. In the `if train_l[i]:` block, add `u_tab_imp[ut_idx] = uti + 1` and inside `if label_l[i] == 1:`, add `u_tab_pos[ut_idx] = utp + 1`.
7. In `FM.__init__`, change `self.W_num = np.zeros(15, dtype=np.float32)` and inside `if self.use_mlp:`, change `self.mlp_in_dim = 7 * k + 15`.

# Current champion files
=== FILE: pipeline.py ===
```python
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
```
=== END FILE ===

# Pipeline contract
`python pipeline.py --data <data_dir> --split val --out preds_val.csv`
- Train ONLY on the train split (dates 20220408-20220421). Validation rows may be used for early stopping / model selection only.
- Write EVERY validation row, in data.load() order, as `row_id,user_id,video_id,score` (row_id from 0, ids echoed exactly as read, finite scores).
- `--split test` must keep working unchanged (it is used once, at finalize, on the champion).
- Exit 0 on success. Single process, no network, no package installs, only pre-installed libraries
  (numpy, pandas, scikit-learn, lightgbm, torch-cpu). Same-row feedback columns are NOT features (leakage).
- Hard wall-clock limit: 3000s for the whole run (load + train + predict).
- TIME BUDGET (hard): `KUAIRAND_TIME_BUDGET_S` is in the environment (3000s here) and the process is killed at it.
  Budget it explicitly with arithmetic, not hope:
  1. Fit the FULL bundle first and WRITE ITS PREDICTIONS before anything else. It must finish inside 40% of the budget.
  2. Only then run ablations, and before each one check `time.time() - t0` against the budget: start a variant only if
     at least 25% of the budget remains, else print `ABLATION <name> skipped: <reason>` and move on.
  3. Ablation variants are DIAGNOSTICS, not submissions: make them cheap (a subsample of the training rows, or a fixed
     small number of rounds/epochs, or one seed). A variant must never cost as much as the full fit.
- In-run attribution: for each variant you do run, score it on validation with the official `evaluate()` and print one
  line `ABLATION <name> primary=<f> gauc=<f> ndcg5=<f>` (real numbers from real fits only). The written predictions are
  always the full bundle.
- Evaluating the metric is expensive (~125k rows). For early stopping, score at most every ~50 boosting rounds or once
  per epoch — NOT every few rounds. Scoring after every 10 rounds turned a 30 s fit into 6 minutes in run ten16.
- If this change replaces or adds a LOSS FUNCTION (e.g. pointwise -> pairwise/BPR, adding an auxiliary
  head): the learning rate and any other optimizer constants were tuned for the OLD objective's gradient
  scale and are not guaranteed to transfer. Reusing them unchanged is a common cause of loss divergence
  (loss climbing epoch over epoch instead of falling). If the change_spec does not already address this,
  pick a conservative LR for the new objective (or add gradient clipping) rather than inheriting the old
  value silently.

# TASK
Implement the change specification above on the champion files. Output the COMPLETE modified file(s)
in the `=== FILE: name === ... === END FILE ===` format described in your role instructions.
Minimal targeted edits only; keep the CLI contract, the train-only rule and the output format intact.

## assistant (response)

```python
=== FILE: pipeline.py ===
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
    n_tabs = len(tab_vc)
    train_mask = (date_arr >= SPLITS["train"][0]) & (date_arr <= SPLITS["train"][1])

    # --- video/author historical-rate features, in chronological (time_ms) order; only train rows update
    #     the running counters -- identical semantics to champion_1k.py's load(), just array-backed. ---
    num_features = np.empty((N, 15), dtype=np.float32)
    u_tab_imp = [0] * (n_users * n_tabs)
    u_tab_pos = [0] * (n_users * n_tabs)
    v_imp = [0] * n_videos; v_pos = [0] * n_videos
    a_imp = [0] * n_authors; a_pos = [0] * n_authors
    u_imp = [0] * n_users; u_pos = [0] * n_users
    v_click = [0] * n_videos; v_like = [0] * n_videos
    a_click = [0] * n_authors; a_like = [0] * n_authors
    u_click = [0] * n_users; u_like = [0] * n_users
    time_order = np.argsort(time_arr, kind="stable").tolist()
    video_l, author_l, user_l, tab_l, label_l, click_l, like_l, train_l = video_arr.tolist(), author_arr.tolist(), user_arr.tolist(), tab_arr.tolist(), label_arr.tolist(), click_arr.tolist(), like_arr.tolist(), train_mask.tolist()
    log1p = np.log1p
    for i in time_order:
        vid, aid, uid, tid = video_l[i], author_l[i], user_l[i], tab_l[i]
        ut_idx = uid * n_tabs + tid
        uti, utp = u_tab_imp[ut_idx], u_tab_pos[ut_idx]
        
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
        num_features[i, 13] = log1p(uti)
        num_features[i, 14] = utp / uti if uti > 0 else 0.0
        
        if train_l[i]:
            v_imp[vid] = vi + 1
            a_imp[aid] = ai + 1
            u_imp[uid] = ui + 1
            u_tab_imp[ut_idx] = uti + 1
            if label_l[i] == 1:
                v_pos[vid] = vp + 1
                a_pos[aid] = ap + 1
                u_pos[uid] = up + 1
                u_tab_pos[ut_idx] = utp + 1
            if click_l[i] == 1:
                v_click[vid] = vc + 1
                a_click[aid] = ac + 1
                u_click[uid] = uc + 1
            if like_l[i] == 1:
                v_like[vid] = vl + 1
                a_like[aid] = al + 1
                u_like[uid] = ul + 1
    del v_imp, v_pos, a_imp, a_pos, u_imp, u_pos, v_click, v_like, a_click, a_like, u_click, u_like, u_tab_imp, u_tab_pos, time_order, video_l, author_l, user_l, tab_l, label_l, click_l, like_l, train_l
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
        self.W_num = np.zeros(15, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mW_num = np.zeros_like(self.W_num); self.vW_num = np.zeros_like(self.W_num)
        self.t = 0
        
        self.use_mlp = use_mlp
        self.k = k
        if self.use_mlp:
            self.mlp_in_dim = 7 * k + 15
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
```
=== END FILE ===
