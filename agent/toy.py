"""Toy task for the Phase-1 skeleton loop and the test-suite: a tiny synthetic dataset in the exact
KuaiRand-Pure starter-kit file format (so the kit's `data.load`, `submit.py --check` and the sealed
`evaluate.py` all run unchanged, in milliseconds) plus a dummy pipeline with one knob (THETA)."""
from __future__ import annotations

import csv
import math
import os
import random
from typing import Dict, List

LOG_COLUMNS = ["user_id", "video_id", "date", "hourmin", "time_ms", "is_click", "is_like", "is_follow", "is_comment",
               "is_forward", "is_hate", "long_view", "play_time_ms", "duration_ms", "profile_stay_time",
               "comment_stay_time", "is_profile_enter", "is_rand", "tab"]
VIDEO_COLUMNS = ["video_id", "author_id", "video_type", "upload_dt", "upload_type", "visible_status", "video_duration",
                 "server_width", "server_height", "music_id", "music_type", "tag"]
USER_COLUMNS = ["user_id", "user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
                "follow_user_num", "follow_user_num_range", "fans_user_num", "fans_user_num_range", "friend_user_num",
                "friend_user_num_range", "register_days", "register_days_range"]

TRAIN_DAYS = list(range(20220408, 20220422))
VALID_DAYS = list(range(20220422, 20220429))
TEST_DAYS = list(range(20220429, 20220431)) + list(range(20220501, 20220509))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_mini_dataset(data_dir: str, seed: int = 0, n_users: int = 40, n_videos: int = 60, n_authors: int = 12,
                      per_user_day: int = 3) -> Dict[str, int]:
    """Write a miniature dataset. Labels come from a latent model where video quality and author
    quality both matter, so blending the two popularity estimates (THETA) changes the ranking quality."""
    rng = random.Random(seed)
    os.makedirs(data_dir, exist_ok=True)
    authors = [rng.randrange(n_authors) for _ in range(n_videos)]
    a_q = [rng.gauss(0, 1.0) for _ in range(n_authors)]
    v_q = [rng.gauss(0, 0.8) for _ in range(n_videos)]
    u_b = [rng.gauss(-0.6, 0.7) for _ in range(n_users)]
    durations = [rng.choice([8000, 15000, 30000, 60000, 120000, 240000]) for _ in range(n_videos)]

    with open(os.path.join(data_dir, "video_features_basic_pure.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(VIDEO_COLUMNS)
        for v in range(n_videos):
            w.writerow([v, authors[v], "NORMAL", "2022-04-01", "LongImport", 0.0, float(durations[v]), 720.0, 1280.0, 1000 + v, 9.0, v % 7])
    with open(os.path.join(data_dir, "user_features_pure.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(USER_COLUMNS)
        for u in range(n_users):
            w.writerow([u, "full_active", 0, 0, 0, 10, "[10,50)", 5, "[0,10)", 1, "[1,5)", 400, "365+"])
    counts = {}

    def write_log(path: str, days: List[int], is_rand: int = 0) -> int:
        n = 0
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(LOG_COLUMNS)
            for d in days:
                for u in range(n_users):
                    for _ in range(per_user_day):
                        v = rng.randrange(n_videos)
                        p = _sigmoid(u_b[u] + v_q[v] + 0.7 * a_q[authors[v]] + rng.gauss(0, 0.5))
                        y = 1 if rng.random() < p else 0
                        play = int(durations[v] * (0.8 if y else 0.2) * rng.random())
                        w.writerow([u, v, d, rng.randrange(0, 2400, 100), 1649000000000 + n, int(rng.random() < 0.2 + 0.3 * y),
                                    int(rng.random() < 0.05), 0, 0, 0, 0, y, play, durations[v], 0, 0, 0, is_rand, rng.randrange(2)])
                        n += 1
        return n
    counts["train"] = write_log(os.path.join(data_dir, "log_standard_4_08_to_4_21_pure.csv"), TRAIN_DAYS)
    counts["valid_test"] = write_log(os.path.join(data_dir, "log_standard_4_22_to_5_08_pure.csv"), VALID_DAYS + TEST_DAYS)
    counts["random"] = write_log(os.path.join(data_dir, "log_random_4_22_to_5_08_pure.csv"), VALID_DAYS + TEST_DAYS, is_rand=1)
    return counts


DUMMY_PIPELINE = '''"""Dummy pipeline (toy task): smoothed video popularity blended with author popularity.
Satisfies the §5.2 contract: python pipeline.py --data <dir> --split val|test --out preds.csv
"""
import argparse, csv, os, collections

THETA = 0.50   # blend weight: 1.0 = pure video popularity, 0.0 = pure author popularity
PRIOR = 5.0
SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428), "test": (20220429, 20220508)}
FILES = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")


def load(data_dir):
    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    rows = []
    for f in FILES:
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r["date"]), r["user_id"], r["video_id"], vid2author.get(r["video_id"], "UNK"),
                             1 if r["long_view"] != "0" else 0))
    return {name: [x for x in rows if lo <= x[0] <= hi] for name, (lo, hi) in SPLITS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="preds_val.csv")
    a = ap.parse_args()
    split = {"val": "valid", "valid": "valid", "test": "test"}[a.split]
    if os.environ.get("TOY_SLEEP_S"):          # test hook: slow run for kill / timeout tests
        import time; time.sleep(float(os.environ["TOY_SLEEP_S"]))
    S = load(a.data)
    vpos, vimp, apos, aimp = (collections.Counter() for _ in range(4))
    for x in S["train"]:
        vimp[x[2]] += 1; vpos[x[2]] += x[4]; aimp[x[3]] += 1; apos[x[3]] += x[4]
    g = sum(vpos.values()) / max(1, sum(vimp.values()))
    vr = lambda v: (vpos[v] + PRIOR * g) / (vimp[v] + PRIOR)
    ar = lambda au: (apos[au] + PRIOR * g) / (aimp[au] + PRIOR)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, x in enumerate(S[split]):
            w.writerow([i, x[1], x[2], f"{THETA * vr(x[2]) + (1 - THETA) * ar(x[3]):.6g}"])
    print(f"wrote {a.out}: {len(S[split])} rows (THETA={THETA})")


if __name__ == "__main__":
    main()
'''


def write_dummy_champion(directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    p = os.path.join(directory, "pipeline.py")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(DUMMY_PIPELINE)
    return p
