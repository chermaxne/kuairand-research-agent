"""Harness tools: kit import, loop-data masking, sealed evaluation of prediction files, rung
predictors, data profile, diffs, submission generation/checking.

Only `sealed/evaluate.py` ever produces a score. The starter kit's `submit.read_submission` is used
verbatim for alignment/NaN validation (it is the organizers' own checker code).
"""
from __future__ import annotations

import csv
import difflib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .schemas import Score

PIPELINE_MAIN = "pipeline.py"
PREDS_VAL = "preds_val.csv"
PREDS_TEST = "preds_test.csv"
CODE_EXCLUDE_PREFIXES = ("preds_", "stdout", "stderr", "__pycache__", "attempts", "llm", ".")


# ---------------------------------------------------------------------------
# starter kit + sealed evaluator import
# ---------------------------------------------------------------------------
_KIT: Dict[str, SimpleNamespace] = {}


def import_kit(kit_dir: str) -> SimpleNamespace:
    """Import the organizers' data.py / submit.py (read-only use) from the kit directory."""
    kit_dir = os.path.realpath(kit_dir)
    if kit_dir in _KIT:
        return _KIT[kit_dir]
    sys.path.insert(0, kit_dir)
    try:
        for name in ("data", "evaluate", "submit", "baseline"):
            mod = sys.modules.get(name)
            if mod is None or os.path.realpath(getattr(mod, "__file__", "") or "") != os.path.join(kit_dir, f"{name}.py"):
                sys.modules.pop(name, None)
                importlib.import_module(name)
        ns = SimpleNamespace(data=sys.modules["data"], submit=sys.modules["submit"], baseline=sys.modules["baseline"],
                             kit_dir=kit_dir)
    finally:
        try:
            sys.path.remove(kit_dir)
        except ValueError:
            pass
    _KIT[kit_dir] = ns
    return ns


def import_sealed_evaluate(sealed_dir: str):
    """Load sealed/evaluate.py under its own module name and return its `evaluate` function."""
    path = os.path.join(os.path.realpath(sealed_dir), "evaluate.py")
    spec = importlib.util.spec_from_file_location("sealed_evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.evaluate


def kit_split_name(split: str) -> str:
    s = split.lower()
    if s in ("val", "valid", "validation"):
        return "valid"
    if s == "test":
        return "test"
    if s == "train":
        return "train"
    raise ValueError(f"unknown split {split}")


# ---------------------------------------------------------------------------
# loop data dir (train + valid only; hidden-test rows masked)
# ---------------------------------------------------------------------------
def _fingerprint(full_dir: str, max_date: int) -> Dict[str, Any]:
    fp = {"max_date": max_date, "files": {}}
    for name in sorted(os.listdir(full_dir)):
        p = os.path.join(full_dir, name)
        if os.path.isfile(p):
            st = os.stat(p)
            fp["files"][name] = [st.st_size, int(st.st_mtime)]
    return fp


def ensure_loop_data_dir(full_dir: str, loop_dir: str, max_date: int, filter_files: Sequence[str] = (
        "log_standard_4_22_to_5_08_pure.csv", "log_random_4_22_to_5_08_pure.csv")) -> Dict[str, Any]:
    """Build (once, fingerprinted) a copy of the data dir whose log files stop at `max_date` (the last
    validation day). Row order is preserved, so validation row_ids are identical to the full dir."""
    full_dir, loop_dir = os.path.realpath(full_dir), os.path.realpath(loop_dir)
    fp = _fingerprint(full_dir, max_date)
    fp_path = os.path.join(loop_dir, ".fingerprint.json")
    if os.path.exists(fp_path):
        try:
            if json.load(open(fp_path)) == fp:
                return {"loop_dir": loop_dir, "rebuilt": False}
        except (OSError, ValueError):
            pass
    tmp = loop_dir + ".building"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    stats: Dict[str, Any] = {}
    for name in sorted(os.listdir(full_dir)):
        src, dst = os.path.join(full_dir, name), os.path.join(tmp, name)
        if not os.path.isfile(src):
            continue
        if name in filter_files:
            kept = dropped = 0
            with open(src, newline="") as fi, open(dst, "w", newline="") as fo:
                r, w = csv.reader(fi), csv.writer(fo)
                header = next(r)
                w.writerow(header)
                di = header.index("date")
                for row in r:
                    if int(row[di]) <= max_date:
                        w.writerow(row)
                        kept += 1
                    else:
                        dropped += 1
            stats[name] = {"kept": kept, "dropped_after_max_date": dropped}
        else:
            shutil.copy2(src, dst)
            stats[name] = "copied"
    with open(os.path.join(tmp, ".fingerprint.json"), "w") as fh:
        json.dump(fp, fh)
    with open(os.path.join(tmp, "README_LOOP_DATA.txt"), "w") as fh:
        fh.write(f"Derived from {full_dir}: log rows with date > {max_date} (hidden-test period) removed.\n"
                 f"Used for every experiment during the loop; only finalize() sees the full directory.\n{json.dumps(stats, indent=1)}\n")
    shutil.rmtree(loop_dir, ignore_errors=True)
    os.replace(tmp, loop_dir)
    return {"loop_dir": loop_dir, "rebuilt": True, "stats": stats}


# ---------------------------------------------------------------------------
# evaluation of prediction files (sealed)
# ---------------------------------------------------------------------------
def evaluate_preds(preds_path: str, rows: Sequence[tuple], sealed_eval, kit) -> Score:
    """Validate the CSV with the kit's own `read_submission` (header, row_id contiguity, user/video
    alignment, NaN/Inf) and score it with sealed evaluate.py. Raises ValueError with a readable message."""
    scores = kit.submit.read_submission(preds_path, rows)
    res = sealed_eval([x[1] for x in rows], [x[6] for x in rows], scores)
    return Score.from_evaluate(res)


def write_preds(path: str, rows: Sequence[tuple], scores: Iterable[float]) -> None:
    """Write a §5.2 prediction file (row_id,user_id,video_id,score) in kit row order."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (x, s) in enumerate(zip(rows, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])


# ---------------------------------------------------------------------------
# reference rung predictors (spec §7.1) — same recipes as starter_kit/baseline.py
# ---------------------------------------------------------------------------
def random_scores(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def popularity_scores(train_rows: Sequence[tuple], eval_rows: Sequence[tuple], prior: float = 20.0) -> List[float]:
    import collections
    pos, imp = collections.Counter(), collections.Counter()
    for x in train_rows:
        imp[x[2]] += 1
        pos[x[2]] += x[6]
    gmean = sum(pos.values()) / max(1, sum(imp.values()))
    return [((pos[x[2]] + prior * gmean) / (imp[x[2]] + prior)) if imp[x[2]] else gmean for x in eval_rows]


# ---------------------------------------------------------------------------
# data profile (goes into the Researcher briefing; computed once per run)
# ---------------------------------------------------------------------------
def data_profile(data_dir: str, splits: Dict[str, list]) -> str:
    lines = ["## Data profile (measured by the harness)", f"data dir: `{data_dir}`", ""]
    for name in ("train", "valid", "test"):
        rws = splits.get(name, [])
        if not rws:
            lines.append(f"- {name}: 0 rows (masked during the loop)" if name == "test" else f"- {name}: 0 rows")
            continue
        users = len({x[1] for x in rws})
        vids = len({x[2] for x in rws})
        pos = sum(x[6] for x in rws) / len(rws)
        dates = (min(x[0] for x in rws), max(x[0] for x in rws))
        lines.append(f"- {name}: {len(rws):,} rows | {users:,} users | {vids:,} videos | long_view rate {pos:.4f} | dates {dates[0]}–{dates[1]}")
    tr = splits.get("train", [])
    if tr:
        import collections
        per_user = collections.Counter(x[1] for x in tr)
        cnts = np.array(list(per_user.values()))
        lines.append(f"- train impressions per user: median {int(np.median(cnts))}, p90 {int(np.percentile(cnts, 90))}, max {int(cnts.max())}")
    try:
        with open(os.path.join(data_dir, "log_standard_4_08_to_4_21_pure.csv")) as fh:
            cols = fh.readline().strip().split(",")
        lines.append(f"- log columns: {', '.join(cols)}")
        for extra in ("user_features_pure.csv", "video_features_basic_pure.csv", "video_features_statistic_pure.csv"):
            p = os.path.join(data_dir, extra)
            if os.path.exists(p):
                with open(p) as fh:
                    c = fh.readline().strip().split(",")
                lines.append(f"- {extra}: {len(c)} columns ({', '.join(c[:12])}{', …' if len(c) > 12 else ''})")
    except OSError:
        pass
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# workspace files + diffs
# ---------------------------------------------------------------------------
def read_code_files(directory: str) -> Dict[str, str]:
    """All text source files of a workspace/champion dir (recursively), keyed by relative path."""
    out: Dict[str, str] = {}
    if not os.path.isdir(directory):
        return out
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(CODE_EXCLUDE_PREFIXES)]
        for f in sorted(files):
            if f.startswith(CODE_EXCLUDE_PREFIXES) or f.endswith((".csv", ".npz", ".npy", ".pkl", ".pt", ".bin", ".txt", ".json")):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, directory)
            try:
                with open(p, encoding="utf-8") as fh:
                    out[rel] = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
    return out


def write_code_files(directory: str, files: Dict[str, str]) -> None:
    os.makedirs(directory, exist_ok=True)
    for rel, code in files.items():
        rel = os.path.normpath(rel)
        if rel.startswith("..") or os.path.isabs(rel):
            raise ValueError(f"refusing to write outside the workspace: {rel}")
        p = os.path.join(directory, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(code if code.endswith("\n") else code + "\n")


def unified_diff(before: Dict[str, str], after: Dict[str, str]) -> Tuple[str, str]:
    """Return (diff_text, summary) e.g. summary 'pipeline.py (+12/-3)'."""
    chunks, summ = [], []
    for name in sorted(set(before) | set(after)):
        a, b = before.get(name, ""), after.get(name, "")
        if a == b:
            continue
        lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(), fromfile=f"champion/{name}", tofile=f"attempt/{name}", lineterm=""))
        chunks.append("\n".join(lines))
        plus = sum(1 for l in lines[2:] if l.startswith("+"))
        minus = sum(1 for l in lines[2:] if l.startswith("-"))
        tag = "new " if name not in before else ("deleted " if name not in after else "")
        summ.append(f"{tag}{name} (+{plus}/-{minus})")
    return ("\n".join(chunks) if chunks else ""), (", ".join(summ) if summ else "no code change")


# ---------------------------------------------------------------------------
# submission (finalize)
# ---------------------------------------------------------------------------
def check_submission(sealed_dir: str, kit_dir: str, data_dir: str, path: str, split: str = "test",
                     python: Optional[str] = None, timeout_s: float = 900) -> Tuple[bool, str]:
    """Run the sealed wrapper around the kit's `submit.py --check`. Returns (passed, output)."""
    py = python or sys.executable
    cmd = [py, os.path.join(sealed_dir, "submit_check.py"), "--split", kit_split_name(split), "--data_dir", os.path.realpath(data_dir),
           "--kit", os.path.realpath(kit_dir), os.path.realpath(path)]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, "submit_check timed out"
    out = (cp.stdout + cp.stderr).strip()
    return cp.returncode == 0, out


def make_official_baseline_preds(kit_dir: str, data_dir: str, out_path: str, split: str = "valid",
                                 python: Optional[str] = None, timeout_s: float = 1800) -> Tuple[bool, str, float]:
    """Run the organizers' `submit.py --make --split <split>` (the official FM, seed 0) unchanged."""
    py = python or sys.executable
    cmd = [py, "submit.py", "--make", "--split", split, "--data_dir", os.path.realpath(data_dir), os.path.realpath(out_path)]
    t0 = time.time()
    try:
        cp = subprocess.run(cmd, cwd=kit_dir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, "official baseline timed out", time.time() - t0
    return cp.returncode == 0, (cp.stdout + cp.stderr).strip()[-4000:], time.time() - t0


def run_pipeline_in_sandbox(workspace: str, data_dir: str, split: str, out_name: str, timeout_s: float, sandbox_cfg: Dict,
                            pythonpath: Sequence[str] = (), deny_read: Sequence[str] = (), log_prefix: str = ""):
    from .sandbox import run_pipeline
    return run_pipeline(workspace, data_dir, split, out_name, timeout_s, sandbox_cfg, pythonpath=pythonpath,
                        deny_read=deny_read, log_prefix=log_prefix)
