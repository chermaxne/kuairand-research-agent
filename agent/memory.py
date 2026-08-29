"""Three-tier memory: ledger (tier 1, append-only), state block (tier 2, regenerated),
per-iteration JSON detail (tier 3). Also the intervention log helpers.

Every fact written here comes from the harness; the only LLM-authored fragment is the LESSON,
which is sanitised and hard-truncated before it is written.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from .schemas import IterationLog, RunState, atomic_write_json, atomic_write_text, read_json, utc_now_iso, CATEGORIES

LEDGER = "ledger.md"
STATE_MD = "state.md"
LOGS_DIR = "logs"
INTERVENTIONS = "interventions.md"
RUN_STATE = "run_state.json"

LEDGER_HEADER = "# Ledger (tier-1 memory, append-only; one line per iteration, harness-written except LESSON)\n"


# ---------------------------------------------------------------------------
# text hygiene
# ---------------------------------------------------------------------------
def one_line(s: Any, max_len: int = 160) -> str:
    s = "" if s is None else str(s)
    s = s.replace("|", "/").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if max_len and len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def truncate_words(s: Any, n: int = 20) -> str:
    words = one_line(s, max_len=0).split()
    return " ".join(words[:n])


def fmt_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}"


def fmt_score(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4f}"


# ---------------------------------------------------------------------------
# tier 1: ledger
# ---------------------------------------------------------------------------
def ledger_line(iteration: int, hyp: str, change: str, result: str, best_after: Optional[float],
                decision: str, lesson: str) -> str:
    """Spec §5.4:
    [itNN] HYP: <short> | CHANGE: <files/summary> | RESULT: <primary or FAILED(reason)> (best <x>) -> PROMOTED|kept|FAILED | LESSON: <≤20 words>
    """
    outcome = {"promoted": "PROMOTED", "kept_champion": "kept", "failed": "FAILED"}[decision]
    return (f"[it{iteration:02d}] HYP: {one_line(hyp, 120)} | CHANGE: {one_line(change, 80)} | "
            f"RESULT: {one_line(result, 80)} (best {fmt_score(best_after)}) -> {outcome} | "
            f"LESSON: {truncate_words(lesson, 20)}")


def result_string(status: str, primary: Optional[float], reason: str = "") -> str:
    if status == "scored" and primary is not None:
        return f"{primary:.4f}"
    if status == "timeout":
        return f"FAILED(timeout{': ' + one_line(reason, 50) if reason else ''})"
    return f"FAILED({one_line(reason, 60) or 'unknown'})"


def ledger_path(run_dir: str) -> str:
    return os.path.join(run_dir, LEDGER)


def init_ledger(run_dir: str) -> None:
    p = ledger_path(run_dir)
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(LEDGER_HEADER)


def append_ledger(run_dir: str, line: str) -> None:
    """Append-only: open in 'a' mode, never rewrite."""
    init_ledger(run_dir)
    with open(ledger_path(run_dir), "a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_ledger(run_dir: str) -> str:
    p = ledger_path(run_dir)
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def ledger_entries(run_dir: str) -> List[str]:
    return [ln for ln in read_ledger(run_dir).splitlines() if ln.startswith("[it")]


# ---------------------------------------------------------------------------
# tier 2: state block (§5.5)
# ---------------------------------------------------------------------------
def active_themes(history: Iterable[Dict[str, Any]], categories: Iterable[str] = CATEGORIES) -> str:
    """Deterministic one-liner: winning / losing / untried directions by category."""
    by_cat: Dict[str, Dict[str, int]] = {}
    for h in history:
        c = h.get("category") or "other"
        d = by_cat.setdefault(c, {"promoted": 0, "kept": 0, "failed": 0})
        dec = h.get("decision")
        if dec == "promoted":
            d["promoted"] += 1
        elif dec == "kept_champion":
            d["kept"] += 1
        else:
            d["failed"] += 1
    winning = [c for c, d in by_cat.items() if d["promoted"] > 0]
    losing = [c for c, d in by_cat.items() if d["promoted"] == 0]
    untried = [c for c in categories if c not in by_cat]

    def fmt(cs):
        return ", ".join(f"{c}[{by_cat[c]['promoted']} promoted/{by_cat[c]['kept']} flat/{by_cat[c]['failed']} failed]" if c in by_cat else c
                         for c in cs) or "none"
    return f"winning: {fmt(winning)}; losing/flat: {fmt(losing)}; untried: {', '.join(untried) or 'none'}"


def research_digest(history: Iterable[Dict[str, Any]], read_detail=None) -> str:
    """Harness-written fact table over EVERY iteration, grouped by category: what was changed, the delta against the
    champion at that time, the decision and the failure/leak status. Deterministic; nothing here is LLM-authored
    except the quoted lesson. `read_detail(iteration)` may return the tier-3 record for the untruncated hypothesis."""
    rows = list(history)
    if not rows:
        return ""
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for h in rows:
        by_cat.setdefault(h.get("category") or "other", []).append(h)
    out = ["# RESEARCH DIGEST — every iteration so far, grouped by direction (harness-measured facts)",
           "| it | direction | what changed | predicted Δ | measured Δ vs then-champion | decision | status | in-run ablations (pipeline-reported, unsealed) | lesson |",
           "|---|---|---|---|---|---|---|---|---|"]
    for cat in sorted(by_cat):
        for h in by_cat[cat]:
            it = h["iteration"]
            d = read_detail(it) if read_detail else None
            x = (d or {}).get("harness_extra", {})
            hyp = (d or {}).get("hypothesis") or h.get("hypothesis", "")
            bb = x.get("best_at_iteration_start")
            delta = (f"{h['primary'] - bb:+.4f}" if (h.get("primary") is not None and bb is not None) else "n/a")
            status = h.get("status", "")
            lt = (x.get("leak_test") or {}).get("verdict")
            if lt and lt != "clean":
                status += f" ({lt.split(' ')[0]})"
            if h.get("error_short") and status != "scored":
                status += f": {one_line(h['error_short'], 60)}"
            eg = x.get("expected_gain")
            predicted = f"{eg:+.4f}" if isinstance(eg, (int, float)) else "n/a"
            abl = format_ablations(x.get("ablations") or [], h.get("primary") if h.get("status") == "scored" else None) or "—"
            out.append(f"| it{it:02d} | {cat} | {one_line(hyp, 220)} | {predicted} | {delta} | {h.get('decision')} | {status} | {one_line(abl, 200)} | {one_line(h.get('lesson', ''), 120)} |")
    promoted = [h for h in rows if h.get("promoted")]
    tried = {c: len(v) for c, v in by_cat.items()}
    out.append("")
    calib = [(read_detail(h["iteration"]) or {}).get("harness_extra", {}) if read_detail else {} for h in rows]
    calib = [(c.get("expected_gain"), h.get("primary") - c.get("best_at_iteration_start"))
             for c, h in zip(calib, rows)
             if isinstance(c.get("expected_gain"), (int, float)) and h.get("primary") is not None and c.get("best_at_iteration_start") is not None]
    if calib:
        bias = sum(p - m for p, m in calib) / len(calib)
        out.append(f"Calibration: over {len(calib)} scored iterations your predicted gain exceeded the measured one by {bias:+.4f} on "
                   f"average (predicted − measured); size the next prediction accordingly.")
    out.append(f"Totals: {len(rows)} iterations; promoted {len(promoted)}" +
               (f" (it{', it'.join(f'{h['iteration']:02d}' for h in promoted)})" if promoted else "") +
               "; attempts per direction: " + ", ".join(f"{c} {n}" for c, n in sorted(tried.items())) +
               "; never attempted: " + (", ".join(c for c in CATEGORIES if c not in by_cat) or "none") + ".")
    return "\n".join(out)


ABLATION_RE = re.compile(r"^\s*ABLATION\s+(\S+)\s+primary=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
                         r"(?:\s+gauc=([-+]?\d*\.?\d+))?(?:\s+ndcg5=([-+]?\d*\.?\d+))?", re.M)


def parse_ablations(stdout: str) -> List[Dict[str, Any]]:
    """In-run attribution: the pipeline may score named variants on validation itself and print
    `ABLATION <name> primary=<f> [gauc=<f>] [ndcg5=<f>]`. These are PIPELINE-REPORTED (unsealed) diagnostics for the
    Researcher; promotion and convergence never read them. Later lines with the same name win."""
    found: Dict[str, Dict[str, Any]] = {}
    for m in ABLATION_RE.finditer(stdout or ""):
        name, primary, gauc, ndcg = m.groups()
        found[name] = {"name": name, "primary": float(primary),
                       "gauc": float(gauc) if gauc else None, "ndcg5": float(ndcg) if ndcg else None}
    return list(found.values())


def format_ablations(ablations: List[Dict[str, Any]], full_primary: Optional[float] = None) -> str:
    if not ablations:
        return ""
    parts = []
    for a in ablations:
        d = f" ({a['primary'] - full_primary:+.4f} vs the full run)" if full_primary is not None else ""
        parts.append(f"{a['name']} {a['primary']:.4f}{d}")
    return "; ".join(parts)


# Model-family bookkeeping (team policy 2026-08-29): develop ONE alternative family until the harness declares it a
# dead end, then move to the best remaining one. The FM is retired (see run.retire_fm).
FAMILY_ALIASES = (("xdeepfm", "xdeepfm"), ("deepfm", "deepfm"), ("dcn", "dcn"), ("autoint", "autoint"), ("nfm", "nfm"),
                  ("dien", "dien"), ("din", "din"), ("deep interest", "din"), ("sasrec", "sasrec"), ("bert4rec", "bert4rec"),
                  ("gru4rec", "gru"), ("gru", "gru"), ("lstm", "lstm"), ("transformer", "transformer"), ("mmoe", "mmoe"),
                  ("ple", "ple"), ("esmm", "esmm"), ("shared.bottom", "shared-bottom"), ("lightgbm", "gbdt"),
                  ("lgbm", "gbdt"), ("gbdt", "gbdt"), ("xgboost", "gbdt"), ("catboost", "gbdt"), ("gradient boost", "gbdt"),
                  ("two.tower", "two-tower"), ("wide", "wide-and-deep"), ("mlp", "mlp"), ("attention", "attention"),
                  ("factori[sz]ation machine", "fm"), ("ffm", "fm"), ("fm", "fm"))
COMPOSITION_WORDS = ("ensemble", "blend", "stack", "hybrid of", "combined with", "rank-avg", "rank average", "averaged with")


def canonical_family(name: str) -> str:
    """Group free-text `model_family` strings into one key per architecture, so 'DIN target attention', 'DIN + session
    fields' and 'deep interest network' are all the same family. The list is priority-ordered: the most specific
    architecture wins, and words like 'attention' only decide when nothing more specific matched. A name that composes
    TWO architectures ('LightGBM ensemble with DIN') is its own combined family — adding a second model is a switch,
    not a deepening of the first."""
    n = (name or "").strip().lower()
    if not n:
        return ""
    keys: List[str] = []
    for pat, key in FAMILY_ALIASES:
        if key not in keys and re.search(r"(?<![a-z0-9])" + pat + r"(?![a-z0-9])", n):
            keys.append(key)
    if not keys:
        return re.sub(r"[^a-z0-9]+", "-", n).strip("-")[:40]
    if len(keys) > 1 and any(w in n for w in COMPOSITION_WORDS):
        return "+".join(sorted(keys))                      # an explicit ensemble of two models = a family of its own
    return keys[0]                                          # otherwise the most specific architecture named


def family_stats(history: Iterable[Dict[str, Any]], epsilon: float, deadend_after: int) -> Dict[str, Any]:
    """Per-family record and the harness's dead-end verdict. A family is a DEAD END once `deadend_after` consecutive
    iterations on it have failed to improve ITS OWN best by more than epsilon (crashes and timeouts count). The ACTIVE
    family is the living non-FM family with the best measured primary (ties: the most recently worked on)."""
    fams: Dict[str, Dict[str, Any]] = {}
    for h in history:
        key = canonical_family(h.get("model_family") or "")
        if not key or key == "fm":
            continue
        f = fams.setdefault(key, {"key": key, "label": h.get("model_family") or key, "iters": [], "best": None,
                                  "best_iter": None, "flat_tail": 0, "scored": 0})
        f["iters"].append(h["iteration"])
        p = h.get("primary") if h.get("status") == "scored" else None
        if p is not None:
            f["scored"] += 1
        if p is not None and (f["best"] is None or p > f["best"] + epsilon):
            f["flat_tail"] = 0
        else:
            f["flat_tail"] += 1
        if p is not None and (f["best"] is None or p > f["best"]):
            f["best"], f["best_iter"] = p, h["iteration"]
    for f in fams.values():
        f["dead"] = f["flat_tail"] >= deadend_after
        f["status"] = ("DEAD END" if f["dead"] else "ACTIVE")
    alive_scored = [f for f in fams.values() if not f["dead"] and f["best"] is not None]
    active = sorted(alive_scored, key=lambda f: (f["best"], max(f["iters"])))[-1] if alive_scored else None
    for f in fams.values():
        if f["dead"]:
            continue
        if active is not None and f["key"] == active["key"]:
            f["status"] = "ACTIVE"
        elif f["best"] is None:
            f["status"] = "unproven (never produced a score — cannot lead)"
        else:
            f["status"] = "alive (not the leader)"
    return {"families": sorted(fams.values(), key=lambda f: max(f["iters"])), "active": active,
            "deadend_after": deadend_after, "epsilon": epsilon}


def render_family_status(stats: Dict[str, Any]) -> str:
    rows = ["# MODEL FAMILY STATUS (harness-measured — which architecture you are developing)",
            "| family | iterations | best primary | consecutive non-improving | status |", "|---|---|---|---|---|"]
    for f in stats["families"]:
        best = f"{f['best']:.4f} (it{f['best_iter']:02d})" if f["best"] is not None else "— (nothing scored)"
        rows.append(f"| {f['label']} | {', '.join(f'it{i:02d}' for i in f['iters'])} | {best} | {f['flat_tail']} | {f['status']} |")
    if not stats["families"]:
        rows.append("| (none yet — the champion is the retired FM baseline) | — | — | 0 | choose the first alternative family |")
    return "\n".join(rows)


def synthesis_numbers_ok(synthesis: str, digest: str) -> bool:
    """Guard: every number the Scribe wrote must appear in the harness digest (no invented figures)."""
    import re as _re
    nums = set(_re.findall(r"\d+\.\d+", synthesis))
    have = set(_re.findall(r"\d+\.\d+", digest))
    return nums <= have


def render_state_block(state: RunState, limits: Dict[str, Any], iteration_display: int, now: Optional[float] = None) -> str:
    """Spec §5.5 — regenerated fresh; every number is harness-measured."""
    best = state.best_primary
    margin = (best - state.baseline_primary) if (best is not None and state.baseline_primary is not None) else None
    lines = [
        f"CURRENT BEST: it{state.best_iter:02d} | val primary {fmt_score(best)} "
        f"(GAUC {fmt_score(state.best_gauc)} / nDCG5 {fmt_score(state.best_ndcg5)}) | "
        f"baseline {fmt_score(state.baseline_primary)} | margin {('%+.4f' % margin) if margin is not None else 'n/a'}",
        f"BUDGET: iteration {iteration_display} of {limits['MAX_ITERS']} | "
        f"{fmt_elapsed(state.elapsed_s(now))} of {fmt_elapsed(float(limits['WALL_CLOCK_HOURS']) * 3600)} elapsed | "
        f"tokens so far {state.tokens_total}",
        f"CONVERGENCE: streak {state.streak} of {limits['N_FLAT']} flat (EPSILON={limits['EPSILON']})",
        "BLOCKED: " + ((", ".join(state.blocked[-8:]) + (f" (+{len(state.blocked) - 8} older, see ledger)" if len(state.blocked) > 8 else "")) if state.blocked else "none"),
        f"ACTIVE THEMES: {active_themes(state.history, limits.get('categories', CATEGORIES))}",
    ]
    return "\n".join(lines) + "\n"


def write_state_block(run_dir: str, text: str) -> None:
    atomic_write_text(os.path.join(run_dir, STATE_MD), text)


def read_state_block(run_dir: str) -> str:
    p = os.path.join(run_dir, STATE_MD)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


# ---------------------------------------------------------------------------
# tier 3: per-iteration JSON (§5.6)
# ---------------------------------------------------------------------------
def iteration_log_path(run_dir: str, iteration: int) -> str:
    return os.path.join(run_dir, LOGS_DIR, f"iter_{iteration:02d}.json")


def write_iteration_log(run_dir: str, entry: IterationLog) -> str:
    os.makedirs(os.path.join(run_dir, LOGS_DIR), exist_ok=True)
    p = iteration_log_path(run_dir, entry.iteration)
    atomic_write_json(p, entry.to_dict())
    return p


def read_iteration_detail(run_dir: str, iteration: int) -> Optional[Dict[str, Any]]:
    p = iteration_log_path(run_dir, iteration)
    return read_json(p) if os.path.exists(p) else None


def write_iteration_narrative(run_dir: str, iteration: int, text: str) -> str:
    os.makedirs(os.path.join(run_dir, LOGS_DIR), exist_ok=True)
    p = os.path.join(run_dir, LOGS_DIR, f"iter_{iteration:02d}.md")
    atomic_write_text(p, text.rstrip() + "\n")
    return p


# ---------------------------------------------------------------------------
# run state persistence
# ---------------------------------------------------------------------------
def run_state_path(run_dir: str) -> str:
    return os.path.join(run_dir, RUN_STATE)


def save_run_state(run_dir: str, state: RunState) -> None:
    atomic_write_json(run_state_path(run_dir), state.to_dict())


def load_run_state(run_dir: str) -> Optional[RunState]:
    p = run_state_path(run_dir)
    return RunState.from_dict(read_json(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------------------
# interventions (§11)
# ---------------------------------------------------------------------------
INTERVENTIONS_TEMPLATE = """# Manual interventions — {run_id}

Every time a human touches this run (restarts it, edits a file, unblocks the agent, changes config),
add a row here — `python -m agent.intervene "what you did" --run-dir {run_dir}` does it for you and bumps
the counter in run_state.json. Honesty is the product.

Count: 0

| timestamp (UTC) | what was stuck | what the human did | scope |
|---|---|---|---|
"""


def init_interventions(run_dir: str, run_id: str) -> None:
    p = os.path.join(run_dir, INTERVENTIONS)
    if not os.path.exists(p):
        atomic_write_text(p, INTERVENTIONS_TEMPLATE.format(run_id=run_id, run_dir=run_dir))


def append_intervention(run_dir: str, what_stuck: str, what_done: str, scope: str, bump: bool = True) -> int:
    """Append a row, update the Count line and (if bump) run_state.interventions. Returns the new count."""
    state = load_run_state(run_dir)
    if state is None:
        raise FileNotFoundError(f"no run_state.json in {run_dir}")
    if bump:
        state.interventions += 1
    p = os.path.join(run_dir, INTERVENTIONS)
    if not os.path.exists(p):
        init_interventions(run_dir, state.run_id)
    text = open(p, encoding="utf-8").read()
    text = re.sub(r"^Count: \d+$", f"Count: {state.interventions}", text, count=1, flags=re.M)
    row = f"| {utc_now_iso()} | {one_line(what_stuck, 200)} | {one_line(what_done, 300)} | {one_line(scope, 40)} |\n"
    atomic_write_text(p, text.rstrip("\n") + "\n" + row)
    save_run_state(run_dir, state)
    return state.interventions
