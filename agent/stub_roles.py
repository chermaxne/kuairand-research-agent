"""STUB agent for Phase 1 and offline mock runs: deterministic handlers for the MockLLMClient.

`default_mock_handlers()`  — the spec's stub: random tiny perturbation of the dummy pipeline's THETA.
The handlers see exactly the prompts the real roles would see and answer in the same formats, so the
harness code path is identical to a real run.
"""
from __future__ import annotations

import json
import random
import re
from typing import Callable, Dict, List

from .roles import parse_file_blocks, render_file_blocks

CATS = ["feature", "model", "training", "multitask", "other"]


def _iteration_from_briefing(text: str) -> int:
    m = re.search(r"BUDGET: iteration (\d+) of", text)
    return int(m.group(1)) if m else 1


def _last_user(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def stub_researcher(role: str, system: List[str], messages: List[Dict[str, str]]) -> str:
    it = _iteration_from_briefing(_last_user(messages))
    rng = random.Random(1000 + it)
    delta = rng.choice([-0.2, -0.1, -0.05, 0.05, 0.1, 0.2])
    plan = {"hypothesis": f"Shift THETA by {delta:+.2f} to rebalance video vs author popularity (stub it{it:02d})",
            "category": CATS[(it - 1) % len(CATS)],
            "change_spec": f"In pipeline.py change the line `THETA = <x>` to THETA = x{delta:+.2f} (clamp to [0,1]). delta={delta:+.2f}",
            "expected_risk": "low", "builds_on": "champion",
            "rationale": "Stub agent: random tiny perturbation of the dummy pipeline (Phase 1 skeleton)."}
    return json.dumps(plan)


def stub_engineer(role: str, system: List[str], messages: List[Dict[str, str]]) -> str:
    user = _last_user(messages)
    # only the champion section carries real files (the task template contains a format example)
    section = user.split("# Current champion files", 1)[-1].split("# Pipeline contract", 1)[0]
    files = parse_file_blocks(section)
    m = re.search(r"delta=([+-]?\d+\.\d+)", user)
    delta = float(m.group(1)) if m else 0.05
    code = files.get("pipeline.py", "")
    def bump(match):
        val = min(1.0, max(0.0, float(match.group(1)) + delta))
        return f"THETA = {val:.2f}"
    code2, n = re.subn(r"THETA = ([0-9.]+)", bump, code, count=1)
    files["pipeline.py"] = code2 if n else code
    return render_file_blocks(files)


def stub_debugger(role: str, system: List[str], messages: List[Dict[str, str]]) -> str:
    return json.dumps({"action": "abandon", "reason": "stub debugger never fixes code"})


def stub_scribe_lesson(role: str, system: List[str], messages: List[Dict[str, str]]) -> str:
    user = _last_user(messages)
    m = re.search(r'"status": "(\w+)"', user)
    d = re.search(r"DECISION \(harness\): (\w+)", user)
    return f"Stub lesson: run {m.group(1) if m else 'unknown'}, harness decided {d.group(1) if d else 'unknown'}."


def stub_scribe_logentry(role: str, system: List[str], messages: List[Dict[str, str]]) -> str:
    return "Stub narrative: see the JSON facts for this iteration (numbers are harness-measured)."


def default_mock_handlers() -> Dict[str, Callable]:
    return {"researcher": stub_researcher, "engineer": stub_engineer, "debugger": stub_debugger,
            "scribe_lesson": stub_scribe_lesson, "scribe_logentry": stub_scribe_logentry}


# ---------------------------------------------------------------------------
# Offline mock for REAL-DATA runs (no API key): a fixed research plan of concrete FM edits, applied by
# deterministic text substitution on the champion pipeline. One planned iteration injects a bug so the
# debugger path is exercised; the mock debugger repairs exactly that bug.
# ---------------------------------------------------------------------------
KUAIRAND_PLAN = [
    {"hypothesis": "Stronger L2 (1e-6 -> 1e-5) to regularise sparse id embeddings under temporal shift",
     "category": "training", "risk": "low", "edits": [("L2 = 1e-6", "L2 = 1e-5")],
     "rationale": "Mock plan step 1 (the Engineer output deliberately contains a NameError to exercise the Debugger path)."},
    {"hypothesis": "Double the FM embedding dimension (K 16 -> 32) to capture richer user x item interactions",
     "category": "model", "risk": "low", "edits": [("K = 16", "K = 32")],
     "rationale": "Mock plan step 2 (organizers report capacity is flat; this measures it on validation)."},
    {"hypothesis": "Raise the learning rate (0.001 -> 0.002) so Adam converges before early stopping triggers",
     "category": "training", "risk": "low", "edits": [("LR = 0.001", "LR = 0.002")],
     "rationale": "Mock plan step 3: cheaper convergence check."},
    {"hypothesis": "Longer patience (4 -> 6) and more epochs (40 -> 60) to avoid stopping on a noisy dip",
     "category": "training", "risk": "low", "edits": [("PATIENCE = 4", "PATIENCE = 6"), ("EPOCHS = 40", "EPOCHS = 60")],
     "rationale": "Mock plan step 4."},
    {"hypothesis": "Finer duration buckets (10 -> 20 train quantiles) to sharpen the duration field",
     "category": "feature", "risk": "low", "edits": [("N_DUR_BUCKETS = 10", "N_DUR_BUCKETS = 20")],
     "rationale": "Mock plan step 5."},
    {"hypothesis": "Smaller batches (8192 -> 4096) for more parameter updates per epoch",
     "category": "training", "risk": "low", "edits": [("BATCH = 8192", "BATCH = 4096")],
     "rationale": "Mock plan step 6."},
    {"hypothesis": "Halve the embedding dimension (K -> 8) to test whether the FM is over-parameterised",
     "category": "model", "risk": "low", "edits": [("K = 16", "K = 8"), ("K = 32", "K = 8")],
     "rationale": "Mock plan step 7."},
]
BUG_STEP = 0   # 0-based index of the plan step whose engineer output contains an injected NameError


def _plan_step(it: int) -> dict:
    return KUAIRAND_PLAN[(it - 1) % len(KUAIRAND_PLAN)]


def kuairand_researcher(role: str, system: List[str], messages: List[Dict[str, str]]) -> str:
    it = _iteration_from_briefing(_last_user(messages))
    step = _plan_step(it)
    edits = "; ".join(f"[[EDIT]] {a} ==> {b}" for a, b in step["edits"])
    plan = {"hypothesis": step["hypothesis"], "category": step["category"],
            "change_spec": f"In pipeline.py apply exactly these line substitutions (leave everything else untouched): {edits}. "
                           f"Keep the CLI, the train-only rule and the output format. [mock it{it:02d}]",
            "expected_risk": step["risk"], "builds_on": "champion", "rationale": step["rationale"] + f" (mock, it{it:02d})"}
    return json.dumps(plan)


def kuairand_engineer(role: str, system: List[str], messages: List[Dict[str, str]]) -> str:
    user = _last_user(messages)
    section = user.split("# Current champion files", 1)[-1].split("# Pipeline contract", 1)[0]
    files = parse_file_blocks(section)
    code = files.get("pipeline.py", "")
    for a, b in re.findall(r"\[\[EDIT\]\] (.+?) ==> (.+?)(?:;|\.|$)", user):
        code = code.replace(a.strip(), b.strip(), 1)
    m = re.search(r"\[mock it(\d+)\]", user)
    it = int(m.group(1)) if m else 0
    if it and (it - 1) % len(KUAIRAND_PLAN) == BUG_STEP:
        code = code.replace("m = FM(dim)", "m = FM(dim, l2=L2_TYPO)", 1)      # injected NameError
    files["pipeline.py"] = code
    return render_file_blocks(files)


def kuairand_debugger(role: str, system: List[str], messages: List[Dict[str, str]]) -> str:
    user = _last_user(messages)
    files = parse_file_blocks(user.split("# Failing files", 1)[-1].split("# Error", 1)[0])
    code = files.get("pipeline.py", "")
    if "L2_TYPO" in code:
        files["pipeline.py"] = code.replace("m = FM(dim, l2=L2_TYPO)", "m = FM(dim)")
        return "FIX SUMMARY: NameError — L2_TYPO was never defined; restored the FM(dim) constructor call.\n" + render_file_blocks(files)
    return json.dumps({"action": "abandon", "reason": "mock debugger only knows how to fix the injected typo"})


def kuairand_scribe_lesson(role: str, system: List[str], messages: List[Dict[str, str]]) -> str:
    user = _last_user(messages)
    status = re.search(r'"status": "(\w+)"', user)
    vs = re.search(r'"vs_best": "([^"]+)"', user)
    hyp = re.search(r"HYPOTHESIS: (.+)", user)
    dec = re.search(r"DECISION \(harness\): (\w+)", user)
    words = (hyp.group(1) if hyp else "experiment").split()[:8]
    return f"{' '.join(words)}: {status.group(1) if status else '?'} {vs.group(1) if vs else ''} -> {dec.group(1) if dec else '?'}"


def kuairand_scribe_logentry(role: str, system: List[str], messages: List[Dict[str, str]]) -> str:
    user = _last_user(messages)
    m = re.search(r"```json\n(.*?)\n```", user, flags=re.S)
    try:
        f = json.loads(m.group(1)) if m else {}
    except json.JSONDecodeError:
        f = {}
    r = f.get("result", {})
    return (f"**Iteration {f.get('iteration')}** — {f.get('hypothesis')}\n\n"
            f"Status `{r.get('status')}`; primary {r.get('primary')} (GAUC {r.get('gauc')}, nDCG@5 {r.get('ndcg5')}), "
            f"{r.get('vs_best')} vs best; runtime {r.get('runtime_s')}s; debug attempts {len(f.get('debug_attempts', []))}. "
            f"Harness decision: **{f.get('decision')}** (streak {f.get('streak_after')}, best after {f.get('best_primary_after')}). "
            f"Lesson: {f.get('lesson')}")


def kuairand_mock_handlers() -> Dict[str, Callable]:
    return {"researcher": kuairand_researcher, "engineer": kuairand_engineer, "debugger": kuairand_debugger,
            "scribe_lesson": kuairand_scribe_lesson, "scribe_logentry": kuairand_scribe_logentry}
