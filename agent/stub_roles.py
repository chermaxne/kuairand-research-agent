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
