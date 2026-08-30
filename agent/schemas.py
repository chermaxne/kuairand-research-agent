"""Dataclasses for every frozen contract in spec §5 plus the persisted run state.

Nothing here is written by an LLM: the harness fills these from measured values.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

CATEGORIES = ("feature", "model", "training", "multitask", "other")
RISKS = ("low", "medium", "high")
STATUSES = ("scored", "failed", "timeout")
DECISIONS = ("promoted", "kept_champion", "failed")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Scores (from sealed evaluate.py only)
# ---------------------------------------------------------------------------
@dataclass
class Score:
    gauc: float
    ndcg5: float
    primary: float
    users: int = 0
    rows: int = 0

    @classmethod
    def from_evaluate(cls, d: Dict[str, Any]) -> "Score":
        return cls(gauc=float(d["GAUC"]), ndcg5=float(d["nDCG@5"]), primary=float(d["primary"]),
                   users=int(d.get("users", 0)), rows=int(d.get("rows", 0)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# §5.1 Researcher output
# ---------------------------------------------------------------------------
class ContractError(ValueError):
    """Raised when a role's output does not satisfy its frozen contract."""


@dataclass
class ResearcherPlan:
    hypothesis: str
    category: str
    change_spec: str
    expected_risk: str
    builds_on: str = "champion"
    rationale: str = ""           # optional superset field; falls back to change_spec
    expected_gain: Optional[float] = None   # the Researcher's predicted primary delta (a number, checked against the measurement)
    gain_evidence: str = ""       # why that number: own measured deltas and/or published results
    ablation_plan: str = ""       # variants the pipeline should also score and print as ABLATION lines (in-run attribution)

    REQUIRED = ("hypothesis", "category", "change_spec", "expected_risk", "builds_on", "expected_gain")

    @classmethod
    def from_obj(cls, obj: Any) -> "ResearcherPlan":
        if not isinstance(obj, dict):
            raise ContractError("researcher output must be a JSON object")
        missing = [k for k in cls.REQUIRED if k not in obj]
        if missing:
            raise ContractError(f"researcher output missing keys: {missing}")
        for k in ("hypothesis", "change_spec"):
            if not isinstance(obj[k], str) or not obj[k].strip():
                raise ContractError(f"researcher field '{k}' must be a non-empty string")
        category = str(obj["category"]).strip().lower()
        if category not in CATEGORIES:
            raise ContractError(f"category must be one of {CATEGORIES}, got {obj['category']!r}")
        risk = str(obj["expected_risk"]).strip().lower()
        if risk not in RISKS:
            raise ContractError(f"expected_risk must be one of {RISKS}, got {obj['expected_risk']!r}")
        rationale = obj.get("rationale") or ""
        if not isinstance(rationale, str):
            rationale = json.dumps(rationale)
        try:
            gain = float(str(obj["expected_gain"]).strip().replace("+", "").rstrip("%"))
        except (TypeError, ValueError):
            raise ContractError(f"expected_gain must be a number (predicted primary delta, e.g. 0.003), got {obj['expected_gain']!r}")
        if not (-1.0 <= gain <= 1.0):
            raise ContractError(f"expected_gain must be a primary delta in [-1, 1], got {gain}")
        evidence = obj.get("gain_evidence") or ""
        ablation = obj.get("ablation_plan") or ""
        return cls(hypothesis=obj["hypothesis"].strip(), category=category,
                   change_spec=obj["change_spec"].strip(), expected_risk=risk,
                   builds_on=str(obj.get("builds_on", "champion")).strip() or "champion",
                   rationale=rationale.strip() or obj["change_spec"].strip(),
                   expected_gain=gain,
                   gain_evidence=(evidence if isinstance(evidence, str) else json.dumps(evidence)).strip(),
                   ablation_plan=(ablation if isinstance(ablation, str) else json.dumps(ablation)).strip())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# §5.3 Harness result
# ---------------------------------------------------------------------------
@dataclass
class HarnessResult:
    status: str                      # scored | failed | timeout
    gauc: float = 0.0
    ndcg5: float = 0.0
    primary: float = 0.0
    runtime_s: float = 0.0
    error_excerpt: str = ""
    vs_best: str = "n/a"

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"bad status {self.status}")

    @property
    def scored(self) -> bool:
        return self.status == "scored"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["runtime_s"] = round(float(self.runtime_s), 1)
        return d


@dataclass
class DebugAttempt:
    attempt: int
    error: str
    fix_summary: str
    status_after: str = ""           # what happened after the fix ran (harness-measured)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# §5.6 Per-iteration log entry (judges' deliverable)
# ---------------------------------------------------------------------------
@dataclass
class IterationLog:
    iteration: int
    timestamp: str
    hypothesis: str
    rationale: str
    category: str
    code_diff: str
    result: Dict[str, Any]
    errors_and_recovery: List[Dict[str, Any]]
    decision: str
    streak_after: int
    tokens_this_iteration: int
    runtime_s: float
    lesson: str = ""
    harness_extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "iteration": self.iteration, "timestamp": self.timestamp,
            "hypothesis": self.hypothesis, "rationale": self.rationale, "category": self.category,
            "code_diff": self.code_diff,
            "result": self.result,
            "errors_and_recovery": self.errors_and_recovery,
            "decision": self.decision, "streak_after": self.streak_after,
            "tokens_this_iteration": self.tokens_this_iteration,
            "runtime_s": round(float(self.runtime_s), 1),
            "lesson": self.lesson,
            "harness_extra": self.harness_extra,
        }
        return d


# ---------------------------------------------------------------------------
# LLM accounting
# ---------------------------------------------------------------------------
@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: Optional[float] = None   # real per-call cost in USD when the provider reports one (OpenRouter
                                        # `usage.include=true`); None when unavailable — never guessed from tokens,
                                        # since that would silently misrepresent a real number as a real one.

    @property
    def total(self) -> int:
        # cache reads/creations are already counted inside input_tokens by some providers and not by
        # others; we bill conservatively: everything the API reports as consumed.
        return int(self.input_tokens + self.output_tokens + self.cache_creation_input_tokens
                   + self.cache_read_input_tokens)

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total"] = self.total
        return d


# ---------------------------------------------------------------------------
# Persisted run state (crash-resume point, written after every iteration)
# ---------------------------------------------------------------------------
@dataclass
class RunState:
    run_id: str
    start_ts: str                     # ISO UTC, human readable
    start_time: float                 # epoch seconds; wall-clock is measured from here (survives resume)
    iteration: int = 0                # last COMPLETED iteration
    streak: int = 0
    best_primary: Optional[float] = None
    best_gauc: Optional[float] = None
    best_ndcg5: Optional[float] = None
    best_iter: int = 0
    baseline_primary: Optional[float] = None   # published validation reference (from the kit json)
    tokens_total: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_by_role: Dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    spend_start: Dict[str, Any] = field(default_factory=dict)   # provider credit snapshot at run start
    spend_end: Dict[str, Any] = field(default_factory=dict)     # ... and at finalize (account-wide; on a shared
                                                                  # key this delta includes any other activity on it)
    real_spend_usd: Optional[float] = None   # sum of real per-call costs (OpenRouter usage.cost) — immune to
                                              # concurrent activity from teammates sharing the same API key;
                                              # None until the first call that actually reports a cost arrives
    interventions: int = 0
    resumes: int = 0
    blocked: List[str] = field(default_factory=list)
    consecutive_failures: int = 0
    stop_reason: Optional[str] = None
    phase0: Dict[str, Any] = field(default_factory=dict)
    finalize: Dict[str, Any] = field(default_factory=dict)
    best_history: List[Optional[float]] = field(default_factory=list)   # champion score after each iteration (window streak mode)
    best_measured: Dict[str, Any] = field(default_factory=dict)        # best leak-clean score seen, even if below the margin
    synthesis: str = ""                                                # Scribe's research synthesis of the digest (interpretive; number-checked)
    history: List[Dict[str, Any]] = field(default_factory=list)   # one compact dict per iteration
    warnings: List[str] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # -- helpers -------------------------------------------------------------
    def elapsed_s(self, now: Optional[float] = None) -> float:
        return (now if now is not None else time.time()) - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunState":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def atomic_write_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
