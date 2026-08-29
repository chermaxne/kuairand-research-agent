"""Promotion, convergence-streak and stopping logic — PURE functions, unit-tested.

Spec §2.5 / §2.6 / §6:
  * promotion   : primary > best + PROMOTE_MARGIN            (checkpoint update)
  * streak reset: primary > best_at_iteration_start + EPSILON (convergence bookkeeping)
  * failed / timed-out / flat / tiny-gain iterations ALL tick the streak.
The two comparisons are deliberately separate functions so nobody can merge them by accident.
No LLM output ever reaches these functions — only harness-measured numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

STOP_CONVERGED = "converged"
STOP_ITER_CAP = "iter_cap"
STOP_WALL_CLOCK = "wall_clock"
STOP_SPEND_GUARD = "spend_guard"


@dataclass(frozen=True)
class RunLimits:
    epsilon: float = 0.002
    n_flat: int = 3
    max_iters: int = 50
    wall_clock_s: float = 6 * 3600.0
    promote_margin: float = 0.0010
    max_total_tokens: Optional[int] = None

    @classmethod
    def from_config(cls, cfg: dict) -> "RunLimits":
        run = cfg["run"]
        llm = cfg.get("llm", {})
        return cls(epsilon=float(run["EPSILON"]), n_flat=int(run["N_FLAT"]), max_iters=int(run["MAX_ITERS"]),
                   wall_clock_s=float(run["WALL_CLOCK_HOURS"]) * 3600.0,
                   promote_margin=float(run["PROMOTE_MARGIN"]),
                   max_total_tokens=(int(llm["max_total_tokens"]) if llm.get("max_total_tokens") else None))


def should_promote(primary: Optional[float], best_primary: Optional[float], margin: float) -> bool:
    """Checkpoint update rule. A failed experiment (primary=None) never promotes."""
    if primary is None:
        return False
    if best_primary is None:
        return True
    return primary > best_primary + margin


def next_streak(primary: Optional[float], best_at_iter_start: Optional[float], streak: int, epsilon: float) -> int:
    """Convergence rule. Reset only on improvement > epsilon over the best known at iteration start;
    everything else (flat, tiny gain, worse, failed, timeout) ticks the streak."""
    if primary is not None and (best_at_iter_start is None or primary > best_at_iter_start + epsilon):
        return 0
    return streak + 1


def decision_label(status: str, promoted: bool) -> str:
    if status != "scored":
        return "failed"
    return "promoted" if promoted else "kept_champion"


def stop_reason(streak: int, iteration: int, elapsed_s: float, tokens_total: int, limits: RunLimits) -> Optional[str]:
    """Checked at the top of the loop BEFORE starting a new iteration (spec §2.5: never start past the ceiling)."""
    if streak >= limits.n_flat:
        return STOP_CONVERGED
    if iteration >= limits.max_iters:
        return STOP_ITER_CAP
    if elapsed_s >= limits.wall_clock_s:
        return STOP_WALL_CLOCK
    if limits.max_total_tokens is not None and tokens_total >= limits.max_total_tokens:
        return STOP_SPEND_GUARD
    return None


@dataclass(frozen=True)
class IterationDecision:
    promoted: bool
    decision: str
    streak_after: int
    best_primary_after: Optional[float]
    best_iter_after: int


def judge_iteration(status: str, primary: Optional[float], best_primary: Optional[float], best_iter: int,
                    streak_before: int, iteration: int, limits: RunLimits) -> IterationDecision:
    """Combine the two separate judgments for one iteration. `best_primary` is the best at iteration start."""
    if status != "scored":
        primary = None
    promoted = should_promote(primary, best_primary, limits.promote_margin)
    streak_after = next_streak(primary, best_primary, streak_before, limits.epsilon)
    return IterationDecision(
        promoted=promoted,
        decision=decision_label(status, promoted),
        streak_after=streak_after,
        best_primary_after=(primary if promoted else best_primary),
        best_iter_after=(iteration if promoted else best_iter),
    )


def vs_best_string(primary: Optional[float], best_primary: Optional[float]) -> str:
    if primary is None or best_primary is None:
        return "n/a"
    return f"{primary - best_primary:+.4f}"
