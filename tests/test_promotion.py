"""Spec §14.1 (convergence boundaries) and §14.2 (promotion vs streak separation) — pure functions."""
import json
import os

import pytest

from agent.promotion import (STOP_CONVERGED, STOP_ITER_CAP, STOP_SPEND_GUARD, STOP_WALL_CLOCK, RunLimits, judge_iteration,
                             next_streak, should_promote, stop_reason, vs_best_string)

LIM = RunLimits(epsilon=0.002, n_flat=3, max_iters=50, wall_clock_s=6 * 3600, promote_margin=0.0010)


def simulate(scores, limits=LIM, start_best=0.6016, seconds_per_iter=0.0, tokens_per_iter=0):
    """Drive the pure functions exactly like the harness does; return (stop_reason, best, best_iter, n_iters)."""
    best, best_iter, streak, it, elapsed, tokens = start_best, 0, 0, 0, 0.0, 0
    while True:
        reason = stop_reason(streak, it, elapsed, tokens, limits)
        if reason:
            return reason, best, best_iter, it
        primary = scores[it] if it < len(scores) else None      # None == failed iteration
        status = "scored" if primary is not None else "failed"
        d = judge_iteration(status, primary, best, best_iter, streak, it + 1, limits)
        best, best_iter, streak = d.best_primary_after, d.best_iter_after, d.streak_after
        it += 1
        elapsed += seconds_per_iter
        tokens += tokens_per_iter


# ---------------------------------------------------------------- §14.1 convergence boundaries
def test_stop_at_streak_with_correct_champion():
    # it1 +0.010 (promote, reset), it2 flat, it3 tiny (+0.0015 promotes but does NOT reset), it4 worse -> streak 3
    seq = [0.6116, 0.6116, 0.6131, 0.6000, 0.9, 0.9]
    reason, best, best_iter, n = simulate(seq)
    assert reason == STOP_CONVERGED
    assert n == 4
    assert best == pytest.approx(0.6131) and best_iter == 3


def test_stop_at_iteration_cap():
    seq = [0.6016 + 0.01 * (i + 1) for i in range(100)]          # keeps improving forever
    reason, best, best_iter, n = simulate(seq)
    assert reason == STOP_ITER_CAP and n == 50
    assert best_iter == 50 and best == pytest.approx(0.6016 + 0.5)


def test_stop_at_wall_clock_with_mock_time():
    seq = [0.6016 + 0.01 * (i + 1) for i in range(100)]
    lim = RunLimits(epsilon=0.002, n_flat=3, max_iters=50, wall_clock_s=6 * 3600, promote_margin=0.001)
    reason, best, best_iter, n = simulate(seq, lim, seconds_per_iter=3600)   # 1h per iteration -> stop before the 7th
    assert reason == STOP_WALL_CLOCK and n == 6 and best_iter == 6


def test_never_start_iteration_past_wall_clock():
    lim = RunLimits(wall_clock_s=100)
    assert stop_reason(0, 0, 99.9, 0, lim) is None
    assert stop_reason(0, 0, 100.0, 0, lim) == STOP_WALL_CLOCK


def test_spend_guard_stops_run():
    lim = RunLimits(max_total_tokens=1000)
    reason, *_ = simulate([0.7 + 0.01 * i for i in range(100)], lim, tokens_per_iter=300)   # always improving
    assert reason == STOP_SPEND_GUARD


def test_stop_reason_priority_converged_before_cap():
    assert stop_reason(3, 50, 0, 0, LIM) == STOP_CONVERGED


# ---------------------------------------------------------------- §14.2 promotion vs streak separation
def test_small_gain_promotes_but_does_not_reset_streak():
    best = 0.6016
    assert should_promote(best + 0.0015, best, LIM.promote_margin) is True
    assert next_streak(best + 0.0015, best, streak=2, epsilon=LIM.epsilon) == 3


def test_failure_ticks_streak_and_never_promotes():
    assert should_promote(None, 0.6016, LIM.promote_margin) is False
    assert next_streak(None, 0.6016, streak=0, epsilon=LIM.epsilon) == 1
    d = judge_iteration("failed", None, 0.6016, 0, 1, 7, LIM)
    assert d.decision == "failed" and d.streak_after == 2 and d.best_primary_after == 0.6016 and not d.promoted


def test_timeout_ticks_streak():
    d = judge_iteration("timeout", None, 0.6016, 0, 0, 1, LIM)
    assert d.streak_after == 1 and d.decision == "failed"


def test_big_gain_resets_streak_and_promotes():
    d = judge_iteration("scored", 0.6046, 0.6016, 0, 2, 5, LIM)
    assert d.promoted and d.streak_after == 0 and d.best_iter_after == 5 and d.best_primary_after == pytest.approx(0.6046)


def test_gain_between_margin_and_epsilon_is_promoted_but_flat():
    d = judge_iteration("scored", 0.6016 + 0.0015, 0.6016, 0, 0, 2, LIM)
    assert d.promoted and d.decision == "promoted" and d.streak_after == 1


def test_gain_below_margin_is_kept_and_flat():
    d = judge_iteration("scored", 0.6016 + 0.0005, 0.6016, 0, 0, 2, LIM)
    assert not d.promoted and d.decision == "kept_champion" and d.streak_after == 1


def test_exact_margin_boundary_is_not_promotion():
    assert should_promote(0.6026, 0.6016, 0.0010) is False       # must be strictly greater
    assert next_streak(0.6036, 0.6016, 0, 0.002) == 1             # strictly greater than epsilon


def test_worse_score_never_overwrites_best():
    d = judge_iteration("scored", 0.55, 0.6016, 3, 0, 4, LIM)
    assert d.best_primary_after == 0.6016 and d.best_iter_after == 3 and not d.promoted


def test_vs_best_string():
    assert vs_best_string(0.6037, 0.6016) == "+0.0021"
    assert vs_best_string(None, 0.6016) == "n/a"


# ---------------------------------------------------------------- config matches the kit's published constants
def test_config_matches_kit_convergence_rule(project_root, base_cfg):
    kit = json.load(open(os.path.join(project_root, "starter_kit", "baseline_scores.json")))
    assert float(base_cfg["run"]["EPSILON"]) == kit["convergence_rule"]["epsilon"]
    assert int(base_cfg["run"]["N_FLAT"]) == kit["convergence_rule"]["N"]
    assert RunLimits.from_config(base_cfg).promote_margin == pytest.approx(0.0005)     # team decision 2026-08-29 (spec default 0.0010)
    assert RunLimits.from_config(base_cfg).promote_margin < RunLimits.from_config(base_cfg).epsilon
