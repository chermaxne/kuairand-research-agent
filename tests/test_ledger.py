"""Spec §14.4 — ledger line golden test (§5.4) + append-only property; state block format (§5.5)."""
import os
import re

from agent.memory import (active_themes, append_ledger, ledger_entries, ledger_line, read_ledger, render_state_block,
                          result_string, truncate_words)
from agent.schemas import RunState

GOLDEN = ("[it03] HYP: add click auxiliary head with weight 0.3 | CHANGE: pipeline.py (+14/-3) | "
          "RESULT: 0.6031 (best 0.6031) -> PROMOTED | LESSON: Auxiliary click signal helps the sparse long_view label.")


def test_ledger_line_matches_golden():
    line = ledger_line(3, "add click auxiliary head with weight 0.3", "pipeline.py (+14/-3)", result_string("scored", 0.6031),
                       0.6031, "promoted", "Auxiliary click signal helps the sparse long_view label.")
    assert line == GOLDEN


def test_ledger_line_failed_and_kept_forms():
    f = ledger_line(4, "h", "pipeline.py (+2/-1)", result_string("failed", None, "KeyError: 'x'"), 0.6031, "failed", "l")
    assert f == "[it04] HYP: h | CHANGE: pipeline.py (+2/-1) | RESULT: FAILED(KeyError: 'x') (best 0.6031) -> FAILED | LESSON: l"
    k = ledger_line(5, "h", "pipeline.py (+2/-1)", result_string("scored", 0.6020), 0.6031, "kept_champion", "l")
    assert k.endswith("RESULT: 0.6020 (best 0.6031) -> kept | LESSON: l")
    t = ledger_line(6, "h", "c", result_string("timeout", None, "limit 900s"), 0.6031, "failed", "l")
    assert "RESULT: FAILED(timeout: limit 900s)" in t


def test_ledger_line_regex_shape():
    pat = re.compile(r"^\[it\d{2}\] HYP: .+ \| CHANGE: .+ \| RESULT: .+ \(best (\d\.\d{4}|n/a)\) -> (PROMOTED|kept|FAILED) \| LESSON: .*$")
    assert pat.match(GOLDEN)


def test_lesson_is_sanitised_and_capped_at_20_words():
    lesson = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one | pipe\nnewline"
    line = ledger_line(1, "hyp | with pipe\nand newline", "c", "0.6", 0.6, "kept_champion", lesson)
    assert line.count("\n") == 0
    assert "HYP: hyp / with pipe and newline" in line
    assert line.endswith("LESSON: " + " ".join(truncate_words(lesson, 20).split()))
    assert len(line.split("LESSON: ")[1].split()) == 20


def test_ledger_is_append_only(tmp_path):
    d = str(tmp_path)
    append_ledger(d, "[it01] HYP: a | CHANGE: c | RESULT: 0.5 (best 0.5) -> PROMOTED | LESSON: x")
    first = read_ledger(d)
    append_ledger(d, "[it02] HYP: b | CHANGE: c | RESULT: 0.4 (best 0.5) -> kept | LESSON: y")
    second = read_ledger(d)
    assert second.startswith(first)                       # nothing before was rewritten
    assert len(ledger_entries(d)) == 2
    assert os.path.getsize(os.path.join(d, "ledger.md")) > len(first)


def test_state_block_format():
    st = RunState(run_id="r", start_ts="t", start_time=1000.0, iteration=3, streak=1, best_primary=0.6037, best_gauc=0.67,
                  best_ndcg5=0.5374, best_iter=2, baseline_primary=0.6016, tokens_total=12345, blocked=["it01: x [timeout 900s]"],
                  history=[{"category": "multitask", "decision": "promoted"}, {"category": "feature", "decision": "failed"}])
    limits = {"MAX_ITERS": 50, "WALL_CLOCK_HOURS": 6, "N_FLAT": 3, "EPSILON": 0.002}
    text = render_state_block(st, limits, 4, now=1000.0 + 3725)
    lines = text.splitlines()
    assert lines[0] == "CURRENT BEST: it02 | val primary 0.6037 (GAUC 0.6700 / nDCG5 0.5374) | baseline 0.6016 | margin +0.0021"
    assert lines[1] == "BUDGET: iteration 4 of 50 | 1:02 of 6:00 elapsed | tokens so far 12345"
    assert lines[2] == "CONVERGENCE: streak 1 of 3 flat (EPSILON=0.002)"
    assert lines[3] == "BLOCKED: it01: x [timeout 900s]"
    assert lines[4].startswith("ACTIVE THEMES: winning: multitask")
    assert "untried: model, training, other" in lines[4]


def test_active_themes_all_untried():
    assert active_themes([]) == "winning: none; losing/flat: none; untried: feature, model, training, multitask, other"
