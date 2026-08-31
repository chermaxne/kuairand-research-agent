# Manual interventions — 20260831_145457_1k_bonus_test

Every time a human touches this run (restarts it, edits a file, unblocks the agent, changes config),
add a row here — `python -m agent.intervene "what you did" --run-dir /home/q3user/kuairand-research-agent/runs/20260831_145457_1k_bonus_test` does it for you and bumps
the counter in run_state.json. Honesty is the product.

Count: 3

| timestamp (UTC) | what was stuck | what the human did | scope |
|---|---|---|---|
| 2026-08-31T07:43:40Z | harness process ended after iteration 0 (stop_reason=None) | harness restarted and resumed from run_state.json | resume (auto-recorded) |
| 2026-08-31T10:26:48Z | harness process ended after iteration 2 (stop_reason=None) | harness restarted and resumed from run_state.json | resume (auto-recorded) |
| 2026-08-31T12:44:07Z | harness process ended after iteration 4 (stop_reason=None) | harness restarted and resumed from run_state.json | resume (auto-recorded) |

**Human-added disambiguation** (the three auto-recorded rows above are logged identically by the harness
and do not distinguish *why* each restart happened; adding the real reasons here for the record):
1. **07:43:40Z** — routine: the transition from the `--phase0-only` dry run process into the iteration-loop
   process proper. Not a failure or a pause; expected two-step launch sequence for this run.
2. **10:26:48Z** — deliberate human-requested pause. A human asked to stop the run cleanly once iteration 2
   finished (leak-test included) so they could leave; the process was sent SIGINT at that safe boundary and
   later resumed on request. No iteration was lost or corrupted.
3. **12:44:07Z** — infrastructure failure, not a pause. Iteration 5's first attempt crashed with
   `openai.APIError: Upstream idle timeout exceeded` (an upstream OpenRouter/provider hiccup) during a
   researcher streaming call — an unhandled exception the harness itself did not catch, retry, or recover
   from. A human/coding-assistant noticed the dead process and relaunched it from the last saved
   `run_state.json`. This is a genuine robustness gap (the harness has no retry around this specific
   exception type), not evidence the harness is unreliable in general — every other transient failure in
   this project's history (rate limits, streaming stalls) is caught by `llm_client.py`'s retry/fallback logic;
   this exception type apparently is not.
