# Evidence behind knowledge/library.md
Scripts and outputs that produced every number in the playbook (validation split, sealed evaluator):
`analyze.py` → `stats.json` (data facts) · `probe.py` (rate features, LightGBM, ensemble) · `probe2.py` (tab rates,
lambdarank, multi-task FM) · `probe3.py` (seed ensembles, recency weighting) · `evaluator_*.py/.out` (the independent
evaluation agent's recomputation and experiments: pairwise loss, session-position field, OOF-stacked ranker) ·
`findings_draft.md` (the draft the evaluator audited). Re-run any script with `.venv/bin/python` from the repo root.
