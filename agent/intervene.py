"""Manual-intervention logger (spec §11): make honesty low-friction.

    python -m agent.intervene "what you did" [--stuck "what was stuck"] [--scope run] [--run-dir runs/X] [--block "direction"]

Appends a row to interventions.md, bumps the counter in run_state.json and optionally adds a
BLOCKED direction the Researcher must route around.
"""
from __future__ import annotations

import argparse
import os
import sys

from .memory import append_intervention, load_run_state, save_run_state


def latest_run_dir(runs_dir: str) -> str:
    p = os.path.join(runs_dir, "LATEST")
    if os.path.exists(p):
        return open(p).read().strip()
    raise SystemExit("no runs/LATEST; pass --run-dir")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what_done", help="what the human did")
    ap.add_argument("--stuck", default="(not stated)", help="what was stuck")
    ap.add_argument("--scope", default="run", help="iteration | run | infra | config | other")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--block", action="append", default=[], help="mark a direction BLOCKED for the Researcher")
    a = ap.parse_args(argv)
    run_dir = a.run_dir or latest_run_dir(a.runs_dir)
    n = append_intervention(run_dir, a.stuck, a.what_done, a.scope, bump=True)
    if a.block:
        st = load_run_state(run_dir)
        for b in a.block:
            st.blocked.append(f"manual: {b}")
        save_run_state(run_dir, st)
    print(f"recorded intervention #{n} in {run_dir}/interventions.md" + (f" (+{len(a.block)} blocked)" if a.block else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
