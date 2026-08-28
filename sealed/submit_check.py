"""Sealed wrapper around the starter kit's own checker (`submit.py --check`).

It never re-implements validation: it executes the organizers' script unchanged and propagates its
exit code and output. Usage:
    python sealed/submit_check.py --split test --data_dir <data> [--kit <starter_kit>] submission.csv
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--kit", default=os.path.join(HERE, "..", "starter_kit"))
    a = ap.parse_args()
    kit = os.path.abspath(a.kit)
    cmd = [sys.executable, os.path.join(kit, "submit.py"), "--check", "--split", a.split,
           "--data_dir", os.path.abspath(a.data_dir), os.path.abspath(a.path)]
    cp = subprocess.run(cmd, cwd=kit)
    sys.exit(cp.returncode)
