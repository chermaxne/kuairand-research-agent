"""Copy a run directory into runs/example_run/ (committed) without the bulky/regenerable artifacts:
prediction CSVs, submission, phase-0 prediction files, __pycache__. Everything judges read stays:
ledger, state block, per-iteration JSON + narratives, diffs, code, stdout/stderr, LLM transcripts,
run_state, results summary, interventions, token log.

    python scripts/make_example_run.py runs/<RUN_ID> [runs/example_run]
"""
import os
import shutil
import sys

SKIP_SUFFIXES = (".csv", ".npz", ".npy", ".pkl", ".pt")
SKIP_DIRS = ("__pycache__", "best.tmp", "best.prev")


def main(src: str, dst: str = "runs/example_run") -> None:
    if os.path.exists(dst):
        shutil.rmtree(dst)
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(root, src)
        for f in files:
            if f.endswith(SKIP_SUFFIXES):
                continue
            out_dir = os.path.join(dst, rel) if rel != "." else dst
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(os.path.join(root, f), os.path.join(out_dir, f))
            n += 1
    with open(os.path.join(dst, "README_EXAMPLE.md"), "w") as fh:
        fh.write(f"# Example run (copied from `{os.path.basename(src)}`)\n\nPrediction CSVs (`preds_*.csv`, `submission.csv`, "
                 f"phase-0 rung files) were dropped to keep the repo small; everything else is verbatim. Regenerate with "
                 f"`python -m agent.harness --mock --label dryrun`.\n")
    print(f"copied {n} files from {src} to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "runs/example_run")
