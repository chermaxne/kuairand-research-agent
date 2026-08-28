# baseline_repro — iteration-0 champion

`pipeline.py` is the organizers' FM baseline (`starter_kit/baseline.py` + `starter_kit/data.py`) ported,
with identical numerics and seed, to the pipeline contract of spec §5.2. Phase 0 runs it through the sandbox,
checks it lands within ±0.005 of the published validation primary (0.6016) and compares it with the
predictions produced by the organizers' own `submit.py --make --split valid`, then installs it as the champion
in `runs/RUN_ID/best/code/`. Every experiment of the run is a minimal edit of this file.

Manual run (from the repo root, venv active):

    PYTHONPATH=sealed python baseline_repro/pipeline.py --data starter_kit/KuaiRand-Pure/data --split val --out /tmp/preds_val.csv
    python starter_kit/submit.py --score --split valid --data_dir starter_kit/KuaiRand-Pure/data /tmp/preds_val.csv
