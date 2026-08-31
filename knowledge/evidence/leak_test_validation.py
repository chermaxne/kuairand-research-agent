"""Validate leak test v2 on real data: two legitimate pipelines (must be clean) and a deliberate label
leak (must be LEAK).

REGENERATED 2026-09-01: the original candidates (ten5/ten6 it01, from a since-deleted scratch dir) no
longer exist -- see git history's "caught a 0.8484 oracle score" commit for that incident. These are
fresh stand-ins that exercise the same three cases with real, currently-available code: the actual
baseline champion, an actual promoted iteration from a saved run, and a newly-built minimal leaker
(knowledge/evidence/leak_validation_candidates/real_leaker/) that adds the label as an input field on
purpose. This validates the leak_test() *mechanism*, not the historical incident's exact numbers."""
import sys, time, json, yaml
sys.path.insert(0, ".")
from agent.task import make_task
SCR = "./knowledge/evidence/leak_validation_candidates"
cfg = yaml.safe_load(open("./config.yaml"))
t = make_task(cfg, ".", toy=False); t.prepare(lambda *_: None)
cases = [("clean_baseline", "baseline FM champion (legit, ~0.6015)"),
         ("clean_it01", "runs/20260830_020718_chermaine_test best/it02 (legit, promoted at 0.6048)"),
         ("real_leaker", "regenerated deliberate label-as-feature leak (must be LEAK)")]
out = {}
for d, label in cases:
    t0 = time.time()
    r = t.leak_test(f"{SCR}/{d}", 900)
    verdict = "clean" if (r.get("ran") and r.get("subset_primary", 0) >= 0.5) else ("LEAK" if r.get("ran") else "INCONCLUSIVE")
    out[d] = {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "attempts"}, "verdict": verdict}
    print(f"{label:<52} verdict={verdict:<12} subset_primary={r.get('subset_primary')} subset_gauc={r.get('subset_gauc')} "
          f"full={r.get('full_primary')} frac={r.get('fraction')} ({time.time()-t0:.0f}s)", flush=True)
    print(f"    attempts: {[{k: (round(v,3) if isinstance(v,float) else v) for k,v in a.items() if k != 'error'} for a in r.get('attempts', [])]}", flush=True)
json.dump(out, open(f"{SCR}/validation.json", "w"), indent=1, default=str)
print("VALIDATION_DONE")
