"""Validate leak test v2 on real data: two legitimate pipelines (must be clean) and the real 0.8484 leaker (must be LEAK)."""
import sys, time, json, yaml
sys.path.insert(0, "/Users/ckwang/Documents/TechJam/kuairand-starter-kit")
from agent.task import make_task
SCR = "/private/tmp/claude-501/-Users-ckwang-Documents-TechJam-kuairand-starter-kit/afbb0d1b-da0b-4c92-b210-73859310e750/scratchpad/leakv2"
cfg = yaml.safe_load(open("/Users/ckwang/Documents/TechJam/kuairand-starter-kit/config.yaml"))
t = make_task(cfg, "/Users/ckwang/Documents/TechJam/kuairand-starter-kit", toy=False); t.prepare(lambda *_: None)
cases = [("clean_baseline", "baseline FM champion (legit, 0.6015)"),
         ("clean_it01", "ten6 it01 bundle (legit 0.6046, v1 FALSELY flagged)"),
         ("real_leaker", "ten5 it01 (real leak, scored the 0.8484 oracle)")]
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
