#!/usr/bin/env python3
"""Study-A experiments re-run on a configurable seed range (default: the 20 fresh seeds 100-119).

Closes the gap that the baseline table, trigger-family table, percentile/oracle decision-rule
sweep and adaptive-attacker grid exist only as 5-8 seed historical outputs with no recovered
runner. Resumable; writes to experiments/results_rerun/study_a_<experiment>.csv.

  python run_study_a.py baselines  100 120      # FedAvg/median/trimmed/normclip/MKrum/FLTrust/MAD/Top-f/oracle
  python run_study_a.py triggers   100 120      # fixed/multi/lowint/large at alpha=10 and 1
  python run_study_a.py rules      100 120      # MAD, 90/95/97.5 pct, Top-f, oracle threshold
  python run_study_a.py adaptive   100 120      # beta in {0,.1,.25,.5}; alpha in {1,10}
Optional 4th arg: comma list of alphas to override.
"""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fl_instrumented as fl

OUT = Path(__file__).resolve().parents[1] / "results_rerun"
K, DELTA, F = 2.0, 0.1, 4

def plans(name):
    if name == "baselines":
        alphas = [0.1, 0.5, 1.0, 10.0]
        D = [("fedavg", {}), ("median", {}), ("trimmed", {}), ("normclip", {}), ("multikrum", {}),
             ("fltrust", {}), ("probeguard", dict(k=K, delta=DELTA)),
             ("probeguard_k", dict(k=K, delta=DELTA, f_assumed=F)), ("clean_oracle", {})]
        return [dict(alpha=a, attack="fixed", defense=d, kw=kw, tag="") for a in alphas for d, kw in D]
    if name == "triggers":
        D = [("fedavg", {}), ("median", {}), ("fltrust", {}), ("probeguard", dict(k=K, delta=DELTA)),
             ("probeguard_k", dict(k=K, delta=DELTA, f_assumed=F)), ("clean_oracle", {})]
        return [dict(alpha=a, attack=t, defense=d, kw=kw, tag="") for a in [10.0, 1.0]
                for t in ["fixed", "multi", "lowint", "large"] for d, kw in D]
    if name == "rules":
        alphas = [0.1, 0.5, 1.0, 10.0]
        D = [("fedavg", {}), ("probeguard", dict(k=K, delta=DELTA)), ("pg_pct90", {}), ("pg_pct95", {}),
             ("pg_pct975", {}), ("probeguard_k", dict(k=K, delta=DELTA, f_assumed=F)), ("pg_oracle", {}),
             ("clean_oracle", {})]
        return [dict(alpha=a, attack="fixed", defense=d, kw=kw, tag="") for a in alphas for d, kw in D]
    if name == "adaptive":
        D = [("fedavg", {}), ("multikrum", {}), ("probeguard", dict(k=K, delta=DELTA)),
             ("probeguard_k", dict(k=K, delta=DELTA, f_assumed=F))]
        return [dict(alpha=a, attack="fixed", defense=d, kw=dict(kw, adaptive=True, beta=b), tag=f"beta={b}")
                for a in [1.0, 10.0] for b in [0.0, 0.1, 0.25, 0.5] for d, kw in D]
    raise SystemExit(__doc__)

def main():
    name, s0, s1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    P = plans(name)
    if len(sys.argv) > 4:
        keep = {float(x) for x in sys.argv[4].split(",")}; P = [p for p in P if p["alpha"] in keep]
    OUT.mkdir(parents=True, exist_ok=True); path = OUT / f"study_a_{name}.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["alpha"], r["attack"], r["defense"], r["tag"] if isinstance(r["tag"], str) else "", r["seed"]) for r in rows}
    t0 = time.time()
    for seed in range(s0, s1):
        for p in P:
            key = (p["alpha"], p["attack"], p["defense"], p["tag"], seed)
            if key in done: continue
            r = fl.run(p["alpha"], 0.2, p["attack"], p["defense"], seed, **p["kw"])
            rows.append(dict(experiment=name, alpha=p["alpha"], attack=p["attack"], defense=p["defense"],
                             tag=p["tag"], seed=seed, asr=r["asr"], clean_accuracy=r["acc"],
                             tpr=r.get("tpr"), fpr=r.get("fpr")))
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"{name} seed={seed} alpha={p['alpha']} {p['attack']} {p['defense']} {p['tag']} "
                  f"ASR={r['asr']:.3f} [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    main()
