#!/usr/bin/env python3
"""Held-out k calibration check on seeds 200-208."""
import sys, time, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fl_instrumented as fl

out = Path(__file__).resolve().parents[1] / "results_rerun" / "k_holdout_200_208_raw.csv"
out.parent.mkdir(parents=True, exist_ok=True)

alphas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0]
ks = [1.0, 1.5, 2.0]

rows = pd.read_csv(out).to_dict("records") if out.exists() else []
done = {
    (round(float(r["alpha"]), 4), int(r["seed"]), str(r["defense"]))
    for r in rows
}

t0 = time.time()

for seed in range(200, 209):
    for a in alphas:
        methods = [
            (f"MAD_k{k}", dict(k=k, delta=0.1), "probeguard")
            for k in ks
        ] + [
            ("TopF_f4", dict(k=2.0, delta=0.1, f_assumed=4), "probeguard_k")
        ]

        for name, kw, d in methods:
            key = (round(a, 4), seed, name)
            if key in done:
                continue

            r = fl.run(a, 0.2, "fixed", d, seed, **kw)

            rows.append({
                "alpha": a,
                "seed": seed,
                "defense": name,
                "asr": r["asr"],
                "clean_accuracy": r["acc"],
                "tpr": r.get("tpr"),
                "fpr": r.get("fpr"),
            })
            done.add(key)

        pd.DataFrame(rows).to_csv(out, index=False)

    print(
        "seed", seed,
        "rows", len(rows),
        "elapsed_sec", round(time.time() - t0, 1),
        flush=True
    )

print("DONE:", out)
