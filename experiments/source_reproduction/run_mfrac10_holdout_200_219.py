#!/usr/bin/env python3
"""Fresh 20-seed robustness check with 10% malicious clients (seeds 200-219)."""

import sys
import time
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fl_instrumented as fl

out = (
    Path(__file__).resolve().parents[1]
    / "results_rerun"
    / "mfrac10_holdout_200_219_raw.csv"
)

out.parent.mkdir(parents=True, exist_ok=True)

alphas = [0.1, 0.5, 1.0]
m_frac = 0.1
seeds = range(200, 220)

rows = pd.read_csv(out).to_dict("records") if out.exists() else []

done = {
    (
        round(float(r["alpha"]), 4),
        round(float(r["m_frac"]), 4),
        int(r["seed"]),
        str(r["defense"]),
    )
    for r in rows
}

t0 = time.time()

for a in alphas:
    for seed in seeds:

        methods = [
            ("fedavg", "fedavg", {}),
            (
                "probeguard",
                "probeguard",
                {"k": 2.0, "delta": 0.1},
            ),
            (
                "probeguard_k",
                "probeguard_k",
                {
                    "k": 2.0,
                    "delta": 0.1,
                    "f_assumed": 2,
                },
            ),
        ]

        for name, defense, kw in methods:

            key = (
                round(a, 4),
                round(m_frac, 4),
                seed,
                name,
            )

            if key in done:
                continue

            r = fl.run(
                a,
                m_frac,
                "fixed",
                defense,
                seed,
                **kw,
            )

            rows.append(
                {
                    "alpha": a,
                    "m_frac": m_frac,
                    "actual_malicious": 2,
                    "seed": seed,
                    "defense": name,
                    "asr": r["asr"],
                    "clean_accuracy": r["acc"],
                    "tpr": r.get("tpr"),
                    "fpr": r.get("fpr"),
                }
            )

            done.add(key)

        pd.DataFrame(rows).to_csv(out, index=False)

        print(
            "alpha", a,
            "seed", seed,
            "rows", len(rows),
            "elapsed_sec",
            round(time.time() - t0, 1),
            flush=True,
        )

print("DONE:", out)
