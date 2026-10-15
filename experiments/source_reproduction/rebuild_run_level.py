#!/usr/bin/env python3
"""Re-derive fullpop_mechanism_run_level.csv (shipped empty) from the round-level log.

Per (alpha, seed, defense): final-round ASR/accuracy and late-round (16-30) mean score-AUROC/TPR/FPR.
Consistency check: final ASR must equal the alpha-sweep runs for the same (alpha, seed, defense).
"""
import sys
from pathlib import Path
import pandas as pd
base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "completed"
Q = pd.read_csv(base / "fullpop_mechanism_round_level.csv")
last = Q[Q["round"] == Q["round"].max()].set_index(["alpha", "seed", "defense"])[["asr", "acc"]]
late = Q[Q["round"] >= 16].groupby(["alpha", "seed", "defense"])[["auc", "tpr", "fpr"]].mean().add_prefix("late_")
R = last.join(late).reset_index().rename(columns={"asr": "final_asr", "acc": "final_acc"})
A = pd.read_csv(base / "alpha_sweep_raw.csv")
A = A[A.defense.isin(["probeguard", "probeguard_k"])].rename(columns={"defense": "defense"})
chk = R.merge(A, on=["alpha", "seed", "defense"])
print(f"{len(R)} runs; {len(chk)} matched to alpha sweep; max |ASR diff| = {(chk.final_asr-chk.asr).abs().max():.2e}")
R.to_csv(base / "fullpop_mechanism_run_level.csv", index=False)
