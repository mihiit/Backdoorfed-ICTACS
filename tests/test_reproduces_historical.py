"""Re-runs single historical digits experiments and requires bit-identical ASR / accuracy."""
import sys, pathlib
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "source_reproduction"))
import fl_instrumented as fl

H = pd.read_csv(ROOT / "experiments" / "completed" / "alpha_sweep_raw.csv")

def _hist(alpha, seed, defense):
    r = H[(H.alpha == alpha) & (H.seed == seed) & (H.defense == defense)]
    assert len(r) == 1; return r.iloc[0]

def test_mad_alpha1_seed100():
    r = fl.run(1.0, 0.2, "fixed", "probeguard", 100, k=2.0, delta=0.1, f_assumed=None); h = _hist(1.0, 100, "probeguard")
    assert abs(r["asr"] - h.asr) < 1e-12 and abs(r["acc"] - h.clean_accuracy) < 1e-12

def test_topf_alpha1_seed100():
    r = fl.run(1.0, 0.2, "fixed", "probeguard_k", 100, k=2.0, delta=0.1, f_assumed=4); h = _hist(1.0, 100, "probeguard_k")
    assert abs(r["asr"] - h.asr) < 1e-12
