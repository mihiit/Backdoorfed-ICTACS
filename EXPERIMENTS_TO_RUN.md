# What to run to strengthen the paper (priority order)

| # | Run | Why | Command | Status |
|---|---|---|---|---|
| 1 | CNN benign learning check | without it CNN ASR is meaningless (paper's own 10k config was dropped for this) | `make cnn-check` | not run |
| 2 | CNN pilot then 30-100 seeds, alpha in {1,10} (+0.5) | replaces 3-seed CNN test; includes closed-loop AUROC at CNN scale | see `experiments/cnn_cifar10/README.md` | not run |
| 3 | Baselines on seeds 100-119 | baseline table (median, FLTrust, Multi-Krum) is 8 seeds; MAD/Top-f is 20 | `python run_study_a.py baselines 100 120` | script tested on 1 seed |
| 4 | Adaptive attacker, 20 seeds | currently 5 seeds, CI +-11 pp at alpha=1 | `python run_study_a.py adaptive 100 120` | tested on 1 seed |
| 5 | Other trigger families on 20 seeds | "robust across four triggers" rests on 8 seeds | `python run_study_a.py triggers 100 120` | tested on 1 seed |
| 6 | Percentile/oracle rules on 20 seeds | Top-f row in Table III disagrees across seed sets (15.0 vs 34.2 at alpha=0.5) | `python run_study_a.py rules 100 120` | tested on 1 seed |
| 7 | Seed hygiene: re-tune k,delta on seeds 200-202 | 100-102 are in both tuning and evaluation | edit `run_mad_threshold_sensitivity.py`; rerun | not run |
| 8 | Include MAD-k sensitivity and malicious-fraction results (already in `experiments/completed/`) | answers "was MAD just badly tuned?" and the 30%/40% attacker question | analysis only | data present, not yet in paper |
| 9 | Reconcile open-loop (0.949) vs closed-loop (0.81) AUROC at alpha=0.5 | the two protocols disagree; paper states this as unresolved | new diagnostic: score the benign-only trajectory and the closed-loop one on the same seeds | not written |
