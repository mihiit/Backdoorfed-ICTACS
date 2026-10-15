# Reproducibility status (updated after code review)

## Verified in review
- 12 tests pass (`make test`): NumPy building blocks of the CNN harness and bit-exact re-runs of historical digits results.
- Digits runs are deterministic: alpha sweep (24 runs), f-sensitivity (12) and malicious-fraction sweep (27) re-ran with **zero** difference from the stored CSVs.
- `fullpop_mechanism_round_level.csv` agrees with `alpha_sweep_raw.csv` on final ASR for all 80 matched runs.

## Bugs fixed
- `run_batch.py`, `run_fresh20.py`, `run_mad_threshold_sensitivity.py`, `run_trajectory_diagnostic.py`: `NameError: Path` (crashed on start); `run_batch.py` also called `.parent` on a `str`.
- `run_malicious_fraction_sweep.py`: hard-coded `/mnt/data/...` path.
- `fullpop_mechanism_run_level.csv` shipped with 0 bytes -> rebuilt (`rebuild_run_level.py`).
- CNN v1 script: seven defects, listed in `experiments/cnn_cifar10/README.md`; replaced by v2.

## Still not verified / not true
- Only spot checks (1 seed) were re-executed; full 20-seed regeneration has not been done by this review.
- No original CNN runner exists; the manuscript's 3-seed CNN table cannot be regenerated. Its FLTrust "unresolved failure" may be an implementation issue of the same kind as v1 defect 3 (check when the original code is found); the digits FLTrust implementation is correct.
- CNN v2 driver `run_cnn.py` has never been run. Pilot first.
- Study A items (baselines, triggers, rules, AUROC, delta-B, ablations, overhead) lack original runners; AUROC/delta-B/ablation/overhead have no re-creation script yet.
- Environment: versions pinned in `requirements.txt` are those used for the verification above (Python 3.12, Linux); other platforms may differ in the last floating-point bits.
- Seeds 100-102 are used both for tuning (k, delta) and in the 20-seed evaluation.


## Update (v6)
- Re-run on the 20 Study B seeds (100-119) and stored in `experiments/results_rerun/`: baselines (720 runs), decision rules (640), adaptive attacker (640), k-sweep over all alphas (720). The k=2 and top-f rows reproduce the historical Study B means exactly.
- Paper Tables I, III, IV are now these 20-seed values; the old 8-seed (and 5-seed adaptive) values are superseded.
- Still not regenerated: open-loop AUROC, benign-spread study, root-size/probe-size ablations, overhead timing (no original runners); CNN driver beyond the user's own runs.
