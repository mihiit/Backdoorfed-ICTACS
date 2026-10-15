# ProbeGuard — code and data for *When Ranking Survives but Filtering Fails*

Behavioural trigger probing for federated backdoor defence under non-IID data: code, raw results and paper sources.

| Path | What it is |
|---|---|
| `paper_ieee/` | IEEE-format manuscript (ICTACS 2026): `main.tex` (self-contained layout), `main_ieeetran.tex` (official `IEEEtran`, uncompiled), figures, PDF |
| `paper/` | Original Springer-format sources and the 8-seed "Study A" result files (`*.jsonl`, `tab*.tex`) |
| `experiments/source_reproduction/` | Digits/MLP simulator (`fl_instrumented.py`) and runners for every Study B experiment, plus `run_study_a.py` |
| `experiments/completed/` | Historical raw/summary CSVs (20-seed Study B) |
| `experiments/cnn_cifar10/` | CIFAR-10 CNN stress test v2 (`run_cnn.py`, `pg_core.py`, `analyze_cnn.py`) |
| `tests/` | Unit tests + bit-exact regression against historical CSVs |
| `scripts/make_paper_figures.py` | Regenerates the IEEE paper figures from the CSVs |
| `REPRODUCIBILITY_STATUS.md`, `REPRODUCIBILITY_MATRIX.csv`, `EXPERIMENTS_TO_RUN.md` | What is verified, what is not, what to run next |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
make test                    # 12 tests, ~5 s; includes bit-exact re-runs of historical digits results
make figures                 # regenerate paper figures from experiments/completed/
```
Digits runs are pure NumPy and deterministic: re-running a seed reproduces the stored ASR/accuracy to the last bit.
Runners write to `experiments/results_rerun/` and never overwrite `experiments/completed/`.

## Running experiments
```bash
cd experiments/source_reproduction
python run_alpha_sweep.py 100 120                 # Table II / Fig. 2   (8 alphas x 3 defences x 20 seeds)
python run_f_sensitivity.py 100 120               # Fig. 3c
python run_study_a.py baselines 100 120           # baselines on the SAME 20 seeds (closes the 8-vs-20-seed gap)
python run_study_a.py adaptive  100 120           # adaptive attacker, 20 seeds (Table IV)
python run_study_a.py rules     100 120           # decision-rule sweep (Table III)
python run_k_sweep_all_alpha.py 100 120           # MAD k in {1,1.5,2} vs top-f, all alphas (f-free calibration)
```
Runners take `seed_start seed_end` and resume from their CSV; run disjoint seed ranges in parallel only with separate checkouts/output dirs (single shared CSV).

## CNN (CIFAR-10)
Download CIFAR-10 (python version), extract to `data/cifar-10-batches-py/` (not committed). See `experiments/cnn_cifar10/README.md`.
**Run `make cnn-check` first**: if the benign-only model does not learn, ASR is uninterpretable.

## Honest status
`experiments/cnn_cifar10/run_cnn.py` has not been executed by the person who wrote it (no torch/CIFAR in that sandbox);
its NumPy logic is unit-tested. Treat the first run as a pilot. The baseline, decision-rule and adaptive-attacker tables were re-run on the 20 Study B seeds with `run_study_a.py` (outputs in `experiments/results_rerun/`). The open-loop AUROC, spread and ablation analyses (Study A, 5-8 seeds) still have no recovered runner. See `REPRODUCIBILITY_STATUS.md`.

License: MIT (change if you prefer another). Cite via `CITATION.cff`.
