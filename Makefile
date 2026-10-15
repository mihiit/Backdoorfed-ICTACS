PY ?= python
test:            ; $(PY) -m pytest -q tests
digits-core:     ; cd experiments/source_reproduction && $(PY) run_alpha_sweep.py 100 120 && $(PY) run_f_sensitivity.py 100 120 && $(PY) run_fresh20.py
digits-extra:    ; cd experiments/source_reproduction && for e in baselines triggers rules adaptive; do $(PY) run_study_a.py $$e 100 120; done
digits-ksweep: ; cd experiments/source_reproduction && $(PY) run_k_sweep_all_alpha.py 100 120
digits-sens:     ; cd experiments/source_reproduction && $(PY) run_mad_threshold_sensitivity.py 100 110 && $(PY) run_malicious_fraction_sweep.py 0 5
figures:         ; $(PY) scripts/make_paper_figures.py
cnn-check:       ; $(PY) experiments/cnn_cifar10/run_cnn.py --data data --seeds 0:1 --alphas 10 --rounds 15 --check-learning
cnn-smoke:       ; $(PY) experiments/cnn_cifar10/run_cnn.py --data data --out experiments/cnn_cifar10/results_smoke --seeds 0:1 --alphas 10 --rounds 1 --clients 10 --root-size 100 --max-client-samples 256 --methods fedavg,mad,topf,oracle
