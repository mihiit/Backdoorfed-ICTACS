# CIFAR-10 CNN stress test (protocol v2)

**v1 (`legacy/run_cnn100_v1_DO_NOT_USE.py`) must not be used.** Defects found in review:

| # | v1 defect | Effect |
|---|---|---|
| 1 | Trigger at pixels 4:8,4:8; probe grid = 9 cells at (0|14|28, 0|14|28) | the trigger cell is **not in the probe bank**, so the defence is probing for the wrong thing (the paper's design includes the trigger cell) |
| 2 | Score = max flip rate **to class 0 only**, no median reference | uses the (secret) target class and is not the paper's score |
| 3 | FLTrust compared a raw gradient with descent-direction updates | cosine is negative -> all trust 0 -> aggregate = 0 -> model frozen at chance |
| 4 | Network has 56,970 parameters | paper says 189k |
| 5 | batch 128, momentum 0.9, lr 0.01 | paper: batch 32 |
| 6 | Multi-Krum used n-f-3 neighbours; no norm clipping baseline; MAD fallback differed from the paper | minor |
| 7 | Single shared output CSV, no sharding | races when parallelised |

v2 (`run_cnn.py`) fixes all of these: 4x4 bottom-right trigger, 64-cell stride-4 probe grid containing the trigger cell,
score over all 10 classes with the cross-client-median reference, correct FLTrust, 188,810-parameter CNN, batch 32,
sharding, per-round closed-loop score AUROC logging (the diagnostic the paper says was "not run at CNN scale").

**Open design choices you must confirm** (the paper does not specify them): learning rate (default 0.02), momentum (0),
trigger intensity (1.0), k/delta (2.0/0.1, tuned on digits). Report whatever you choose.

## Procedure
```bash
make cnn-check      # benign FedAvg, 15 rounds, prints test accuracy per round. Need clearly above chance (aim >= 50%)
                    # If not: raise --rounds / --lr / --local-epochs; do NOT launch the sweep.
make cnn-smoke      # 1 round, 10 clients, tiny: proves the code path end to end
# pilot: time ONE task, then extrapolate
python experiments/cnn_cifar10/run_cnn.py --data data --seeds 0:1 --alphas 10 --out experiments/cnn_cifar10/results_pilot
# full run, 4 shards in parallel (repeat with --shard 1/4 ... 3/4, ideally on separate cores/GPUs)
python experiments/cnn_cifar10/run_cnn.py --data data --seeds 0:100 --alphas 1,10 --shard 0/4 --threads 4
python experiments/cnn_cifar10/analyze_cnn.py --in experiments/cnn_cifar10/results --out experiments/cnn_cifar10/analysis
```
## Compute (order-of-magnitude estimate, NOT measured)
One (alpha, seed) task trains ~50k samples x 15 rounds per method, and MAD/Top-f add 64 probes x 200 images x 20 clients per round
(about as costly as training). 9 methods is roughly 11 training-equivalents per task. Time the pilot; on a CPU laptop expect
hours per task, so 100 seeds x 3 alphas x 9 methods is likely hundreds of CPU-hours. Tiers if that is too much:
(A) `--methods fedavg,mad,topf,oracle --alphas 1,10 --seeds 0:30` (B) add `multikrum,coord_median` (C) full grid on a GPU.
`--device` is not exposed yet: move tensors/model to CUDA in `run_cnn.py` if you have a GPU (one-line changes in `local_train`/`predict`).
