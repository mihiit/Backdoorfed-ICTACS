#!/usr/bin/env python3
"""CIFAR-10 federated backdoor stress test for ProbeGuard (v2, replaces the v1 re-implementation).

STATUS: written to match the manuscript protocol; the NumPy logic it relies on (pg_core.py) is unit-tested,
but this torch driver has NOT been executed by its author-assistant (no torch/CIFAR-10 in the sandbox).
Run `--check-learning` and the smoke test in README before any long run.

Protocol (matches Section "CNN-scale stress test"):
  20 clients, 20% malicious (random ids), Dirichlet(alpha) label skew, 200-image clean server root set,
  1 local epoch/round, batch 32, fixed trigger = 4x4 white patch bottom-right (28:32, 28:32),
  malicious clients stamp+relabel 50% of their non-target samples (target class 0),
  probe bank = all 4x4 patches on a stride-4 grid of the 32x32 image (64 cells, includes the trigger cell),
  score = max_{p,c}(R_i[p,c] - median_j R_j[p,c]) over ALL 10 classes (defender is not told the target),
  MAD rule k=2, delta=0.1 (values tuned on digits; not re-tuned), Top-f with true f.
Methods: fedavg, coord_median, trimmed_mean, norm_clip, multikrum, fltrust, mad, topf, oracle (benign-only avg).
Sharding: --shard i/n splits (alpha, seed) tasks; each shard writes its own CSV (no write races); merge with analyze_cnn.py.
"""
import argparse, csv, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pg_core as pg

TARGET, PATCH, TRIG = 0, 4, (28, 28)
ALL_METHODS = "fedavg,coord_median,trimmed_mean,norm_clip,multikrum,fltrust,mad,topf,oracle"


class CNN(nn.Module):                                  # 188,810 parameters (manuscript: 189k)
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x): return self.c(self.f(x))


def get_vec(m): return torch.nn.utils.parameters_to_vector(m.parameters()).detach().cpu().numpy().astype(np.float64)
def set_vec(m, v): torch.nn.utils.vector_to_parameters(torch.as_tensor(v, dtype=torch.float32), m.parameters())


def stamp(x, r=TRIG[0], c=TRIG[1], size=PATCH, val=1.0):
    x = x.clone(); x[:, :, r:r + size, c:c + size] = val; return x


def load_cifar(base):
    base = Path(base)
    if (base / "cifar-10-batches-py").exists(): base = base / "cifar-10-batches-py"
    def rb(p):
        with open(p, "rb") as f: d = pickle.load(f, encoding="bytes")
        return np.asarray(d[b"data"], np.uint8).reshape(-1, 3, 32, 32), np.asarray(d[b"labels"], np.int64)
    parts = [rb(base / f"data_batch_{i}") for i in range(1, 6)]
    Xtr = np.concatenate([p[0] for p in parts]); ytr = np.concatenate([p[1] for p in parts])
    Xte, yte = rb(base / "test_batch")
    t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0)
    return t(Xtr), ytr, t(Xte), yte


def local_train(model, w, X, y, seed, args):
    set_vec(model, w); model.train()
    g = torch.Generator().manual_seed(int(seed))
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    n = len(y); yt = torch.as_tensor(y)
    for _ in range(args.local_epochs):
        perm = torch.randperm(n, generator=g)
        if args.max_client_samples: perm = perm[:args.max_client_samples]
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad(); nn.functional.cross_entropy(model(X[idx]), yt[idx]).backward(); opt.step()
    return get_vec(model)


@torch.inference_mode()
def predict(model, X, bs=1024):
    model.eval(); return torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)]).numpy()


def probe_R(model, w, Xroot, yroot, grid):
    set_vec(model, w); model.eval()      # parameters are set OUTSIDE inference mode; predict() enters it
    N = len(Xroot); big = torch.cat([stamp(Xroot, r, c) for r, c in grid])
    preds = predict(model, big).reshape(len(grid), N)
    return pg.flip_rate_tensor(preds, yroot, 10)


def evaluate(model, w, Xte, yte, Xte_trig):
    set_vec(model, w)
    acc = float((predict(model, Xte) == yte).mean())
    asr = float((predict(model, Xte_trig) == TARGET).mean())
    return acc, asr


def build_task(seed, alpha, args, Xtr, ytr):
    rng = np.random.default_rng(seed)
    root = rng.choice(len(ytr), args.root_size, replace=False)
    rest = np.setdiff1d(np.arange(len(ytr)), root)
    parts = pg.dirichlet_partition(ytr[rest], args.clients, alpha, rng)
    n_mal = int(round(args.mal_frac * args.clients)) if not args.no_attack else 0
    mal = set(rng.choice(args.clients, n_mal, replace=False).tolist()) if n_mal else set()
    clients = []
    for i, p in enumerate(parts):
        idx = rest[p]; X = Xtr[idx]; y = ytr[idx].copy()
        if i in mal:
            cand = np.where(y != TARGET)[0]
            ch = np.random.default_rng(seed * 1009 + i).choice(cand, int(round(args.poison_rate * len(cand))), replace=False)
            X = X.clone(); X[ch] = stamp(X[ch]); y[ch] = TARGET
        clients.append((X, y))
    return root, clients, mal


def run_task(seed, alpha, args, data, writer_rows, round_rows):
    Xtr, ytr, Xte, yte = data
    root, clients, mal = build_task(seed, alpha, args, Xtr, ytr)
    Xroot, yroot = Xtr[root], ytr[root]
    nt = yte != TARGET
    Xte_trig = stamp(Xte[nt]); 
    grid = pg.probe_grid(32, PATCH, args.probe_stride)
    torch.manual_seed(seed); model = CNN(); w0 = get_vec(model)
    mal_mask = np.array([i in mal for i in range(args.clients)])
    f = int(mal_mask.sum())
    for method in args.methods:
        w = w0.copy(); tp = fp = 0; nr = 0
        for rnd in range(args.rounds):
            U = np.stack([local_train(model, w, *clients[i], seed * 1_000_003 + rnd * 101 + i, args) - w
                          for i in range(args.clients)])
            flag = np.zeros(args.clients, bool)
            if method == "fedavg": d = pg.agg_fedavg(U)
            elif method == "coord_median": d = pg.agg_median(U)
            elif method == "trimmed_mean": d = pg.agg_trimmed(U)
            elif method == "norm_clip": d = pg.agg_normclip(U)
            elif method == "multikrum": d = pg.agg_multikrum(U, max(f, 1))
            elif method == "fltrust":
                g0 = local_train(model, w, Xroot, yroot, seed * 1_000_003 + rnd * 101 + 99, args) - w
                d = pg.agg_fltrust(U, g0)
            elif method == "oracle": flag = mal_mask.copy(); d = U[~flag].mean(0)
            elif method in ("mad", "topf"):
                R = np.stack([probe_R(model, w + U[i], Xroot, yroot, grid) for i in range(args.clients)])
                s = pg.client_scores(R)
                flag = pg.mad_flag(s, args.k, args.delta) if method == "mad" else pg.topf_flag(s, f)
                d = U[~flag].mean(0) if (~flag).any() else U.mean(0)
                tp += int((flag & mal_mask).sum()); fp += int((flag & ~mal_mask).sum()); nr += 1
                round_rows.append(dict(seed=seed, alpha=alpha, method=method, round=rnd + 1,
                                       auroc=pg.auroc(s, mal_mask), tpr=float((flag & mal_mask).sum() / max(f, 1)),
                                       fpr=float((flag & ~mal_mask).sum() / max(args.clients - f, 1))))
            else: raise ValueError(method)
            w = w + d
        acc, asr = evaluate(model, w, Xte, yte, Xte_trig)
        writer_rows.append(dict(seed=seed, alpha=alpha, method=method, asr=asr, clean_accuracy=acc,
                                cum_tpr=(tp / max(f * nr, 1) if nr else ""), cum_fpr=(fp / max((args.clients - f) * nr, 1) if nr else ""),
                                rounds=args.rounds, clients=args.clients, malicious=f, root_size=args.root_size,
                                lr=args.lr, batch_size=args.batch_size, protocol="v2"))
        print(f"seed={seed} alpha={alpha} {method:13s} ASR={asr:.3f} acc={acc:.3f}", flush=True)


def append(path, rows):
    if not rows: return
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); 
        if new: w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="experiments/cnn_cifar10/results")
    ap.add_argument("--seeds", default="0:100"); ap.add_argument("--alphas", default="0.5,1,10")
    ap.add_argument("--methods", default=ALL_METHODS)
    ap.add_argument("--clients", type=int, default=20); ap.add_argument("--mal-frac", type=float, default=0.2)
    ap.add_argument("--poison-rate", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=15); ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--root-size", type=int, default=200); ap.add_argument("--probe-stride", type=int, default=4)
    ap.add_argument("--max-client-samples", type=int, default=0)
    ap.add_argument("--k", type=float, default=2.0); ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-attack", action="store_true")
    ap.add_argument("--check-learning", action="store_true",
                    help="benign-only FedAvg on one seed; prints test accuracy per round and exits")
    args = ap.parse_args(); args.methods = args.methods.split(",")
    torch.set_num_threads(args.threads)
    a, b = map(int, args.seeds.split(":")); alphas = [float(x) for x in args.alphas.split(",")]
    si, sn = map(int, args.shard.split("/"))
    data = load_cifar(args.data)
    m = CNN(); print("CNN parameters:", sum(p.numel() for p in m.parameters()))
    if args.check_learning:
        args.no_attack = True; Xtr, ytr, Xte, yte = data
        root, clients, _ = build_task(a, alphas[0], args, Xtr, ytr)
        torch.manual_seed(a); w = get_vec(CNN()); model = CNN()
        for rnd in range(args.rounds):
            U = np.stack([local_train(model, w, *clients[i], a * 1_000_003 + rnd * 101 + i, args) - w for i in range(args.clients)])
            w = w + U.mean(0); set_vec(model, w)
            print(f"round {rnd+1}: clean acc {(predict(model, Xte) == yte).mean():.3f}", flush=True)
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = out / f"raw_shard{si}of{sn}.csv"; rr = out / f"rounds_shard{si}of{sn}.csv"
    done = set()
    if raw.exists():
        with raw.open() as fh: 
            cnt = {}
            for r in csv.DictReader(fh): cnt[(int(r["seed"]), float(r["alpha"]))] = cnt.get((int(r["seed"]), float(r["alpha"])), 0) + 1
            done = {k for k, v in cnt.items() if v >= len(args.methods)}
    tasks = [(s, al) for al in alphas for s in range(a, b)]
    for j, (s, al) in enumerate(tasks):
        if j % sn != si or (s, al) in done: continue
        t0 = time.time(); rows, rounds = [], []
        run_task(s, al, args, data, rows, rounds)
        append(raw, rows); append(rr, rounds)                      # a task is committed only when all its methods finished
        print(f"committed seed={s} alpha={al} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
