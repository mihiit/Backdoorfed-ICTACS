"""Federated backdoor study: ProbeGuard (trigger-bank behavioural probing) vs. baselines.
Pure NumPy, deterministic given seed. Data: scikit-learn digits (8x8, 1797 images)."""
import numpy as np, json, sys, time
from sklearn.datasets import load_digits

D, H, C = 64, 32, 10
TARGET = 0
N_CLIENTS = 20
ROUNDS = 30
LOCAL_EP = 2
LR = 0.1
BS = 32

# ---------------- model ----------------
def init_params(rng):
    return np.concatenate([
        (rng.standard_normal((D, H)) * np.sqrt(2 / D)).ravel(), np.zeros(H),
        (rng.standard_normal((H, C)) * np.sqrt(2 / H)).ravel(), np.zeros(C)])

def unpack(w):
    i = 0
    W1 = w[i:i + D * H].reshape(D, H); i += D * H
    b1 = w[i:i + H]; i += H
    W2 = w[i:i + H * C].reshape(H, C); i += H * C
    b2 = w[i:i + C]
    return W1, b1, W2, b2

def forward(w, X, hidden=False):
    W1, b1, W2, b2 = unpack(w)
    h = np.maximum(X @ W1 + b1, 0)
    z = h @ W2 + b2
    return (z, h) if hidden else z

def predict(w, X):
    return forward(w, X).argmax(1)

def local_train(w, X, y, rng, epochs=LOCAL_EP, lr=LR):
    w = w.copy()
    W1, b1, W2, b2 = [a.copy() for a in unpack(w)]
    n = len(X)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, BS):
            b = idx[s:s + BS]
            xb, yb = X[b], y[b]
            h = np.maximum(xb @ W1 + b1, 0)
            z = h @ W2 + b2
            z -= z.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            p[np.arange(len(b)), yb] -= 1
            p /= len(b)
            gW2 = h.T @ p; gb2 = p.sum(0)
            dh = (p @ W2.T) * (h > 0)
            gW1 = xb.T @ dh; gb1 = dh.sum(0)
            W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

def estimate_worst_cell(w, X_proxy, y_proxy, target, stealth_positions):
    """Attacker-side estimate of which non-trigger probe-bank cell currently has
    the highest flip-rate-to-target, using its own local data as a root-set
    proxy (full white-box score knowledge, but no access to the real root set)."""
    best_pos, best_rate = stealth_positions[0], -1.0
    mask = y_proxy != target
    if not mask.any():
        return best_pos
    for pos in stealth_positions:
        Xp = stamp(X_proxy[mask], pos, 1.0)
        pred = predict(w, Xp)
        rate = (pred == target).mean()
        if rate > best_rate:
            best_rate, best_pos = rate, pos
    return best_pos

def local_train_adaptive(w, X, y, y_true_for_stealth, rng, stealth_positions, beta,
                          epochs=LOCAL_EP, lr=LR, targeted=False):
    """Probe-aware adaptive malicious training. Each batch combines two losses:
    (1) the standard backdoor CE loss on (X,y) [y already carries poisoned labels
        where triggered], and
    (2) a stealth CE loss that pushes bank-patched CLEAN copies of the same batch
        (patched at bank positions OTHER than the true trigger cell) back toward
        their TRUE label y_true_for_stealth, directly penalising the behavioural
        score the attacker is assumed to know. beta weights (2) relative to (1).
    stealth_positions: list of (r,c) bank cells excluding the true trigger cell."""
    w0 = w.copy()
    W1, b1, W2, b2 = [a.copy() for a in unpack(w0)]
    n = len(X)
    size = 2
    fixed_target_pos = None
    if targeted:
        cur_w = np.concatenate([W1.ravel(), b1, W2.ravel(), b2])
        fixed_target_pos = estimate_worst_cell(cur_w, X, y_true_for_stealth, TARGET, stealth_positions)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, BS):
            b = idx[s:s + BS]
            xb, yb = X[b], y[b]
            # backdoor gradient
            h = np.maximum(xb @ W1 + b1, 0)
            z = h @ W2 + b2
            z -= z.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            p[np.arange(len(b)), yb] -= 1
            p /= len(b)
            gW2 = h.T @ p; gb2 = p.sum(0)
            dh = (p @ W2.T) * (h > 0)
            gW1 = xb.T @ dh; gb1 = dh.sum(0)
            # stealth gradient: patch the (unpoisoned-label) batch at a bank
            # cell, push toward TRUE label. targeted=True re-estimates and
            # attacks the single highest-flip-rate non-trigger cell each
            # round (an explicit, optimization-driven evasion); targeted=False
            # (the original variant) samples a random non-trigger cell each
            # batch, a diffuse/non-optimized stealth strategy.
            pos = fixed_target_pos if targeted else stealth_positions[rng.integers(0, len(stealth_positions))]
            xs = xb.copy().reshape(-1, 8, 8)
            r, c = pos
            xs[:, r:r + size, c:c + size] = 1.0
            xs = xs.reshape(-1, D)
            ys = y_true_for_stealth[b]
            hs = np.maximum(xs @ W1 + b1, 0)
            zs = hs @ W2 + b2
            zs -= zs.max(1, keepdims=True)
            ps = np.exp(zs); ps /= ps.sum(1, keepdims=True)
            ps[np.arange(len(b)), ys] -= 1
            ps /= len(b)
            sW2 = hs.T @ ps; sb2 = ps.sum(0)
            sh = (ps @ W2.T) * (hs > 0)
            sW1 = xs.T @ sh; sb1 = sh.sum(0)
            W1 -= lr * (gW1 + beta * sW1); b1 -= lr * (gb1 + beta * sb1)
            W2 -= lr * (gW2 + beta * sW2); b2 -= lr * (gb2 + beta * sb2)
    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

# ---------------- data / attack ----------------
def load_data(seed, root_size=100):
    d = load_digits()
    X = d.data / 16.0
    y = d.target
    rng = np.random.default_rng(1000 + seed)
    idx = rng.permutation(len(X))
    root, rest = idx[:root_size], idx[root_size:]
    test, train = rest[:497], rest[497:]
    return X[train], y[train], X[root], y[root], X[test], y[test]

def dirichlet_split(y, alpha, rng):
    parts = [[] for _ in range(N_CLIENTS)]
    for c in range(C):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        pr = rng.dirichlet(alpha * np.ones(N_CLIENTS))
        cuts = (np.cumsum(pr) * len(ids)).astype(int)[:-1]
        for k, part in enumerate(np.split(ids, cuts)):
            parts[k] += part.tolist()
    return [np.array(p, dtype=int) for p in parts]

CORNERS = [(0, 0), (0, 6), (6, 0), (6, 6)]

def stamp(X, pos, val, size=2):
    X = X.copy().reshape(-1, 8, 8)
    r, c = pos
    X[:, r:r + size, c:c + size] = val
    return X.reshape(-1, D)

def make_trigger_fn(attack, rng):
    """returns f(X) -> triggered X. attack in fixed | multi | lowint | large."""
    if attack == "fixed":
        return lambda X: stamp(X, (6, 6), 1.0)
    if attack == "lowint":
        return lambda X: stamp(X, (6, 6), 0.5)
    if attack == "large":   # 3x3 patch, moderate intensity
        return lambda X: stamp(X, (5, 5), 0.75, size=3)
    if attack == "multi":   # randomly one of four corner locations per sample
        def f(X):
            out = X.copy()
            ch = rng.integers(0, 4, len(X))
            for k in range(4):
                m = ch == k
                if m.any():
                    out[m] = stamp(X[m], CORNERS[k], 1.0)
            return out
        return f
    raise ValueError(attack)

# ---------------- ProbeGuard ----------------
def build_bank(vals, size=2, stride=None):
    """All size x size patches at intensities `vals`: returns list of (pos,val)."""
    stride = stride or 1
    return [((r, c), v) for v in vals for r in range(0, 8 - size + 1, stride) for c in range(0, 8 - size + 1, stride)]

class Probe:
    def __init__(self, Xroot, yroot, vals, stride=1):
        bank = build_bank(vals, stride=stride)
        self.Xp = np.concatenate([stamp(Xroot, p, v) for p, v in bank])   # (P*N, D)
        self.P, self.N = len(bank), len(Xroot)
        self.y = yroot
        self.Xroot = Xroot
        self.nonc = np.stack([(yroot != c) for c in range(C)])            # (C,N)

    def rates(self, w):
        pr = predict(w, self.Xp).reshape(self.P, self.N)
        R = np.zeros((self.P, C))
        for c in range(C):
            m = self.nonc[c]
            R[:, c] = ((pr == c) & m).sum(1) / m.sum()
        return R

    def base(self, w):
        base = predict(w, self.Xroot)
        b = np.zeros(C)
        for c in range(C):
            m = self.nonc[c]
            b[c] = ((base == c) & m).sum() / m.sum()
        return b

    def score(self, w):
        pr = predict(w, self.Xp).reshape(self.P, self.N)
        base = predict(w, self.Xroot)
        best = 0.0
        for c in range(C):
            m = self.nonc[c]
            rate = ((pr == c) & m).sum(1) / m.sum()
            b = ((base == c) & m).sum() / m.sum()
            best = max(best, rate.max() - b)
        return best

def mad_flag(s, k, delta):
    s = np.asarray(s)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return s > med + max(k * mad, delta)

def pg_scores(probe, w, U, mode, K=6):
    Rs = np.array([probe.rates(w + u) for u in U])          # (n,P,C)
    B = np.array([probe.base(w + u) for u in U])            # (n,C)
    n = len(U)
    if mode in ("probeguard", "probeguard+"):                # cross-client median reference
        return (Rs - np.median(Rs, 0)).reshape(n, -1).max(1)
    if mode == "pg_abs":                                     # own no-patch baseline only
        return (Rs - B[:, None, :]).reshape(n, -1).max(1)
    if mode == "pg_local":                                   # spatial localisation, self-contained
        return (Rs.max(1) - np.median(Rs, 1)).max(1)
    if mode == "pg_knn":                                     # bias-matched neighbour reference
        dist = ((B[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        out = np.zeros(n)
        for i in range(n):
            nb = np.argsort(dist[i])[1:K + 1]
            out[i] = (Rs[i] - np.median(Rs[nb], 0)).max()
        return out
    raise ValueError(mode)

# ---------------- aggregators ----------------
def agg_fedavg(U, **kw):
    return U.mean(0)

def agg_median(U, **kw):
    return np.median(U, 0)

def agg_trimmed(U, beta=0.2, **kw):
    n = len(U); k = int(beta * n)
    S = np.sort(U, 0)
    return S[k:n - k].mean(0)

def agg_normclip(U, **kw):
    nrm = np.linalg.norm(U, axis=1)
    tau = np.median(nrm)
    return (U * np.minimum(1, tau / (nrm + 1e-12))[:, None]).mean(0)

def agg_multikrum(U, f, **kw):
    n = len(U)
    d2 = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    m = max(n - f - 2, 1)
    sc = np.array([np.sort(d2[i])[1:m + 1].sum() for i in range(n)])
    sel = np.argsort(sc)[:max(n - f, 1)]
    return U[sel].mean(0)

def agg_fltrust(U, g0, **kw):
    n0 = np.linalg.norm(g0) + 1e-12
    cs = (U @ g0) / (np.linalg.norm(U, axis=1) * n0 + 1e-12)
    ts = np.maximum(cs, 0)
    if ts.sum() < 1e-12:
        return np.zeros_like(g0)
    Un = U * (n0 / (np.linalg.norm(U, axis=1) + 1e-12))[:, None]
    return (ts[:, None] * Un).sum(0) / ts.sum()

# ---------------- one FL run ----------------
def run(alpha, m_frac, attack, defense, seed, k=3.5, delta=0.2, poison=0.5, boost=1.0,
        root_size=100, probe_stride=1, trajectory=False, adaptive=False, beta=1.0,
        f_assumed=None, targeted=False):
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xr, yr, Xte, yte = load_data(seed, root_size=root_size)
    parts = dirichlet_split(ytr, alpha, rng)
    n_mal = int(round(m_frac * N_CLIENTS))
    mal = set(rng.choice(N_CLIENTS, n_mal, replace=False).tolist()) if n_mal else set()
    trig = make_trigger_fn(attack, np.random.default_rng(seed + 7))
    tmask = yte != TARGET
    Xte_t = trig(Xte[tmask])
    probe = None
    if defense in ("probeguard", "probeguard_k", "pg_abs", "pg_local", "pg_knn",
                    "pg_pct90", "pg_pct95", "pg_pct975", "pg_oracle"):
        probe = Probe(Xr, yr, [1.0], stride=probe_stride)
    elif defense == "probeguard+":
        probe = Probe(Xr, yr, [0.5, 1.0], stride=probe_stride)
    w = init_params(rng)
    tp = fp = fn = tn = 0
    traj = []
    for r in range(ROUNDS):
        U, flags_true, scores = [], [], []
        for i in range(N_CLIENTS):
            idx = parts[i]
            if len(idx) == 0:
                U.append(np.zeros_like(w)); flags_true.append(i in mal); continue
            X, y = Xtr[idx], ytr[idx]
            if i in mal:
                X, y = X.copy(), y.copy()
                y_true = y.copy()
                cand = np.where(y != TARGET)[0]
                pk = rng.choice(cand, int(poison * len(cand)), replace=False) if len(cand) else []
                if len(pk):
                    X[pk] = trig(X[pk]); y[pk] = TARGET
                if adaptive:
                    true_pos = (6, 6)  # fixed trigger's own bank cell -- excluded from stealth set
                    stealth_positions = [(rr, cc) for rr in range(7) for cc in range(7)
                                         if (rr, cc) != true_pos]
                    wi = local_train_adaptive(w, X, y, y_true, rng, stealth_positions, beta, targeted=targeted)
                else:
                    wi = local_train(w, X, y, rng)
            else:
                wi = local_train(w, X, y, rng)
            u = (wi - w) * (boost if i in mal else 1.0)
            U.append(u); flags_true.append(i in mal)
        U = np.array(U); ft = np.array(flags_true)
        f = n_mal if f_assumed is None else f_assumed
        if defense == "fedavg" or defense == "clean_oracle":
            upd = agg_fedavg(U) if defense == "fedavg" else U[~ft].mean(0)
        elif defense == "median": upd = agg_median(U)
        elif defense == "trimmed": upd = agg_trimmed(U)
        elif defense == "normclip": upd = agg_normclip(U)
        elif defense == "multikrum": upd = agg_multikrum(U, f=max(f, 1))
        elif defense == "fltrust":
            g0 = local_train(w, Xr, yr, rng) - w
            upd = agg_fltrust(U, g0)
        elif defense in ("probeguard", "probeguard+", "probeguard_k", "pg_abs", "pg_local", "pg_knn",
                          "pg_pct90", "pg_pct95", "pg_pct975", "pg_oracle"):
            score_mode = "probeguard" if defense in ("probeguard_k", "pg_pct90", "pg_pct95", "pg_pct975", "pg_oracle") else defense
            s = pg_scores(probe, w, U, score_mode)
            if defense == "probeguard_k":          # rank-based rule: drop the f highest scores
                f_used = n_mal if f_assumed is None else f_assumed
                fl = np.zeros(len(s), bool)
                if f_used > 0: fl[np.argsort(-s)[:f_used]] = True
            elif defense == "pg_pct90":
                fl = s > np.percentile(s, 90)
            elif defense == "pg_pct95":
                fl = s > np.percentile(s, 95)
            elif defense == "pg_pct975":
                fl = s > np.percentile(s, 97.5)
            elif defense == "pg_oracle":
                fl = ft.copy()   # upper-bound reference only: uses ground-truth labels, not a deployable rule
            else:
                fl = mad_flag(s, k, delta)
            if fl.all(): fl[:] = False
            upd = U[~fl].mean(0)
            tp += int((fl & ft).sum()); fp += int((fl & ~ft).sum())
            fn += int((~fl & ft).sum()); tn += int((~fl & ~ft).sum())
        else:
            raise ValueError(defense)
        w = w + upd
        if trajectory:
            r_acc = float((predict(w, Xte) == yte).mean())
            r_asr = float((predict(w, Xte_t) == TARGET).mean())
            traj.append((r, r_acc, r_asr))
    acc = float((predict(w, Xte) == yte).mean())
    asr = float((predict(w, Xte_t) == TARGET).mean())
    out = dict(alpha=alpha, m=m_frac, attack=attack, defense=defense, seed=seed,
               acc=acc, asr=asr)
    if trajectory:
        out["traj"] = traj
    if defense in ("probeguard", "probeguard+", "probeguard_k", "pg_abs", "pg_local", "pg_knn",
                    "pg_pct90", "pg_pct95", "pg_pct975", "pg_oracle"):
        out.update(tpr=tp / max(tp + fn, 1) if (tp + fn) else None,
                   fpr=fp / max(fp + tn, 1) if (fp + tn) else None)
    return out

if __name__ == "__main__":
    t = time.time()
    for d in ["fedavg", "probeguard", "multikrum", "fltrust"]:
        print(run(0.5, 0.2, "fixed", d, 0), round(time.time() - t, 1))
