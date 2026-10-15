"""NumPy-only ProbeGuard building blocks (score, decision rules, aggregators, partitioning).

Kept free of torch so the logic can be unit-tested without a deep-learning stack
(see tests/test_pg_core.py). The CNN runner (run_cnn.py) imports everything from here.
"""
import numpy as np


# ---------------------------------------------------------------- probe geometry
def probe_grid(size, patch, stride):
    """Top-left corners of all patch x patch windows on a stride grid."""
    pos = list(range(0, size - patch + 1, stride))
    return [(r, c) for r in pos for c in pos]


# ---------------------------------------------------------------- score
def flip_rate_tensor(preds, y, n_classes=10):
    """R[p, c] = #{x: y!=c, pred_p(x)=c} / #{x: y!=c}.

    preds: int array [P, N], predictions of one candidate model on root images stamped with
    each of the P probe patches. y: true labels [N].
    """
    P, N = preds.shape
    R = np.zeros((P, n_classes))
    for c in range(n_classes):
        m = y != c
        denom = max(int(m.sum()), 1)
        R[:, c] = (preds[:, m] == c).sum(1) / denom
    return R


def client_scores(R_all):
    """s_i = max_{p,c} (R_i[p,c] - median_j R_j[p,c]).  R_all: [n_clients, P, C]."""
    ref = np.median(R_all, axis=0)
    return (R_all - ref[None]).reshape(len(R_all), -1).max(1)


# ---------------------------------------------------------------- decision rules
def mad_flag(s, k=2.0, delta=0.1):
    s = np.asarray(s, float)
    med = np.median(s); mad = np.median(np.abs(s - med))
    flag = s > med + max(k * mad, delta)
    if flag.all():            # paper: if every client is flagged, none is
        flag[:] = False
    return flag


def topf_flag(s, f):
    s = np.asarray(s, float); flag = np.zeros(len(s), bool)
    if f > 0:
        flag[np.argsort(s, kind="stable")[-f:]] = True
    return flag


def auroc(scores, labels):
    """P(score_malicious > score_benign) + 0.5 P(tie); labels 1 = malicious."""
    s = np.asarray(scores, float); l = np.asarray(labels).astype(bool)
    a, b = s[l], s[~l]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(((a[:, None] > b[None]) + 0.5 * (a[:, None] == b[None])).mean())


# ---------------------------------------------------------------- aggregators (rows of U = client updates)
def agg_fedavg(U):
    return U.mean(0)


def agg_median(U):
    return np.median(U, axis=0)


def agg_trimmed(U, beta=0.2):
    n = len(U); t = max(1, int(round(beta * n)))
    return np.sort(U, axis=0)[t:n - t].mean(0)


def agg_normclip(U):
    norms = np.linalg.norm(U, axis=1); tau = np.median(norms)
    return (U * np.minimum(1.0, tau / (norms + 1e-12))[:, None]).mean(0)


def agg_multikrum(U, f):
    n = len(U); d2 = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    m = max(n - f - 2, 1)
    sc = np.array([np.sort(d2[i])[1:m + 1].sum() for i in range(n)])
    return U[np.argsort(sc)[:max(n - f, 1)]].mean(0)


def agg_fltrust(U, g0):
    """FLTrust. g0 MUST be the server's *update* (w_after_local_training - w), i.e. the same
    sign convention as client updates. (The v1 CNN script passed a raw gradient, whose cosine
    with descent-direction updates is negative -> all trust scores 0 -> a frozen model.)"""
    n0 = np.linalg.norm(g0) + 1e-12
    cs = (U @ g0) / (np.linalg.norm(U, axis=1) * n0 + 1e-12)
    ts = np.maximum(cs, 0)
    if ts.sum() < 1e-12:
        return np.zeros_like(g0)
    Un = U * (n0 / (np.linalg.norm(U, axis=1) + 1e-12))[:, None]
    return (ts[:, None] * Un).sum(0) / ts.sum()


# ---------------------------------------------------------------- data partition
def dirichlet_partition(y, n_clients, alpha, rng, min_size=10):
    """Per-class Dirichlet(alpha) label-skew split; resamples until every client has >= min_size."""
    classes = np.unique(y)
    for _ in range(200):
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            idx = np.where(y == c)[0]; rng.shuffle(idx)
            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for i, part in enumerate(np.split(idx, cuts)):
                parts[i].extend(part.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(p, dtype=int) for p in parts]
    raise RuntimeError("could not build a partition with the requested minimum client size")
