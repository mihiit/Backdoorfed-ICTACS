import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments" / "cnn_cifar10"))
import pg_core as pg

def test_median_containment_lemma():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n, m = 20, 4
        R = rng.random((n, 5, 3)); R[:m] += rng.random((m, 5, 3)) * 5     # arbitrary malicious values
        med = np.median(R, axis=0)
        assert np.all(med >= R[m:].min(0) - 1e-12) and np.all(med <= R[m:].max(0) + 1e-12)

def test_flip_rate_tensor_by_hand():
    y = np.array([0, 1, 2, 3]); preds = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])   # 2 probes, 4 classes
    R = pg.flip_rate_tensor(preds, y, n_classes=4)
    assert R.shape == (2, 4)
    assert R[0, 1] == 1.0          # y!=1 are {0,2,3}; all three are predicted 1 under patch 0
    assert R[1, 0] == 1.0          # y!=0 are {1,2,3}; all three are predicted 0 under patch 1
    assert R[0, 0] == 0.0 and R[1, 1] == 0.0

def test_scores_flag_planted_outlier():
    rng = np.random.default_rng(1)
    R = rng.random((20, 9, 10)) * 0.1; R[3, 4, 0] = 0.95
    s = pg.client_scores(R); assert s.argmax() == 3
    assert pg.mad_flag(s)[3]

def test_mad_flags_nobody_when_scores_identical_or_all_flagged():
    assert not pg.mad_flag(np.zeros(6)).any()
    s = np.array([1., 2., 3., 4.])
    assert not pg.mad_flag(s, k=-10, delta=-10).any()      # threshold below every score -> all flagged -> none

def test_topf_count_and_order():
    s = np.array([.1, .9, .3, .8, .2]); fl = pg.topf_flag(s, 2)
    assert fl.sum() == 2 and fl[1] and fl[3]
    assert pg.topf_flag(s, 0).sum() == 0

def test_auroc_known_values():
    assert pg.auroc([.9, .8, .1, .2], [1, 1, 0, 0]) == 1.0
    assert pg.auroc([.1, .1, .1, .1], [1, 1, 0, 0]) == 0.5

def test_fltrust_descent_convention_is_not_zero():
    rng = np.random.default_rng(2); g = rng.standard_normal(500)
    U = np.stack([-0.01 * g + 1e-3 * rng.standard_normal(500) for _ in range(20)])
    g0_update = -0.01 * g                      # server update, same convention as clients
    out = pg.agg_fltrust(U, g0_update)
    assert np.linalg.norm(out) > 0 and (out @ g0_update) > 0
    out_bad = pg.agg_fltrust(U, g)             # raw gradient: documents the v1 bug
    assert np.linalg.norm(out_bad) == 0

def test_aggregators_resist_outliers():
    rng = np.random.default_rng(3); U = rng.standard_normal((20, 50)) * 0.01; U[:4] += 10
    for fn in (pg.agg_median, pg.agg_trimmed, pg.agg_normclip):
        assert np.abs(fn(U)).max() < np.abs(pg.agg_fedavg(U)).max()
    assert np.abs(pg.agg_multikrum(U, 4)).max() < 1.0

def test_partition_covers_everything_once():
    y = np.repeat(np.arange(10), 100); rng = np.random.default_rng(4)
    parts = pg.dirichlet_partition(y, 20, 0.5, rng)
    allidx = np.concatenate(parts); assert len(allidx) == len(y) and len(set(allidx.tolist())) == len(y)

def test_probe_grid_contains_trigger_cell():
    assert (28, 28) in pg.probe_grid(32, 4, 4) and len(pg.probe_grid(32, 4, 4)) == 64
    assert len(pg.probe_grid(8, 2, 1)) == 49
