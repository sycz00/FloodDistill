"""Unit tests for v10_reference.py. Run: python test_v10_reference.py (CPU, seconds)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v10_reference as v  # noqa: E402

REV = HERE.parent.parent
SPEC = json.loads((REV / 'capacity_v7_20261011T210730Z' / 'training_spec.json').read_text())
sys.path.insert(0, SPEC['capsule'])
sys.path.insert(0, str(REV / 'research_v9_recovery_20261013T075000Z'))


def test_soft_focal_equals_focal_on_hard_labels():
    from training.losses import focal_loss
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 2, 17, 19, generator=g)
    target = torch.randint(0, 2, (3, 17, 19), generator=g)
    target[0, :2] = -1
    ref = focal_loss(logits, target, 2.0, -1)
    got = v.soft_focal_loss(logits, target.clamp_min(0).float(), target != -1, 2.0)
    assert torch.allclose(ref, got, atol=1e-6), (ref, got)


def test_unmixing_recovers_synthetic_fractions():
    rng = np.random.default_rng(1)
    H = W = 64
    water = np.array([.05, .06, .04, .02, .01, .01], np.float32)     # reflectance endmembers
    land = np.array([.08, .10, .12, .30, .25, .15], np.float32)
    truth = np.zeros((H, W), np.float32); truth[:, :30] = 1.0; truth[:, 30] = 0.7; truth[:, 31] = 0.3
    rho = truth[None] * water[:, None, None] + (1 - truth[None]) * land[:, None, None]
    rho += rng.normal(0, 0.001, rho.shape).astype(np.float32)
    x = (rho - v.MEAN[:, None, None]) / v.STD[:, None, None]
    mask = truth >= 0.5
    f, used = v.unmix_fraction(x, mask, k=2, radius=7)
    assert used[:, 30].all() and used[:, 31].all()
    assert np.abs(f[:, 30] - 0.7).max() < 0.05 and np.abs(f[:, 31] - 0.3).max() < 0.05
    assert np.isnan(f[:, 10]).all()                                      # outside the band: no target


def test_edge_distance_and_decode_match_edge_v9():
    import edge_v9 as e
    rng = np.random.default_rng(2)
    pred = ndi_blobs(rng)
    assert np.array_equal(np.isinf(v.edge_distance(pred)), np.isinf(e.edge_distance(pred)))
    assert np.allclose(np.nan_to_num(v.edge_distance(pred), posinf=-1), np.nan_to_num(e.edge_distance(pred), posinf=-1))
    m = rng.uniform(-1, 1, pred.shape).astype(np.float32); av = rng.uniform(size=pred.shape) > 0.1
    assert np.array_equal(v.decode(pred, m, av, 2, 0.1), e.decode(pred, m, av, 2, 0.1))


def test_source_matched_tau():
    m = np.linspace(-5, 5, 10001)
    tau = v.source_matched_tau(m, 0.3)
    assert abs((m > tau).mean() - 0.3) < 1e-3


def test_native_grid_score():
    a = np.repeat(np.repeat(np.random.default_rng(3).uniform(size=(16, 16)), 2, 0), 2, 1)
    assert v.native_grid_score(a) == 1.0
    assert v.native_grid_score(np.random.default_rng(4).uniform(size=(32, 32))) < 0.01


def ndi_blobs(rng):
    from scipy import ndimage
    return ndimage.gaussian_filter(rng.normal(size=(64, 64)), 4) > 0




def test_tiny_models():
    count = lambda m: sum(p.numel() for p in m.parameters())
    stats = (torch.zeros(6), torch.ones(6))
    expect = {'R2': 418, 'R2_log': 514, 'R2_ctx': 610, 'R4_micro': 1266}
    x = torch.randn(2, 6, 64, 64)
    for kind, n in expect.items():
        m = v.build_tiny(kind, *stats)
        assert count(m) == n, (kind, count(m))
        assert m(x).shape == (2, 2, 64, 64)
    # R2 here is the same function class as rungs.R2 (identical parameter shapes)
    import importlib.util
    s = importlib.util.spec_from_file_location('r', SPEC['rungs_dir'] + '/rungs.py'); r = importlib.util.module_from_spec(s); s.loader.exec_module(r)
    assert [p.shape for p in r.build('R2').parameters()] == [p.shape for p in v.build_tiny('R2').parameters()]
    # box context is parameter-free and local means are exact on a constant image
    c = torch.full((1, 6, 40, 40), 3.0)
    assert torch.allclose(v.box_context(c, 33), c)


def test_offline_kd_equals_distill_loss():
    from training.losses import focal_loss, kd_loss
    g = torch.Generator().manual_seed(5)
    s = torch.randn(2, 2, 16, 16, generator=g); t = 3 * torch.randn(2, 2, 16, 16, generator=g)
    m = torch.randint(0, 2, (2, 16, 16), generator=g); m[0, 0] = -1
    ref = 0.5 * focal_loss(s, m, 2.0, -1) + 0.5 * kd_loss(s, t, 4.0, valid_mask=m != -1)
    assert torch.allclose(v.offline_kd_loss(s, m, t), ref, atol=1e-6)
    # shifting both teacher logits by a constant leaves the target unchanged (two-class softmax)
    assert torch.allclose(v.offline_kd_loss(s, m, t + 5.0), ref, atol=1e-5)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('PASS', name)
