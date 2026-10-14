"""Frozen affine-margin adaptation. No model/data side effects on import."""
import time
import numpy as np


def reliability(p, perturbed):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1-1e-4)
    entropy = -(p*np.log2(p)+(1-p)*np.log2(1-p))
    return .1+.9*(1-entropy)*np.exp(-np.abs(p-perturbed)/.1)


def target(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1-1e-4)
    return np.clip(np.log(p/(1-p)), -8, 8)


def design(features, mean, scale):
    f = (np.asarray(features, dtype=np.float64)-mean)/scale
    return np.column_stack([f, np.ones(len(f))])


def solve(phi, residual, weights=None, difference=None, lam=.01, gamma=0, mass=None):
    """Float64 sufficient statistics and Cholesky; intercept is regularized too."""
    began = time.perf_counter()
    phi = np.asarray(phi, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    assert lam > 0 and len(phi) > 0
    assert np.isfinite(phi).all() and np.isfinite(residual).all()
    alpha = np.full(len(phi), 1/len(phi)) if mass is None else np.asarray(mass, dtype=np.float64)
    alpha = alpha/alpha.sum()
    w = alpha if weights is None else alpha*np.asarray(weights, dtype=np.float64)
    assert np.isfinite(w).all() and np.all(w > 0)
    w = w/w.sum()
    d = np.zeros_like(phi) if difference is None else np.asarray(difference, dtype=np.float64)
    a = phi.T@(w[:, None]*phi)+lam*np.eye(phi.shape[1])+gamma*d.T@(alpha[:, None]*d)
    b = phi.T@(w*residual)
    chol = np.linalg.cholesky(a)
    delta = np.linalg.solve(chol.T, np.linalg.solve(chol, b))
    gradient = 2*(a@delta-b)
    rms = np.sqrt(np.sum(alpha*(phi@delta)**2))
    eta = min(1., 1/max(float(rms), 1e-12))
    return eta*delta, dict(undamped_delta=delta.tolist(), eta=eta, undamped_RMS=float(rms),
        objective_gradient_max=float(np.max(np.abs(gradient))), condition=float(np.linalg.cond(a)),
        solve_seconds=time.perf_counter()-began, samples=len(phi), lambda_=lam, gamma=gamma)


def unstandardize(delta, mean, scale):
    dv = np.asarray(delta[:-1])/scale
    return np.r_[dv, delta[-1]-mean@dv]


def sgd(phi, base_margin, teacher_probability, steps, lr, temperature=4, lam=.01):
    import torch
    began = time.perf_counter()
    x = torch.as_tensor(phi, dtype=torch.float64)
    base = torch.as_tensor(base_margin, dtype=torch.float64)
    p = torch.as_tensor(teacher_probability, dtype=torch.float64)
    delta = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([delta], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        z = (base+x@delta)/temperature
        loss = temperature**2*(torch.nn.functional.softplus(z)-p*z).mean()+lam*delta.square().sum()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(delta.grad).all()
        opt.step()
    raw=delta.detach().numpy()
    rms=float(np.sqrt(np.mean((np.asarray(phi)@raw)**2)))
    eta=min(1.,1/max(rms,1e-12))
    return eta*raw, dict(solve_seconds=time.perf_counter()-began, steps=steps,
        lr=lr, temperature=temperature, loss=float(loss.detach()), finite_gradients=True,
        undamped_delta=raw.tolist(),undamped_RMS=rms,eta=eta)


def sample_indices(valid, seed, maximum=4096):
    """One spatially stratified draw: one valid point per 8x8 cell, capped 4096."""
    valid = np.asarray(valid, dtype=bool)
    rng = np.random.default_rng(seed)
    h, w = valid.shape
    points = []
    for row in range(0, h, 8):
        for col in range(0, w, 8):
            ys, xs = np.where(valid[row:row+8, col:col+8])
            if len(xs):
                j = rng.integers(len(xs)); points.append((row+int(ys[j]))*w+col+int(xs[j]))
    if len(points) > maximum:
        points = np.asarray(points)[np.sort(rng.choice(len(points), maximum, replace=False))].tolist()
    assert points and np.all(valid.reshape(-1)[points])
    return np.asarray(points, dtype=np.int64)


def focused_checks():
    rng = np.random.default_rng(1729)
    f = rng.normal(size=(256,16)); mean = rng.normal(size=16); scale = rng.uniform(.1, 2, 16)
    x = design(f,mean,scale); r = rng.normal(size=256)
    d, info = solve(x,r)
    d0, _ = solve(x,r,difference=np.zeros_like(x),gamma=1)
    raw = unstandardize(d,mean,scale)
    error = float(np.max(np.abs(x@d-(f@raw[:-1]+raw[-1]))))
    assert error < 1e-12 and info['objective_gradient_max'] < 1e-10 and np.allclose(d,d0)
    assert np.sqrt(np.mean((x@d)**2)) <= 1+1e-12
    mask = np.ones((32,32),bool);mask[:8,:8]=False
    ids = sample_indices(mask,42); assert mask.reshape(-1)[ids].all()
    for bits in (4,8):
        lo,hi = -(2**(bits-1)),2**(bits-1)-1
        z = np.clip(np.rint(np.array([-1e3,1e3])),lo,hi)
        assert z.tolist()==[lo,hi]
    return dict(outcome='passed',fold_error=error,gradient_max=info['objective_gradient_max'],
        gamma_zero_distortion_parity=True,invalid_pixel_exclusion=True,orientation='water minus land',
        test_seed=1729)
