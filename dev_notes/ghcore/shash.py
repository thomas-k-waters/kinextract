"""Sinh-arcsinh (Jones & Pewsey 2009) core x penalized spline correction.

    x = (v - V) / sigma
    S = sinh(delta * asinh(x) - eps)
    log b(v) = -S^2/2 + 0.5*log1p(S^2) - 0.5*log1p(x^2) + spline(v; c)

Only `c` is penalized, so the penalty's null space is exactly the
4-parameter core family {V, sigma, eps, delta} (amplitude is absorbed by
the template weights). Heavy regularization collapses to a parametric
SHASH fit; light regularization gives full non-parametric freedom.

**Why SHASH and not Gauss-Hermite.** The GH version of this idea was
built and measured, and fails structurally: in the exponent H4(w) grows
as w^4 and beats the -w^2/2 term, so `b` diverges in the wings for ANY
h4 > 0 (confirmed -- fits drove the core h4 to its NEGATIVE bound on a
truth of h4=+0.05, fleeing the divergence). In linear space the same
family goes negative instead. SHASH has neither failure mode:

  * strictly positive for every parameter value, by construction;
  * both tails always decay -- as |x| -> inf, S^2 ~ |x|^(2 delta), so
    exp(-S^2/2) decays for any delta > 0;
  * reduces exactly to a Gaussian at (eps=0, delta=1), so the current
    order-3 behaviour is contained as a special case;
  * `eps` maps onto skewness (h3) and `delta` onto tail weight /
    kurtosis (h4) monotonically near the Gaussian.

`log1p` forms are used for the normalising terms because
`log cosh(z) = 0.5*log1p(sinh(z)^2)` is stable where `cosh` itself
overflows.

(V, sigma, eps, delta) are NUISANCE parameters, degenerate with the
spline -- only the product is physical. Reported moments still come from
summarizing the recovered curve.
"""
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize as scipy_minimize, nnls


def shash_log_density(v, V, sigma, eps, delta):
    """Unnormalised log SHASH density (numpy, for inspection/plots)."""
    x = (np.asarray(v, float) - V) / sigma
    S = np.sinh(delta * np.arcsinh(x) - eps)
    return -0.5 * S**2 + 0.5 * np.log1p(S**2) - 0.5 * np.log1p(x**2)


def fit_shash_core_spline(A_stack_w, y_w, v_grid, basis, D, lam, ridge=1e-3,
                          n_iter=12, sigma_starts=(60.0, 150.0), v_start=0.0,
                          eps_bound=2.0, log_delta_bound=1.1):
    """Alternating (core+spline, weights) fit.

    Parameters are internally rescaled to O(1): V is in km/s (hundreds)
    while the spline coefficients are O(1), and handed to L-BFGS-B raw the
    core simply does not move -- measured on the GH prototype, where V
    stayed pinned at its start at every lam.
    """
    n_coef = basis.shape[1]
    bj, Dj = jnp.array(basis), jnp.array(D)
    vj = jnp.array(np.asarray(v_grid, float))
    nT = A_stack_w.shape[0]
    V_SCALE = max(1.0, 0.25 * (float(v_grid.max()) - float(v_grid.min())))

    def log_b(p):
        V = p[0] * V_SCALE
        sigma = jnp.exp(p[1])
        eps = p[2]
        delta = jnp.exp(p[3])
        c = p[4:]
        x = (vj - V) / sigma
        S = jnp.sinh(delta * jnp.arcsinh(x) - eps)
        return -0.5 * S**2 + 0.5 * jnp.log1p(S**2) - 0.5 * jnp.log1p(x**2) + bj @ c

    def make_vg(A_eff):
        Aj, yj = jnp.array(A_eff), jnp.array(y_w)
        def obj(p):
            b = jnp.exp(log_b(p))
            r = Aj @ b - yj
            c = p[4:]
            # `ridge` closes the curvature penalty's own null space so that
            # lam -> inf drives c -> 0 exactly; a curvature term alone would
            # leave the spline free to re-add a quadratic, degenerate with sigma.
            pen = lam * (jnp.sum((Dj @ c) ** 2) + ridge * jnp.sum(c ** 2))
            return 0.5 * jnp.sum(r**2) + 0.5 * pen
        return jax.jit(jax.value_and_grad(obj))

    bounds = ([(float(v_grid.min()) / V_SCALE, float(v_grid.max()) / V_SCALE),
               (np.log(5.0), np.log(600.0)),
               (-eps_bound, eps_bound),
               (-log_delta_bound, log_delta_bound)]
              + [(None, None)] * n_coef)

    best = None
    for s0 in sigma_starts:
        p = np.concatenate([[v_start / V_SCALE, np.log(s0), 0.0, 0.0], np.zeros(n_coef)])
        wl = np.ones(nT)
        res = None
        for _ in range(n_iter):
            A_eff = np.tensordot(wl, A_stack_w, axes=([0], [0]))
            vg_fn = make_vg(A_eff)
            f = lambda z: (lambda t: (float(t[0]), np.array(t[1])))(vg_fn(jnp.array(z)))
            res = scipy_minimize(f, p, jac=True, method="L-BFGS-B", bounds=bounds)
            p = res.x
            b = np.exp(np.asarray(log_b(jnp.array(p))))
            design = np.stack([A_stack_w[k] @ b for k in range(nT)], axis=1)
            wl, _ = nnls(design, y_w)
            if not np.any(wl > 0):
                break
        if best is None or res.fun < best[0]:
            best = (float(res.fun), p.copy(), wl.copy())

    obj, p, w = best
    b = np.exp(np.asarray(log_b(jnp.array(p))))
    return dict(b=b, w=w, objective=obj, coeffs=p[4:],
                core=dict(V=float(p[0] * V_SCALE), sigma=float(np.exp(p[1])),
                          eps=float(p[2]), delta=float(np.exp(p[3]))))
