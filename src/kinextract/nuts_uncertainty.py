"""NUTS (via NumPyro) -- an opt-in, slow "gold standard" reference sampler
over the same posterior `laplace_uncertainty.py` linearizes, at a single
fixed `lam`. Two purposes (`dev_notes/V1_HARDENING_ROADMAP.md` Phase 1c):

1. **Cross-checking the Laplace approximation.** The Laplace engine's own
   coverage sweep (`dev_notes/uncertainty/FINDINGS.md`, 2026-08-01) found
   h4 coverage still short of the 68% target, especially at low sigma
   (12-25%). NUTS samples the *actual* posterior (no Gaussian
   linearization), so comparing its `sherm_std`/`h4_std` against
   `laplace_losvd_uncertainty`'s on the same case, same `lam`, is a
   direct way to see whether the residual gap is the Laplace
   approximation's own limit (Gaussian curvature is a poor local
   description of a skewed/heavy-tailed posterior) or something upstream
   of both engines (e.g. the fixed-`w` simplification they share).
2. **An opt-in slow mode for end users** who want a non-Gaussian
   posterior check on a specific bin, not a routine per-bin default.

**Same scope limits as the Laplace engine, same reason (`laplace_uncertainty`'s
own "Honest scope" section).** Template weights `w` are held fixed at
their converged (MAP) value, not sampled -- NNLS-constrained, non-Gaussian,
not addressed by either uncertainty engine yet. `cfg.continuum_knots`
must be `None` (pass `dataclasses.replace(cfg, continuum_knots=None)` and
inputs that need no continuum). This module validates the Laplace engine's
*coefficient-space* approximation specifically; it is not meant to
leapfrog it into modeling everything neither engine addresses yet.

**Reuses the existing objective directly as the log-density -- no new
forward-model code.** `losvd._neg_log_post` already *is* the negative
log-posterior (chi2 + curvature-penalty-as-Gaussian-prior,
Eilers & Marx 1996's own framing) that `fit_log_spline_map` minimizes and
that `laplace_covariance_at_lam` takes the Hessian of. Passed to NumPyro
via `numpyro.factor` against an improper flat prior on `coeffs`
(`dist.ImproperUniform`) -- the standard NumPyro pattern for sampling
directly from an existing potential function rather than reformulating
the model in NumPyro's own generative-model style.

**No divergent-edge rejection filter, unlike `_sample_curves_and_moments`.**
The Laplace engine needs one because it samples from a *Gaussian*
approximation that assigns nonzero density to the curvature penalty's
divergent null-space directions even where the true posterior doesn't.
NUTS samples the true (penalty-inclusive) posterior directly, which
already assigns those directions the same low density the point-estimate
optimizer implicitly avoids -- so no post-hoc filtering is needed here.
A large `n_divergences` in the returned dict would indicate NUTS itself
struggling with that geometry; treat it as a diagnostic, not something to
routinely filter around.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

import dataclasses

from .config import PENALTY_ORDER, FitConfig
from .gh_moments import compute_losvd_moments
from .losvd import _neg_log_post, build_bspline_basis, difference_matrix


def _model(basis, D, A_eff_w, y_w, lam_vec, n_coef):
    coeffs = numpyro.sample(
        "coeffs", dist.ImproperUniform(dist.constraints.real, (), event_shape=(n_coef,))
    )
    numpyro.factor("neg_log_post", -_neg_log_post(coeffs, basis, D, A_eff_w, y_w, lam_vec))


def nuts_losvd_uncertainty(A_stack_w: np.ndarray, y_w: np.ndarray, cfg: FitConfig,
                            lam: float | None = None, n_samples: int = 1000, n_warmup: int = 500,
                            seed: int = 0, fit_h3h4: bool = True,
                            target_accept_prob: float = 0.9) -> dict:
    """Full NUTS sampling of the LOSVD spline coefficients at a single
    fixed `lam` -- the slow reference counterpart to
    `laplace_losvd_uncertainty`.

    `lam` : `None` (default) samples at `cfg.lam`; pass the point
        estimate's `lam_used`.

    **Slow: budget minutes, not seconds**, even for a modest `n_samples`
    -- this is the opt-in "gold standard", not a per-bin-loop default.

    Returns a dict shaped like `laplace_losvd_uncertainty`'s (`result0`,
    `point0`, `lam_used`, `b_samples`, `b_median`, `b_std`, `b_p16`,
    `b_p84`, `vherm_samples`/`sherm_samples`/`h3_samples`/`h4_samples` and
    their `_std` summaries) for direct side-by-side comparison, plus
    `n_divergences` (NUTS-specific -- see module docstring).
    """
    if cfg.continuum_knots is not None:
        raise ValueError("nuts_losvd_uncertainty: cfg.continuum_knots must be None -- NUTS samples the LOSVD "
                         "coefficients only (module docstring).")
    from .multi_template import fit_losvd

    lam = float(cfg.lam if lam is None else lam)
    result0 = fit_losvd(A_stack_w, y_w, dataclasses.replace(cfg, lam_rule="fixed", lam=lam))
    v_grid = cfg.build_velocity_grid()
    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D = difference_matrix(n_coef, order=PENALTY_ORDER, dv=dv, dv_ref=cfg.knot_spacing_kms)
    lam_vec = np.full(D.shape[0], lam)  # uniform lam across every penalty row
    A_eff_w = np.tensordot(result0["w"], A_stack_w, axes=([0], [0]))
    coeffs_map = np.asarray(result0["coeffs"])

    point0 = compute_losvd_moments(v_grid, result0["b"], fit_h3h4=fit_h3h4)

    kernel = NUTS(_model, target_accept_prob=target_accept_prob,
                  init_strategy=init_to_value(values={"coeffs": jnp.array(coeffs_map)}))
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), basis=jnp.array(basis), D=jnp.array(D),
              A_eff_w=jnp.array(A_eff_w), y_w=jnp.array(y_w), lam_vec=jnp.array(lam_vec),
              n_coef=n_coef)
    coeffs_samples = np.asarray(mcmc.get_samples()["coeffs"])
    n_divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())

    b_samples = np.exp(coeffs_samples @ basis.T)
    areas = np.trapezoid(b_samples, v_grid, axis=1)
    b_samples = b_samples / areas[:, None]

    n = len(b_samples)
    vherm = np.empty(n)
    sherm = np.empty(n)
    h3 = np.empty(n)
    h4 = np.empty(n)
    for i in range(n):
        m = compute_losvd_moments(v_grid, b_samples[i], fit_h3h4=fit_h3h4)
        vherm[i], sherm[i], h3[i], h4[i] = m["vherm"], m["sherm"], m["h3"], m["h4"]

    return dict(
        method="nuts", result0=result0, point0=point0, lam_used=float(lam),
        n_divergences=n_divergences,
        b_samples=b_samples,
        b_median=np.median(b_samples, axis=0),
        b_std=b_samples.std(axis=0, ddof=1),
        b_p16=np.percentile(b_samples, 16, axis=0),
        b_p84=np.percentile(b_samples, 84, axis=0),
        vherm_samples=vherm, sherm_samples=sherm, h3_samples=h3, h4_samples=h4,
        vherm_std=float(vherm.std(ddof=1)), sherm_std=float(sherm.std(ddof=1)),
        h3_std=float(h3.std(ddof=1)), h4_std=float(h4.std(ddof=1)),
    )
