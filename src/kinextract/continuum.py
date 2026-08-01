"""Continuum co-fitting: a shared, structurally-positive multiplicative
correction, jointly recovered with the LOSVD and template mixture
weights via three-block alternating minimization.

    f(lambda) = C(lambda) * [ sum_k w_k * (T_k conv b)(lambda) ]

See `kinextract_test_suite/CONTINUUM_ARCHITECTURE_PLAN.md` for the full
design rationale (why the continuum is shared across templates rather
than per-template, why LSF matching isn't a fourth fitted block, the
template-continuum-trust hierarchy noted below).

**`C(lambda) = exp(spline_continuum(lambda))`** -- the same
structurally-positive `exp`-link construction as `b(v)` itself, over
wavelength instead of velocity, reusing the same B-spline/P-spline
machinery (`kinextract.losvd.build_bspline_basis`, `difference_matrix`)
on the wavelength grid. Deliberately much lower flexibility than the
LOSVD's own spline (a handful of knots by default) -- its job is mopping
up smooth residual gaps, not carrying real spectral information.

**Three-block alternating minimization**, fixing two of `(b, w, C)` and
solving the third each step, in forward-model form throughout (never
dividing the data by the current continuum estimate, which would silently
mis-weight pixel noise by how large the local continuum happens to be):

1. **Fix `(w, C)`, solve `b`.** Fold both into the effective design
   matrix, `A_eff[i,j] = C(lambda_i) * (sum_k w_k * A_stack[k])[i,j]`,
   and reuse `fit_log_spline_map` unchanged.
2. **Fix `(b, C)`, solve `w`.** NNLS against continuum-multiplied
   per-template columns, `design[i,k] = C(lambda_i) * (A_stack[k]@b)[i]`.
3. **Fix `(b, w)`, solve `C`.** `M(lambda) = sum_k w_k*(T_k conv b)(lambda)`
   is now fully known; fit `C(lambda)*M(lambda) ~ y_w(lambda)` via
   `fit_continuum_block` (a dedicated, smaller nonlinear solve).

**Requires templates already at a consistent continuum-trust level**
(e.g. already continuum-normalized) -- the template-continuum-trust
hierarchy (library-keyed presets for how much of each raw template's own
continuum to trust) is not implemented. Best suited to a clean,
already-normalized template library; not recommended for raw,
non-normalized templates without independently re-checking agreement
against a reference. Wavelength-dependent LSF matching is out of scope.

**`fit_losvd_with_continuum` is a thin wrapper over
`kinextract.multi_template.fit_losvd_basis`'s shared alternation
driver** (given `continuum_basis`/`D2_c`), not a separately-maintained
copy of the loop -- the tapered roughness penalty is applied in the
LOSVD sub-fit like any other fit.

**Regularization has two independent terms, and both matter.** The
continuum's curvature penalty (`lam_c * D2_c`) controls smoothness, but a
second-order-difference penalty has an exact null space -- the constant
and *linear* directions in coefficient space, a standard property of any
such P-spline penalty -- so `lam_c` alone cannot bound an arbitrary
near-linear trend in the recovered continuum. `ridge_c` (see
`fit_continuum_block`'s own docstring) is a direct L2 penalty on the
coefficients themselves that closes exactly that gap. Both are always
active; treat them as a pair, not `lam_c` alone, when reasoning about how
constrained the recovered continuum is. Substantially de-risked by a
recent fix (see `dev_notes/VALIDATION_CAMPAIGN.md` Stage 15 for the investigation
this addressed); a first multi-seed bias sweep now exists
(`dev_notes/continuum/FINDINGS.md`, 2026-07-30, 288 fits, one mock
regime) -- treat continuum-cofitting as still maturing (one mock regime
is not full production validation) rather than production-hardened, but
no longer entirely unvalidated.

**Penalty-order recommendation, updated by that sweep.** Adaptive
`lam_selection` doesn't support continuum-cofitting yet, so a fixed `lam`
is required either way. Earlier guidance (this module's docstring and
`examples/notebooks/02_realistic_mock_fit.ipynb`'s own Section 8) pinned
`penalty_order=2, lam=2.0` as "the documented-working combination" and
called `penalty_order=3` untested. The sweep found the opposite: at every
S/N tested (30/60/100), a fixed order-3 `lam` (4096-32768 range) beat
order-2/lam=2.0 on both `sigma` bias and, especially, `h4` bias (order-2's
h4 bias was 3-4x larger, at every S/N -- worse than not cofitting a
continuum at all). Prefer `penalty_order=3` with a fixed `lam` in that
range for new continuum-cofitting work; `penalty_order=2, lam=2.0` is no
longer the recommended default, though it still works and is still the
`FitConfig` default's own automatic fallback when adaptive selection is
left on with continuum-cofitting active (see `multi_template.fit_losvd`'s
own `UserWarning` for that fallback -- it does not yet know about this
finding and still falls back to order 2).
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from .losvd import build_bspline_basis, difference_matrix
from .multi_template import fit_losvd_basis


def _neg_log_post_continuum(coeffs, M, y_w, basis, D2, lam_c, ridge_c):
    C = jnp.exp(basis @ coeffs)
    resid = M * C - y_w
    chi2 = 0.5 * jnp.sum(resid ** 2)
    rough = 0.5 * lam_c * jnp.sum((D2 @ coeffs) ** 2)
    ridge = 0.5 * ridge_c * jnp.sum(coeffs ** 2)
    return chi2 + rough + ridge


# Module-level, not defined fresh inside fit_continuum_block: a closure captures
# M/y_w/basis/D2/lam_c as baked-in constants, which are a *new* Python object every
# call even when their shapes are identical -- jax.jit's cache is keyed on the jitted
# function object's identity plus argument shapes/dtypes, so a fresh closure defeats
# it entirely and forces a full retrace+recompile on every single call (every
# alternating-minimization iteration, every bootstrap replicate). Same fix already
# applied to losvd.fit_log_spline_map's identical anti-pattern -- see that module for
# the measured (modest but real) speedup. `lam_c` is passed as a traced argument
# (not a Python/static one) so different lam_c values don't themselves force
# recompilation either.
_val_fn_continuum = jax.jit(_neg_log_post_continuum)
_vg_fn_continuum = jax.jit(jax.value_and_grad(_neg_log_post_continuum))


def fit_continuum_block(M: np.ndarray, y_w: np.ndarray, basis_c: np.ndarray, D2_c: np.ndarray,
                         lam_c: float = 2.0, ridge_c: float = 0.1,
                         coeffs0: np.ndarray | None = None) -> dict:
    """Block 3 of the alternating minimization: given a known "model"
    `M(lambda)` (the current template-mixture-and-LOSVD prediction) and
    target `y_w`, fit `C(lambda) = exp(basis_c @ coeffs)` minimizing
    `0.5*||M*C - y_w||^2 + 0.5*lam_c*||D2_c @ coeffs||^2 + 0.5*ridge_c*
    ||coeffs||^2` via L-BFGS-B with JAX autodiff -- structurally the
    *same kind* of problem as `fit_log_spline_map` (an `exp`-linked
    spline fit by curvature-penalized nonlinear least squares) but not
    literally reusable as that function, since here the spline multiplies
    `M` elementwise at each wavelength rather than acting as `b(v)`
    convolved through a Doppler-shift design matrix -- a genuinely
    different forward-model structure, not just a renamed call.

    **`ridge_c` closes the curvature penalty's zero-cost null space.** A
    second-order difference penalty (`D2_c`) assigns zero cost to the
    constant and *linear* directions in coefficient space (a standard
    property of any such P-spline penalty), so `lam_c` alone cannot bound
    a near-linear trend in the recovered continuum. `ridge_c` is a direct
    L2 penalty on the coefficients themselves, which has a real, nonzero
    cost along that same direction -- see `dev_notes/VALIDATION_CAMPAIGN.md` Stage
    15 for the investigation this closes. Default `0.1`: recovery RMS
    against a known injected continuum is unaffected by `ridge_c` across
    its whole 0-1.0 range, so the value mainly needs to be decisively
    nonzero, not precisely tuned.

    **`result["success"]` is frequently False for reasons unrelated to fit
    quality -- don't filter on it.** This function's own L-BFGS-B call
    intermittently reports an "ABNORMAL" line-search termination on an
    already-well-converged solution (every multi-start candidate typically
    agrees to several decimal places regardless of the flag). Measured
    directly: mean recovered sigma-bias is statistically indistinguishable
    between `success=True`- and `success=False`-flagged fits. Treat this
    flag as a diagnostic curiosity, not a hard filter for discarding fits.

    **Multi-start** guards against competing local minima the same way
    `fit_log_spline_map`'s own multi-start does, even though the continuum
    block's low flexibility (a handful of knots) makes that less of a
    concern here. Unlike `fit_log_spline_map`, an explicit `coeffs0` (a
    warm start from the previous alternating-minimization iteration) is
    *added* to the multi-start pool rather than used exclusively, since
    this function is always called with a warm start in practice (from
    `_alternate_once`'s own state) -- using it exclusively would mean this
    never actually multi-starts.

    Parameters
    ----------
    M : ndarray, shape (n_pix,)
        Current template-mixture-and-LOSVD model, `sum_k w_k*(T_k conv b)`.
    y_w : ndarray, shape (n_pix,)
        Noise-whitened observed flux (the *same*, undivided array used
        throughout this package -- never `y_w / M`, see module docstring).
    basis_c, D2_c : B-spline design matrix and roughness-penalty operator
        on the wavelength grid, from `build_bspline_basis`/
        `difference_matrix` called on `wave_fit` instead of `v_grid`.
    lam_c : curvature penalty weight for the continuum spline. Default
        2.0, unvalidated -- inherited from `lam`'s own default as a
        starting point, not independently calibrated (see
        `tests/test_continuum.py` for what has and hasn't been checked).
    ridge_c : direct L2 penalty on the continuum coefficients themselves,
        default 0.1 -- see this function's own docstring above for why
        this exists.
    coeffs0 : optional warm-start coefficients (e.g. the previous
        alternating-minimization iteration's own continuum coefficients).
        Added to the multi-start pool alongside a flat (`C=1`) start and a
        couple of small perturbed starts -- see module docstring above for
        why this doesn't disable the pool the way `fit_log_spline_map`'s
        `coeffs0` does.

    Returns
    -------
    dict with `C` (ndarray, shape (n_pix,), the recovered continuum
    curve), `coeffs`, `success`, `message`, `n_coef`, `objective`,
    `objective_gap` (gap between the best and second-best start's final
    objective -- small means every start agreed, large means this result
    only survived because of the multi-start search).
    """
    n_coef = basis_c.shape[1]
    starts = _default_continuum_starts(n_coef, coeffs0)

    M_j, y_w_j, basis_j, D2_j = jnp.array(M), jnp.array(y_w), jnp.array(basis_c), jnp.array(D2_c)
    lam_c_j, ridge_c_j = jnp.asarray(lam_c), jnp.asarray(ridge_c)

    # Fused value_and_grad, not separate val_fn/grad_fn calls -- see
    # losvd.fit_log_spline_map's identical change for the measured 1.45x
    # speedup and the reasoning (computing the gradient already requires
    # the forward pass, so separate calls pay for it twice per iteration).
    val_fn = lambda c: _val_fn_continuum(c, M_j, y_w_j, basis_j, D2_j, lam_c_j, ridge_c_j)
    vg_fn = lambda c: _vg_fn_continuum(c, M_j, y_w_j, basis_j, D2_j, lam_c_j, ridge_c_j)

    def _obj_and_grad(c):
        v, g = vg_fn(jnp.array(c))
        return float(v), np.array(g)

    fits = []
    for c0 in starts:
        res = scipy_minimize(_obj_and_grad, c0, jac=True, method='L-BFGS-B')
        fits.append(res)

    objectives = [float(val_fn(jnp.array(res.x))) for res in fits]
    best_idx = int(np.argmin(objectives))
    res = fits[best_idx]
    objective_gap = None
    if len(fits) > 1:
        sorted_obj = sorted(objectives)
        objective_gap = sorted_obj[1] - sorted_obj[0]

    coeffs = res.x
    C = np.exp(basis_c @ coeffs)
    return dict(C=C, coeffs=coeffs, success=bool(res.success), message=str(res.message),
                n_coef=n_coef, objective=objectives[best_idx], objective_gap=objective_gap)


def _default_continuum_starts(n_coef: int, coeffs0: np.ndarray | None) -> list[np.ndarray]:
    """Multi-start pool for `fit_continuum_block`: a flat (`C=1`) start,
    a couple of small perturbed starts, and the given warm-start
    `coeffs0` (if any) -- see `fit_continuum_block`'s own docstring for
    why a warm start is added to the pool rather than used exclusively.
    """
    starts = [np.zeros(n_coef), np.full(n_coef, 0.05), np.full(n_coef, -0.05)]
    if coeffs0 is not None and not np.allclose(coeffs0, 0.0):
        starts.append(np.asarray(coeffs0, dtype=float))
    return starts


def fit_losvd_with_continuum(A_stack: np.ndarray, y_w: np.ndarray, wave_fit: np.ndarray,
                              basis: np.ndarray, D2: np.ndarray,
                              n_continuum_knots: int = 4, lam: float = 2.0, lam_c: float = 2.0,
                              ridge_c: float = 0.1,
                              ridge_lam: float = 1.0, w0: np.ndarray | None = None,
                              max_iter: int = 15, tol: float = 1e-4, patience: int = 2) -> dict:
    """Three-block alternating minimization -- LOSVD, template weights,
    shared multiplicative continuum. See this module's own docstring for
    the algorithm. A thin wrapper: builds the continuum's own B-spline
    basis/penalty from `wave_fit`/`n_continuum_knots`, then calls
    `kinextract.multi_template.fit_losvd_basis` with them -- the
    actual three-block alternation lives there now (shared with the
    plain two-block path), not duplicated here.

    Parameters
    ----------
    A_stack, y_w, basis, D2, lam, ridge_lam : as in `fit_losvd_basis`.
    wave_fit : ndarray, shape (n_pix,), the wavelength (not velocity)
        grid `y_w`/`A_stack` correspond to -- used to build the
        continuum's own B-spline basis.
    n_continuum_knots : interior knots for the continuum spline. Default
        4 -- deliberately much coarser than the LOSVD's own knot count.
    lam_c : curvature penalty for the continuum block (see
        `fit_continuum_block`).
    ridge_c : direct L2 penalty on the continuum coefficients themselves
        (see `fit_continuum_block`'s own docstring for why this exists).
        Default 0.1.
    w0 : optional starting weight vector (shape `(n_templates,)`).
        Defaults to equal weights. Note: unlike `fit_losvd_basis`'s own
        `w0_pool`, this takes a single vector, not a pool -- continuum-
        cofitting does not multi-start over the weight vector.
    max_iter, tol, patience : as in `fit_losvd_basis`'s alternation loop.

    Returns
    -------
    dict with `b`, `w`, `C`, `coeffs` (LOSVD), `continuum_coeffs`,
    `success`, `n_iter`, `objective`, `objective_history`, plus
    `n_coef_continuum`, `lam_c`, and `runtime_s` (wall time for this call).
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1 (got {max_iter}).")

    n_templates = A_stack.shape[0]
    basis_c, n_coef_c = build_bspline_basis(wave_fit, n_interior_knots=n_continuum_knots, degree=3)
    dw = (wave_fit.max() - wave_fit.min()) / (n_continuum_knots + 1)
    D2_c = difference_matrix(n_coef_c, order=2, dv=dw, dv_ref=dw)  # no cross-grid rescaling needed here
    w0 = np.ones(n_templates) if w0 is None else np.asarray(w0, dtype=float)

    t0 = time.perf_counter()
    result = fit_losvd_basis(A_stack, y_w, basis, D2, lam=lam, ridge_lam=ridge_lam,
                              w0_pool=[w0], max_iter=max_iter, tol=tol, patience=patience,
                              continuum_basis=basis_c, D2_c=D2_c, lam_c=lam_c, ridge_c=ridge_c)
    result["runtime_s"] = time.perf_counter() - t0
    return result
