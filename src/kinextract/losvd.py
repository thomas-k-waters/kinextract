"""Non-parametric LOSVD recovery: a low-dimensional, smoothly-positive
log-density spline, fit by a deterministic MAP optimization from a small
pool of starting points.

**Representation.** The line-of-sight velocity distribution `b(v)` is
represented as `b(v) = exp(spline(v))`: a cubic B-spline in `log(b)`,
with a modest number of coefficients matched to how much velocity
information the spectrum actually contains, rather than one free
parameter per fine histogram bin. Two properties of this matter for
correctness, not just convenience:

- **Positivity is structural, not enforced.** `exp(...)` is positive
  everywhere by construction -- there is no boundary at `b=0` for the
  optimizer to get stuck against, and no post-hoc clipping step that
  would make the reported `b` the solution to a different problem than
  the one actually optimized.
- **Dimensionality matches identifiability.** A handful of spline
  coefficients spanning the full velocity range keeps the inverse
  problem well-posed without needing heavy explicit shrinkage on top.

Each candidate start is optimized deterministically by L-BFGS-B, with JAX
autodiff providing exact gradients through the `exp` link, minimizing
data chi^2 plus a P-spline (Eilers & Marx 1996) curvature penalty on the
spline coefficients' second differences. The penalty acts on curvature,
not raw coefficient magnitude -- a plain L2-on-magnitude penalty measurably
underperforms the curvature form.

**Multi-start initialization.** Because `b = exp(spline(v))` is a
nonlinear reparameterization, the optimization landscape in coefficient
space is not convex, and a single starting point -- even a good,
data-driven one -- can converge to a real but meaningfully-suboptimal
local minimum. `default_initial_coeffs` provides a cheap, data-driven
start (a closed-form ridge estimate of `b`, clipped positive,
log-transformed, and projected onto the spline basis), which is a
substantially better average starting point than a flat one, but is not
by itself a safe substitute for trying more than one basin: in validation
testing this kind of failure -- reported as `success=True` by the
optimizer, with no other visible warning sign -- occurs in roughly
1-in-150 noisy fits, and the basin that wins can depend on which start is
used. `fit_log_spline_map` therefore runs a small pool of different
starts by default (the ridge estimate plus two flat starts at different
scales, see `_default_multi_starts`) and keeps whichever converges to the
lowest final objective, rather than trusting any single starting point.

**Knot density scales with the velocity grid.** A fixed
`n_interior_knots` under-resolves the LOSVD once the velocity grid widens
for a broader target (large sigma requires `vmin`/`vmax` far out to hold
the wings) -- validated: an absolute knot spacing coarser than roughly
150 km/s/knot produces 47-58% shape residual, versus comfortably under 5%
at ~95 km/s/knot or finer. `FitConfig` (see `config.py`) auto-derives
`n_interior_knots` from a target *absolute* knot spacing
(`knot_spacing_kms`, default 95 km/s) rather than exposing a bare count as
the primary knob, so the same configuration stays well-resolved across
different target sigmas.

**The curvature penalty is scaled by real velocity spacing, not raw
coefficient index.** A plain difference operator on coefficient index
treats every pair of adjacent coefficients as one "unit" apart regardless
of the actual km/s between their knots, so the same `lam` would represent
different physical smoothing strengths on different grids -- this was the
root cause of a systematic sigma bias that grew with grid width even
after knot density was auto-scaled to a safe absolute spacing, and of why
a given `lam` value could be catastrophically unsafe on one grid and fine
on another. `difference_matrix` accepts `dv` (the grid's actual knot
spacing) and `dv_ref` (`FitConfig.knot_spacing_kms`, the spacing `lam` is
calibrated against) and scales the penalty by `dv_ref/dv`, removing this
grid-width dependence; `config.fit_losvd` always passes these. `lam`'s
default (2.0) is calibrated against this scaled penalty -- see its help
text in `config.py`.

**Sigma bias is dominated by two mechanisms, not one.** (1)
`compute_losvd_moments`'s windowing (see `gh_moments.py`) truncates real
tail variance -- a deterministic, shape-independent ~-1.2% bias for a
near-Gaussian LOSVD, present even on the exact noiseless truth. (2) A
roughness penalty cannot tell a compact LOSVD from one carrying a broad,
low-amplitude pedestal across the velocity grid; any moment estimator
integrates that pedestal, so it inflates the reported sigma while leaving
the *core* correct. Mechanism (2) is the dominant one on real data --
measured on real MUSE spectra at 81 km/s recovered against an independent
~54 km/s reference measurement -- and is essentially invisible in this package's own
synthetic validation, because mocks built from the same design matrix the
fit inverts give the pedestal nothing to absorb.
`FitConfig.max_edge_amplitude_frac` attacks mechanism (2) directly, by
rejecting any candidate `lam` whose fit is still diverging at the
velocity-grid edge, rather than trying to suppress the pedestal via a
wing-specific penalty (an earlier per-row wing-taper approach was tried,
needed per-target retuning to actually work, and was removed once this
constraint superseded it); see `dev_notes/real_data_sigma/FINDINGS.md`
for the full diagnosis and `dev_notes/PORT_STATUS.md` for the wing
taper's own retired history.

- **Strongly negative h4 (platykurtic, flat-topped LOSVDs) recovers less
  well than other shapes.** At h4=-0.1 (S/N~100), worst-case shape
  residual is ~12%, versus comfortably under 5% for the rest of the
  standard validation grid. Partly a structural limit, not a fit
  artifact: the raw Gauss-Hermite curve at this parameter combination is
  not a valid probability density (it goes negative over more than half
  the velocity grid), a known pathology of the truncated GH expansion.
- **No continuum model at this layer.** This module expects a template
  mixture (and, if active, a continuum) already fixed before the design
  matrix is built -- see `kinextract.multi_template` for the
  template-mixture extension and `kinextract.continuum` for
  continuum-cofitting.
- **JAX runs in float64 here as of 2026-07** (`kinextract/__init__.py`
  enables `jax_enable_x64` at import; opt out with
  KINEXTRACT_NO_X64=1). This changed from the historical float32
  default after the shape-residual investigation measured a systematic
  optimizer failure, not just basin jitter: in the stiff heavy-penalty
  regime (wide grids, sigma >~ 200, large `lam` -- exactly where the
  now-default order-3 penalty operates), float32 L-BFGS-B reliably
  stalls tens of objective units above the true minimum while reporting
  `success=True` (measured at sigma=300/S N=30, lam=65536: 35% vs 6%
  shape residual, float32 vs float64 on identical data/starts). The
  float64 switch was validated as statistically neutral for the
  order-2 penalty path on a full 480-case paired grid before becoming
  the default, at ~2x runtime (see
  `dev_notes/REGULARIZATION_INVESTIGATION_LOG.md`, 2026-07-26/27).
  Related trap, still relevant at any precision: whitening by errors
  smaller than ~1e-4 of the signal (the old "noise=1e-6 noiseless
  benchmark" convention) produces enormous whitened values that degrade
  the optimizer -- use S/N ~ 1e4 as the noiseless proxy in benchmarks.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize as scipy_minimize

from .config import FitConfig


# Module-level jitted objective/gradient, not defined fresh inside
# `fit_log_spline_map` -- confirmed directly to matter: a per-call closure
# (capturing `basis`/`D2`/`A_w`/`y_w`/`lam` as baked-in constants, the
# original design here) gives JAX no way to recognize "this exact
# computation was already compiled" across repeated calls with the same
# array shapes, so every one of `bootstrap_losvd`'s `n_boot` replicates
# (and every point in `select_lambda_aic_local`'s own `lam` grid) paid a
# full XLA retrace+recompile, not just a forward/backward pass. Taking
# `basis`/`D2`/`A_w`/`y_w`/`lam` as ordinary (traced, not closed-over)
# arguments to a function jitted exactly once at import time lets JAX's
# own shape/dtype-keyed compilation cache do its job: the same compiled
# executable is reused across every bootstrap replicate and every `lam`
# grid point, since none of those change the array shapes, only the
# values. `lam` is passed as a traced scalar (not marked static) for the
# same reason -- different `lam` values must not themselves trigger a
# recompile.
def _neg_log_post(coeffs, basis, D, A_w, y_w, lam_vec):
    """chi2 + a *per-row* roughness penalty.

    `lam_vec` carries one regularization strength per row of `D` -- a
    single scalar broadcasts to every row identically (`fit_log_spline_
    map`'s usual case), but a caller wanting a non-uniform penalty
    (varying regularization strength by velocity) can pass a full
    per-row array instead.
    """
    b = jnp.exp(basis @ coeffs)
    resid = A_w @ b - y_w
    chi2 = 0.5 * jnp.sum(resid ** 2)
    rough = 0.5 * jnp.sum(lam_vec * (D @ coeffs) ** 2)
    return chi2 + rough


_val_fn = jax.jit(_neg_log_post)
_vg_fn = jax.jit(jax.value_and_grad(_neg_log_post))


def build_bspline_basis(v_grid: np.ndarray, n_interior_knots: int = 16, degree: int = 3) -> tuple[np.ndarray, int]:
    """Cubic B-spline design matrix evaluating each basis function on
    `v_grid`, clamped at both boundaries.

    Size `n_interior_knots` to roughly match the target's own information
    content (e.g. an SVD effective-rank diagnostic on the whitened design
    matrix) rather than assuming one value is universal -- and note that
    the right count scales with `v_grid`'s width; `FitConfig.
    resolved_n_interior_knots()` handles this automatically by default
    rather than requiring a fixed count here (see module docstring).
    """
    vmin, vmax = float(v_grid.min()), float(v_grid.max())
    interior = np.linspace(vmin, vmax, n_interior_knots + 2)[1:-1]
    knots = np.concatenate([[vmin] * (degree + 1), interior, [vmax] * (degree + 1)])
    n_coef = len(knots) - degree - 1
    design = BSpline.design_matrix(v_grid, knots, degree).toarray()
    return design, n_coef


def build_spline_truth(v_grid: np.ndarray, vel: float, sigma: float, degree: int = 3,
                        n_interior_knots: int | None = None, wiggle_scale: float = 0.15,
                        seed: int | np.random.Generator = 0) -> tuple[np.ndarray, np.ndarray]:
    """A "truth" LOSVD defined directly as `exp(spline)`, with no
    Gauss-Hermite step at all -- the recommended way to build a mock LOSVD
    with known, unambiguous moments.

    **Why this exists.** `gauss_hermite_model`'s `vel`/`sig`/`h3`/`h4`
    parameters are basis-expansion coefficients, not the resulting curve's
    actual mean/std/skewness/kurtosis, except in the `h3, h4 -> 0` limit --
    away from that limit the gap is large and exact, not a fitting
    artifact (see `gauss_hermite_model`'s own docstring, and
    `dev_notes/validation_suite/FINDINGS.md` for the full derivation and
    numbers). A spline truth has no such gap by construction: there is no
    parametric family being truncated, so whatever `compute_losvd_moments`
    reports on the resulting curve *is* the truth, not an approximation to
    it.

    A Gaussian core is projected onto the spline basis by least squares,
    then perturbed by a random log-space wiggle at each knot (`wiggle_scale`
    is that perturbation's standard deviation, in units of the Gaussian
    core's own local log-density curvature scale). **`wiggle_scale=0.0`
    reproduces a pure Gaussian core** (up to the spline basis's own
    representation error) -- the spline-truth analog of `h3=h4=0`, so h3/h4
    recovery is still meaningful to validate against a near-zero-shape
    truth. 0.15 (the default) keeps the shape recognizably a single
    asymmetric peak for the sigma/knot-spacing regime this package
    validates against; larger values increasingly produce multi-modal or
    heavy-noise curves, a deliberate stress option, not the default.

    `seed` may be an int (seeds a fresh `np.random.Generator`) or an
    existing `np.random.Generator` (used directly, e.g. to keep replicate
    generation reproducible from a caller's own RNG stream).

    Returns
    -------
    (b_unit, coeffs) : the unit-area LOSVD on `v_grid`, and the spline
        coefficients that produced it (`exp(basis @ coeffs)`, before
        normalization).
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    if n_interior_knots is None:
        n_interior_knots = FitConfig(vmin=float(v_grid.min()), vmax=float(v_grid.max())).resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_interior_knots, degree=degree)
    gauss_core = -0.5 * ((v_grid - vel) / sigma) ** 2
    coeffs0, *_ = np.linalg.lstsq(basis, gauss_core, rcond=None)
    wiggle = rng.normal(0.0, wiggle_scale, size=coeffs0.shape) if wiggle_scale > 0 else np.zeros_like(coeffs0)
    coeffs = coeffs0 + wiggle
    b = np.exp(basis @ coeffs)
    b_unit = b / np.trapezoid(b, v_grid)
    return b_unit, coeffs


def difference_matrix(n_coef: int, order: int = 2, dv: float | None = None, dv_ref: float | None = None) -> np.ndarray:
    """Discrete difference operator on the spline coefficient sequence --
    the standard P-spline (Eilers & Marx 1996) roughness-penalty operator,
    `sum((D @ coeffs)**2)`. Operates on the spline *coefficients*, not on
    `b` itself -- a plain L2-on-magnitude penalty measurably hurt
    recovery relative to this curvature form.

    **Scaled by real knot spacing, not just coefficient index.** A plain
    `np.diff` treats adjacent coefficients as one "unit" apart regardless
    of how many actual km/s separate their knots, so the same `lam` would
    represent different physical smoothing strengths on different grids
    -- see the module docstring for why this matters. Pass `dv` (the
    actual knot spacing on this grid) and `dv_ref` (the calibration
    reference, `FitConfig.knot_spacing_kms`) to scale `D` by `dv_ref / dv`
    -- when `dv == dv_ref` this is a no-op, so `lam`'s calibration is
    preserved exactly at the reference spacing and only corrected where
    the actual grid deviates from it. Omit both (the default) to leave
    `D` unscaled.
    """
    D = np.eye(n_coef)
    for _ in range(order):
        D = np.diff(D, axis=0)
    if dv is not None and dv_ref is not None:
        D = D * (dv_ref / dv)
    return D


def default_initial_coeffs(A_w: np.ndarray, y_w: np.ndarray, basis: np.ndarray, ridge_lam: float = 1.0) -> np.ndarray:
    """A cheap, data-driven starting point for `fit_log_spline_map`'s
    optimization: a closed-form ridge estimate of `b` (clipped to a small
    positive floor, since the true LOSVD is non-negative), log-transformed,
    and projected onto the spline basis by least squares.

    This is not a sophisticated estimate -- it doesn't need to be. Its job
    is only to land the optimizer in the right basin; the L-BFGS-B step
    that follows does the actual precision fitting. See the module
    docstring for why a single (even data-driven) starting point is not
    used alone.
    """
    n_bins = A_w.shape[1]
    M = A_w.T @ A_w + ridge_lam * np.eye(n_bins)
    b0 = np.linalg.solve(M, A_w.T @ y_w)
    b0_positive = np.clip(b0, 1e-6 * max(float(b0.max()), 1e-6), None)
    coeffs0, *_ = np.linalg.lstsq(basis, np.log(b0_positive), rcond=None)
    return coeffs0


def _default_multi_starts(A_w: np.ndarray, y_w: np.ndarray, basis: np.ndarray, ridge_lam: float = 1.0) -> list[np.ndarray]:
    """The default pool of starting points for `fit_log_spline_map`'s
    multi-start search: the data-driven ridge estimate plus two flat
    starts at different scales. The ridge estimate is a much better
    average start than a flat one, but is not by itself a safe substitute
    for trying more than one basin -- see the module docstring.
    """
    n_coef = basis.shape[1]
    return [
        default_initial_coeffs(A_w, y_w, basis, ridge_lam=ridge_lam),
        np.full(n_coef, np.log(0.01)),
        np.full(n_coef, np.log(0.001)),
    ]


def fit_log_spline_map(A_w: np.ndarray, y_w: np.ndarray, basis: np.ndarray, D: np.ndarray,
                        lam, coeffs0: np.ndarray | None = None,
                        ridge_lam: float = 1.0) -> dict:
    """Fit `b(v) = exp(basis @ coeffs)` by minimizing
    `0.5*||A_w @ b - y_w||^2 + 0.5*sum(lam_j * (D @ coeffs)_j^2)` via
    L-BFGS-B, with JAX autodiff for exact gradients through the `exp` link.

    `lam` may be a scalar (uniform smoothing, what every caller in this
    package currently passes) or a vector with one entry per row of `D`,
    for a caller wanting a non-uniform penalty strength across the
    velocity grid.

    **Multi-start by default.** The optimization landscape is not convex
    (see module docstring), and no single starting point is safe on its
    own -- roughly 1-in-150 noisy fits land in a real, meaningfully-worse
    local minimum with no other visible warning sign. Unless `coeffs0` is
    given explicitly, this runs from a small pool of different starts
    (`_default_multi_starts`: one data-driven, two flat at different
    scales) and keeps whichever converges to the lowest final objective.
    Pass `coeffs0` explicitly (e.g. to warm-start from a neighbouring
    iteration's answer) to skip the search and run a single deterministic
    optimization instead -- which is what `multi_template._alternate_once`
    does after its first iteration, since the basin has been chosen by
    then.

    Parameters
    ----------
    A_w : ndarray, shape (n_pix, n_bins)
        Noise-whitened design matrix (template mixture x LOSVD-grid
        convolution, divided by the per-pixel error) with template
        weights and continuum already fixed.
    y_w : ndarray, shape (n_pix,)
        Noise-whitened observed flux (`flux / err`).
    basis : ndarray, shape (n_bins, n_coef)
        From `build_bspline_basis`, evaluated on the same velocity grid
        `A_w`'s columns correspond to.
    D : ndarray
        From `difference_matrix`.
    lam : float or ndarray
        Roughness penalty strength: scalar, or one value per row of `D`.
    coeffs0 : ndarray, optional
        A single initial guess in coefficient (log-amplitude) space. If
        given, disables the multi-start search.
    ridge_lam : float
        Ridge strength for the one data-driven start within the
        multi-start pool (see `default_initial_coeffs`). Ignored if
        `coeffs0` is given.

    Returns
    -------
    dict with `b` (the recovered LOSVD on the same grid `basis` was built
    on), `coeffs`, `success`, `message`, `runtime_s`, `n_coef`, `lam`,
    `objective`, and `objective_gap` (None if `coeffs0` was given --
    otherwise the gap between the best and second-best start's final
    objective; large means this fit only survived because of the
    multi-start search and is worth a second look).
    """
    if not (np.all(np.isfinite(A_w)) and np.all(np.isfinite(y_w))):
        raise ValueError(
            "A_w/y_w contain non-finite values (NaN or inf). fit_log_spline_map has no bad-"
            "pixel masking of its own -- exclude bad pixels (cosmic rays, masked emission "
            "lines, saturated detector regions, etc.) from A_w/y_w before calling this, e.g. "
            "`A_w, y_w = A_w[good], y_w[good]` for a boolean `good` mask. Silently fitting "
            "through non-finite input would otherwise propagate NaN into the entire recovered "
            "LOSVD with no indication of why."
        )

    n_coef = basis.shape[1]
    starts = [coeffs0] if coeffs0 is not None else _default_multi_starts(A_w, y_w, basis, ridge_lam=ridge_lam)

    lam_vec = np.broadcast_to(np.asarray(lam, dtype=float), (D.shape[0],))
    basis_j, D_j, A_w_j, y_w_j = (jnp.array(basis), jnp.array(D), jnp.array(A_w), jnp.array(y_w))
    lam_j = jnp.asarray(lam_vec)

    # A single fused value_and_grad call, not separate value/grad calls per
    # L-BFGS-B iteration -- jax.grad computes the forward pass internally
    # anyway, so calling them separately pays for it twice. Confirmed on the
    # real MUSE-template setup: 1.45x faster, objectives matching to better
    # than 1e-8 (mathematically identical, not an approximation).
    def _obj_and_grad(c):
        v, g = _vg_fn(jnp.array(c), basis_j, D_j, A_w_j, y_w_j, lam_j)
        return float(v), np.array(g)

    t0 = time.perf_counter()
    fits = [scipy_minimize(_obj_and_grad, c0, jac=True, method='L-BFGS-B') for c0 in starts]
    runtime_s = time.perf_counter() - t0

    objectives = [float(_val_fn(jnp.array(res.x), basis_j, D_j, A_w_j, y_w_j, lam_j)) for res in fits]
    best_idx = int(np.argmin(objectives))
    res = fits[best_idx]
    objective_gap = None
    if len(fits) > 1:
        srt = sorted(objectives)
        objective_gap = srt[1] - srt[0]

    coeffs = res.x
    return dict(b=np.exp(basis @ coeffs), coeffs=coeffs, success=bool(res.success),
                message=str(res.message), runtime_s=runtime_s, n_coef=n_coef,
                lam=lam, objective=objectives[best_idx], objective_gap=objective_gap)
