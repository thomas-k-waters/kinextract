"""Data-driven regularization-strength selection via the generalized AIC
of Lipka & Thomas (2021, MNRAS 504, 4599; arXiv:2104.10168), extended to
LOSVD-specific regularization in Thomas & Lipka (2022).

**The criterion (their eq. 14)**: minimize `chi^2 + w_m * m_eff` over a
grid of `lam`, where `m_eff` is the fit's own *effective* degrees of
freedom at that `lam` -- not the spline's raw coefficient count, which a
smoothness penalty shrinks well below in a `lam`-dependent way that has
no closed form for a non-negativity-constrained, jointly-alternating
(LOSVD + template-weight) fit like this package's. `w_m = 2.0` (their
eq. 13, the classical Akaike weight) is the value their own simulations
(Sec. 7 of the paper) calibrate as the right choice -- not an arbitrary
default copied without checking.

**`m_eff` estimation (their eq. 10/11, the "covariance" form they settle
on as their production method, cross-checked against a direct chi^2-
reduction estimator in their eq. 9 and found equivalent)**:

    m_eff = (1/K) * sum_k sum_i (bootfit_i - fit_i)(bootdata_i - fit_i) / sigma_i^2

At each of `K` bootstrap iterations: add Gaussian noise (matching the
data's own per-pixel sigma) to the *original* fit's own model
prediction to make synthetic "bootstrap data", refit at the *same*
`lam`, and measure how much the new fit's prediction covaries with the
noise it was just handed. A flexible (low-`lam`) model tracks injected
noise closely (`m_eff` large); a stiff (high-`lam`) model barely moves
(`m_eff` near 0).

Because kinextract already works in noise-whitened units throughout
(`A_stack_w = A_stack / errors`, `y_w = y / errors`, so every pixel's
noise sigma is already 1), `sigma_i^2` in the formula above is just 1 at
every pixel -- no extra rescaling needed at the call site.

**Cost**: `n_boot` extra full fits per `lam` in the grid, by design (the
source paper flags the same tradeoff -- "computationally expensive"). This
applies to `estimate_m_eff`/`select_lambda_aic` specifically -- the closed-
form `estimate_m_eff_local`/`select_lambda_aic_local` below costs only one
extra fit per `lam` in the grid (no bootstrap), which is cheap enough that
**`select_lambda_aic_local` is `FitConfig`'s default `lam_selection` mode
as of the comparison this module exists to provide**: measured 20-57%
sigma-bias reduction vs. fixed `lam=2.0` across sigma=50/150/300 and
h4_true=0/0.15 (`dev_notes/regularization/aic_local_firming_up_results.txt`),
consistent enough on the primary target (sigma) to promote, per this
project's own "tested better becomes standard" rule -- h4 bias is a
genuine mixed result (better at some sigma, worse at others), not silently
smoothed over by that promotion. `estimate_m_eff`/`select_lambda_aic`
(the expensive bootstrap originals) remain available for cross-checking
`_local`'s closed-form approximation, not as the default path.

**What this does NOT claim.** Nothing in Lipka & Thomas (2021)/Thomas &
Lipka (2022) addresses whether AIC-selected smoothing specifically
preserves genuine non-Gaussian structure (e.g. real high-velocity wings)
as opposed to noise -- their own validation is against toy galaxies with
realistic orbital anisotropy on aggregate mass/inclination-recovery
metrics, not a wing-preservation test. "AIC picks the statistically
best-supported smoothing level" is not automatically the same claim as
"never suppresses a real high-velocity wing" -- test that directly (a
mock with genuine non-Gaussian wings, not just the Gaussian-truth mocks
used elsewhere in this package's validation) before trusting this for
black-hole-mass work, where wing fidelity is itself the science.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np

from .losvd import build_bspline_basis, difference_matrix
from .multi_template import fit_losvd


@dataclass
class AICSelectionResult:
    best_lam: float
    lam_grid: np.ndarray
    chi2: np.ndarray
    m_eff: np.ndarray
    criterion: np.ndarray
    w_m: float
    # The full `fit_losvd` result already computed at `best_lam` during the
    # scan. Callers that would otherwise refit at the selected lam (i.e.
    # `multi_template.fit_losvd`) can reuse this instead -- the selection
    # loop has to fit at every candidate lam anyway, so refitting at the
    # winner repeats work that was already done and thrown away. `None`
    # only for the bootstrap-based `select_lambda_aic` below, which does
    # not retain per-lam fits.
    best_fit: dict | None = None


def _model_prediction(A_stack_w, result):
    w = result["w"]
    b = result["b"]
    return sum(w[k] * (A_stack_w[k] @ b) for k in range(len(w)))


def estimate_m_eff(A_stack_w, y_w, cfg, lam, n_boot=20, seed=0, L=None):
    """Bootstrap estimate of effective degrees of freedom at a fixed
    `lam` (Lipka & Thomas 2021 eq. 11).

    `L` : **not needed for the mainline correlated-noise case -- read this
        before assuming it is.** If the caller whitened `A_stack_w`/`y_w`
        correctly via `noise_model.whiten(..., rho=rho)` using the *true*
        `rho`, the whitened residuals are already i.i.d. N(0,1) by
        construction (GLS pre-whitening removes the correlation, it
        doesn't just rescale it away), so the plain
        `rng.standard_normal(len(y_w))` default below is already correct
        in that case, with no change needed here at all. `L` exists only
        for the deliberate-mismatch case: testing this criterion's
        behavior when the *assumed* whitening covariance differs from the
        *true* one (the real-world case where `rho` is itself only ever
        an estimate) -- pass the *true* generating process's Cholesky
        factor here to inject noise from a different covariance than
        whatever `A_stack_w`/`y_w` were actually whitened against, for
        that specific diagnostic. Default `None` (no deliberate
        mismatch).

    Returns `(m_eff, chi2, fit_result)`: `chi2` is the *original* fit's
    own whitened residual sum of squares at this `lam`; `fit_result` is
    its full `fit_losvd` output, returned so callers don't need to
    refit for it.
    """
    # FitConfig's parameter-hazard warnings are suppressed for this construction --
    # see estimate_m_eff_local's own note on why (this lam is an internal scan
    # point, not a user choice; whatever the scan actually selects is checked
    # normally when fit_losvd rebuilds its config with it).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg_lam = replace(cfg, lam=lam, lam_selection="fixed")
    result = fit_losvd(A_stack_w, y_w, cfg_lam)
    model_fit = _model_prediction(A_stack_w, result)
    resid = y_w - model_fit
    chi2 = float(np.sum(resid ** 2))

    rng = np.random.default_rng(seed)
    cov_sum = 0.0
    for _ in range(n_boot):
        z = rng.standard_normal(len(y_w))
        noise = L @ z if L is not None else z
        y_boot = model_fit + noise
        boot_result = fit_losvd(A_stack_w, y_boot, cfg_lam)
        model_boot = _model_prediction(A_stack_w, boot_result)
        cov_sum += float(np.sum((model_boot - model_fit) * (y_boot - model_fit)))
    m_eff = cov_sum / n_boot
    return m_eff, chi2, result


def estimate_m_eff_local(A_stack_w, y_w, cfg, lam, initial_coeffs=None):
    """Closed-form *local* approximation to `estimate_m_eff`'s bootstrap
    quantity, via the classical ridge-regression/smoothing-spline
    effective-degrees-of-freedom formula (Hastie & Tibshirani -- the
    "hat matrix trace"), avoiding bootstrap refitting entirely.

    **The idea.** `b = exp(basis @ coeffs)` is a smooth reparametrisation,
    so at the converged solution, `d(b)/d(coeffs) = diag(b) @ basis` (the
    pointwise derivative of exp). Linearizing the LOSVD sub-problem at
    that point turns it into an ordinary penalized linear regression in
    `coeffs`-space with effective design matrix `X = A_eff_w @ diag(b) @
    basis`, and the standard ridge/smoothing-spline result gives:

        df = trace[ (X'X + lam*D2'D2)^-1 X'X ]

    computed as a single `n_coef x n_coef` linear solve -- no refitting
    at all, versus `estimate_m_eff`'s `n_boot` extra nonlinear fits.

    **What this does and does not capture, honestly.** This measures the
    *local* flexibility of the LOSVD spline coefficients alone, holding
    the converged template weights `w` fixed at their point-estimate
    value. `estimate_m_eff`'s bootstrap measures the flexibility of the
    *whole* alternating (LOSVD + weight) fit, since weights are also
    re-solved on every bootstrap replicate -- these are not guaranteed to
    be the same quantity, only related ones. Also ignores which
    non-negativity constraints on `w` are active/binding, treating the
    linearization as if `w` were fixed and unconstrained. Validate
    against `estimate_m_eff` before trusting this as a substitute, not
    just a curiosity -- that comparison is this function's whole reason
    to exist.

    `initial_coeffs` : optional single starting guess in LOSVD coefficient
    space, passed straight through to `fit_losvd` -- see its own
    docstring. Used by `select_lambda_aic_local`'s `warm_start_lam_grid`
    option to chain each grid point's fit from its neighbour's converged
    solution instead of cold-starting every point independently.

    Returns `(m_eff_local, chi2, fit_result)`, same shape as
    `estimate_m_eff`'s return (minus the bootstrap cost).
    """
    # lam_selection="fixed" is required here, not just consistent with the rest of this
    # function's intent -- without it, this function's own per-lam-grid-point fit_losvd
    # call would each trigger a *nested* adaptive-lam search (fit_losvd's own default),
    # since select_lambda_aic_local/estimate_m_eff_local exists specifically to scan a
    # lam_grid at fixed, known lam values, not to have `fit_losvd` silently override them.
    #
    # FitConfig's parameter-hazard warnings are suppressed for this one construction:
    # they exist to tell a *user* that a value they chose is dangerous, but these lam
    # values are internal scan points, not user choices -- and warning about each of
    # them turns one ordinary adaptive fit into a wall of identical warnings naming
    # values the caller never set. Whatever lam the scan actually selects is checked
    # normally when `fit_losvd` rebuilds its config with it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg_lam = replace(cfg, lam=lam, lam_selection="fixed")
    result = fit_losvd(A_stack_w, y_w, cfg_lam, initial_coeffs=initial_coeffs)
    b = result["b"]
    w = result["w"]

    v_grid = cfg_lam.build_velocity_grid()
    n_knots = cfg_lam.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg_lam.degree)
    dv = (cfg_lam.vmax - cfg_lam.vmin) / (n_knots + 1)
    D2 = difference_matrix(n_coef, order=cfg_lam.penalty_order, dv=dv, dv_ref=cfg_lam.knot_spacing_kms)

    A_eff = sum(w[k] * A_stack_w[k] for k in range(len(w)))  # (n_pix, n_bins)
    J_b = b[:, None] * basis  # (n_bins, n_coef): d(b)/d(coeffs) = diag(b) @ basis
    X = A_eff @ J_b  # (n_pix, n_coef), the local linear design in coeffs-space

    XtX = X.T @ X
    DtD = D2.T @ D2
    M = XtX + lam * DtD
    # `np.linalg.solve` raises LinAlgError if M is exactly singular -- a real,
    # reachable case at lam=0 (no regularization at all: M=XtX, which can be
    # rank-deficient whenever some basis functions' columns of J_b=diag(b)@basis
    # are ~0, e.g. b decaying to ~0 in the wings of a narrow-sigma grid).
    # `lstsq` solves the same M @ Z = XtX system via SVD -- identical to
    # `solve`'s exact answer whenever M is non-singular, and a well-defined
    # minimum-norm solution instead of a crash when it isn't.
    df_local = float(np.trace(np.linalg.lstsq(M, XtX, rcond=None)[0]))

    model_fit = A_eff @ b
    chi2 = float(np.sum((y_w - model_fit) ** 2))
    return df_local, chi2, result


def _edge_amplitude_frac(b, n_edge=5):
    """Largest LOSVD amplitude within `n_edge` bins of either end of the
    velocity grid, as a fraction of the LOSVD's own peak.

    The diagnostic for a divergent edge pedestal. A well-posed fit has
    decayed to numerically nothing here; a pedestal sits at tens of percent.
    """
    b = np.asarray(b, dtype=float)
    peak = float(np.max(b)) if b.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0 or b.size < 2 * n_edge:
        return np.inf
    return float(max(b[:n_edge].max(), b[-n_edge:].max())) / peak


def select_lambda_aic_local(A_stack_w, y_w, cfg, lam_grid, w_m=2.0, verbose=False,
                            parsimony_delta=0.0, min_resolvable_sigma_frac=0.0,
                            max_edge_amplitude_frac=0.0, warm_start_lam_grid=True,
                            self_normalize_lam_grid=False):
    """Same criterion and interface as `select_lambda_aic`, but using
    `estimate_m_eff_local`'s closed-form df instead of bootstrap -- no
    `n_boot`/`seed` needed since there's no randomness. Cross-check
    against `select_lambda_aic` before trusting this in place of it.

    `warm_start_lam_grid` : if True (the default), each grid point after
        the first is seeded (via `estimate_m_eff_local`'s `initial_coeffs`)
        from the immediately preceding (sorted) `lam` value's own converged
        coefficients, instead of cold-starting `fit_losvd`'s full
        multi-start search at every one of the 6 default grid points.
        Only the smallest `lam` still pays the full cold multi-start cost.
        `dev_notes/speed/RESULTS.md` measured the un-warm-started scan as
        ~9.5x a single fixed-`lam` fit; this closes most of that gap.
        Validated quality-neutral-to-better, not just faster
        (`dev_notes/speed/verify03_lamgrid_warmstart_ab.py`, 12 seeds x
        sigma in {50,140,300}): 3.7-4.2x faster scan, 0-1 `best_lam`
        mismatches out of 12 seeds per sigma, and recovered sigma/shape
        residual both mildly *improved*, not degraded, in every cell
        tested. Pass `False` to reproduce the original independently-
        cold-started-per-point behavior (e.g. for cross-checking a
        specific case against the pre-2026-07-30 result).

    `parsimony_delta` : if > 0, pick the *largest* `lam` whose criterion
        is within `parsimony_delta` of the grid minimum, instead of the
        exact argmin -- the Burnham & Anderson "models within delta-AIC
        of the best are indistinguishable" convention, resolved toward
        the more-regularized (more parsimonious) candidate. Default 0.0
        (exact argmin, the original behavior). This exists for the
        order-3 penalty specifically: in strongly ill-posed regimes
        (e.g. sigma ~ 20 at low S/N) the criterion goes flat (m_eff
        pins near 3 at every lam) and the argmin degenerates to a
        noise-driven lottery whose small-lam picks produce catastrophic
        delta-spike LOSVDs under a third-difference penalty (measured:
        max core shape residual 167-712% across validation seeds;
        `parsimony_delta=2.0` cut the sigma=20/S N=30 mean residual
        from 51% to 26% and the sigma=300 worst case from 45% to 15%
        with no Gaussian-truth regression -- see
        `dev_notes/REGULARIZATION_INVESTIGATION_LOG.md`, 2026-07-26).
        Do NOT enable it for the order-2 penalty: there, heavier
        smoothing actively inflates the wings (the D2 log-linearity
        mechanism), and the same tiebreak measurably *hurt* (tested
        directly on the same sweeps).

    `min_resolvable_sigma_frac` : if > 0, a candidate `lam` is excluded
        from selection when its own fit's recovered direct-moment sigma
        is below `min_resolvable_sigma_frac` x the spline knot spacing
        -- such a fit claims velocity structure narrower than the basis
        can legitimately resolve, which is the signature of the
        delta-spike collapse mode: in the strongly ill-posed sigma~20 /
        S/N~30 regime, a spike solution can "win" the AIC comparison
        outright (its chi2 gain exceeds any reasonable delta) because
        the *linearized* m_eff badly underestimates a spike's true
        flexibility. Measured on the 480-case production validation
        (`dev_notes/shape_residual/06c_*`): at 0.5 this flags exactly
        the 4 catastrophic spike fits (recovered sigma 2.4-4.4 km/s vs
        16.5 km/s knot spacing; 344-712% shape residual) and zero
        healthy fits at any sigma, either penalty order. If every
        candidate is excluded, the largest lam is used. Default 0.0
        (off, original behavior).

    `max_edge_amplitude_frac` : if > 0, a candidate `lam` is excluded when
        its fitted LOSVD still has amplitude above this fraction of its own
        peak at either end of the velocity grid. This is the *only* rule
        found that fixes the real-data sigma inflation without damaging
        mocks (`dev_notes/real_data_sigma/FINDINGS.md`, 2026-07-28).

        It is a statement about validity, not smoothness: the moment
        integrals are only meaningful if the velocity grid contains the
        LOSVD, so a fit whose density is still 20-60% of peak at the grid
        edge is being integrated over a range where it was never
        constrained. That is exactly the divergent edge pedestal that
        inflates sigma by ~2x on real NGC 5102 spectra.

        Why this and not the alternatives: AIC cannot see the pedestal,
        because on systematics-dominated residuals (chi2_red 11-16) the
        extra flexibility genuinely does buy large chi2 gains. OG
        kinextract's unimodality + jaggedness rule cannot see it either --
        measured directly, the pedestal solutions are unimodal (1 peak) and
        smooth (roughness 0.09 vs its 0.25 threshold), so that rule selects
        the same broken lam AIC does. Edge amplitude separates the two
        classes with no overlap at all: 21.9%/42.4% at the failing lam,
        0.0% at every good one.

        Crucially it is *inert* where there is no pedestal. On the NB02
        mock (+-660 km/s grid, sigma 90) every candidate passes, nothing is
        excluded, and results are unchanged -- unlike the wing-mass and
        `lam_wing` penalties, which fixed real data only at strengths that
        destroyed mocks. If every candidate is excluded, the largest lam is
        used. Default 0.0 (off); `FitConfig.max_edge_amplitude_frac`
        supplies 0.01 on the mainline path.

    `self_normalize_lam_grid` : if True, rescale `lam_grid` by
        `(dv/cfg.knot_spacing_kms)^2` (`dv` = this fit's actual knot
        spacing, from `cfg.vmin`/`cfg.vmax`/`cfg.resolved_n_interior_knots()`)
        before scanning -- cancels the `(knot_spacing_kms/dv)^2` factor
        `difference_matrix` applies to the penalty operator, so the
        *effective* smoothing strength explored stays centered on the
        validated range regardless of knot spacing. See
        `FitConfig.self_normalize_lam_grid`'s own help text for why this
        defaults to False (changes `lam_used`'s numeric meaning on narrow
        grids, which the existing bias-correction calibration tables don't
        yet account for).
    """
    from .gh_moments import compute_direct_moments

    lam_grid = np.asarray(lam_grid, dtype=float)
    if self_normalize_lam_grid:
        n_knots = cfg.resolved_n_interior_knots()
        dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
        lam_grid = lam_grid * (dv / cfg.knot_spacing_kms) ** 2
    order = np.argsort(lam_grid)
    lam_grid = lam_grid[order]
    chi2_arr = np.empty(len(lam_grid))
    m_eff_arr = np.empty(len(lam_grid))
    fits = [None] * len(lam_grid)
    valid = np.ones(len(lam_grid), dtype=bool)
    reason = [""] * len(lam_grid)
    if min_resolvable_sigma_frac > 0.0:
        v_grid = cfg.build_velocity_grid()
        dv_knot = (cfg.vmax - cfg.vmin) / (cfg.resolved_n_interior_knots() + 1)
    prev_coeffs = None
    for i, lam in enumerate(lam_grid):
        seed = prev_coeffs if warm_start_lam_grid else None
        m_eff, chi2, result = estimate_m_eff_local(A_stack_w, y_w, cfg, lam, initial_coeffs=seed)
        chi2_arr[i] = chi2
        m_eff_arr[i] = m_eff
        fits[i] = result
        if warm_start_lam_grid:
            prev_coeffs = result["coeffs"]
        if min_resolvable_sigma_frac > 0.0:
            sherm = compute_direct_moments(v_grid, result["b"])["sherm"]
            if not (np.isfinite(sherm) and sherm >= min_resolvable_sigma_frac * dv_knot):
                valid[i] = False
                reason[i] = "sub-knot-scale sigma"
        if max_edge_amplitude_frac > 0.0 and valid[i]:
            edge = _edge_amplitude_frac(result["b"])
            if not (np.isfinite(edge) and edge <= max_edge_amplitude_frac):
                valid[i] = False
                reason[i] = f"edge pedestal {edge * 100:.1f}% of peak"
        if verbose:
            print(f"  lam={lam:8.4g}  chi2={chi2:10.2f}  m_eff_local={m_eff:8.2f}  "
                  f"criterion={chi2 + w_m * m_eff:10.2f}"
                  + ("" if valid[i] else f"  [excluded: {reason[i]}]"), flush=True)
    criterion = chi2_arr + w_m * m_eff_arr
    if not valid.any():
        # every candidate is either a sub-knot spike or still pedestalled at the
        # grid edge: fall back to the heaviest smoothing, which is the least bad
        # of both failure modes (and, for a pedestal, usually means the velocity
        # grid itself is too narrow for this spectrum).
        best_idx = len(lam_grid) - 1
    else:
        crit_valid_min = criterion[valid].min()
        if parsimony_delta > 0.0:
            within = np.flatnonzero(valid & (criterion <= crit_valid_min + parsimony_delta))
            best_idx = int(within[-1])  # largest lam among the indistinguishable set
        else:
            candidates = np.flatnonzero(valid & (criterion == crit_valid_min))
            best_idx = int(candidates[0])

    best_fit = fits[best_idx]
    # The warm-started scan seeds every point after the first from its neighbour's
    # converged coeffs (see the loop above), so the WINNING point's own fit is only
    # a genuine, from-scratch multi-start result if it happened to be lam_grid[0] --
    # every other winner inherited its starting basin rather than searching for one,
    # so its own `losvd_objective_gap` (the "did multi-start actually matter here"
    # diagnostic `fit_log_spline_map` reports) is None, not a real gap. Confirm the
    # winner with one fresh cold fit (full multi-start) so that diagnostic -- and the
    # returned `best_fit` itself -- reflects an actual multi-start result, not a
    # warm-started shortcut, for the one candidate callers actually go on to use.
    # Costs at most one extra full fit per selection call (not one per grid point),
    # so this keeps most of `warm_start_lam_grid`'s own speedup.
    #
    # Guarded on `not cfg.losvd_single_start`: when single-start is active (e.g.
    # `bootstrap_losvd`'s own replicates, `replicate_single_start=True` by default),
    # `_alternate_once` ignores `initial_coeffs`/warm-starting entirely and always
    # uses the same deterministic ridge start -- `losvd_objective_gap` is *always*
    # None there regardless of warm-starting, since there is no multi-start
    # ambiguity to report in the first place. Without this guard, every single-
    # start bootstrap replicate's lam-selection call redundantly re-fit its own
    # winning lam a second time for literally no benefit -- found and fixed the
    # same day the confirmation logic was added, before it shipped in that state.
    if (warm_start_lam_grid and not cfg.losvd_single_start
            and best_fit is not None and best_fit.get("losvd_objective_gap") is None):
        m_eff_confirm, chi2_confirm, best_fit_confirm = estimate_m_eff_local(
            A_stack_w, y_w, cfg, float(lam_grid[best_idx]), initial_coeffs=None)
        chi2_arr[best_idx] = chi2_confirm
        m_eff_arr[best_idx] = m_eff_confirm
        criterion[best_idx] = chi2_confirm + w_m * m_eff_confirm
        best_fit = best_fit_confirm

    return AICSelectionResult(
        best_lam=float(lam_grid[best_idx]), lam_grid=lam_grid,
        chi2=chi2_arr, m_eff=m_eff_arr, criterion=criterion, w_m=w_m,
        best_fit=best_fit,
    )


def select_lambda_aic(A_stack_w, y_w, cfg, lam_grid, w_m=2.0, n_boot=20, seed=0, verbose=False, L=None):
    """Scan `lam_grid`, compute `chi^2 + w_m*m_eff` at each value (Lipka
    & Thomas 2021 eq. 14; `w_m=2.0` is their calibrated final choice --
    eq. 13 / Thomas & Lipka 2022), and return an `AICSelectionResult`
    with the minimizing `best_lam`.

    `L` : passed straight through to `estimate_m_eff` -- see its own
    docstring; not needed for the mainline correlated-noise case, only for
    deliberately testing an assumed-vs-true covariance mismatch.
    """
    lam_grid = np.asarray(lam_grid, dtype=float)
    chi2_arr = np.empty(len(lam_grid))
    m_eff_arr = np.empty(len(lam_grid))
    for i, lam in enumerate(lam_grid):
        m_eff, chi2, _ = estimate_m_eff(A_stack_w, y_w, cfg, lam, n_boot=n_boot, seed=seed + i, L=L)
        chi2_arr[i] = chi2
        m_eff_arr[i] = m_eff
        if verbose:
            print(f"  lam={lam:8.4g}  chi2={chi2:10.2f}  m_eff={m_eff:8.2f}  "
                  f"criterion={chi2 + w_m * m_eff:10.2f}", flush=True)
    criterion = chi2_arr + w_m * m_eff_arr
    best_idx = int(np.argmin(criterion))
    return AICSelectionResult(
        best_lam=float(lam_grid[best_idx]), lam_grid=lam_grid,
        chi2=chi2_arr, m_eff=m_eff_arr, criterion=criterion, w_m=w_m,
    )
