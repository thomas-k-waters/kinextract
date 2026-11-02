"""LOSVD fitting robust to unmodeled contamination (sky-subtraction
residuals, cosmic rays, an under-scaled noise vector): `fit_losvd_robust`.

**`cleaning_method="inflate"` (the default) -- soft, per-wavelength error
inflation.** On real MUSE data,
real chi2_red commonly runs well above 1 (11-28 measured on one target,
`dev_notes/real_data_sigma`) -- i.e. the input noise vector is
*structurally* under-scaled at many wavelengths, not just a handful of
genuinely bad pixels, which a hard sigma-clip cannot fix and which made a
plain clip worse on real data (below; the clip mode has been removed). A
hard clip can also make a model bias worse: the pixels that disagree most
with an already-biased fit are often the ones correctly arguing against
it. Variance rescaling comes first; clipping is kept only as loose cleanup
for genuine one-off outliers. Algorithm, per bin/spectrum (not a blanket wavelength
list shared across every spectrum -- sky-residual amplitude scales with a
bin's own S/N and falls off with radius):

1. Preliminary fit with a loose, cleanup-only clip (catches genuine
   single-pixel spikes -- cosmic-ray-like outliers -- without doing the
   heavy lifting), at the fit's `lam`.
2. Whitened residual at *every* pixel (not just the surviving subset) from
   that fit's own effective model.
3. A running (median-filtered) robust local scale of `|residual|` in
   wavelength -- smooths past single-pixel shot noise while preserving
   sky-line-width features (OH residuals are a few pixels wide at MUSE's
   resolution, not single-pixel).
4. That local scale becomes a per-wavelength inflation factor on the
   *input* error array (floored at 1 -- never deflate; capped, so a single
   badly-contaminated window is downweighted, not zeroed out entirely).
5. Re-whiten with the inflated errors, refit -- this final fit is the one
   returned. A second loose clip pass on top is still applied (cleanup,
   not primary defense, per the hierarchy above).

The whole procedure runs under `cfg.lam_rule` (`multi_template.
with_lam_rule`): with the default `'sigma_scaled'`, broad LOSVDs and fits
still at the grid edge are redone, inflation and all, at a larger `lam`.
No change to the fitting engine itself: this wraps `whiten` + the
internal clip loop, so it composes with continuum-cofitting, AR(1)
whitening and parallel per-bin fitting unchanged.

**Validation (`cleaning_method="inflate"`).** Developed and validated on real
NGC 4751 MUSE NFM data (the raw ESO-pipeline cube -- ZAP was tried and
ruled out for this target: masked-source self-calibration still destroys
signal at the galactic center on a galaxy-filling field with no dedicated
sky exposures or blank-sky regions, consistent with Soto et al. 2016's own
stated requirement for either). Full record:
`dev_notes/variance_inflation/FINDINGS.md`. Headline full-field numbers
(113 bins), inflate vs. a hard 3-sigma clip:

- Median whitened residual RMS: 1.41 -> 0.96 (ideal is ~1).
- Bins with more than 10 % of pixels clipped: 22/113 -> 0/113.
- Recovered kinematics (V, sigma, h3, h4) track the clip mode closely at
  every radius -- inflation changes how well-calibrated the fit's
  residuals are and how much of the spectrum survives to inform it, not
  the recovered physics.

**Open, not yet resolved**: a small-sample (18-trial) injection-recovery
test did *not* show a clear reduction in absolute kinematic bias between
the two -- both carried a comparable, tens-of-km/s bias attributed
to factors other than sky-residual handling (regularization choice,
template mismatch). Treat inflation as a calibration/robustness
improvement, not a demonstrated accuracy improvement, until a larger
injection-test sample says otherwise. Also open: the per-spectrum
inflation profile is not narrowly confined to a few sky-line wavelengths
at low S/N/high radius -- it behaves as a general adaptive variance
recalibration there, not a pure sky-line isolator.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from scipy.ndimage import median_filter

from .cleaning import CAII_PROTECTED_WINDOWS, _failed_fit_result, _fit_losvd_clip
from .config import FitConfig
from .losvd import build_bspline_basis
from .noise_model import whiten

_PRELIM_CLIP_SIGMA = 5.0   # loose -- cleanup only, not primary defense
_PRELIM_MAX_ITER = 1
_FINAL_CLIP_SIGMA = 5.0    # loose -- see module docstring, inflate step 5
_FINAL_MAX_ITER = 2


def reconstruct_continuum(result: dict, wave_fit: np.ndarray, cfg: FitConfig):
    """The fitted continuum `C(wave_fit)`, evaluated at *every* pixel in
    `wave_fit` -- not `result["C"]` directly, which is only as long as
    whatever pixel subset survived the hard-clip loop's own last
    iteration (a shape mismatch against a full-length array whenever
    *any* clipping happened). Rebuilds the continuum spline's basis on
    the full `wave_fit` from `result["continuum_coeffs"]` instead (the
    coefficients themselves have no length dependence on the kept
    subset, only the basis they're evaluated against does).

    Returns a `(len(wave_fit),)` array, or the scalar `1.0` if
    `cfg.continuum_knots` is `None` (no continuum-cofitting) -- either way,
    directly usable as a multiplicative factor against any model built on
    this same `wave_fit`, including one using a *different* LOSVD `b` than
    `result["b"]` (e.g. a known injected LOSVD sharing this bin's own real
    template mixture/continuum, for an injection-recovery test).
    """
    continuum_coeffs = result.get("continuum_coeffs")
    if continuum_coeffs is None:
        return 1.0
    continuum_basis_full, _ = build_bspline_basis(wave_fit, n_interior_knots=cfg.continuum_knots, degree=3)
    return np.exp(continuum_basis_full @ continuum_coeffs)


def full_pixel_model(result: dict, A_stack_w: np.ndarray, wave_fit: np.ndarray, cfg: FitConfig) -> np.ndarray:
    """The fitted model evaluated at *every* pixel in `wave_fit`/`A_stack_w`
    (not just whatever subset survived the hard-clip loop's own last
    iteration) -- see `reconstruct_continuum` for why this isn't simply
    `result["C"] * (A_eff @ result["b"])`."""
    A_eff_full = np.tensordot(result["w"], A_stack_w, axes=([0], [0]))
    C_full = reconstruct_continuum(result, wave_fit, cfg)
    return C_full * (A_eff_full @ result["b"])


def compute_local_inflation(wave_fit: np.ndarray, resid_w: np.ndarray,
                             smooth_window_A: float = 8.0, cap: float = 8.0) -> np.ndarray:
    """The per-wavelength inflation factor itself, exposed separately from
    `fit_losvd_robust` so it can be inspected/plotted directly (e.g.
    against a real airglow line list, as a validation cross-check).

    Parameters
    ----------
    wave_fit : wavelength array (Å, any consistent frame) the residual
        corresponds to.
    resid_w : whitened residual (`y_w - model_w`) at every pixel in
        `wave_fit` -- **at every pixel, not just a clipping survivor
        subset**: a clipped pixel's own wavelength is exactly where
        inflation is most needed, so it must contribute to the local
        scale estimate, not be silently excluded from it.
    smooth_window_A : median-filter window width (Å), converted internally
        to an odd pixel count from `wave_fit`'s own median spacing.
        Default 8.0 Å -- a few times a MUSE OH-residual's own width at
        this instrument's resolution, wide enough to smooth single-pixel
        shot noise, narrow enough not to blur two nearby-but-distinct
        contaminated windows into one.
    cap : maximum inflation factor. Default 8.0 -- large enough to
        substantially downweight a badly-contaminated window, small enough
        that pixels there still contribute rather than being effectively
        zero-weighted ("capped, not erased" -- preferring this over a hard
        mask).

    Returns
    -------
    inflation : ndarray, same shape as `wave_fit`. `1.0` where the local
        residual scale is already <=1 (nothing to inflate); up to `cap`
        where it's high.
    """
    wave_fit = np.asarray(wave_fit, float)
    resid_w = np.asarray(resid_w, float)
    dw = float(np.median(np.diff(np.sort(wave_fit))))
    win_px = max(3, int(round(smooth_window_A / dw)))
    if win_px % 2 == 0:
        win_px += 1
    # 1.4826x converts a local median-absolute-deviation-like amplitude into a
    # normal-consistent sigma estimate, same convention as `cleaning._robust_sigma`.
    local_scale = 1.4826 * median_filter(np.abs(resid_w), size=win_px, mode="nearest")
    return np.clip(local_scale, 1.0, cap)


def _normalize_losvd_area(result: dict, v_grid: np.ndarray) -> dict:
    """Rescale `result["b"]`/`result["w"]` in place so `b` is unit-area
    (matching the Laplace sampler's own convention --
    `_sample_curves_and_moments` unit-area-normalizes every posterior
    draw), without changing the fit itself.

    `fit_log_spline_map` (`losvd.py`) returns `b=np.exp(basis @ coeffs)`
    with no area normalization -- an arbitrary byproduct of the `(w, b)
    -> (c*w, b/c)` scale non-identifiability this module's own docstring
    already describes (the template weights and the LOSVD amplitude trade
    off an overall scale the data alone can't pin down). Found directly on
    real N4751 data: a single-spaxel bin's `result["b"]` had area=331.6,
    not ~1 -- 300+x off from every uncertainty-ensemble curve
    (`b_samples`/`b_median`/`b_p16`/`b_p84`) computed for the exact same
    bin, which are already correctly unit-area-normalized. The point
    estimates (V/sigma/h3/h4) themselves were unaffected (`compute_losvd_
    moments` normalizes internally), but `result["b"]` was silently on a
    different footing than everything else in a bin's own result -- any
    future code plotting or otherwise comparing it directly against the
    uncertainty ensemble would look wildly (and misleadingly) inconsistent
    for a reason that has nothing to do with the actual fit.

    Rescaling `w` by the reciprocal factor is required, not optional --
    `tensordot(w, A_stack) @ b` (the actual reconstructed model spectrum,
    used throughout this module and in `laplace_uncertainty.py`/
    `nuts_uncertainty.py`/`cleaning.py`) is bilinear in `(w, b)`, so
    `w -> w*area, b -> b/area` leaves it mathematically identical, not
    approximately so -- this is a relabeling of the same fit, not a
    different one. Every existing consumer of `result["w"]` either pairs
    it with `result["b"]` this same way (safe) or normalizes it by its own
    sum first (`w / w.sum()`, already scale-invariant) -- confirmed by
    checking every call site in this package before adding this function.
    """
    area = np.trapezoid(np.maximum(result["b"], 0.0), v_grid)
    if area > 0:
        result["b"] = result["b"] / area
        result["w"] = result["w"] * area
    return result


def fit_losvd_robust(A_stack: np.ndarray, flux_fit: np.ndarray, errors: np.ndarray, cfg: FitConfig,
                      wave_fit: np.ndarray | None = None, cleaning_method: str | None = "inflate",
                      smooth_window_A: float = 3.0, inflation_cap: float = 100.0, n_passes: int = 1,
                      prelim_clip_sigma: float = _PRELIM_CLIP_SIGMA,
                      final_clip_sigma: float = _FINAL_CLIP_SIGMA,
                      final_max_iter: int = _FINAL_MAX_ITER,
                      excluded_wavelength_windows: tuple[tuple[float, float], ...] | None = CAII_PROTECTED_WINDOWS,
                      measure_rho: bool = False,
                      min_inflation_to_apply: float | None = 1.1) -> dict:
    """LOSVD (+ template-mixture-weight) fit with per-wavelength error
    inflation against unmodeled contamination -- see this module's own
    docstring for the method -- under `cfg.lam_rule`.

    Always takes **unwhitened** `flux_fit`/`errors` (plus `A_stack`, also
    unwhitened) and whitens internally, per pass (the inflation needs the
    original `errors`). The defaults are the settings validated on the
    NGC 4751 MUSE data (dev_notes/regularization_v2/FINDINGS.md).

    Parameters
    ----------
    A_stack : unwhitened per-template design-matrix stack, shape
        `(n_templates, n_pix, n_bins)` (`kinextract.forward_model.
        build_template_stack`'s own output).
    flux_fit, errors : unwhitened observed flux and per-pixel error,
        shape `(n_pix,)`.
    cfg : `FitConfig` -- `cfg.rho` is used for whitening; `cfg.continuum_knots`
        turns on continuum-cofitting orthogonally to `cleaning_method`, exactly
        as for `kinextract.multi_template.fit_losvd`.
    wave_fit : the observed wavelength grid `A_stack`/`flux_fit`/`errors`
        correspond to (rest frame, if `excluded_wavelength_windows` are rest
        frame). Required for `cleaning_method="inflate"`
        (`compute_local_inflation` needs it) and for continuum-cofitting
        (`cfg.continuum_knots`, on by default).
    cleaning_method : `"inflate"` (default, see module docstring) or `None`
        `None` is one plain fit with no cleaning -- a quick check, and the
        preliminary fit `noise_model_errors` measures residuals from.
    smooth_window_A, inflation_cap, n_passes, prelim_clip_sigma, final_clip_sigma, final_max_iter : float, int
        `"inflate"` mode only -- `smooth_window_A` (default 3 A, a few MUSE sky-line widths) and
        `inflation_cap` (default 100: a badly contaminated window is
        down-weighted up to 100x, not zeroed) are forwarded to
        `compute_local_inflation`; `n_passes` re-derives the residual/
        inflation from the just-inflated fit and repeats, this many times
        total (default 1 -- the point of this mode is to soften the
        primary defense, not converge to a fixed point; set >1 only if a
        single pass leaves a visibly under-inflated region on
        inspection); `prelim_clip_sigma`/`final_clip_sigma`/`final_max_iter`
        are loose cleanup-only clip settings for the two internal clip
        passes -- not the primary defense, do not tune these down to do
        the inflation's job.
    excluded_wavelength_windows : optional `(lo, hi)` rest-frame windows
        (e.g. `kinextract.cleaning.CAII_TRIPLET_CENTERS_A` +/-
        `CAII_TRIPLET_HALF_WIDTH_A`) never clipped, regardless of mode or
        how large their residual is. **Without this, a real, narrow (1-2
        pixel) absorption core the model doesn't perfectly reproduce is
        indistinguishable from a cosmic ray to a clip pass, and gets
        silently removed** -- `compute_local_inflation`'s own
        median-filtered local scale is deliberately robust to exactly
        this kind of narrow, single-feature residual (so a genuine
        multi-pixel sky-residual bump still gets inflated, not diluted by
        the smoothing window), which means it does not detect or protect
        a narrow real absorption line either; only this parameter does.
        Confirmed directly on real NGC 5102 bins (`dev_notes/
        real_data_sigma/FINDINGS.md`'s own CaII-clipping investigation,
        2026-07-31, plus this package's real full-galaxy runs): every
        clipped pixel in several real bins landed within 1-2 pixels of a
        CaII triplet center and nowhere else, with `inflation_factor`
        never engaging at all -- i.e. clipping was the *only* contamination
        defense active for those bins, and it was removing the actual
        absorption feature being measured, not contamination. Default
        `CAII_PROTECTED_WINDOWS` (the three CaII triplet cores, rest frame;
        inert for spectra that do not cover them). Pass `None` to protect
        nothing, or your own windows for other strong lines.
    measure_rho : if True, measure *this bin's own* pixel correlation
        (`noise_model.estimate_correlation_length`'s lag-1 ACF, same
        method `05_noise_correlation.ipynb` uses) from a preliminary
        `rho=0`-whitened fit, and use that measured value in place of
        `cfg.rho` for every whitening step in this call -- instead of
        trusting one `cfg.rho` measured once (typically on a single
        high-S/N representative spectrum) and applied uniformly across
        bins spanning a much wider S/N range. Confirmed directly on real
        NGC 5102 bins that rho is S/N-dependent (0.68-0.89 across
        representative high/med/low-S/N apertures) -- see
        `dev_notes/uncertainty/FINDINGS.md`'s 2026-08-07/08-08 entries.
        Costs one extra continuum-off preliminary fit (a residual good
        enough for a lag-1 ACF estimate doesn't need an accurate continuum,
        and skipping it avoids a third continuum-cofit pass per bin).
        Measured at each `lam` the lam rule tries. Default False (use
        `cfg.rho` as given). NGC 4751 measured it for
        bins of 20 or more spaxels and used a fixed fallback below that,
        where the residual is too noisy for the ACF.
    min_inflation_to_apply : `"inflate"` mode only. If given, the
        preliminary pass's own local inflation profile
        (`compute_local_inflation`'s output, 1.0-floored by construction
        -- rises only where a real local excess exists) is checked
        before refitting; if its *max* is `<= min_inflation_to_apply`
        (nothing anywhere needs inflating), the refit is skipped and
        this bin gets a plain final fit, with `inflation_factor` all-ones and
        `errors_inflated == errors`. Guards against stacking inflation
        on top of an already well-calibrated error budget for bins with
        no local contamination at all -- confirmed directly on real
        NGC 5102 bins: chi2/dof 0.08-0.13 with unconditional inflation
        on top of `rho=0.88` vs. 0.4-1.3 without it (see
        `dev_notes/real_data_sigma/FINDINGS.md`'s 2026-08-08 entries).
        **Deliberately not gated on a global chi2/dof average** -- an
        earlier version of this parameter did that and was wrong: a bin
        can look fine on average (global chi2/dof ~0.7-0.9) while still
        having a real, localized contaminated window (confirmed
        directly: 12-14% of pixels with local `inflation_factor > 1.2`,
        up to 3x, on bins whose global chi2/dof looked fine) -- a global
        gate wrongly skipped inflation there entirely, pushing that same
        contamination onto the loose final cleanup clip instead, which
        then clipped 15% of pixels, not the few percent "cleanup" is
        supposed to mean. Default 1.1 (the NGC 4751 setting); `None`
        always inflates. `chi2_dof_prelim` is always computed and
        returned regardless of this parameter, as a diagnostic.

    Returns
    -------
    The underlying `fit_losvd` result dict (`b`, `w`, `coeffs`, `success`,
    `lam_used`, `lam_tried`, ...), with the same extra keys regardless of
    `cleaning_method` so callers never need to branch on mode to read a
    common field: `clipped_mask` (bool array, all `False` when
    `cleaning_method is None`), `n_clipped`, `n_iterations`,
    `inflation_factor` (the final pass's per-pixel inflation factor, all ones
    when `cleaning_method is None`) and `errors_inflated` (the errors the
    final fit used). `rho_used` is always present -- `cfg.rho` as given,
    unless `measure_rho=True` replaced it with this bin's own measured value;
    re-whiten with *this*, not the original `cfg.rho`, in any downstream
    step that needs to reproduce the exact fit (e.g. the Laplace
    uncertainty -- `losvd_error_bands` does this for you). `chi2_dof_prelim` (`cleaning_method="inflate"` only) is the
    preliminary pass's own global whitened chi2/dof, always present --
    a diagnostic only, not what `min_inflation_to_apply` gates on.
    """
    if cleaning_method not in (None, "inflate"):
        raise ValueError(f"fit_losvd_robust: cleaning_method must be 'inflate' or None (got {cleaning_method!r}).")
    if cleaning_method == "inflate" and wave_fit is None:
        raise ValueError("fit_losvd_robust: wave_fit is required for cleaning_method='inflate' -- "
                          "compute_local_inflation needs a wavelength grid.")
    from .multi_template import with_lam_rule

    A_stack = np.asarray(A_stack, float)
    flux_fit = np.asarray(flux_fit, float)
    errors = np.asarray(errors, float)
    wave_fit_arr = np.asarray(wave_fit, float) if wave_fit is not None else None
    v_grid = cfg.build_velocity_grid()

    def fit_at(c):
        return _fit_at_lam(A_stack, flux_fit, errors, c, wave_fit_arr, cleaning_method, smooth_window_A,
                           inflation_cap, n_passes, prelim_clip_sigma, final_clip_sigma, final_max_iter,
                           excluded_wavelength_windows, min_inflation_to_apply, v_grid, measure_rho)

    return with_lam_rule(fit_at, cfg)


def _fit_at_lam(A_stack, flux_fit, errors, cfg, wave_fit_arr, cleaning_method, smooth_window_A, inflation_cap,
                n_passes, prelim_clip_sigma, final_clip_sigma, final_max_iter, excluded_wavelength_windows,
                min_inflation_to_apply, v_grid, measure_rho) -> dict:
    """`fit_losvd_robust`'s procedure at exactly `cfg.lam` (`cfg.lam_rule == 'fixed'`). With `measure_rho`, the
    pixel correlation is measured at this `lam` -- every fit the lam rule makes re-measures it."""
    if measure_rho:
        from .noise_model import estimate_correlation_length
        # continuum_knots=None here regardless of cfg's own setting -- this pass only
        # needs a residual good enough for a lag-1 ACF estimate, not an accurate
        # continuum. Confirmed real cost for continuum-cofitting targets (N4751): this
        # is a third continuum-cofit pass per bin on top of the inflate mode's own two
        # (preliminary + final) if left on.
        rho0_cfg = dataclasses.replace(cfg, continuum_knots=None)
        A_stack_w0, y_w0, _ = whiten(A_stack, flux_fit, errors, rho=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rho0_result = _fit_losvd_clip(A_stack_w0, y_w0, rho0_cfg, clip_sigma=_PRELIM_CLIP_SIGMA,
                                           max_iter=_PRELIM_MAX_ITER, wave_fit=wave_fit_arr, robust_scale=True,
                                           max_clip_fraction=1.0,
                                           excluded_wavelength_windows=excluded_wavelength_windows)
        resid_w0 = y_w0 - full_pixel_model(rho0_result, A_stack_w0, wave_fit_arr, rho0_cfg)
        cfg = dataclasses.replace(cfg, rho=max(0.0, float(estimate_correlation_length(resid_w0))))

    if cleaning_method == "inflate":
        errors_current = errors.copy()
        inflation_factor = np.ones_like(errors)
        result = None
        chi2_dof_prelim = None

        for _pass in range(max(1, n_passes)):
            A_stack_w, y_w, _L = whiten(A_stack, flux_fit, errors_current, rho=cfg.rho)
            pass_clip_sigma = prelim_clip_sigma if _pass == 0 else final_clip_sigma
            pass_max_iter = _PRELIM_MAX_ITER if _pass == 0 else final_max_iter
            result = _fit_losvd_clip(A_stack_w, y_w, cfg, clip_sigma=pass_clip_sigma,
                                      max_iter=pass_max_iter, wave_fit=wave_fit_arr, robust_scale=True,
                                      max_clip_fraction=1.0,  # never warn here -- inflation is the point
                                      excluded_wavelength_windows=excluded_wavelength_windows)

            model_w_full = full_pixel_model(result, A_stack_w, wave_fit_arr, cfg)
            resid_w_full = y_w - model_w_full

            if _pass == 0:
                dof = max(len(resid_w_full) - A_stack.shape[0] - 5, 1)
                chi2_dof_prelim = float(np.sum(resid_w_full ** 2)) / dof
            inflation_factor = compute_local_inflation(wave_fit_arr, resid_w_full,
                                                         smooth_window_A=smooth_window_A, cap=inflation_cap)

            if (_pass == 0 and min_inflation_to_apply is not None
                    and float(inflation_factor.max()) <= min_inflation_to_apply):
                # Gate on the *local* inflation profile itself, not a global chi2/dof average --
                # a bin can look fine on average (global chi2/dof ~1) while still having a real,
                # localized contaminated window (confirmed directly on real NGC 5102 bins: global
                # chi2/dof 0.72-0.86, "fine" by a global-average gate, yet 12-14% of pixels still
                # had local inflation_factor > 1.2, up to 3x -- a global gate wrongly skipped
                # inflation for those bins entirely, pushing that same real contamination onto
                # the loose final cleanup clip instead, which clipped 15% of pixels, not the few
                # percent "cleanup" is supposed to mean). Only skip when *nothing* anywhere needs
                # inflating (`inflation_factor` is 1.0-floored by construction, so its max is the
                # single number that actually answers "is there anything to inflate").
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    final_result = _fit_losvd_clip(A_stack_w, y_w, cfg, clip_sigma=final_clip_sigma,
                                                    max_iter=final_max_iter, wave_fit=wave_fit_arr,
                                                    robust_scale=True,
                                                    excluded_wavelength_windows=excluded_wavelength_windows)
                final_result["inflation_factor"] = np.ones_like(errors)
                final_result["errors_inflated"] = errors.copy()
                final_result["chi2_dof_prelim"] = chi2_dof_prelim
                final_result["rho_used"] = cfg.rho
                return _normalize_losvd_area(final_result, v_grid)

            errors_current = errors * inflation_factor

        # Final fit against the last pass's inflated errors -- the loose clip here is cleanup
        # only (per-pixel outliers the smoothed local inflation wouldn't catch, e.g. a single
        # cosmic ray), not the primary defense; if it fires often, `inflation_cap`/
        # `smooth_window_A` need retuning, not this threshold.
        A_stack_w, y_w, _L = whiten(A_stack, flux_fit, errors_current, rho=cfg.rho)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # max_clip_fraction sanity warning -- expected quiet now
            result = _fit_losvd_clip(A_stack_w, y_w, cfg, clip_sigma=final_clip_sigma, max_iter=final_max_iter,
                                      wave_fit=wave_fit_arr, robust_scale=True,
                                      excluded_wavelength_windows=excluded_wavelength_windows)
        result["inflation_factor"] = inflation_factor
        result["errors_inflated"] = errors_current
        result["rho_used"] = cfg.rho
        result["chi2_dof_prelim"] = chi2_dof_prelim
        return _normalize_losvd_area(result, v_grid)

    # cleaning_method is None -- plain fit, no inflate loop, but the same result-dict shape
    # so callers never need to branch on mode to read a common field.
    from .multi_template import fit_losvd
    A_stack_w, y_w, _L = whiten(A_stack, flux_fit, errors, rho=cfg.rho)
    result = fit_losvd(A_stack_w, y_w, cfg, wave_fit=wave_fit_arr)
    result["clipped_mask"] = np.zeros(A_stack.shape[1], dtype=bool)
    result["n_clipped"] = 0
    result["n_iterations"] = 0
    result["n_flagged_not_refit"] = 0
    result["clip_converged"] = True
    result["inflation_factor"] = np.ones_like(errors)
    result["errors_inflated"] = errors.copy()
    result["rho_used"] = cfg.rho
    return _normalize_losvd_area(result, v_grid)


def _fit_losvd_robust_worker(args):
    """Top-level (picklable) per-bin worker for `fit_losvd_robust_parallel` --
    process pools need a module-level function, not a closure, since
    workers unpickle it by import path (same reason as `multi_template.
    _fit_losvd_worker`). Catches any exception from this bin's own fit
    and returns `_failed_fit_result` instead of propagating it, so a
    single bad bin cannot crash an entire parallel batch."""
    (A_stack, flux_fit, errors, cfg, wave_fit, kwargs) = args
    try:
        return fit_losvd_robust(A_stack, flux_fit, errors, cfg, wave_fit=wave_fit, **kwargs)
    except Exception as e:
        warnings.warn(
            f"fit_losvd_robust_parallel: a bin's fit raised {type(e).__name__}: {e} -- "
            "returning a NaN-filled, success=False result for this bin instead of "
            "crashing the whole batch. Inspect this bin's own input data (a common real "
            "cause: sky-subtraction residuals washing out the real signal at low S/N).",
            UserWarning, stacklevel=2,
        )
        return _failed_fit_result(A_stack, f"{type(e).__name__}: {e}")


def fit_losvd_robust_parallel(A_stack_list, flux_fit_list, errors_list, cfg_list,
                               wave_fit_list=None, n_jobs: int | None = None, **kwargs) -> list:
    """`fit_losvd_robust`, run across many independent bins in parallel --
    same `ProcessPoolExecutor`/`spawn`/memory-aware-worker-count pattern as
    `kinextract.multi_template.fit_losvd_parallel`; see that function's own
    docstring for the threading/worker-startup details.

    Parameters
    ----------
    A_stack_list, flux_fit_list, errors_list : sequences of the same
        length, one entry per bin, each matching `fit_losvd_robust`'s own
        unwhitened `A_stack`/`flux_fit`/`errors`.
    cfg_list : a single `FitConfig` (reused for every bin) or a sequence of
        one per bin.
    wave_fit_list : forwarded per bin -- a single array (shared) or a
        sequence of one per bin. Required for the default inflate cleaning
        and continuum-cofitting -- see `fit_losvd_robust`'s own docstring.
    n_jobs : worker process count -- `None` (default) uses
        `kinextract.multi_template.resolve_n_jobs`'s own memory-aware
        default; `1` disables the process pool and runs serially in-process.
    **kwargs : forwarded to every bin's own `fit_losvd_robust` call
        unchanged (`cleaning_method`, `smooth_window_A`, `inflation_cap`,
        `excluded_wavelength_windows`, ...).

    Returns
    -------
    list of `fit_losvd_robust`'s own result dicts, one per bin, in the
    same order as the inputs.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from .multi_template import _limit_worker_threads, resolve_n_jobs, single_threaded_worker_env

    n_bins = len(A_stack_list)
    cfgs = cfg_list if isinstance(cfg_list, (list, tuple)) else [cfg_list] * n_bins
    waves = wave_fit_list if isinstance(wave_fit_list, (list, tuple)) else [wave_fit_list] * n_bins
    if len(waves) != n_bins:
        raise ValueError(f"wave_fit_list, given as a sequence, must have one entry per bin "
                          f"(got {len(waves)} for {n_bins} bins).")

    jobs = [(A_stack_list[i], flux_fit_list[i], errors_list[i], cfgs[i], waves[i], kwargs)
            for i in range(n_bins)]

    if n_jobs == 1:
        return [_fit_losvd_robust_worker(job) for job in jobs]

    per_job_bytes = 4 * max((a.nbytes for a in A_stack_list), default=0)
    n_jobs = resolve_n_jobs(n_jobs, per_job_bytes)
    with single_threaded_worker_env(), ProcessPoolExecutor(max_workers=n_jobs, mp_context=get_context("spawn"),
                                                           initializer=_limit_worker_threads) as pool:
        return list(pool.map(_fit_losvd_robust_worker, jobs))
