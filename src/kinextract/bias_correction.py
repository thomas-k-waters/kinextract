"""Pinkney et al. (2003)-style regularization-bias correction, calibrated
from synthetic data with known truth, as prescribed by the literature this
package's whole MPL/Tikhonov-regularized lineage traces back to (Saha &
Williams 1994 -> Gebhardt et al. 2000b -> Pinkney et al. 2003): the
roughness penalty biases recovered kinematics (worse at low S/N and, per
this package's own Stage 1 sweep, also at high sigma/low S/N), and naive
Monte Carlo/bootstrap intervals built around a biased point estimate do
not, by themselves, reveal or correct that bias -- it has to be estimated
separately from synthetic data with known truth and applied explicitly.

**Status (2026-07): opt-in, not the default, and not yet trustworthy enough
for unsupervised use on real data.** `uncertainty.bootstrap_losvd`'s
`apply_bias_correction` now defaults to `False` -- two distinct
overcorrection failure modes were found under real testing in short
succession (see `bootstrap_losvd`'s own docstring for both), and a real
user applying this to real data has no ground truth to catch a bad
correction the way synthetic-mock testing does. Kept available and
documented (it does genuinely help in some regimes, confirmed directly),
but treat any `_corrected` value as provisional until the calibration
pipeline itself has been validated far more thoroughly -- see this
project's own calibration-validation strategy notes (`dev_notes/VALIDATION_CAMPAIGN.md`,
the stage documenting this decision) for what "thoroughly" needs to mean
before that changes.

This is a *different* mechanism from `gh_moments.shape_aware_taper_correction`
(a self-referential geometric correction for `compute_direct_moments`'s
windowing taper, independent of noise level) -- the two are complementary,
not competing: this module targets the regularization/curvature-penalty
shrinkage itself, calibrated empirically as a function of (recovered sigma,
S/N).

**Default calibration: the shipped E-MILES SSP grid, one fixed
representative multi-template mixture**
(`calibration_data/regularization_bias_emiles_multitemplate.txt`, built by
`dev_notes/calibration/build_emiles_calibration.py` via
`kinextract.calibrate_bias.run_calibration_sweep` against a 40-template
E-MILES subset spanning ages 0.1-14 Gyr and [M/H] -1.49 to +0.40, h3=h4=0
truth, V=0, MUSE pixel sampling, n=30 seeds/cell, sigma in 20-400 km/s x
S/N in 10-200).

**Why E-MILES, specifically.** A calibration is only valid for template
shapes similar to what it was built from -- E-MILES is synthetic,
parametrized, portable, and LSF-matchable to any instrument
(`kinextract.emiles`), so it generalizes as a package-wide default
in a way a fixed real-star or idealized-template calibration doesn't (see
`dev_notes/VALIDATION_CAMPAIGN.md` Stages 9-10 for the earlier calibrations this
one replaced, and why).

**Regularization bias is calibration-specific, not a universal constant --
if your own template library/mixture/instrument differs meaningfully from
the shipped E-MILES default, build your own table.** This is not a
special/advanced path: `calibrate_bias.run_calibration_sweep` was already
fully template-library-agnostic before the E-MILES default existed, and
`dev_notes/calibration/build_emiles_calibration.py` is the literal, adaptable
recipe used to build the shipped table -- swap its template-loading
function for your own library (a different E-MILES subset, an HST/STIS
LSF-matched grid, a different star catalog) and rerun. Pass the result as
`bias_calibration=...` rather than trusting the shipped table uncritically
for a materially different setup.

**Scope, stated plainly because the calibration data itself has this
scope, not more:**
- Corrects `sigma` (the dominant, well-tabulated effect) and `h4` (an
  additive floor at true h4=0, also tabulated). Does **not** correct `V`
  (found consistent with pure noise, no systematic trend to calibrate) or
  `h3` (no zero-truth calibration needed by construction -- true h3=0
  gives no h3 bias to speak of -- but nonzero-truth h3 recovery bias is
  not correctable here at all; no calibration data for it exists yet).
- Calibrated at V=0 truth, MUSE pixel sampling (step=1.25 Angstrom/pix),
  and ONE fixed E-MILES template mixture. Applying it to strongly
  non-zero V, non-MUSE pixel sampling, or a template shape very different
  from E-MILES SSPs (see `calibrate_bias`'s module docstring) is an
  extrapolation beyond what's actually validated -- treat results there as
  provisional, and prefer building your own calibration when in doubt.
- Interpolation is 2-D linear over (recovered sigma, S/N) via
  `scipy.interpolate.griddata`, clamped to the calibration grid's own
  range (`sigma_range`, `snr_range` on `BiasCalibration`) with a
  `UserWarning` if a query falls outside it, since extrapolation beyond
  the calibrated grid is not validated at all. Under adaptive
  regularization (`lam` varying per fit), the correction is looked up
  against a `PerLamCalibration` instead -- a set of these same 2-D tables,
  one per canonical `lam` value, with a further nearest-two-`lam`
  interpolation on top (log-linear in `lam`) -- see that class's own
  docstring for why this, not a single 3-D `(sigma, S/N, lam)`-scattered
  table, is the validated design.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata

_DEFAULT_TABLE_PATH = os.path.join(
    os.path.dirname(__file__), "calibration_data", "regularization_bias_emiles_multitemplate.txt"
)


@dataclass
class BiasCalibration:
    """Synthetic bias-vs-(recovered sigma, S/N) calibration table, at one
    fixed `lam`.

    `recovered_sigma`/`snr` are the two lookup keys always present (both
    directly observable on real data, unlike the underlying true sigma
    used to generate the calibration grid). `sig_bias`/`h4_bias` are
    `recovered - true` at each calibration point (true h4=0 throughout
    unless the sweep used `h4_true!=0`); `*_std` is the scatter of that
    bias across the calibration sweep's seeds, used to inflate reported
    uncertainty (see `correct_moments`).

    `lam` (default `None`): the single `lam` this table was calibrated at.
    `uncertainty.bootstrap_losvd` uses this to decide whether switching to
    a `PerLamCalibration` (see below) is actually warranted: only if the
    fit's own selected `lam` meaningfully differs from this value -- a
    single-`lam` table has far more (sigma, S/N) reps concentrated at its
    one `lam` than any one table inside a `PerLamCalibration` set (same
    total rep budget spread over more `lam` values), so switching when the
    `lam` values already match makes a correction worse, not better (see
    `dev_notes/VALIDATION_CAMPAIGN.md` Stage 12).
    """

    recovered_sigma: np.ndarray
    snr: np.ndarray
    sig_bias: np.ndarray
    sig_bias_std: np.ndarray
    h4_bias: np.ndarray
    h4_bias_std: np.ndarray
    sigma_range: tuple[float, float]
    snr_range: tuple[float, float]
    lam: float | None = None
    h3_bias: np.ndarray | None = None
    h3_bias_std: np.ndarray | None = None


def _parse_calibration_rows(path: str) -> np.ndarray:
    """Shared row-parsing step for `load_calibration_table`/
    `load_per_lam_calibration_table`: returns the raw
    `(sigma_true, snr, sig_bias_mean, sig_bias_std, h4_mean, h4_std, lam,
    m_eff_mean, h3_mean, h3_std)` array parsed from a calibration sweep
    file (see `calibrate_bias.save_calibration_table` for the on-disk
    format). `lam`/`m_eff_mean` are `nan` for the original 10-column
    format (no `lam_grid` varied when that table was built); `h3_mean`/
    `h3_std` are `nan` for any file written before those became two more
    trailing columns (13-column format or earlier).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "sigma":  # column-header line, no leading '#'
                continue
            # sig_bias_pct (column index 7) carries a trailing '%' and is
            # redundant with sigma_true/sig_bias_mean -- skip it rather
            # than parse the '%' string.
            sigma_true, snr, _n, _v_mean, _v_std, sig_bias_mean, sig_bias_std = (
                float(parts[i]) for i in range(7)
            )
            h4_mean, h4_std = float(parts[8]), float(parts[9])
            if len(parts) >= 13:
                lam, m_eff_mean = float(parts[10]), float(parts[11])
            else:
                lam, m_eff_mean = np.nan, np.nan
            if len(parts) >= 15:
                h3_mean, h3_std = float(parts[13]), float(parts[14])
            else:
                h3_mean, h3_std = np.nan, np.nan
            rows.append((sigma_true, snr, sig_bias_mean, sig_bias_std, h4_mean, h4_std, lam, m_eff_mean,
                         h3_mean, h3_std))
    if not rows:
        raise ValueError(f"{path}: no calibration rows parsed")
    return np.array(rows)


def _build_calibration_from_rows(arr: np.ndarray, lam: float | None = None) -> BiasCalibration:
    """Build a single 2-D `(sigma, S/N)` `BiasCalibration` from a
    `_parse_calibration_rows`-shaped array (already restricted to one
    `lam` value's rows, if this is one slice of a per-lam set)."""
    (sigma_true, snr, sig_bias_mean, sig_bias_std, h4_mean, h4_std, _lam_col, _m_eff_col,
     h3_mean, h3_std) = arr.T
    recovered_sigma = sigma_true + sig_bias_mean
    h3_bias = h3_mean if np.isfinite(h3_mean).any() else None
    h3_bias_std = h3_std if np.isfinite(h3_std).any() else None
    return BiasCalibration(
        recovered_sigma=recovered_sigma,
        snr=snr,
        sig_bias=sig_bias_mean,
        sig_bias_std=sig_bias_std,
        h4_bias=h4_mean,
        h4_bias_std=h4_std,
        sigma_range=(float(recovered_sigma.min()), float(recovered_sigma.max())),
        snr_range=(float(snr.min()), float(snr.max())),
        lam=lam,
        h3_bias=h3_bias,
        h3_bias_std=h3_bias_std,
    )


def load_calibration_table(path: str | None = None) -> BiasCalibration:
    """Parse a calibration sweep file into a single 2-D `(sigma, S/N)`
    `BiasCalibration`.

    Expected whitespace-delimited columns (see `calibrate_bias.
    save_calibration_table`): `sigma S/N n V_bias_mean V_bias_std
    sig_bias_mean sig_bias_std sig_bias_pct h4_mean h4_std` (the original
    10-column format), optionally followed by 3 more columns `lam
    m_eff_mean m_eff_std` for a table that also recorded `lam`/`m_eff` per
    row. `sigma` is the *true* sigma used to generate that row's synthetic
    ensemble. Lines starting with `#` (including the trailing timing
    comment) and the column-header line are skipped.

    **If the file's rows actually vary `lam`** (a real multi-lam sweep,
    not just a single fixed `lam` recorded incidentally), this collapses
    them into one table anyway -- use `load_per_lam_calibration_table`
    instead for that case, which builds one 2-D table *per* `lam` value
    plus nearest-`lam` interpolation (see that function and
    `PerLamCalibration`'s own docstring for why this replaced an earlier
    single-table, 3-D `(sigma, S/N, m_eff)`-scattered-interpolation
    design). This function's own `lam` field on the returned
    `BiasCalibration` is set only when every row shares one `lam` value.
    """
    path = path or _DEFAULT_TABLE_PATH
    arr = _parse_calibration_rows(path)
    lam_col = arr[:, 6]
    single_lam = float(lam_col[0]) if (np.isfinite(lam_col).all() and len(np.unique(lam_col)) == 1) else None
    return _build_calibration_from_rows(arr, lam=single_lam)


@dataclass
class PerLamCalibration:
    """A set of 2-D `(sigma, S/N)` `BiasCalibration` tables, one per
    canonical `lam` value in `lam_values`, plus nearest-two-`lam`
    interpolation at query time (`correct_moments`'s `lam` argument) --
    `CALIBRATION_TRUST_STRATEGY.md`'s own "Fix #1" design: build a full,
    densely-sampled `(sigma, S/N)` table AT EACH canonical `lam` value
    (the same grid `select_lambda_aic_local` actually scans), then
    interpolate between the two nearest `lam` tables at query time, rather
    than spreading one fixed compute budget across an added `lam`/`m_eff`
    axis and interpolating in 3-D.

    **Replaces an earlier, different design** (a single `BiasCalibration`
    with an `m_eff` column, interpolated via 3-D `scipy.interpolate.
    griddata` over `(recovered_sigma, snr, m_eff)` jointly) that used the
    *same* densely-sampled data this class also uses, but queried it
    differently: `m_eff` depends on both `lam` and the local `(sigma,
    S/N)` regime, so two calibration points with similar `m_eff` for
    unrelated reasons (different `lam`, different regime) could end up
    interpolated together in 3-D even though they don't actually
    represent nearby fits. Per-`lam` tables with 2-D interpolation avoid
    that by construction: only the two `lam` values actually bracketing
    the query ever contribute. `dev_notes/calibration/CALIBRATION_TRUST_
    STRATEGY.md`'s 4-gate acceptance test found the old design failed
    gate 1 (per-cell bias) even against data dense enough to pass it --
    this is the fix, not a data rebuild (the same underlying rows are
    reused, just partitioned and queried differently).
    """
    lam_values: np.ndarray  # sorted, ascending
    tables: list  # one BiasCalibration per lam_values entry, same order


def load_per_lam_calibration_table(path: str) -> PerLamCalibration:
    """Parse a calibration sweep file built with `lam_grid` genuinely
    varied (more than one distinct `lam` across its rows) into a
    `PerLamCalibration` -- one 2-D `(sigma, S/N)` `BiasCalibration` per
    `lam` value, sorted ascending. Raises `ValueError` if the file has
    only one `lam` value (use `load_calibration_table` for that case).
    """
    arr = _parse_calibration_rows(path)
    lam_col = arr[:, 6]
    if not np.isfinite(lam_col).all():
        raise ValueError(f"{path}: rows are missing a lam column -- this is not a "
                          "multi-lam calibration sweep file.")
    lam_values = np.unique(lam_col)
    if len(lam_values) < 2:
        raise ValueError(f"{path}: only one distinct lam value ({lam_values}) -- use "
                          "load_calibration_table for a single-lam file.")
    tables = [_build_calibration_from_rows(arr[lam_col == lam], lam=float(lam)) for lam in lam_values]
    return PerLamCalibration(lam_values=lam_values, tables=tables)


_default_calibration: BiasCalibration | None = None

_MEFF_TABLE_PATH = os.path.join(
    os.path.dirname(__file__), "calibration_data", "regularization_bias_emiles_multitemplate_meff.txt"
)
_default_meff_calibration: BiasCalibration | None = None


def _get_default_calibration() -> BiasCalibration:
    global _default_calibration
    if _default_calibration is None:
        _default_calibration = load_calibration_table()
    return _default_calibration


def get_default_lam_axis_calibration() -> PerLamCalibration:
    """The shipped per-`lam` calibration set
    (`regularization_bias_emiles_multitemplate_meff.txt` -- same file the
    old `m_eff`-axis design used, reparsed as a `PerLamCalibration`; see
    that class's own docstring for why), cached the same way
    `_get_default_calibration` caches the plain (sigma, S/N) one.

    This is `uncertainty.bootstrap_losvd`'s default bias-correction table
    when `cfg.lam_selection == 'aic_local'` (the `FitConfig` default) and
    no `bias_calibration` was passed explicitly -- the plain 2-D table was
    built at fixed `lam=2.0` and is no longer a domain match once `lam`
    varies per fit under adaptive selection."""
    global _default_meff_calibration
    if _default_meff_calibration is None:
        _default_meff_calibration = load_per_lam_calibration_table(_MEFF_TABLE_PATH)
    return _default_meff_calibration


def calibration_has_lam_axis(calibration=None) -> bool:
    """Whether `calibration` (or the shipped default, if `None`) is a
    `PerLamCalibration` with a `lam` axis to interpolate over. Callers
    that have their own fit context (e.g. `uncertainty.bootstrap_losvd`)
    use this to decide *whether* `correct_moments` needs a `lam` value at
    all -- passing one unconditionally would raise `ValueError` against a
    plain single-`lam` `BiasCalibration`. Resolves the same shipped
    default `correct_moments` itself falls back to, so this always
    reflects what a `calibration=None` call would actually use.
    """
    calib = calibration or _get_default_calibration()
    return isinstance(calib, PerLamCalibration)


def estimate_snr_whitened(model_w: np.ndarray) -> float:
    """S/N estimate matching the calibration sweep's own convention: the
    median best-fit model amplitude in noise-whitened units (where unit
    noise is assumed), i.e. `median(model_w)` for `model_w = A_eff @ b`
    evaluated on the same whitened data `fit_losvd` was given. This equals
    the sweep's `median(galaxy_noiseless) / noise_level` definition once
    the model is expressed in whitened units.
    """
    return float(np.median(np.abs(np.asarray(model_w, float))))


def _interp(query_point: tuple, xy: np.ndarray, values: np.ndarray,
            calib: BiasCalibration, quantity_name: str) -> float:
    lo_s, hi_s = calib.sigma_range
    lo_n, hi_n = calib.snr_range
    query_sigma, query_snr = query_point[0], query_point[1]
    in_range = (lo_s <= query_sigma <= hi_s) and (lo_n <= query_snr <= hi_n)
    range_msg = f"sigma {lo_s:.0f}-{hi_s:.0f}, S/N {lo_n:.0f}-{hi_n:.0f}"
    if not in_range:
        warnings.warn(
            f"bias_correction: {quantity_name} query {query_point} falls outside the "
            f"calibration grid's own range ({range_msg}) -- nearest-point extrapolation, "
            "not validated.",
            UserWarning, stacklevel=4,
        )
        method = "nearest"
    else:
        method = "linear"
    result = griddata(xy, values, query_point, method=method)
    if np.isnan(result):
        # Inside the bounding box but outside the convex hull of scattered
        # points -- fall back to nearest rather than propagate a NaN.
        result = griddata(xy, values, query_point, method="nearest")
    return float(result)


def _correct_moments_2d(vherm, sherm, h4, snr, calib: BiasCalibration) -> dict:
    """The plain 2-D (sigma, S/N) correction -- shared by `correct_moments`
    (single-lam `BiasCalibration`) and `_correct_moments_per_lam` (each of
    the two bracketing lam tables it interpolates between)."""
    xy = np.column_stack([calib.recovered_sigma, calib.snr])
    query_point = (sherm, snr)
    sig_bias = _interp(query_point, xy, calib.sig_bias, calib, "sigma")
    sig_bias_std = _interp(query_point, xy, calib.sig_bias_std, calib, "sigma_std")
    h4_bias = _interp(query_point, xy, calib.h4_bias, calib, "h4")
    h4_bias_std = _interp(query_point, xy, calib.h4_bias_std, calib, "h4_std")
    return dict(
        sherm_corrected=sherm - sig_bias, h4_corrected=h4 - h4_bias,
        sig_bias_used=sig_bias, h4_bias_used=h4_bias,
        sig_uncertainty_inflation=sig_bias_std, h4_uncertainty_inflation=h4_bias_std,
    )


def _correct_moments_per_lam(vherm, sherm, h4, snr, calib: PerLamCalibration, lam: float) -> dict:
    """`correct_moments`'s `PerLamCalibration` path: find the two `lam`
    values in `calib.lam_values` bracketing the query `lam`, correct
    against each of their own 2-D tables independently, then linearly
    interpolate the two results in `log10(lam)` (the grid's own natural
    spacing -- `FitConfig`'s default lam grids are log-spaced) rather than
    folding `lam` into a single 3-D interpolation -- see
    `PerLamCalibration`'s own docstring for why that distinction matters.
    """
    lam_values = calib.lam_values
    lo, hi = float(lam_values[0]), float(lam_values[-1])
    if lam <= lo:
        if lam < lo:
            warnings.warn(
                f"bias_correction: lam query {lam:.4g} is below the per-lam calibration "
                f"grid's own range ({lo:.4g}-{hi:.4g}) -- snapping to the smallest "
                "calibrated lam, not validated as an extrapolation.",
                UserWarning, stacklevel=3,
            )
        return _correct_moments_2d(vherm, sherm, h4, snr, calib.tables[0])
    if lam >= hi:
        if lam > hi:
            warnings.warn(
                f"bias_correction: lam query {lam:.4g} is above the per-lam calibration "
                f"grid's own range ({lo:.4g}-{hi:.4g}) -- snapping to the largest "
                "calibrated lam, not validated as an extrapolation.",
                UserWarning, stacklevel=3,
            )
        return _correct_moments_2d(vherm, sherm, h4, snr, calib.tables[-1])

    hi_idx = int(np.searchsorted(lam_values, lam))
    lo_idx = hi_idx - 1
    lam_lo, lam_hi = float(lam_values[lo_idx]), float(lam_values[hi_idx])
    weight_hi = (np.log10(lam) - np.log10(lam_lo)) / (np.log10(lam_hi) - np.log10(lam_lo))

    r_lo = _correct_moments_2d(vherm, sherm, h4, snr, calib.tables[lo_idx])
    r_hi = _correct_moments_2d(vherm, sherm, h4, snr, calib.tables[hi_idx])
    return {key: (1.0 - weight_hi) * r_lo[key] + weight_hi * r_hi[key] for key in r_lo}


def correct_moments(vherm: float, sherm: float, h4: float, snr: float,
                     calibration=None, lam: float | None = None) -> dict:
    """Apply the Pinkney-style regularization-bias correction to a single
    recovered (sherm, h4), using `snr` (see `estimate_snr_whitened`) as the
    second calibration key.

    `calibration` : a plain `BiasCalibration` (single fixed `lam`) or a
    `PerLamCalibration` (a set of `BiasCalibration` tables spanning
    several `lam` values, plus nearest-two-`lam` interpolation -- see that
    class's own docstring). `None` (default) uses the shipped plain table.

    `lam` (default `None`): the fit's own `lam` (whatever
    `select_lambda_aic_local` chose, or a caller-supplied fixed value) --
    **required if and only if `calibration` is a `PerLamCalibration`**;
    omit it (leave `None`) for a plain single-`lam` `BiasCalibration`, even
    if you happen to know that table's own `lam`. Raises `ValueError` on
    either mismatch (a `PerLamCalibration` with no `lam` given, or a `lam`
    given against a plain `BiasCalibration`).

    Returns a dict with `sherm_corrected`, `h4_corrected` (`vherm`/`h3`
    passed through unchanged -- not corrected here, see this module's
    docstring), plus `sig_bias_used`/`h4_bias_used` (the calibrated bias
    actually subtracted) and `sig_uncertainty_inflation`/
    `h4_uncertainty_inflation` (the calibration's own bias scatter at this
    query point -- add in quadrature to a bootstrap-derived std to
    avoid an overly narrow reported interval, per Pinkney's own
    prescription: naive MC/bootstrap intervals are too narrow unless the
    bias itself, and its own uncertainty, are folded in).
    """
    calib = calibration if calibration is not None else _get_default_calibration()
    is_per_lam = isinstance(calib, PerLamCalibration)
    if is_per_lam and lam is None:
        raise ValueError(
            "correct_moments: calibration is a PerLamCalibration but no lam was given -- "
            "pass the fit's own lam (e.g. result['lam_used']) so the two bracketing lam "
            "tables can be selected."
        )
    if not is_per_lam and lam is not None:
        raise ValueError(
            "correct_moments: lam was supplied but calibration is a plain single-lam "
            "BiasCalibration, not a PerLamCalibration -- omit lam for this table, or pass "
            "a PerLamCalibration (see get_default_lam_axis_calibration) if you meant to "
            "correct against several lam values."
        )
    if is_per_lam:
        return _correct_moments_per_lam(vherm, sherm, h4, snr, calib, lam)
    return _correct_moments_2d(vherm, sherm, h4, snr, calib)


def correct_losvd(v_grid: np.ndarray, vherm: float, sherm: float, h3: float, h4: float,
                   amp: float, snr: float, calibration=None,
                   lam: float | None = None) -> dict:
    """Full-curve counterpart of `correct_moments`: corrects `sherm`/`h4`,
    then reconstructs a bias-corrected Gauss-Hermite LOSVD curve on
    `v_grid` from the corrected moments (`vherm`/`h3` unchanged -- see
    `correct_moments`).

    This is necessarily a *parametric* reconstruction (the correction is
    calibrated at the moment level, not the per-bin level), not a
    bias-corrected version of the original nonparametric `b(v)` bin-by-bin
    -- appropriate for comparing against a GH-model truth curve, and for
    reporting corrected V/sigma/h3/h4, but not a substitute for the raw
    recovered `b(v)` if per-bin shape detail beyond the 4 GH moments
    matters for a specific use case.

    **`b_corrected` is clipped non-negative.** The truncated Gauss-Hermite
    series is not guaranteed non-negative everywhere for arbitrary (h3, h4)
    -- a known property of this expansion in general (see `losvd.py`'s own
    docstring on h4=-0.1 going negative over more than half the grid), not
    something specific to this correction. But the calibrated correction
    shifts h4 by an offset with no constraint keeping the result inside the
    non-negative region, so it can push a curve that would otherwise have
    stayed positive into going negative near the core edges -- observed
    directly on real fits. Clipping matches the convention already used
    everywhere else in this package a Gauss-Hermite curve is treated as a
    final density (`ablation.py`, `calibrate_bias.py`,
    `shape_aware_taper_correction`'s own `b_ref`); this was the one
    reconstruction site that hadn't been brought in line with it. Not
    renormalized after clipping (the point is a minimal correction to a
    physically-required constraint, not a change to the reported
    amplitude/normalization).

    Returns a dict with `b_corrected` (the reconstructed curve on
    `v_grid`) plus everything `correct_moments` returns.
    """
    from .gh_moments import gauss_hermite_model  # local import: gh_moments imports this module at top level

    corr = correct_moments(vherm, sherm, h4, snr, calibration, lam=lam)
    b_corrected = gauss_hermite_model(
        np.asarray(v_grid, float), amp, vherm, corr["sherm_corrected"], h3, corr["h4_corrected"]
    )
    corr["b_corrected"] = np.maximum(b_corrected, 0.0)
    return corr
