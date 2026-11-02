"""All tunable parameters controlling a LOSVD fit, in one place.

**Maintenance rule, not just a suggestion**: every field here must
correspond to something the current code actually reads. When a feature
is removed or reworked, remove or update its field in the same change --
don't leave a stale field behind "in case," and don't let two fields
silently interact in a way that isn't obvious from reading this file
top to bottom. A config field that exists but does nothing, or two fields
whose combined effect depends on an interaction not documented right
here, are exactly the kind of accumulated confusion this file exists to
prevent.

Each field's help text lives in its own `metadata`, next to the field
itself -- not in a separate lookup table that could drift out of sync --
and follows a fixed shape: what the parameter controls, its default, what
raising and lowering it does to the fit, and any known interaction with
another field. Use `FitConfig.describe()` to print every field grouped by
subsystem, or `FitConfig.describe("lam")` to filter by a substring.

**Known-dangerous values warn, not just document.** Every hazard recorded
in a help string below is backed by a concrete validation result (see
`kinextract_test_suite/notebooks/kinextract_validation/`), not a
guess -- and `FitConfig.__post_init__` raises a `UserWarning` for the
combinations known to cause the specific failure the help text describes,
so a bad value doesn't fail silently.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields

import numpy as np

from .forward_model import build_velocity_grid

# Difference order of the LOSVD roughness penalty. Third differences: the penalty's null space is any Gaussian, so
# heavy smoothing shrinks toward the best-fitting Gaussian instead of the exponential tails of the second-difference
# penalty, which inflated wing mass and sigma at low S/N and gave 3-4x larger h4 bias with continuum-cofitting
# (dev_notes/REGULARIZATION_INVESTIGATION_LOG.md; dev_notes/continuum/FINDINGS.md).
PENALTY_ORDER = 3


def _f(default, group: str, help: str):
    return field(default=default, metadata={"group": group, "help": help})


# Genuine once-per-process dedup for the "knot spacing much finer than
# knot_spacing_kms" warning below, independent of Python's own warnings-module
# "once" action. That action's dedup key is the *exact* formatted message text
# (message, category, module, lineno) -- but this warning embeds per-call values
# (vmin, vmax, dv, effective_lam), so on a real multi-bin run, every bin (and
# every lam the lam rule tries) produces a distinct message text and "once"
# barely deduplicates anything, flooding the notebook exactly the way it's meant
# to prevent. A plain module-level flag, checked before ever calling
# `warnings.warn`, dedupes correctly regardless of what the message says.
# Deliberately a plain flag, not registered via `warnings.filterwarnings`, so it
# also covers serial (non-`_parallel`) usage -- `_limit_worker_threads` (which
# used to own this dedup) only ever ran inside a `ProcessPoolExecutor` worker.
_KNOT_SPACING_WARNING_SHOWN = False


def _reset_knot_spacing_warning_dedup() -> None:
    """Test-only: re-arm the once-per-process knot-spacing warning so a test
    can assert it fires, independent of whichever earlier test may have
    already triggered it in this same process."""
    global _KNOT_SPACING_WARNING_SHOWN
    _KNOT_SPACING_WARNING_SHOWN = False


@dataclass
class FitConfig:
    """Configuration for a single non-parametric LOSVD fit. See module
    docstring for the maintenance rule this file follows.
    """

    # -- LOSVD velocity grid --------------------------------------------
    vmin: float = _f(-600.0, "LOSVD grid",
                      "Lower edge of the velocity grid the LOSVD is recovered on, km/s. "
                      "Default -600. Lowering it (more negative) extends coverage for "
                      "blueshifted/broad targets at the cost of spline resolution spent on "
                      "empty velocity range (see `knot_spacing_kms`, which reacts to this). "
                      "Raising it (less negative) risks truncating real flux if the true "
                      "LOSVD's blue wing extends past it -- truncation biases the recovered "
                      "moments, it does not just clip visually, and isn't fully caught by "
                      "shape-residual metrics that only score the region near the true peak. "
                      "**Validated convention: `V - 7*sigma`**, using your best available "
                      "estimate of the target's own V/sigma -- see `vmax`'s help text for how "
                      "this was chosen.")
    vmax: float = _f(600.0, "LOSVD grid",
                      "Upper edge of the velocity grid, km/s. Default 600. Same tradeoff as "
                      "`vmin`, mirrored -- **validated convention: `V + 7*sigma`** (i.e. "
                      "`vmin`/`vmax = V -+ 7*sigma`, NSIG=7). A grid width sweep against a "
                      "broad (V, sigma) validation grid found worst-case shape residual is "
                      "minimized at NSIG=7 (not monotonic in NSIG -- narrower grids visibly "
                      "truncate real LOSVD wings even when their aggregate bias metric looks "
                      "good, wider grids waste spline resolution on near-zero LOSVD density). "
                      "Too narrow truncates real flux and biases recovered moments; too wide "
                      "wastes spline resolution on empty range -- and can push the auto-derived "
                      "knot spacing far finer than `knot_spacing_kms`'s reference if "
                      "`min_interior_knots` ends up the binding constraint (opposite direction: "
                      "this makes `lam` behave *stronger* than calibrated -- see the "
                      "effective-lam warning this triggers).")
    bin_spacing_kms: float = _f(8.0, "LOSVD grid",
                                 "Target velocity spacing, km/s, for the grid the LOSVD is "
                                 "evaluated on and the design matrix is built on -- auto-derives "
                                 "the point count via `resolved_n_bins()`, floored at `min_bins`. "
                                 "There is no `n_bins` field to set directly: this grid is *not* "
                                 "the number of free parameters in the fit (that's "
                                 "`n_interior_knots`) -- it is purely a numerical-quadrature "
                                 "resolution for the template-shift integral. Recovered V/sigma "
                                 "converge and stop changing once spacing is finer than ~40 km/s, "
                                 "with no accuracy gain from going finer past that point; the "
                                 "current default carries a large safety margin below it. Only "
                                 "lower this (finer spacing) if you have a specific reason to "
                                 "resample the output curve more densely; raising it (coarser) "
                                 "risks re-entering untested territory above ~40 km/s/bin.")
    min_bins: int = _f(149, "LOSVD grid",
                        "Floor on the auto-derived point count from `bin_spacing_kms`, so a "
                        "narrow velocity grid doesn't get too few points for the quadrature to "
                        "be reliable. Default 149, comfortably above the ~100-point convergence "
                        "point found in testing.")

    # -- LOSVD spline basis ----------------------------------------------
    n_interior_knots: int | None = _f(None, "LOSVD spline",
                                       "Interior knot count for the cubic B-spline representing "
                                       "log(LOSVD). Default None, meaning auto-derive it from "
                                       "`knot_spacing_kms` and the grid width -- see "
                                       "`resolved_n_interior_knots()`. Set an explicit int to "
                                       "override. Raising it lets the spline represent sharper "
                                       "features but re-invites the instability a low-dimensional "
                                       "basis exists to avoid; lowering it (or pinning a fixed "
                                       "count that ends up too coarse for a given `vmin`/`vmax`) "
                                       "under-resolves the LOSVD -- absolute knot spacing coarser "
                                       "than roughly 150 km/s/knot has produced 47-58% shape "
                                       "residual in validation testing. If you pin this "
                                       "explicitly, check the implied km/s-per-knot spacing "
                                       "yourself against `knot_spacing_kms`'s own guidance below.")
    knot_spacing_kms: float = _f(95.0, "LOSVD spline",
                                  "Target absolute knot spacing, km/s, used to auto-derive "
                                  "`n_interior_knots` when it's left as None. Default 95.0 -- "
                                  "empirically the coarsest spacing that still hit <5% RMS shape "
                                  "residual across a broad (V, sigma) grid at S/N~100. Raising "
                                  "this (coarser spacing, fewer knots) trades recovery precision "
                                  "for a smaller/faster fit; values above roughly 150-190 km/s per "
                                  "knot were the ones that produced the 47-58% residual failure "
                                  "documented under `n_interior_knots` above. Lowering it (finer "
                                  "spacing, more knots) raises the coefficient count and can "
                                  "reintroduce noise sensitivity the low-dimensional basis is "
                                  "meant to avoid, though this hasn't been stress-tested as "
                                  "thoroughly as the coarse direction.")
    min_interior_knots: int = _f(16, "LOSVD spline",
                                  "Floor on the auto-derived `n_interior_knots`, so a narrow "
                                  "velocity grid (small vmax-vmin) doesn't get too few knots to "
                                  "represent a compact LOSVD shape. Default 16. Lowering it risks "
                                  "under-resolving compact/narrow targets on a narrow grid; "
                                  "raising it only matters when `knot_spacing_kms` would already "
                                  "auto-derive fewer knots than the new floor, i.e. it has no "
                                  "effect on wide grids where the auto-derived count already "
                                  "exceeds it.")
    degree: int = _f(3, "LOSVD spline",
                      "B-spline polynomial degree. Default 3 (cubic), the standard choice; "
                      "untested at other values in this package's own validation, so treat "
                      "changing it as unverified territory rather than a supported tuning knob.")

    # -- Continuum --------------------------------------------------------
    continuum_knots: int | None = _f(4, "Continuum",
                                      "Interior knot count for the shared multiplicative continuum "
                                      "spline (see `kinextract.continuum`), co-fitted with the LOSVD. "
                                      "Default 4, the setting validated on the NGC 4751 MUSE data "
                                      "(dev_notes/regularization_v2/FINDINGS.md). Fits then need `wave_fit` "
                                      "(the observed wavelength grid), since the continuum spline is built "
                                      "on wavelength, not velocity. Set None to switch continuum-cofitting "
                                      "off (for spectra that are already exactly continuum-normalized, e.g. "
                                      "mocks built from the fit's own templates). Requires templates at a "
                                      "consistent continuum level (e.g. continuum-normalized) -- see "
                                      "`continuum.py`'s module docstring. `result['success']` is often False "
                                      "with this on but carries no accuracy information -- see "
                                      "`continuum.fit_continuum_block`. Pairs with `lam_c` and `ridge_c`: "
                                      "both regularization terms are needed to keep the continuum "
                                      "well-constrained.")
    lam_c: float = _f(2.0, "Continuum",
                       "Curvature penalty weight for the continuum spline. Default 2.0 (the "
                       "NGC 4751 setting; recovery is insensitive to it -- see "
                       "`tests/test_continuum.py`). Only relevant if `continuum_knots` is set.")
    ridge_c: float = _f(0.1, "Continuum",
                        "Direct L2 penalty on the continuum spline coefficients themselves, "
                        "in addition to `lam_c`'s curvature penalty -- a second-order curvature "
                        "penalty alone assigns zero cost to a constant or linear trend in "
                        "coefficient space, so `lam_c` cannot bound one; `ridge_c` does. See "
                        "`continuum.fit_continuum_block`'s own docstring for the mechanism. "
                        "Default 0.1: recovery accuracy against a known continuum is unaffected "
                        "by the exact value, so it mainly needs to be decisively nonzero, not "
                        "precisely tuned. Only relevant if `continuum_knots` is set.")

    # -- Regularization ---------------------------------------------------
    lam: float = _f(4096.0, "Regularization",
                     "Base weight of the P-spline roughness penalty on the third differences of the "
                     "log-LOSVD spline coefficients (scaled by the real knot spacing, see "
                     "`kinextract.losvd.difference_matrix`, so it means the same physical smoothing "
                     "on any grid). The third-difference penalty's null space is any Gaussian, so "
                     "heavy smoothing shrinks toward the best-fitting Gaussian, never toward fat "
                     "tails. Default 4096, validated on 480 NGC 4751-like MUSE mocks and the real "
                     "NGC 4751 bins (dev_notes/regularization_v2/FINDINGS.md): no adaptive "
                     "selector beat it by more than a few per cent on mocks, and adaptive "
                     "selection failed on the real spectra. With `lam_rule='sigma_scaled'` (the "
                     "default) this is the starting value, raised for broad LOSVDs. Larger = "
                     "smoother; smaller = more flexible and more prone to a broad pedestal at the "
                     "grid edges.")
    lam_rule: str = _f("sigma_scaled", "Regularization",
                        "How `fit_losvd`/`fit_losvd_robust` set the `lam` of a fit. 'sigma_scaled' "
                        "(default): fit at `lam`, then refit at `config.sigma_scaled_lam(sigma, "
                        "lam0=lam)` = lam * (sigma/350)^6 (capped at 131072) if that is larger -- "
                        "broad LOSVDs need more smoothing for the same relative shape -- and, while "
                        "the LOSVD still holds more than `max_edge_amplitude_frac` of its peak at the "
                        "grid edge, refit at `lam` times each of `edge_retry_factors` in turn. "
                        "'fixed': one fit at exactly `lam` (for reproducing a given `lam`, and for "
                        "the package's internal sub-fits). The `lam` used is returned as "
                        "`result['lam_used']`.")
    max_edge_amplitude_frac: float = _f(0.01, "Regularization",
                            "Largest LOSVD amplitude within 5 grid points of either velocity-grid "
                            "edge, as a fraction of its peak (`losvd.edge_amplitude_frac`), that "
                            "counts as contained by the grid. Default 0.01. Above it the fit has a "
                            "divergent edge pedestal: the moment integrals then run over a range the "
                            "data never constrained (it inflated sigma about 2x on real NGC 5102 "
                            "MUSE spectra). `lam_rule='sigma_scaled'` refits such fits at a larger "
                            "`lam` (`edge_retry_factors`), and the Laplace sampler rejects posterior "
                            "draws above it. 0.0 disables both.")
    edge_retry_factors: tuple = _f((2.0, 4.0), "Regularization",
                            "With `lam_rule='sigma_scaled'`: factors applied in turn to the fit's "
                            "`lam` while its LOSVD is still at the grid edge (see "
                            "`max_edge_amplitude_frac`) -- by default 2x, then a further 4x. Empty "
                            "disables the retry.")
    ridge_lam: float = _f(1.0, "Regularization",
                           "Ridge strength used only for the optimizer's own starting point "
                           "(see `kinextract.losvd.default_initial_coeffs`) -- not part "
                           "of the fitted objective itself, so it does not trade off against "
                           "`lam` the way it might look like it should. Rarely needs changing; "
                           "its only job is landing the optimizer in a reasonable basin before "
                           "the real (lam-regularized) optimization takes over.")

    # -- Noise --------------------------------------------------------------
    rho: float = _f(0.0, "Noise",
                     "AR(1) lag-1 pixel-noise correlation, in [0, 1). Default 0.0 "
                     "(independent pixels -- every fit in this package has always assumed "
                     "this until now). Real MUSE star-spectrum residuals measure ~0.3-0.5 "
                     "here, surviving even after continuum-detrending -- see "
                     "kinextract.noise_model's own module docstring, and "
                     "`estimate_correlation_length` for measuring it from real residuals. "
                     "**Not automatically applied by `fit_losvd` or anything else in this "
                     "package** -- this field only documents/carries the value alongside the "
                     "rest of a fit's configuration; the caller must still build "
                     "`A_stack_w`/`y_w` via `noise_model.whiten(A_stack, y, errors, "
                     "rho=cfg.rho)` before calling `fit_losvd`, exactly as with `rho=0.0`'s "
                     "plain `A_stack/errors` division today. This is deliberate: the fitting "
                     "engines only ever consume already-whitened inputs, so correcting for "
                     "correlated noise is entirely a matter of how those inputs are built, "
                     "not a change to the fit itself -- see "
                     "`examples/notebooks/05_noise_correlation.ipynb` for the worked pattern "
                     "(GLS pre-whitening makes residuals i.i.d. by construction, regardless "
                     "of `rho`). `fit_losvd_robust(..., measure_rho=True)` measures it per bin.")

    # -- Reporting ----------------------------------------------------------
    fit_h3h4: bool = _f(True, "Reporting",
                         "Whether the Gauss-Hermite summary (kinextract.gh_moments) "
                         "fits h3/h4 in addition to V and sigma, or fixes them to zero. "
                         "Default True. Turning it off only changes what's *reported* -- the "
                         "LOSVD recovery itself (`result['b']`) is identical either way, since "
                         "this step runs after the fit, not as part of it.")

    def __post_init__(self) -> None:
        if self.vmin >= self.vmax:
            raise ValueError(f"vmin ({self.vmin}) must be < vmax ({self.vmax}).")
        if self.lam_rule not in ("sigma_scaled", "fixed"):
            raise ValueError(f"lam_rule must be 'sigma_scaled' or 'fixed', got {self.lam_rule!r}.")
        if any(f <= 1.0 for f in self.edge_retry_factors):
            raise ValueError(f"edge_retry_factors must all be > 1, got {self.edge_retry_factors!r}.")
        if self.lam < 1.0:
            warnings.warn(
                f"FitConfig.lam={self.lam} is below 1.0, the observed sweet spot under the "
                "current velocity-spacing-scaled curvature penalty. Worst-case shape residual "
                "was measured at 18% (lam=0.5) rising to 43% (lam=0.1) in validation testing "
                "from noise overfitting -- see `lam`'s own help text "
                "(`FitConfig.describe('lam')`). Only proceed if you've validated this value "
                "against more than the one case you're tuning for.",
                UserWarning, stacklevel=2,
            )
        if self.n_interior_knots is not None:
            km_per_knot = (self.vmax - self.vmin) / self.n_interior_knots
            if km_per_knot > 150.0:
                warnings.warn(
                    f"FitConfig.n_interior_knots={self.n_interior_knots} implies "
                    f"{km_per_knot:.0f} km/s per knot on this grid (vmin={self.vmin}, "
                    f"vmax={self.vmax}) -- coarser than the ~95 km/s/knot validated as safe. "
                    "A fixed knot count this coarse relative to grid width caused 47-58% shape "
                    "residual in validation testing. Leave `n_interior_knots=None` to "
                    "auto-scale via `knot_spacing_kms` instead of pinning a count explicitly.",
                    UserWarning, stacklevel=2,
                )
        elif self.knot_spacing_kms > 150.0:
            warnings.warn(
                f"FitConfig.knot_spacing_kms={self.knot_spacing_kms} is coarser than the "
                "~95 km/s/knot validated as safe -- spacing this coarse caused 47-58% shape "
                "residual in validation testing. See `knot_spacing_kms`'s own help text.",
                UserWarning, stacklevel=2,
            )
        n_knots = self.resolved_n_interior_knots()
        dv = (self.vmax - self.vmin) / (n_knots + 1)
        # The penalty is scaled by (knot_spacing_kms / dv) (see `difference_matrix`), so a grid narrow enough that
        # `min_interior_knots` sets the knot count makes every `lam` act stronger than its nominal value. Warn once
        # the effective lam passes 3x the top of the lam rule's range (131072), where validation testing saw severe
        # oversmoothing.
        effective_lam = self.lam * (self.knot_spacing_kms / dv)
        if effective_lam > 3.0 * 131072.0:
            global _KNOT_SPACING_WARNING_SHOWN
            if not _KNOT_SPACING_WARNING_SHOWN:
                _KNOT_SPACING_WARNING_SHOWN = True
                warnings.warn(
                    f"FitConfig with vmin={self.vmin}, vmax={self.vmax} resolves to a knot "
                    f"spacing of {dv:.1f} km/s -- much finer than the {self.knot_spacing_kms:.0f} "
                    "km/s `knot_spacing_kms` is calibrated against (this happens when the grid is "
                    "narrow enough that `min_interior_knots` sets the knot count, not "
                    "`knot_spacing_kms`). Because the curvature penalty is scaled by real velocity "
                    "spacing (see `kinextract.losvd.difference_matrix`), this makes lam="
                    f"{self.lam} behave like an effective lam of {effective_lam:.0f} -- confirmed to "
                    "cause severe oversmoothing (>100% sigma bias) at this kind of extreme scale "
                    "factor in validation testing. This does not happen at the validated NSIG=7 "
                    "velocity-grid convention (`vmin/vmax = V -+ 7*sigma`) even down to sigma~20 "
                    "km/s -- it means this grid is unusually narrow for its `min_interior_knots`. "
                    "Consider a wider grid, a smaller `min_interior_knots`, or an explicit "
                    "`n_interior_knots` override. (This warning prints at most once per process -- "
                    "the underlying condition may still be affecting other bins/candidates.)",
                    UserWarning, stacklevel=2,
                )

    def build_velocity_grid(self) -> np.ndarray:
        """The LOSVD velocity grid implied by `vmin`/`vmax` and the point
        count auto-derived by `resolved_n_bins()`.
        """
        return build_velocity_grid(self.vmin, self.vmax, self.resolved_n_bins())

    def resolved_n_bins(self) -> int:
        """The number of points the velocity grid is built with, auto-
        derived from `bin_spacing_kms` and the grid width, floored at
        `min_bins`. This is a numerical-quadrature resolution, not a
        model-flexibility parameter -- see `bin_spacing_kms`'s help text.
        """
        auto = round((self.vmax - self.vmin) / self.bin_spacing_kms)
        return max(self.min_bins, auto)

    def resolved_n_interior_knots(self) -> int:
        """`n_interior_knots` if explicitly set, otherwise auto-derived from
        `knot_spacing_kms` and the velocity grid width, floored at
        `min_interior_knots`. Always use this (not the raw field) when the
        actual knot count is needed -- see `n_interior_knots`'s help text
        for why a fixed count is unsafe across different `vmin`/`vmax`.
        """
        if self.n_interior_knots is not None:
            return self.n_interior_knots
        auto = round((self.vmax - self.vmin) / self.knot_spacing_kms)
        return max(self.min_interior_knots, auto)

    @classmethod
    def describe(cls, filter_substring: str | None = None) -> None:
        """Print every field, grouped by subsystem, with its default and
        help text. Pass a substring (matched against field name, group,
        or help text, case-insensitively) to show only matching fields.
        """
        by_group: dict[str, list] = {}
        for fld in fields(cls):
            group = fld.metadata.get("group", "other")
            by_group.setdefault(group, []).append(fld)

        needle = filter_substring.lower() if filter_substring else None
        for group, group_fields in by_group.items():
            rows = []
            for fld in group_fields:
                help_text = fld.metadata.get("help", "")
                haystack = f"{fld.name} {group} {help_text}".lower()
                if needle is not None and needle not in haystack:
                    continue
                rows.append(fld)
            if not rows:
                continue
            print(f"\n[{group}]")
            for fld in rows:
                print(f"  {fld.name} = {fld.default!r}")
                print(f"      {fld.metadata.get('help', '')}")
        print("\nCall FitConfig(...).resolved_n_interior_knots() to see the actual knot "
              "count that would be used (auto-derived from vmin/vmax unless overridden), "
              "and .resolved_n_bins() for the auto-derived velocity-grid point count.")


def sigma_scaled_lam(sigma: float, lam0: float = 4096.0, sigma0: float = 350.0, power: float = 6.0,
                     lam_max: float = 131072.0) -> float:
    """Fixed-lam rule for the order-3 penalty: `lam0 * max(1, (sigma/sigma0)**power)`, capped at `lam_max`.

    The order-3 penalty on log b scales as sigma**-6 for a given relative shape, so broad LOSVDs need a larger lam
    for the same smoothing. Only ever raises lam above `lam0`. Calibrated on NGC 4751-like MUSE mocks (sigma 250-550,
    S/N 40-300) and checked on the real NGC 4751 bins (dev_notes/regularization_v2/FINDINGS.md, findings 21 and 25):
    never worse than fixed `lam0` in any mock cell, 4-6 % better shapes at sigma 550, and it repairs real broad bins
    that fixed `lam0` leaves with a broad pedestal. `sigma` is the fit's own (e.g. `compute_losvd_moments` 'sherm'
    of a `lam0` fit). `FitConfig(lam_rule='sigma_scaled')`, the default, applies it inside every fit."""
    if not (np.isfinite(sigma) and sigma > 0):
        return float(lam0)
    return float(min(lam_max, lam0 * max(1.0, (sigma / sigma0) ** power)))
