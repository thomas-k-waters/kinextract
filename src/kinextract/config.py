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


def _f(default, group: str, help: str):
    return field(default=default, metadata={"group": group, "help": help})


_WING_LAM_UNSAFE_SIGMA = 100.0


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
    continuum_knots: int | None = _f(None, "Continuum",
                                      "Interior knot count for the shared multiplicative continuum "
                                      "spline (see `kinextract.continuum`). Default None -- "
                                      "continuum-cofitting off. Set an int (4 is a reasonable "
                                      "starting point) to turn continuum-cofitting on for "
                                      "`fit_losvd`/`bootstrap_losvd`, which then require `wave_fit` "
                                      "(the observed wavelength grid) to be passed explicitly, since "
                                      "the continuum spline is built on wavelength, not velocity. "
                                      "Requires templates already at a consistent continuum-trust "
                                      "level (e.g. already continuum-normalized) -- see "
                                      "`continuum.py`'s own module docstring. `result['success']` is "
                                      "frequently False with this on but carries no real accuracy "
                                      "information -- see `continuum.fit_continuum_block`'s own "
                                      "docstring before treating it as a hard failure filter. Pairs "
                                      "with `lam_c` and `ridge_c` (see their own help text) -- both "
                                      "regularization terms are needed to keep the recovered "
                                      "continuum well-constrained. Still maturing: see "
                                      "`dev_notes/VALIDATION_CAMPAIGN.md` Stage 15 for its validation status.")
    lam_c: float = _f(2.0, "Continuum",
                       "Curvature penalty weight for the continuum spline. Default 2.0, "
                       "inherited from `lam`'s own default as a starting point -- not "
                       "independently calibrated (see `tests/test_continuum.py` for what has "
                       "and hasn't been checked). Only relevant if `continuum_knots` is set.")
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
    lam: float = _f(2.0, "Regularization",
                     "P-spline roughness penalty weight on the spline coefficients' "
                     "`penalty_order`-th differences (only used when lam_selection='fixed'; "
                     "the default adaptive selection ignores it). NOTE: this default (2.0) and "
                     "everything below in this help text are calibrated for penalty_order=2 -- "
                     "an order-3 penalty is far weaker per unit lam (useful values sit orders "
                     "of magnitude higher, see `lam_grid`), and FitConfig warns if you combine "
                     "penalty_order=3 with fixed lam=2.0. Original order-2 guidance: penalty on "
                     "second differences, scaled by real velocity spacing (see "
                     "`kinextract.losvd.difference_matrix`) so this represents an actual "
                     "physical smoothing strength, not one that silently changes meaning with "
                     "grid width. Default 2.0 -- the best worst-case point across a broad (V, "
                     "sigma, h3, h4) validation grid (~12% worst-case shape residual). Values "
                     "below ~1.0 degrade quickly (worst-case shape residual 18% at lam=0.5, 43% "
                     "at lam=0.1) from noise overfitting; values above ~10 degrade more "
                     "gradually from oversmoothing. Larger = smoother; smaller = more flexible, "
                     "more sensitive to noise. This value is calibrated against the "
                     "velocity-spacing-scaled penalty specifically -- it is not portable to an "
                     "unscaled curvature penalty, where the same nominal value represents a "
                     "different physical smoothing strength depending on grid width.")
    ridge_lam: float = _f(1.0, "Regularization",
                           "Ridge strength used only for the optimizer's own starting point "
                           "(see `kinextract.losvd.default_initial_coeffs`) -- not part "
                           "of the fitted objective itself, so it does not trade off against "
                           "`lam` the way it might look like it should. Rarely needs changing; "
                           "its only job is landing the optimizer in a reasonable basin before "
                           "the real (lam-regularized) optimization takes over.")
    lam_selection: str = _f("aic_local", "Regularization",
                             "'aic_local' (default) or 'fixed'. 'aic_local' picks `lam` per fit "
                             "from `lam_grid` via `kinextract.aic_regularization."
                             "select_lambda_aic_local` (minimizing chi2 + 2*m_eff, Lipka & Thomas "
                             "2021/Thomas & Lipka 2022), instead of always using the fixed `lam` "
                             "value above -- measurably reduces sigma bias relative to a fixed "
                             "`lam=2.0` across a broad (sigma, h4) validation grid (see "
                             "`dev_notes/VALIDATION_CAMPAIGN.md` Stage 12); h4 bias is a genuine mixed "
                             "result (better in some regimes, worse in others). Set 'fixed' to "
                             "get the old, always-`lam` behavior exactly -- required when "
                             "`continuum_knots` is set (adaptive selection doesn't support "
                             "continuum-cofitting yet; `fit_losvd` falls back to 'fixed' with a "
                             "warning if you leave this at 'aic_local' anyway), and useful for "
                             "reproducing a specific `lam` deliberately (e.g. `ablation.py`-style "
                             "sweeps, which always force 'fixed' internally regardless of this "
                             "field). `fit_losvd`'s result dict reports whatever `lam` was "
                             "actually used as `lam_used`. `bootstrap_losvd` runs the selection "
                             "once (at the point estimate) and holds that `lam` fixed for every "
                             "bootstrap replicate, rather than re-selecting per replicate -- "
                             "cheaper, at the cost of not propagating lam-selection uncertainty "
                             "into the reported band. Also affects `bootstrap_losvd`'s default "
                             "bias-correction table when no `bias_calibration` was passed "
                             "explicitly: it switches to the `m_eff`-axis table "
                             "(`regularization_bias_emiles_multitemplate_meff.txt`) whenever the "
                             "fit's own `lam` (however it got there) is more than 25% away from "
                             "the plain table's own fixed `lam=2.0` -- see `bootstrap_losvd`'s own "
                             "docstring/body comment for why the check is against the actual "
                             "`lam` value, not against whether selection ran.")
    lam_grid: tuple | None = _f(None, "Regularization",
                          "Candidate `lam` values `select_lambda_aic_local` scans over when "
                          "`lam_selection='aic_local'`. Only used in that mode. Default None "
                          "auto-selects per `penalty_order` via `resolved_lam_grid()`: "
                          "(0.5, 1, 2, 4, 8, 16) for the order-2 penalty (the grid this "
                          "package's original validation sweep used) and (1, 8, 64, 512, 4096, "
                          "32768) for order 3 (a third-difference penalty is much weaker per "
                          "unit lam, so its useful range sits orders of magnitude higher -- "
                          "measured directly in `dev_notes/shape_residual/`, scripts 02-06). "
                          "Set an explicit tuple to override either. Widen it if a target's own "
                          "true sigma is well outside the 20-400 km/s validated range.")

    penalty_order: int = _f(3, "Regularization",
                          "Difference order of the P-spline roughness penalty on the log-LOSVD "
                          "spline coefficients. Default 3 (as of the 2026-07 shape-residual "
                          "investigation -- promoted on a 480-case paired validation: better "
                          "mean shape residual in 23/32 (sigma, S/N, shape, recipe) cells, "
                          "|sigma bias| 5.6 vs 15.5 km/s overall, wing-mass inflation "
                          "eliminated; see dev_notes/REGULARIZATION_INVESTIGATION_LOG.md). "
                          "2 penalizes second differences -- its "
                          "null space is *linear* log-b, so heavy smoothing pushes the recovered "
                          "shape toward exponential (slowly-decaying) tails; this is the root "
                          "mechanism of the documented low-S/N wing-mass/sigma inflation AND of "
                          "the core shape residual (the two failure modes squeeze `lam` from "
                          "both sides -- see dev_notes/REGULARIZATION_SHAPE_RESIDUAL_HANDOFF.md, "
                          "2026-07-26 status update); order 2 remains the right choice ONLY "
                          "when reproducing pre-2026-07 validated behavior exactly, and keeps a "
                          "modest edge on shape residual for genuinely non-Gaussian LOSVDs "
                          "(|h3|/h4 ~ 0.06+) at sigma 50-140 and S/N ~ 100, at the cost of its "
                          "wing/sigma inflation everywhere else. 3 penalizes third differences -- null "
                          "space is *quadratic* log-b, i.e. any Gaussian (any V, sigma, "
                          "amplitude) is penalty-free, so heavy smoothing shrinks toward the "
                          "best-fitting Gaussian instead of fat tails: wing inflation is "
                          "eliminated by construction and shape residual improves nearly "
                          "everywhere on Gaussian-truth validation; the tradeoff is stronger "
                          "attenuation of real h3/h4 structure at low S/N (where that structure "
                          "is near the information limit anyway). Interacts with `lam`/"
                          "`lam_grid`: order-3 penalties need lam values ~2-3 orders of "
                          "magnitude larger (see `lam_grid`); the fixed-`lam` default (2.0) is "
                          "calibrated for order 2 ONLY -- with penalty_order=3 use "
                          "lam_selection='aic_local' (the default) or retune lam explicitly. "
                          "IMPORTANT: order 3 drives fits into a heavy-penalty regime where "
                          "JAX's float32 objective can stall L-BFGS-B on wide grids "
                          "(sigma >~ 200; measured: 35% vs 6% shape residual at sigma=300, "
                          "S/N=30, lam=65536, float32 vs float64) -- the package therefore "
                          "enables JAX float64 at import time (see `kinextract/__init__.py`; "
                          "opt out with KINEXTRACT_NO_X64=1, but never combine that "
                          "opt-out with penalty_order=3 on broad targets). Order-3 selection "
                          "also activates two selection safeguards in `fit_losvd` "
                          "(delta-AIC parsimony tiebreak + sub-knot-scale sigma candidate "
                          "guard) -- see `aic_regularization.select_lambda_aic_local`.")

    taper_sfac: float = _f(1.8, "Regularization",
                            "Where the roughness penalty starts ramping up in the wings, in "
                            "units of the LOSVD's own recovered width (see "
                            "`kinextract.losvd.tapered_lam_vector`). Default 1.8, the "
                            "value the original `kinextract` pipeline uses. Beyond this "
                            "distance the penalty grows as `taper_power`, which is what "
                            "suppresses a broad low-amplitude pedestal -- a shape a uniform "
                            "penalty cannot distinguish from a compact LOSVD, and which "
                            "inflates every reported moment. Measuring the onset in units of "
                            "the recovered width (not the velocity grid) is what makes one "
                            "setting work across sigma~20-400 without retuning. Raising it "
                            "leaves more of the wings lightly smoothed (keeps genuine "
                            "high-velocity structure, risks the pedestal returning); lowering "
                            "it clamps the wings harder and closer in, at the cost of real "
                            "tail signal.")
    lam_wing: float = _f(0.0, "Regularization",
                          "Absolute strength of the wing ramp, independent of `lam` (see "
                          "`kinextract.losvd.tapered_lam_vector`). Default 0.0 (off). The core "
                          "keeps whatever strength `lam`/`lam_selection` chooses; this controls "
                          "only how hard the wings beyond `taper_sfac` are clamped. It is a "
                          "separate number because `lam` is legitimately O(1) for a "
                          "well-constrained core, and a wing ramp scaled by that is far too "
                          "weak to suppress a pedestal -- measured: with the two coupled, the "
                          "real NGC 5102 nucleus stayed at 80.6 km/s against an independent ~54 "
                          "km/s reference measurement. **Default 0.0 (off), deliberately: this mechanism is "
                          "ported and mechanically verified but NOT yet validated to fix the "
                          "real-data sigma inflation it was ported for.** With the taper "
                          "engaged and its onset seeded from the LOSVD core width, the real "
                          "NGC 5102 nucleus still returned 79 km/s against an independent ~54 km/s "
                          "reference measurement -- the wing ramp demonstrably fires (verified per-row) but "
                          "the excess width is not, in that configuration, coming from where "
                          "the ramp reaches. Leaving it off keeps the default path exactly the "
                          "behaviour the 480-case validation grid covers. Turn it on only for "
                          "deliberate experiments, and check the recovered curve, not just the "
                          "moments. See dev_notes/PORT_STATUS.md.")
    taper_power: float = _f(4.0, "Regularization",
                             "Exponent of the wing ramp beyond `taper_sfac` (see "
                             "`taper_sfac`). Default 4.0, matching the original `kinextract` "
                             "pipeline: the penalty is 59x its core strength at 5 sigma and "
                             "229x at 7 sigma. Lower values ramp more gently and leave more "
                             "wing freedom; 0.0 disables the taper entirely, recovering a "
                             "uniform penalty (not recommended -- that is the configuration "
                             "under which a divergent wing pedestal was measured to inflate "
                             "recovered sigma by ~60% on real MUSE data).")
    max_edge_amplitude_frac: float = _f(0.01, "Regularization",
                            "A candidate `lam` is rejected during adaptive selection if its "
                            "fitted LOSVD still exceeds this fraction of its own peak at "
                            "either end of the velocity grid. Default 0.01 (1%). This is a "
                            "validity condition rather than a smoothness preference: the "
                            "moment integrals only mean something if the velocity grid "
                            "actually contains the LOSVD, so a fit that is still at tens of "
                            "percent of peak at the edge is being integrated over a range "
                            "where it was never constrained. That divergent edge pedestal is "
                            "what inflated recovered sigma ~2x on real NGC 5102 MUSE spectra "
                            "(84.5 and 100.7 km/s against independent reference measurements of 54.3 and "
                            "40-45); with this constraint the same data give 49.1 and 40.3. "
                            "Neither the AIC criterion nor a unimodality/jaggedness rule can "
                            "detect it -- the pedestal solutions are smooth and single-peaked, "
                            "and on systematics-dominated residuals the extra flexibility "
                            "genuinely does lower chi2. The constraint is inert on data with "
                            "no pedestal (it excludes nothing on the mock grids), which is why "
                            "it does not repeat the non-portability that killed `lam_wing`. "
                            "0.0 disables it. See dev_notes/real_data_sigma/FINDINGS.md.")
    self_normalize_lam_grid: bool = _f(False, "Regularization",
                            "If True, `lam_selection='aic_local'` rescales its candidate "
                            "`lam_grid` by `(dv/knot_spacing_kms)^2` (`dv` = this fit's actual "
                            "knot spacing) before scanning, so the *effective* penalty strength "
                            "explored is invariant to knot spacing -- cancels exactly the "
                            "`(knot_spacing_kms/dv)^2` factor `kinextract.losvd.difference_matrix` "
                            "applies to the penalty operator itself (see `difference_matrix`'s own "
                            "docstring). Default False, deliberately: this fixes the mechanism "
                            "behind a real, confirmed bug -- on a narrow grid (small vmax-vmin "
                            "relative to `min_interior_knots`, e.g. sigma 25-55 under the "
                            "validated NSIG=7 convention), the fixed absolute lam_grid is "
                            "uniformly inflated in effective strength, and AIC selection lands at "
                            "the grid boundary (severe oversmoothing, +33% to +178% sigma bias "
                            "measured) -- but turning this on changes what numeric value "
                            "`result['lam_used']` reports for exactly those narrow-grid fits, and "
                            "`bias_correction.PerLamCalibration`'s tables were built assuming "
                            "today's (unscaled) lam_used values. Enabling this without rebuilding "
                            "those tables would silently feed the wrong canonical-lam bracket to "
                            "bias correction for narrow-grid fits. Opt in only for direct "
                            "characterization of the oversmoothing bug itself, or once the "
                            "calibration tables have been rebuilt to match -- see "
                            "dev_notes/regularization/FINDINGS.md.")
    n_sigl0_iter: int = _f(3, "Regularization",
                            "Fixed-point iterations used to make the wing taper "
                            "self-consistent: fit, measure the recovered width, rebuild the "
                            "taper around it, refit. Default 3. The taper's onset is defined "
                            "in units of the LOSVD's own width, which is not known before the "
                            "fit -- this is how that circularity is resolved without needing "
                            "a prior sigma estimate from the user. 1 disables the iteration "
                            "(taper built from the first-pass width only).")
    losvd_single_start: bool = _f(False, "Regularization",
                                   "If True, the FIRST alternating iteration's LOSVD sub-fit uses "
                                   "a single, deterministic starting point instead of "
                                   "`fit_log_spline_map`'s own default multi-start pool -- later "
                                   "iterations still warm-start from the previous iteration's own "
                                   "converged coefficients exactly as they do when this is False "
                                   "(only the iteration-0 multi-start *search* is ablated, not the "
                                   "separate warm-start mechanism). Exists for isolating the "
                                   "multi-start search's own marginal effect (ablation testing); "
                                   "not recommended for a single point-estimate fit -- multi-start "
                                   "exists to catch a real, otherwise-invisible failure mode "
                                   "(~1-in-150 fits landing in a meaningfully-worse local minimum). "
                                   "`bootstrap_losvd`'s own `replicate_single_start=True` default "
                                   "sets this True for every bootstrap replicate deliberately, "
                                   "though: an occasional bad replicate there is just an outlier in "
                                   "an n_boot-sized distribution, not the reported point estimate, "
                                   "so the risk profile differs from a single fit. Default False "
                                   "(multi-start). Bug fixed 2026-07-30: this used to force the "
                                   "deterministic start on EVERY iteration, not just the first, "
                                   "which disabled warm-starting entirely and made True fits ~3-5x "
                                   "*slower* than multi-start on some real cases -- the opposite of "
                                   "its own intent.")

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
                     "`examples/notebooks/`'s noise-correlation notebook for the full "
                     "worked pattern, including why `bootstrap_losvd` needs no changes at "
                     "all once whitening is done correctly (GLS pre-whitening makes "
                     "residuals i.i.d. by construction, regardless of `rho`).")

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
        if self.lam_selection not in ("aic_local", "fixed"):
            raise ValueError(
                f"lam_selection must be 'aic_local' or 'fixed', got {self.lam_selection!r}."
            )
        if self.penalty_order not in (2, 3):
            raise ValueError(
                f"penalty_order must be 2 or 3, got {self.penalty_order!r} -- see its help text "
                "(FitConfig.describe('penalty_order'))."
            )
        if self.penalty_order == 3 and self.lam_selection == "fixed" and self.lam == 2.0:
            warnings.warn(
                "FitConfig: penalty_order=3 with lam_selection='fixed' and the order-2-"
                "calibrated default lam=2.0 -- a third-difference penalty is far weaker per "
                "unit lam, so lam=2.0 will barely regularize at all. Use "
                "lam_selection='aic_local' (recommended) or set lam explicitly for order 3 "
                "(useful values sit orders of magnitude higher; see FitConfig.lam_grid's "
                "help text).",
                UserWarning, stacklevel=2,
            )
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
        effective_lam = self.lam * (self.knot_spacing_kms / dv)
        # The threshold has to be expressed on THIS penalty order's own lam
        # scale. The original constant (50) was calibrated when order 2 was
        # the only option, where the sweet spot is lam~2 and the candidate
        # grid tops out at 16 -- so "effective lam > 50" meant roughly 3x
        # past the top of the useful range. An order-3 penalty is far weaker
        # per unit lam and its own validated grid tops out at 32768, so the
        # same constant fires on every ordinary order-3 fit and says
        # "severe oversmoothing" about values that are simply correct --
        # observed directly in every notebook after the order-3 default
        # landed. Anchoring to `3 * max(default grid for this order)` keeps
        # the order-2 threshold at 48 (i.e. the validated behaviour, within
        # rounding) and makes it meaningful rather than constant-spam for
        # order 3.
        effective_lam_limit = 3.0 * max(self._default_lam_grid())
        if effective_lam > effective_lam_limit:
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
                "`n_interior_knots` override.",
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

    def _default_lam_grid(self) -> tuple:
        """The built-in candidate grid for this `penalty_order`, ignoring
        any explicit `lam_grid` override. Separate from
        `resolved_lam_grid()` because it also defines this order's own
        *lam scale*, which `__post_init__`'s effective-lam check needs
        even when the user has overridden the grid itself.
        """
        if self.penalty_order == 3:
            return (1.0, 8.0, 64.0, 512.0, 4096.0, 32768.0)
        return (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

    def resolved_lam_grid(self) -> tuple:
        """`lam_grid` if explicitly set, otherwise the default grid for
        this `penalty_order` -- see `lam_grid`'s help text. Always use
        this (not the raw field) when the actual candidate grid is
        needed.
        """
        if self.lam_grid is not None:
            return tuple(self.lam_grid)
        return self._default_lam_grid()

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
