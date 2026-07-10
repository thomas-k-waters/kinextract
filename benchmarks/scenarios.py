"""Scenario grid definition + FitConfig builder for the kinextract validation suite.

Defines a one-axis-at-a-time sweep (not full factorial -- see module
docstring in run_suite.py for the runtime-budget rationale) around a single
representative BASE scenario, plus two named "motivating bug" scenarios that
directly reproduce the real failures found while debugging NGC 4751
(high-sigma, wide-window V/continuum bug) and NGC 5102 (low-sigma
cross-validation disagreement against the legacy Fortran pipeline).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Optional

from kinextract import FitConfig

from benchmarks.mock_spectrum import MUSE_CAII, STIS_SETUP, InstrumentSetup


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    axis: str
    instrument: InstrumentSetup
    v_true: float
    sigma_true: float
    snr_target: float
    template_role: Literal["matched", "mismatched"] = "matched"
    losvd_shape: Literal["gaussian", "gh_moderate", "gh_strong", "double_peak"] = "gaussian"
    h3_true: float = 0.0
    h4_true: float = 0.0
    emission_level: Literal["none", "moderate", "strong"] = "none"
    xlam_max_peaks_override: Optional[int] = None
    wavefitmin_override: Optional[float] = None
    wavefitmax_override: Optional[float] = None
    sigl_override: Optional[float] = None
    notes: str = ""


# BASE point: matches notebook 01/02's established convention (V=+80, sigma=140,
# MUSE CaII, matched template, Gaussian LOSVD, S/N=50, no emission), so any
# deviation seen in a swept axis is attributable to that axis, not an
# unfamiliar starting point.
_BASE = dict(
    instrument=MUSE_CAII, v_true=80.0, sigma_true=140.0, snr_target=50.0,
    template_role="matched", losvd_shape="gaussian", h3_true=0.0, h4_true=0.0,
    emission_level="none",
)


def build_scenario_grid() -> list[Scenario]:
    scenarios: list[Scenario] = []

    scenarios.append(Scenario(scenario_id="base", axis="base", notes="BASE reference point", **_BASE))

    # --- Axis 1: velocity dispersion (low-mass star cluster -> high-mass BH) ---
    for sigma in (15.0, 60.0, 200.0, 350.0, 500.0):  # 140 is BASE, not repeated
        s = dict(_BASE); s["sigma_true"] = sigma
        scenarios.append(Scenario(scenario_id=f"sigma_{sigma:.0f}", axis="sigma", **s))

    # --- Axis 2: S/N ---
    for snr in (10.0, 20.0, 100.0, 300.0):  # 50 is BASE
        s = dict(_BASE); s["snr_target"] = snr
        scenarios.append(Scenario(scenario_id=f"snr_{snr:.0f}", axis="snr", **s))

    # --- Axis 3: instrument ---
    s = dict(_BASE); s["instrument"] = STIS_SETUP
    scenarios.append(Scenario(scenario_id="instrument_stis", axis="instrument", **s))

    # --- Axis 4: template match quality (MUSE only -- STIS has 1 template) ---
    s = dict(_BASE); s["template_role"] = "mismatched"
    scenarios.append(Scenario(scenario_id="template_mismatched", axis="template", **s))

    # --- Axis 5: emission contamination ---
    for level in ("moderate", "strong"):
        s = dict(_BASE); s["emission_level"] = level
        scenarios.append(Scenario(scenario_id=f"emission_{level}", axis="emission", **s))

    # --- Axis 6: true LOSVD shape ---
    s = dict(_BASE); s["losvd_shape"] = "gh_moderate"; s["h3_true"] = 0.05; s["h4_true"] = 0.05
    scenarios.append(Scenario(scenario_id="losvd_gh_moderate", axis="losvd_shape", **s))
    s = dict(_BASE); s["losvd_shape"] = "gh_strong"; s["h3_true"] = 0.15; s["h4_true"] = 0.10
    scenarios.append(Scenario(scenario_id="losvd_gh_strong", axis="losvd_shape", **s))
    for max_peaks in (1, 2):
        s = dict(_BASE); s["losvd_shape"] = "double_peak"
        scenarios.append(Scenario(
            scenario_id=f"losvd_double_peak_maxpeaks{max_peaks}", axis="losvd_shape",
            xlam_max_peaks_override=max_peaks, **s,
        ))

    # --- Named bug scenarios -------------------------------------------------
    # bug_ngc4751_like: reproduces the real high-sigma bug found this session --
    # widening the fit window from 8400-8750 to 7750-8900 destabilized the fit
    # (V jumped from +68.6 to +158.4 km/s, chi2_red worsened 0.39->0.78, on the
    # real NGC 4751 data). V_true=0 here since the actual complaint was that a
    # galactic-center bin should recover V~0.
    for window_label, wmin, wmax in (("narrow", 8400.0, 8750.0), ("wide", 7750.0, 8900.0)):
        for sigl_label, sigl_val in (("scaled", 380.0), ("default", 100.0)):
            scenarios.append(Scenario(
                scenario_id=f"bug_ngc4751_like__{window_label}_window__sigl_{sigl_label}",
                axis="bug", instrument=MUSE_CAII, v_true=0.0, sigma_true=380.0,
                snr_target=40.0, template_role="matched", losvd_shape="gaussian",
                wavefitmin_override=wmin, wavefitmax_override=wmax, sigl_override=sigl_val,
                notes="Reproduces the real NGC 4751 high-sigma window-widening bug",
            ))

    # bug_ngc5102_like: closest synthetic analog to the real Fortran-vs-kinextract
    # cross-validation case (low sigma, mild h3/h4, since the real disagreement
    # was in both V and sigma, ~10% fractional sigma error).
    scenarios.append(Scenario(
        scenario_id="bug_ngc5102_like", axis="bug", instrument=MUSE_CAII,
        v_true=50.0, sigma_true=60.0, snr_target=50.0, template_role="matched",
        losvd_shape="gh_moderate", h3_true=0.03, h4_true=0.02,
        notes="Closest synthetic analog to the real NGC 5102 Fortran cross-validation case",
    ))

    return scenarios


def scenario_to_fitconfig(scenario: Scenario, continuum_mode: Literal["joint", "poly_amp", "none"],
                           outdir: str, template_list_file: str, template_dir: str,
                           xlam_criterion: str = "chi2",
                           xlam_discrepancy_nsigma: float = 1.0) -> FitConfig:
    """Translate one Scenario + continuum_mode into a concrete FitConfig.

    `xlam_criterion`/`xlam_discrepancy_nsigma` default to the package
    default ("chi2") so the existing continuum-mode comparison suite
    (run_suite.py/report.py) is unaffected; passed through explicitly by
    calibrate_xlam.py's xlam-method comparison, which holds continuum_mode
    fixed at "none" instead and varies these two.
    """
    inst = scenario.instrument
    sigma_true = scenario.sigma_true
    sigl = scenario.sigl_override if scenario.sigl_override is not None else sigma_true
    # No floor: a floor here (e.g. 500 km/s) silently forces an oversized LOSVD
    # grid at low sigma -- verified to make recovery *worse*, not better, since
    # it spends the fixed nl=29 bins on velocity range the true LOSVD doesn't
    # need instead of resolution it does. This matches make_fit_state's own
    # default (xl = linspace(-4.5*sigl, 4.5*sigl, nl)) when losvd_vmin/vmax
    # aren't overridden, so a real user relying on that default wouldn't hit
    # this floor at all.
    vgrid_half = 4.5 * sigl

    wavefitmin = scenario.wavefitmin_override if scenario.wavefitmin_override is not None else inst.wavefitmin
    wavefitmax = scenario.wavefitmax_override if scenario.wavefitmax_override is not None else inst.wavefitmax

    xlam_auto_grid = tuple(10.0 ** k for k in range(2, 8)) if sigma_true >= 250 \
        else tuple(10.0 ** k for k in range(1, 6))

    common = dict(
        template_list_file=template_list_file, template_dir=template_dir, outdir=outdir,
        wavemin_full=inst.wavemin_full, step=inst.step,
        wavefitmin=wavefitmin, wavefitmax=wavefitmax, zgal=0.0,
        sigl=sigl, losvd_vmin=-vgrid_half, losvd_vmax=vgrid_half,
        xlam_auto=True, xlam_criterion=xlam_criterion, xlam_chi2_tolerance=0.02,
        xlam_discrepancy_nsigma=xlam_discrepancy_nsigma,
        xlam_auto_grid=xlam_auto_grid,
        xlam_max_peaks=scenario.xlam_max_peaks_override or 1,
        clean=False, use_spectrum_errors=True,
        map_maxiter=8000, print_every=999999,
        mask_emission_lines_in_fit=True,
    )

    if continuum_mode == "joint":
        mode_kwargs = dict(fit_continuum=True, fit_global_amp=False, continuum_poly_mode="none")
    elif continuum_mode == "poly_amp":
        mode_kwargs = dict(fit_continuum=False, fit_global_amp=True,
                            continuum_poly_mode="multiplicative", continuum_poly_bound=0.1)
    else:  # "none" -- oracle/ceiling case, mock is pre-divided by true continuum by the caller
        mode_kwargs = dict(fit_continuum=False, fit_global_amp=False, continuum_poly_mode="none")

    return FitConfig(**common, **mode_kwargs)
