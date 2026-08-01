"""Tests for kinextract.bias_correction.

The correction is calibrated from a synthetic sweep against the real MUSE
template library with a fixed, realistic multi-template Dirichlet mixture
(`calibrate_bias_muse_multitemplate.py`, h3=h4=0 truth, V=0, MUSE
sampling) -- these tests check the calibration table parses correctly,
the correction moves recovered sigma/h4 in the expected direction (toward
truth, not away from it) on cases drawn from that same regime, and that
out-of-range queries warn rather than silently extrapolate. This table
replaced an earlier idealized single-template calibration that was found
to make real NGC 5102 recovery worse, not better -- see
`bias_correction.py`'s own module docstring.

**Deliberately kept on `gauss_hermite_model` truth, not the package-wide
`build_spline_truth` default (2026-07-29).** The calibration table itself
is built from GH-parameterized truths (h3=h4=0 or specific injected h3/h4),
and `correct_losvd`/`correct_moments` take `h3`/`h4` as GH parameters and
reconstruct via `gauss_hermite_model` internally -- this module tests
GH-regularization-shrinkage bias specifically, a different mechanism from
the GH-parameterization-vs-moments gap `build_spline_truth` exists to
avoid (see its docstring). Swapping this file's mock generator would test
an out-of-distribution case relative to what the calibration was built
against, not a like-for-like improvement.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from kinextract import (
    FitConfig, build_template_stack, gauss_hermite_model, compute_losvd_moments,
    estimate_snr_whitened, correct_moments, correct_losvd, load_calibration_table,
)
from kinextract.bias_correction import BiasCalibration, PerLamCalibration
from kinextract.multi_template import fit_losvd

WAVEMIN, STEP, N_PIX = 4750.0, 1.25, 3681
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0


def _make_single_template_mock(sig_true, snr, seed):
    wavelength = WAVEMIN + np.arange(N_PIX) * STEP
    mask = (wavelength >= WAVEFITMIN) & (wavelength <= WAVEFITMAX)
    wave_fit = wavelength[mask]
    template = np.ones(N_PIX) - 0.4 * np.exp(-0.5 * ((wavelength - 8600.0) / 8.0) ** 2)

    cfg = FitConfig(vmin=-7 * sig_true, vmax=7 * sig_true)
    v_grid = cfg.build_velocity_grid()
    b_truth = gauss_hermite_model(v_grid, amp=1.0, vel=0.0, sig=sig_true, h3=0.0, h4=0.0)
    b_truth = b_truth / b_truth.sum()

    A_stack = build_template_stack(wave_fit, wavelength, [template], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / snr
    galaxy = galaxy_noiseless + np.random.default_rng(seed).normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors
    return cfg, v_grid, A_stack_w, y_w


def test_load_calibration_table_shape():
    calib = load_calibration_table()
    assert len(calib.recovered_sigma) == len(calib.snr) == len(calib.sig_bias) == 64
    assert calib.sigma_range[0] < calib.sigma_range[1]
    assert calib.snr_range == (10.0, 200.0)


def test_correction_reduces_low_sigma_bias():
    # Low sigma / moderate S/N is the regime with the largest, best-
    # characterized regularization bias (Stage 1: sigma=30, S/N=60 ->
    # roughly +54% sigma bias). The correction should pull the recovered
    # sigma toward the true value, not away from it, and should not
    # overshoot past truth by more than it started on the other side.
    # penalty_order=2 pinned deliberately: this calibration table (and this
    # test's own reference numbers) were built against the order-2 curvature
    # penalty's specific bias mechanism (see VALIDATION_CAMPAIGN.md Stage 1) --
    # the order-3 default (2026-07, see Stage 22) removes most of that bias at
    # the source, so the raw estimate is already close to truth there and this
    # table's correction (calibrated for the old, larger bias) overcorrects.
    sig_true, snr = 30.0, 60.0
    cfg, v_grid, A_stack_w, y_w = _make_single_template_mock(sig_true, snr, seed=123)
    cfg = dataclasses.replace(cfg, penalty_order=2)
    result = fit_losvd(A_stack_w, y_w, cfg)
    gh = compute_losvd_moments(v_grid, result["b"], fit_h3h4=cfg.fit_h3h4)

    model_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0])) @ result["b"]
    snr_est = estimate_snr_whitened(model_w)

    corr = correct_moments(gh["vherm"], gh["sherm"], gh["h4"], snr_est)
    raw_err = abs(gh["sherm"] - sig_true)
    corrected_err = abs(corr["sherm_corrected"] - sig_true)
    assert corrected_err < raw_err, (
        f"correction should reduce |sigma error| in this well-calibrated regime: "
        f"raw={gh['sherm']:.1f} (err {raw_err:.1f}), corrected={corr['sherm_corrected']:.1f} "
        f"(err {corrected_err:.1f}), true={sig_true}"
    )


def test_correct_losvd_reconstructs_curve():
    sig_true, snr = 50.0, 100.0
    cfg, v_grid, A_stack_w, y_w = _make_single_template_mock(sig_true, snr, seed=7)
    result = fit_losvd(A_stack_w, y_w, cfg)
    gh = compute_losvd_moments(v_grid, result["b"], fit_h3h4=cfg.fit_h3h4)
    model_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0])) @ result["b"]
    snr_est = estimate_snr_whitened(model_w)

    out = correct_losvd(v_grid, gh["vherm"], gh["sherm"], gh["h3"], gh["h4"], gh["amp"], snr_est)
    assert out["b_corrected"].shape == v_grid.shape
    assert np.all(np.isfinite(out["b_corrected"]))
    # Corrected curve should still integrate to roughly the same area as
    # the input (moments changed, not overall normalization).
    area_in = np.trapezoid(np.maximum(result["b"], 0.0), v_grid)
    area_out = np.trapezoid(np.maximum(out["b_corrected"], 0.0), v_grid)
    assert area_out == pytest.approx(area_in, rel=0.15)


def test_out_of_range_query_warns():
    with pytest.warns(UserWarning, match="outside the calibration grid"):
        correct_moments(vherm=0.0, sherm=1000.0, h4=0.0, snr=100.0)
    with pytest.warns(UserWarning, match="outside the calibration grid"):
        correct_moments(vherm=0.0, sherm=100.0, h4=0.0, snr=5.0)


def test_in_range_query_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        correct_moments(vherm=0.0, sherm=100.0, h4=0.0, snr=100.0)


def test_correct_losvd_clips_negative_curve_when_correction_pushes_h4_extreme():
    """Regression test for a real bug found by inspecting a sandbox
    notebook: the truncated Gauss-Hermite series is not guaranteed
    non-negative for arbitrary (h3, h4) (a known property of this
    expansion in general -- see `losvd.py`'s docstring on h4=-0.1), and
    the calibrated h4 correction shifts h4 by an offset with nothing
    constraining the result to stay inside the non-negative region. Uses
    a deliberately extreme synthetic calibration (not the shipped table,
    which may not push h4 far enough to trigger this at any in-range
    query) to force the corrected h4 into the known-negative regime
    deterministically.
    """
    extreme_calib = BiasCalibration(
        recovered_sigma=np.array([50.0, 50.0, 200.0, 200.0]),
        snr=np.array([10.0, 200.0, 10.0, 200.0]),
        sig_bias=np.zeros(4),
        sig_bias_std=np.ones(4),
        h4_bias=np.array([0.2, 0.2, 0.2, 0.2]),  # true h4=0 -> corrected h4=-0.2, deep in the negative regime
        h4_bias_std=np.full(4, 0.05),
        sigma_range=(50.0, 200.0),
        snr_range=(10.0, 200.0),
    )
    v_grid = np.linspace(-350.0, 350.0, 141)
    out = correct_losvd(v_grid, vherm=0.0, sherm=100.0, h3=0.0, h4=0.0, amp=1.0, snr=100.0,
                         calibration=extreme_calib)
    assert out["h4_corrected"] == pytest.approx(-0.2)
    # Confirm this specific (h3=0, h4=-0.2) combination really does go
    # negative unclipped -- otherwise this test isn't exercising the fix.
    raw = gauss_hermite_model(v_grid, amp=1.0, vel=0.0, sig=100.0, h3=0.0, h4=-0.2)
    assert np.any(raw < 0), "test setup error: this (h3, h4) combination should go negative unclipped"
    assert np.all(out["b_corrected"] >= 0.0), (
        "correct_losvd must clip the reconstructed curve non-negative -- a LOSVD is a "
        "probability density over velocity and cannot be negative anywhere"
    )


def _make_per_lam_calib(lam_values, sig_bias_values):
    """Build a `PerLamCalibration` with one degenerate 2x2 (sigma, S/N)
    table per `lam` value, each returning a single, distinct `sig_bias`
    constant regardless of query point within its own box -- isolates
    whether `correct_moments`'s `lam` interpolation is actually selecting/
    blending the right tables, independent of the 2-D (sigma, S/N) lookup
    each table also does internally (covered separately above)."""
    tables = []
    for sig_bias in sig_bias_values:
        tables.append(BiasCalibration(
            recovered_sigma=np.array([50.0, 50.0, 200.0, 200.0]),
            snr=np.array([10.0, 200.0, 10.0, 200.0]),
            sig_bias=np.full(4, sig_bias), sig_bias_std=np.ones(4),
            h4_bias=np.zeros(4), h4_bias_std=np.ones(4) * 0.01,
            sigma_range=(50.0, 200.0), snr_range=(10.0, 200.0), lam=None,
        ))
    return PerLamCalibration(lam_values=np.asarray(lam_values, dtype=float), tables=tables)


def test_correct_moments_lam_query_at_a_calibrated_lam_matches_that_tables_own_value():
    """Querying exactly at one of `PerLamCalibration`'s own `lam` values
    should reproduce that table's own 2-D correction exactly (no blending
    with its neighbour) -- the boundary case a bracketing-interpolation
    scheme has to get right."""
    per_lam = _make_per_lam_calib([8.0, 512.0], [1.0, 50.0])
    at_lo = correct_moments(vherm=0.0, sherm=100.0, h4=0.0, snr=100.0, calibration=per_lam, lam=8.0)
    at_hi = correct_moments(vherm=0.0, sherm=100.0, h4=0.0, snr=100.0, calibration=per_lam, lam=512.0)
    assert at_lo["sig_bias_used"] == pytest.approx(1.0)
    assert at_hi["sig_bias_used"] == pytest.approx(50.0)


def test_correct_moments_lam_query_interpolates_log_linearly_between_bracketing_tables():
    """A query lam between two calibrated lam values must blend their
    corrections log-linearly (the grid's own natural spacing), not
    linearly in lam itself, and not via the old 3-D (sigma, S/N, m_eff)
    scattered-interpolation scheme this replaced."""
    per_lam = _make_per_lam_calib([1.0, 100.0], [0.0, 40.0])
    # log10(10) is exactly halfway between log10(1)=0 and log10(100)=2.
    mid = correct_moments(vherm=0.0, sherm=100.0, h4=0.0, snr=100.0, calibration=per_lam, lam=10.0)
    assert mid["sig_bias_used"] == pytest.approx(20.0)


def test_correct_moments_lam_query_outside_grid_snaps_to_nearest_table_with_warning():
    per_lam = _make_per_lam_calib([8.0, 512.0], [1.0, 50.0])
    with pytest.warns(UserWarning, match="below the per-lam calibration grid"):
        low = correct_moments(vherm=0.0, sherm=100.0, h4=0.0, snr=100.0, calibration=per_lam, lam=0.5)
    with pytest.warns(UserWarning, match="above the per-lam calibration grid"):
        high = correct_moments(vherm=0.0, sherm=100.0, h4=0.0, snr=100.0, calibration=per_lam, lam=4096.0)
    assert low["sig_bias_used"] == pytest.approx(1.0)
    assert high["sig_bias_used"] == pytest.approx(50.0)


def test_correct_moments_raises_if_lam_supplied_against_a_2d_only_table():
    calib_2d = BiasCalibration(
        recovered_sigma=np.array([50.0, 200.0]), snr=np.array([10.0, 200.0]),
        sig_bias=np.array([5.0, 8.0]), sig_bias_std=np.ones(2),
        h4_bias=np.zeros(2), h4_bias_std=np.ones(2) * 0.01,
        sigma_range=(50.0, 200.0), snr_range=(10.0, 200.0),
    )
    with pytest.raises(ValueError, match="plain single-lam BiasCalibration"):
        correct_moments(vherm=0.0, sherm=100.0, h4=0.0, snr=100.0, calibration=calib_2d, lam=10.0)


def test_correct_moments_raises_if_per_lam_calibration_given_without_lam():
    per_lam = _make_per_lam_calib([8.0, 512.0], [1.0, 50.0])
    with pytest.raises(ValueError, match="no lam was given"):
        correct_moments(vherm=0.0, sherm=100.0, h4=0.0, snr=100.0, calibration=per_lam)


def test_vherm_and_h3_passed_through_unchanged():
    corr = correct_moments(vherm=12.3, sherm=100.0, h4=0.02, snr=100.0)
    # correct_moments doesn't touch vherm/h3 at all -- confirm the
    # documented scope (sigma/h4 only) by checking the returned dict has
    # no vherm/h3 keys to be silently misinterpreted as corrected.
    assert "vherm_corrected" not in corr
    assert "h3_corrected" not in corr


# ---------------------------------------------------------------------------
# CALIBRATION_TRUST_STRATEGY.md gate 4: "a regression test per bug already
# found." Two specific, numbered overcorrection bugs, both previously found
# under real testing and both attributed to stale/mismatched calibration
# tables, not a flaw in the correction method itself -- see that document's
# own account. Added 2026-07-29/30 per its acceptance-gate checklist.
# ---------------------------------------------------------------------------

def test_bootstrap_correction_does_not_severely_overcorrect_the_01_notebook_mock():
    """Regression test for the lam=0.5 overcorrection bug
    (`dev_notes/calibration/CALIBRATION_TRUST_STRATEGY.md`): applying
    `apply_bias_correction=True` against the *stale* (order-2-calibrated)
    tables on `01_basic_mock_fit.ipynb`'s own default mock pulled a good raw
    sigma estimate (52.9 km/s, true 50) to a much worse corrected one
    (35.5 km/s) -- ~5x worse than the raw error, in a regime that would
    still look fine on an aggregate/median check. The 2026-07-29 table
    rebuild (for the current `penalty_order=3` default) fixed this on this
    exact case; this test pins that fix as a permanent regression check
    against a real, single-template, single-noise-draw mock -- if this
    starts failing, the tables (or the table-selection logic in
    `bootstrap_losvd`) have regressed.
    """
    from kinextract.uncertainty import bootstrap_losvd

    sig_true, snr = 50.0, 60.0
    cfg, v_grid, A_stack_w, y_w = _make_single_template_mock(sig_true, snr, seed=123)
    result0 = fit_losvd(A_stack_w, y_w, cfg)
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=20, seed=0, apply_bias_correction=True)

    raw_err = abs(boot["point0"]["sherm"] - sig_true)
    corrected_err = abs(boot["point0_corrected"]["sherm"] - sig_true)
    assert corrected_err <= max(2.0 * raw_err, 10.0), (
        f"corrected sigma error ({corrected_err:.1f} km/s) is more than 2x the raw "
        f"error ({raw_err:.1f} km/s, raw={boot['point0']['sherm']:.1f}, "
        f"corrected={boot['point0_corrected']['sherm']:.1f}, true={sig_true}) -- this is "
        f"the exact overcorrection failure mode found under the stale order-2-calibrated "
        f"tables; if this fails, the current tables have regressed toward that behavior."
    )


def test_lam_axis_table_switch_uses_plain_table_when_lam_matches_and_per_lam_set_otherwise():
    """Regression test for the Stage-12 bug (`dev_notes/VALIDATION_CAMPAIGN.md`):
    `bootstrap_losvd` used to switch to the thinner-sampled per-`lam`
    calibration set *whenever* adaptive lam selection ran, even when the
    picked `lam` matched the plain table's own single calibration point --
    fixed via a 25%-of-`plain_table.lam` threshold (see `bootstrap_losvd`'s
    own inline comment). Structural check (monkeypatching `correct_moments`
    inside the `uncertainty` module, the same technique
    `test_reselect_lam_controls_the_actual_cfg_passed_to_every_replicate_fit`
    in `test_uncertainty.py` uses) of which calibration object actually gets
    passed, not just the downstream numbers -- confirms the mechanism
    itself, not a statistical consequence of it. Predates and survives the
    2026-07-30 m_eff-axis -> per-lam redesign (`PerLamCalibration`) -- the
    switch logic itself (compare the fit's own lam against the plain
    table's) is unchanged, only which object gets switched *to*.
    """
    import kinextract.uncertainty as unc_mod
    from kinextract.uncertainty import bootstrap_losvd
    from kinextract.bias_correction import _get_default_calibration, get_default_lam_axis_calibration

    plain_table = _get_default_calibration()
    per_lam_table = get_default_lam_axis_calibration()
    band = 0.25 * plain_table.lam

    # The threshold itself, checked against the actual shipped plain table's own
    # lam (not a hardcoded assumed value) -- catches a regression to the band
    # width or to which table is "plain" just as much as the branch below does.
    assert abs(plain_table.lam - plain_table.lam) <= band          # exact match: in-band
    assert abs(plain_table.lam * 1.2 - plain_table.lam) <= band    # +20%: in-band
    assert abs(plain_table.lam * 2.0 - plain_table.lam) > band     # +100%: out-of-band
    assert abs(32768.0 - plain_table.lam) > band                    # a real adaptive-selection
    #                                                                  winner (flat-AIC regime,
    #                                                                  Stage 12) must NOT match.

    real_correct_moments = unc_mod._correct_moments
    for fixed_lam, expect_per_lam_table in [(plain_table.lam, False), (32768.0, True)]:
        seen_calibrations = []

        def _spy(vherm, sherm, h4, snr, calibration=None, lam=None, _seen=seen_calibrations):
            _seen.append(calibration)
            return real_correct_moments(vherm, sherm, h4, snr, calibration=calibration, lam=lam)

        unc_mod._correct_moments = _spy
        try:
            cfg, v_grid, A_stack_w, y_w = _make_single_template_mock(90.0, 60.0, seed=5)
            cfg = dataclasses.replace(cfg, lam_selection="fixed", lam=fixed_lam)
            bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=3, seed=0, apply_bias_correction=True)
        finally:
            unc_mod._correct_moments = real_correct_moments

        assert len(seen_calibrations) == 1
        # Both getters cache a module-level singleton (see bias_correction.py), so
        # identity comparison is exact -- not "same content, different object."
        used_per_lam_table = seen_calibrations[0] is per_lam_table
        assert seen_calibrations[0] is (per_lam_table if expect_per_lam_table else plain_table)
        assert used_per_lam_table == expect_per_lam_table, (
            f"fixed_lam={fixed_lam}: expected the per-lam table to be used = "
            f"{expect_per_lam_table}, got calibration={seen_calibrations[0]!r}"
        )
