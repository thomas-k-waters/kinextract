"""Tests for kinextract.robust_fit.fit_losvd_robust -- see that module's
own docstring for what each `cleaning_method` mode does and its real-data
validation record. Consolidates what were previously separate
`test_cleaning.py` (`fit_losvd_clean`) and `test_variance_inflation.py`
(`fit_losvd_inflate`) test files, now that both are one function with a
`cleaning_method` parameter -- parametrized over `cleaning_method` wherever the two
modes make the same underlying claim, kept as separate mode-specific
tests where they don't.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from kinextract import (
    FitConfig, build_template_stack, build_spline_truth, compute_fit_rms, compute_local_inflation,
    compute_losvd_moments, fit_losvd_robust, fit_losvd_robust_parallel, full_pixel_model,
    reconstruct_continuum,
)
from kinextract.cleaning import CAII_TRIPLET_CENTERS_A
from kinextract.multi_template import fit_losvd
from kinextract.noise_model import whiten

CA_CENTERS = [8498.02, 8542.09, 8662.14]


def _make_templates(wavelength, n_templates=1):
    depths_list = [[0.55, 0.70, 0.65], [0.30, 0.65, 0.35], [0.60, 0.25, 0.55]][:n_templates]
    widths = [5.0, 3.0, 7.0][:n_templates]
    templates = []
    for depths, width in zip(depths_list, widths):
        t = np.ones_like(wavelength)
        for cen, d in zip(CA_CENTERS, depths):
            t -= d * np.exp(-0.5 * ((wavelength - cen) / width) ** 2)
        templates.append(t)
    return templates


def _mock_case(vel=0.0, sig=140.0, snr=100.0, seed=0, n_templates=1, continuum_knots=None):
    """Unwhitened `(A_stack, flux, errors, wave_fit, cfg, v_grid, sig)` --
    `fit_losvd_robust`'s own input contract regardless of `cleaning_method`."""
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]
    templates = _make_templates(wavelength, n_templates)
    true_w = np.ones(n_templates) / n_templates

    cfg = FitConfig(vmin=vel - 7 * sig, vmax=vel + 7 * sig, continuum_knots=continuum_knots)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=vel, sigma=sig, wiggle_scale=0.0, seed=seed)
    A_stack = build_template_stack(wave_fit, wavelength, templates, v_grid)
    galaxy_noiseless = sum(true_w[k] * (A_stack[k] @ b_truth) for k in range(n_templates))
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    flux = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    return A_stack, flux, errors, wave_fit, cfg, v_grid, sig


@pytest.mark.parametrize("cleaning_method", ["clip", "inflate"])
def test_runs_and_produces_sane_output_with_the_expected_extra_fields(cleaning_method):
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0)
    result = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method=cleaning_method)
    assert np.all(np.isfinite(result["b"])) and np.all(result["b"] > 0)
    assert "clipped_mask" in result and result["clipped_mask"].shape == wave_fit.shape
    assert result["clipped_mask"].dtype == bool
    assert result["n_clipped"] == int(result["clipped_mask"].sum())
    if cleaning_method == "inflate":
        assert "inflation_factor" in result and result["inflation_factor"].shape == wave_fit.shape
        assert "errors_inflated" in result and result["errors_inflated"].shape == errors.shape
    else:
        assert 1 <= result["n_iterations"] <= 3


def test_plain_fit_has_no_clip_inflate_fields_but_the_same_common_ones():
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0)
    result = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method=None)
    assert result["n_clipped"] == 0 and result["n_iterations"] == 0
    assert not np.any(result["clipped_mask"])
    assert "inflation_factor" not in result


def test_invalid_cleaning_method_raises_a_clear_error():
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0)
    with pytest.raises(ValueError, match="cleaning_method"):
        fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method="bogus")


def test_inflate_without_wave_fit_raises_a_clear_error():
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0)
    with pytest.raises(ValueError, match="wave_fit"):
        fit_losvd_robust(A_stack, flux, errors, cfg, cleaning_method="inflate")


def test_clean_data_gets_barely_clipped_and_matches_a_plain_fit():
    """On genuinely clean (Gaussian-noise-only) data, sigma-clipping at
    the conventional 3-sigma threshold should clip a tiny, statistically
    expected fraction of pixels and barely change the result."""
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=1)
    A_stack_w, y_w, _ = whiten(A_stack, flux, errors, rho=cfg.rho)
    plain = fit_losvd(A_stack_w, y_w, cfg)
    cleaned = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method="clip")

    assert cleaned["n_clipped"] <= max(3, int(0.02 * len(wave_fit))), (
        f"expected only a handful of pixels clipped on clean data, got {cleaned['n_clipped']}"
    )
    m_plain = compute_losvd_moments(v_grid, plain["b"])
    m_cleaned = compute_losvd_moments(v_grid, cleaned["b"])
    assert abs(m_plain["sherm"] - m_cleaned["sherm"]) < 3.0


def test_clean_data_is_barely_inflated_and_matches_a_plain_clean_fit():
    """On genuinely clean data, the whitened residual is already ~N(0,1)
    almost everywhere, so the local inflation factor should sit near its
    1.0 floor and the result should be close to a plain clip-mode fit."""
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=2)
    plain = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method="clip", robust_scale=True)
    inflated = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method="inflate")

    assert np.median(inflated["inflation_factor"]) < 1.5, (
        f"expected clean data to stay near the 1.0 floor, "
        f"got median inflation {np.median(inflated['inflation_factor']):.2f}"
    )
    m_plain = compute_losvd_moments(v_grid, plain["b"])
    m_inflated = compute_losvd_moments(v_grid, inflated["b"])
    assert abs(m_plain["sherm"] - m_inflated["sherm"]) < 5.0


@pytest.mark.parametrize("n_templates", [1, 3])
def test_localized_contaminant_is_identified_and_clipped(n_templates):
    """The actual motivating scenario: a cluster of severely deviant
    pixels (mimicking an unmodeled sky-subtraction residual or cosmic
    ray) get correctly flagged and excluded, and the cleaned fit is
    dramatically closer to truth than the uncleaned one on average.

    Uses S/N=30 (not the module default 100) and a 10-pixel-wide
    contaminant: a heavily curvature-regularized fit is already fairly
    robust to a tiny contaminant at high S/N, so this uses a scenario
    that actually threatens fit accuracy.
    """
    _, _, _, wave_fit0, cfg, v_grid, true_sig = _mock_case(sig=140.0, snr=30.0, seed=0, n_templates=n_templates)
    contaminated_idx = np.zeros(len(wave_fit0), dtype=bool)
    contaminated_idx[50:60] = True  # 10 pixels, arbitrary location

    uncleaned_errs, cleaned_errs, all_clipped = [], [], True
    for seed in range(8 if n_templates == 1 else 5):
        A_stack, flux, errors, wave_fit, _, _, _ = _mock_case(sig=140.0, snr=30.0, seed=seed, n_templates=n_templates)
        flux_contaminated = flux.copy()
        flux_contaminated[contaminated_idx] += 50.0 * errors[contaminated_idx]

        A_stack_w, y_w, _ = whiten(A_stack, flux_contaminated, errors, rho=cfg.rho)
        uncleaned = fit_losvd(A_stack_w, y_w, cfg)
        cleaned = fit_losvd_robust(A_stack, flux_contaminated, errors, cfg, wave_fit=wave_fit,
                                    cleaning_method="clip", clip_sigma=3.0)
        all_clipped &= bool(np.all(cleaned["clipped_mask"][contaminated_idx]))

        uncleaned_errs.append(abs(compute_losvd_moments(v_grid, uncleaned["b"])["sherm"] - true_sig))
        cleaned_errs.append(abs(compute_losvd_moments(v_grid, cleaned["b"])["sherm"] - true_sig))

    assert all_clipped, "the injected outlier pixels should be clipped in every seed"
    mean_uncleaned, mean_cleaned = np.mean(uncleaned_errs), np.mean(cleaned_errs)
    threshold = mean_uncleaned / 2 if n_templates == 1 else mean_uncleaned
    assert mean_cleaned < threshold, (
        f"cleaning should substantially improve mean accuracy on this contaminated case "
        f"(n_templates={n_templates}), got uncleaned={mean_uncleaned:.2f} cleaned={mean_cleaned:.2f}"
    )


def test_localized_contaminant_produces_localized_capped_inflation():
    """The actual motivating scenario for inflate mode: a cluster of
    severely deviant pixels should drive a large, *capped* (not
    unbounded) inflation factor concentrated near that cluster, not
    smeared uniformly across the whole spectrum."""
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(sig=140.0, snr=30.0, seed=0)
    contaminated_idx = np.zeros(len(wave_fit), dtype=bool)
    contaminated_idx[50:60] = True
    flux_contaminated = flux.copy()
    flux_contaminated[contaminated_idx] += 50.0 * errors[contaminated_idx]

    cap = 6.0
    result = fit_losvd_robust(A_stack, flux_contaminated, errors, cfg, wave_fit=wave_fit,
                               cleaning_method="inflate", inflation_cap=cap)

    assert np.all(result["inflation_factor"] <= cap + 1e-9), "inflation must never exceed the configured cap"
    near_contaminant = result["inflation_factor"][40:70]
    far_from_contaminant = np.concatenate([result["inflation_factor"][:20], result["inflation_factor"][-20:]])
    assert np.max(near_contaminant) > 1.5 * np.median(far_from_contaminant), (
        "expected inflation to concentrate near the injected contaminant, not spread uniformly"
    )


def test_inflation_never_deflates_below_the_input_errors():
    """The whole point is a floor at 1x -- inflation only ever grows the
    error bar, never shrinks it below what was measured."""
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=1)
    result = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method="inflate")
    assert np.all(result["errors_inflated"] >= errors - 1e-12)
    assert np.all(result["inflation_factor"] >= 1.0 - 1e-12)


def test_a_severe_localized_contaminant_is_far_better_calibrated_than_an_unclipped_fit():
    """Inflate mode's actual validated benefit: a much better-calibrated
    residual RMS on contaminated data. The whitened RMS of the *inflated*
    fit (evaluated against its own inflated errors) should land close to
    1, while the same contaminated data fit plainly should not."""
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(sig=140.0, snr=30.0, seed=0)
    contaminated_idx = np.zeros(len(wave_fit), dtype=bool)
    contaminated_idx[50:60] = True
    flux_contaminated = flux.copy()
    flux_contaminated[contaminated_idx] += 50.0 * errors[contaminated_idx]

    A_stack_w, y_w, _ = whiten(A_stack, flux_contaminated, errors, rho=cfg.rho)
    plain = fit_losvd(A_stack_w, y_w, cfg)
    model_plain_w = full_pixel_model(plain, A_stack_w, wave_fit, cfg)
    rms_plain = compute_fit_rms(y_w, model_plain_w, np.ones_like(y_w))

    inflated = fit_losvd_robust(A_stack, flux_contaminated, errors, cfg, wave_fit=wave_fit, cleaning_method="inflate")
    errors_inflated = inflated["errors_inflated"]
    A_stack_w_i, y_w_i, _ = whiten(A_stack, flux_contaminated, errors_inflated, rho=cfg.rho)
    model_inflated_w = full_pixel_model(inflated, A_stack_w_i, wave_fit, cfg)
    rms_inflated = compute_fit_rms(y_w_i, model_inflated_w, np.ones_like(y_w_i))

    assert rms_plain > 2.0, "sanity check: the injected contamination should visibly inflate the plain-fit RMS"
    assert rms_inflated < rms_plain, (
        f"expected variance inflation to bring the residual RMS much closer to 1 than an "
        f"unclipped fit on the same contaminated data, got plain={rms_plain:.2f} inflated={rms_inflated:.2f}"
    )
    assert rms_inflated < 2.0


def test_excluded_wavelength_windows_protect_a_contaminant_inside_them_clip():
    """A severe outlier landing inside a protected window must survive
    every iteration regardless of `clip_sigma`, while an equally severe
    outlier outside any protected window is clipped normally."""
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(sig=140.0, snr=30.0, seed=0)
    protected_idx = np.argmin(np.abs(wave_fit - CAII_TRIPLET_CENTERS_A[0]))
    unprotected_idx = np.argmin(np.abs(wave_fit - 8600.0))  # between the 2nd/3rd CaII lines

    flux_contaminated = flux.copy()
    flux_contaminated[protected_idx] += 50.0 * errors[protected_idx]
    flux_contaminated[unprotected_idx] += 50.0 * errors[unprotected_idx]

    windows = tuple((c - 8.0, c + 8.0) for c in CAII_TRIPLET_CENTERS_A)
    result = fit_losvd_robust(A_stack, flux_contaminated, errors, cfg, wave_fit=wave_fit,
                               cleaning_method="clip", clip_sigma=3.0, excluded_wavelength_windows=windows)

    assert not result["clipped_mask"][protected_idx], (
        "a contaminant inside an excluded window must never be clipped"
    )
    assert result["clipped_mask"][unprotected_idx], (
        "a contaminant outside any excluded window should still be clipped normally"
    )


def test_excluded_wavelength_windows_protects_a_narrow_line_core_from_the_cleanup_clip_inflate():
    """The motivating real-data bug (dev_notes/real_data_sigma/FINDINGS.md):
    `compute_local_inflation`'s own median-filtered local scale is
    deliberately robust to a narrow (1-2 pixel) residual spike -- exactly
    what a real CaII line core looks like when the model doesn't
    perfectly reproduce its depth. Without protection, neither the
    preliminary nor the final cleanup clip can tell that apart from a
    cosmic ray, and removes it. Confirm directly: an injected single-pixel
    deficit at a CaII center is clipped without `excluded_wavelength_windows`,
    and survives with it."""
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(sig=140.0, snr=60.0, seed=3)
    ca_idx = int(np.argmin(np.abs(wave_fit - CA_CENTERS[0])))
    flux_deficit = flux.copy()
    flux_deficit[ca_idx] -= 20.0 * errors[ca_idx]  # a narrow, severe single-pixel deviation

    unprotected = fit_losvd_robust(A_stack, flux_deficit, errors, cfg, wave_fit=wave_fit, cleaning_method="inflate")
    assert unprotected["clipped_mask"][ca_idx], (
        "sanity check: an unprotected 20-sigma single-pixel deficit at a CaII center "
        "should be clipped by the loose cleanup pass"
    )

    windows = tuple((c - 8.0, c + 8.0) for c in CA_CENTERS)
    protected = fit_losvd_robust(A_stack, flux_deficit, errors, cfg, wave_fit=wave_fit,
                                  cleaning_method="inflate", excluded_wavelength_windows=windows)
    assert not protected["clipped_mask"][ca_idx], (
        "excluded_wavelength_windows should keep the CaII-center pixel in the fit "
        "regardless of its residual size"
    )


def test_excluded_wavelength_windows_without_wave_fit_raises_a_clear_error():
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0)
    with pytest.raises(ValueError, match="wave_fit"):
        fit_losvd_robust(A_stack, flux, errors, cfg, cleaning_method="clip",
                          excluded_wavelength_windows=((8490.0, 8506.0),))


def test_max_clip_fraction_warns_on_a_large_systematic_mismatch():
    """A wildly wrong velocity grid (mimicking a large redshift error or
    similar systematic mismatch) should clip a large fraction of pixels
    and trigger the sanity warning -- distinguishing "a few bad pixels"
    from "something fundamental is wrong," which iterative clipping alone
    cannot fix and shouldn't be allowed to quietly paper over.
    """
    A_stack, flux, errors, wave_fit, cfg, v_grid, true_sig = _mock_case(seed=3)
    rng = np.random.default_rng(4)
    flux_bad = flux.copy()
    bad_idx = rng.choice(len(flux), size=int(0.5 * len(flux)), replace=False)
    flux_bad[bad_idx] += rng.normal(0.0, 20.0, len(bad_idx)) * errors[bad_idx]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fit_losvd_robust(A_stack, flux_bad, errors, cfg, wave_fit=wave_fit,
                                   cleaning_method="clip", max_clip_fraction=0.1)
        assert any("above the" in str(x.message) and "sanity threshold" in str(x.message) for x in w)
    assert result["n_clipped"] / len(wave_fit) > 0.1


class TestRobustScale:
    """`robust_scale=True` (`"clip"` mode only) -- found necessary on real
    MUSE data where the default (raw whitened residual, assuming a
    correctly-calibrated noise vector) clipped 90%+ of pixels and
    collapsed to a biased, unconstrained baseline, because real chi2_red
    there runs 11-28: a nominal 3-sigma cut was actually well under 1
    true sigma almost everywhere, not just at the genuine contamination.
    """

    def test_matches_plain_on_well_calibrated_noise(self):
        A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=2)
        plain = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit,
                                  cleaning_method="clip", robust_scale=False)
        robust = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit,
                                   cleaning_method="clip", robust_scale=True)
        assert abs(plain["n_clipped"] - robust["n_clipped"]) <= 3

    def test_avoids_runaway_under_global_noise_miscalibration(self):
        """The actual motivating failure mode: the whole spectrum's noise
        is under-estimated (not a few localized bad pixels), mimicking
        real chi2_red >> 1."""
        A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=3, snr=100.0)
        rng = np.random.default_rng(3)
        flux_mismatched = flux + rng.normal(0.0, 4.0, len(flux)) * errors  # uniform excess noise

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # robust_scale=False explicitly -- reproduces the naive behavior this test
            # exists to contrast against, now that robust_scale=True is the default.
            plain = fit_losvd_robust(A_stack, flux_mismatched, errors, cfg, wave_fit=wave_fit,
                                      cleaning_method="clip", robust_scale=False)  # default max_clip_fraction=0.1
            assert any("sanity threshold" in str(x.message) for x in w)
        robust = fit_losvd_robust(A_stack, flux_mismatched, errors, cfg, wave_fit=wave_fit,
                                   cleaning_method="clip", robust_scale=True, max_clip_fraction=0.9)

        assert plain["n_clipped"] > 0.3 * len(wave_fit)
        assert robust["n_clipped"] < 0.05 * len(wave_fit)

    def test_still_catches_a_real_localized_contaminant_when_robust(self):
        A_stack, flux, errors, wave_fit, cfg, v_grid, true_sig = _mock_case(sig=140.0, snr=30.0, seed=0)
        contaminated_idx = np.zeros(len(wave_fit), dtype=bool)
        contaminated_idx[50:60] = True
        flux_contaminated = flux.copy()
        flux_contaminated[contaminated_idx] += 50.0 * errors[contaminated_idx]

        robust = fit_losvd_robust(A_stack, flux_contaminated, errors, cfg, wave_fit=wave_fit,
                                   cleaning_method="clip", robust_scale=True)
        assert np.all(robust["clipped_mask"][contaminated_idx])


@pytest.mark.parametrize("cleaning_method", ["clip", "inflate"])
def test_continuum_knots_can_be_combined_with_cleaning_method(cleaning_method):
    """Confirms continuum-cofitting runs end to end under both cleaning_method
    modes and returns a sane, near-flat continuum for this mock's true
    continuum of exactly 1.0."""
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0, continuum_knots=4)
    result = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method=cleaning_method)
    assert result["success"]
    assert "C" in result and "continuum_coeffs" in result
    C_full = reconstruct_continuum(result, wave_fit, cfg)
    assert C_full.shape == wave_fit.shape
    np.testing.assert_allclose(C_full, np.ones_like(C_full), atol=0.2)


def test_continuum_knots_without_wave_fit_raises_a_clear_error():
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0, continuum_knots=4)
    with pytest.raises(ValueError, match="wave_fit"):
        fit_losvd_robust(A_stack, flux, errors, cfg, cleaning_method="clip")


def test_reconstruct_continuum_returns_scalar_one_without_continuum_knots():
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0)
    result = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method="inflate")
    assert reconstruct_continuum(result, wave_fit, cfg) == 1.0


class TestComputeLocalInflation:
    """`compute_local_inflation` in isolation -- the core smoothing/
    flooring/capping logic, decoupled from the outer fit loop."""

    def test_pure_noise_residual_stays_near_the_floor(self):
        rng = np.random.default_rng(0)
        wave = np.arange(8400.0, 8800.0, 1.25)
        resid = rng.normal(0.0, 1.0, size=wave.size)
        inflation = compute_local_inflation(wave, resid)
        assert np.all(inflation >= 1.0)
        assert np.median(inflation) < 2.0

    def test_a_wide_severe_excursion_is_capped_not_unbounded(self):
        """A multi-pixel excursion (OH residuals are a few pixels wide at
        MUSE's resolution, not single-pixel) survives the median filter
        and should saturate the cap at its center; a genuinely
        single-pixel spike is what the preliminary loose clip's job is,
        not this smoothing step's."""
        wave = np.arange(8400.0, 8800.0, 1.25)
        resid = np.zeros_like(wave)
        resid[97:104] = 500.0  # a wide, severe excursion -- survives median-filtering
        inflation = compute_local_inflation(wave, resid, cap=8.0)
        assert np.max(inflation) <= 8.0 + 1e-9
        assert inflation[100] == pytest.approx(8.0)

    def test_never_deflates_below_one(self):
        wave = np.arange(8400.0, 8800.0, 1.25)
        resid = np.full_like(wave, 0.01)  # residual far smaller than 1 sigma everywhere
        inflation = compute_local_inflation(wave, resid)
        np.testing.assert_array_equal(inflation, np.ones_like(inflation))


class TestFitLosvdRobustParallel:
    """`fit_losvd_robust_parallel`'s own core promise: fitting bins across
    worker processes must be bit-identical to fitting them one at a time
    in a serial loop -- pure wall-clock orchestration, zero accuracy cost.
    Parametrized over `cleaning_method`, mirroring the two previously-separate
    parallel test classes."""

    @staticmethod
    def _bins():
        return [_mock_case(sig=s, seed=seed) for s, seed in [(80.0, 0), (140.0, 1), (60.0, 2)]]

    @pytest.mark.parametrize("cleaning_method", ["clip", "inflate"])
    def test_n_jobs_1_matches_a_serial_loop(self, cleaning_method):
        bins = self._bins()
        A_stack_list = [b[0] for b in bins]
        flux_list = [b[1] for b in bins]
        errors_list = [b[2] for b in bins]
        wave_fit_list = [b[3] for b in bins]
        cfg_list = [b[4] for b in bins]

        kwargs = dict(cleaning_method=cleaning_method, robust_scale=True) if cleaning_method == "clip" else dict(cleaning_method=cleaning_method)
        serial = [fit_losvd_robust(A, f, e, cfg, wave_fit=w, **kwargs)
                  for A, f, e, w, cfg in zip(A_stack_list, flux_list, errors_list, wave_fit_list, cfg_list)]
        parallel = fit_losvd_robust_parallel(A_stack_list, flux_list, errors_list, cfg_list,
                                              wave_fit_list=wave_fit_list, n_jobs=1, **kwargs)

        assert len(parallel) == len(serial) == 3
        for s, p in zip(serial, parallel):
            np.testing.assert_array_equal(s["b"], p["b"])
            np.testing.assert_array_equal(s["clipped_mask"], p["clipped_mask"])
            assert s["n_clipped"] == p["n_clipped"]

    @pytest.mark.parametrize("cleaning_method", ["clip", "inflate"])
    def test_a_real_process_pool_is_bit_identical_to_serial(self, cleaning_method):
        bins = self._bins()
        A_stack_list = [b[0] for b in bins]
        flux_list = [b[1] for b in bins]
        errors_list = [b[2] for b in bins]
        wave_fit_list = [b[3] for b in bins]
        cfg_list = [b[4] for b in bins]

        kwargs = dict(cleaning_method=cleaning_method, robust_scale=True) if cleaning_method == "clip" else dict(cleaning_method=cleaning_method)
        serial = [fit_losvd_robust(A, f, e, cfg, wave_fit=w, **kwargs)
                  for A, f, e, w, cfg in zip(A_stack_list, flux_list, errors_list, wave_fit_list, cfg_list)]
        parallel = fit_losvd_robust_parallel(A_stack_list, flux_list, errors_list, cfg_list,
                                              wave_fit_list=wave_fit_list, n_jobs=2, **kwargs)

        for i, (s, p) in enumerate(zip(serial, parallel)):
            np.testing.assert_array_equal(s["b"], p["b"], err_msg=f"bin {i} mismatch")
            np.testing.assert_array_equal(s["clipped_mask"], p["clipped_mask"], err_msg=f"bin {i} mismatch")

    def test_mismatched_wave_fit_list_length_raises_a_clear_error(self):
        bins = self._bins()
        A_stack_list = [b[0] for b in bins]
        flux_list = [b[1] for b in bins]
        errors_list = [b[2] for b in bins]
        wave_fit_list = [b[3] for b in bins]
        cfg_list = [b[4] for b in bins]

        with pytest.raises(ValueError, match="one entry per bin"):
            fit_losvd_robust_parallel(A_stack_list, flux_list, errors_list, cfg_list,
                                       wave_fit_list=wave_fit_list[:-1], n_jobs=1, cleaning_method="inflate")


def test_fit_losvd_robust_raises_a_clear_error_when_too_few_pixels_survive():
    """A spectrum with fewer pixels than free LOSVD coefficients must fail
    loudly and specifically, not silently 'succeed' on a fit driven
    entirely by the roughness penalty with no real data behind it."""
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0)
    n_coef = cfg.resolved_n_interior_knots() + cfg.degree + 1
    tiny = max(1, n_coef - 5)
    with pytest.raises(RuntimeError, match="pixels survived clipping"):
        fit_losvd_robust(A_stack[:, :tiny, :], flux[:tiny], errors[:tiny], cfg,
                          wave_fit=wave_fit[:tiny], cleaning_method="clip")


@pytest.mark.parametrize("cleaning_method", ["clip", "inflate"])
def test_fit_losvd_robust_parallel_returns_a_failed_result_instead_of_crashing_the_batch(cleaning_method):
    """A per-bin failure (here: too few pixels, matching the guard above)
    must produce a NaN-filled, success=False result for that one bin, not
    crash the entire parallel batch -- confirms fit_losvd_robust_parallel's
    own worker-level try/except actually works end to end, not just the
    exception it's designed to catch."""
    good = _mock_case(seed=1)
    A_stack_good, flux_good, errors_good, wave_fit_good, cfg, v_grid, sig = good
    n_coef = cfg.resolved_n_interior_knots() + cfg.degree + 1
    tiny = max(1, n_coef - 5)
    A_stack_bad = A_stack_good[:, :tiny, :]
    flux_bad = flux_good[:tiny]
    errors_bad = errors_good[:tiny]
    wave_fit_bad = wave_fit_good[:tiny]

    with pytest.warns(UserWarning, match="returning a NaN-filled"):
        results = fit_losvd_robust_parallel(
            [A_stack_good, A_stack_bad], [flux_good, flux_bad], [errors_good, errors_bad], cfg,
            wave_fit_list=[wave_fit_good, wave_fit_bad], n_jobs=1, cleaning_method=cleaning_method)

    assert results[0]["success"] is True
    assert np.all(np.isfinite(results[0]["b"]))
    assert results[1]["success"] is False
    assert np.all(np.isnan(results[1]["b"]))
    assert np.all(np.isnan(results[1]["w"]))
