"""Tests for kinextract.robust_fit.fit_losvd_robust -- see that module's
own docstring for the inflate cleaning and its real-data validation
record. Parametrized over `cleaning_method` (`"inflate"`, the default, and
`None`, a plain fit) wherever the two make the same claim.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from conftest import run_jobs_in_one_worker_process

from kinextract import (
    FitConfig, build_template_stack, build_spline_truth, compute_fit_rms, compute_local_inflation,
    compute_losvd_moments, fit_losvd_robust, fit_losvd_robust_parallel, full_pixel_model,
    reconstruct_continuum,
)
from kinextract.cleaning import CAII_TRIPLET_CENTERS_A
from kinextract.multi_template import fit_losvd
from kinextract.noise_model import whiten
from kinextract.robust_fit import _fit_losvd_robust_worker

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


@pytest.mark.parametrize("cleaning_method", [None, "inflate"])
def test_runs_and_produces_sane_output_with_the_expected_extra_fields(cleaning_method):
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0)
    result = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method=cleaning_method)
    assert np.all(np.isfinite(result["b"])) and np.all(result["b"] > 0)
    assert "clipped_mask" in result and result["clipped_mask"].shape == wave_fit.shape
    assert result["clipped_mask"].dtype == bool
    assert result["n_clipped"] == int(result["clipped_mask"].sum())
    assert result["inflation_factor"].shape == wave_fit.shape
    assert result["errors_inflated"].shape == errors.shape
    if cleaning_method is None:
        assert result["n_iterations"] == 0


@pytest.mark.parametrize("max_iter", [1, 2, 3])
def test_clipped_mask_is_exactly_the_pixel_set_the_returned_fit_excluded(max_iter):
    """Regression (2026-10-15): when the last allowed iteration still flagged new
    outliers, they used to be added to `clipped_mask` without a refit, so the
    returned `b` had been fit WITH pixels its own mask called clipped. Refitting on
    `~clipped_mask` must now reproduce the returned fit exactly, at any max_iter."""
    from kinextract.cleaning import _fit_losvd_clip
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=4)
    flux = flux.copy()
    flux[[20, 120, 220]] += 40.0 * errors[0]   # obvious outliers, flagged on iteration 1
    A_stack_w, y_w, _ = whiten(A_stack, flux, errors, rho=0.0)
    res = _fit_losvd_clip(A_stack_w, y_w, cfg, clip_sigma=3.0, max_iter=max_iter, wave_fit=wave_fit)
    keep = ~res["clipped_mask"]
    refit = fit_losvd(A_stack_w[:, keep, :], y_w[keep], cfg, wave_fit=wave_fit[keep])
    np.testing.assert_allclose(refit["b"], res["b"], rtol=1e-8, atol=1e-12)
    assert res["n_clipped"] == int(res["clipped_mask"].sum())
    assert res["clip_converged"] == (res["n_flagged_not_refit"] == 0)
    if max_iter == 1:
        # one fit, no refit: nothing was actually excluded, the spikes are only flagged
        assert res["n_clipped"] == 0 and res["n_flagged_not_refit"] >= 3
        assert not res["clip_converged"]
    else:
        assert np.all(res["clipped_mask"][[20, 120, 220]])


def test_plain_fit_clips_and_inflates_nothing():
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0)
    result = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method=None)
    assert result["n_clipped"] == 0 and result["n_iterations"] == 0
    assert not np.any(result["clipped_mask"])
    assert np.all(result["inflation_factor"] == 1.0) and np.array_equal(result["errors_inflated"], errors)


def test_invalid_cleaning_method_raises_a_clear_error():
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0)
    with pytest.raises(ValueError, match="cleaning_method"):
        fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method="clip")


def test_inflate_without_wave_fit_raises_a_clear_error():
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0)
    with pytest.raises(ValueError, match="wave_fit"):
        fit_losvd_robust(A_stack, flux, errors, cfg, cleaning_method="inflate")


def test_clean_data_is_barely_inflated_and_matches_a_plain_clean_fit():
    """On genuinely clean data, the whitened residual is already ~N(0,1)
    almost everywhere, so the local inflation factor should sit near its
    1.0 floor and the result should be close to a plain fit."""
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=2)
    plain = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method=None)
    inflated = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method="inflate")

    assert np.median(inflated["inflation_factor"]) < 1.5, (
        f"expected clean data to stay near the 1.0 floor, "
        f"got median inflation {np.median(inflated['inflation_factor']):.2f}"
    )
    m_plain = compute_losvd_moments(v_grid, plain["b"])
    m_inflated = compute_losvd_moments(v_grid, inflated["b"])
    assert abs(m_plain["sherm"] - m_inflated["sherm"]) < 5.0


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


def test_excluded_wavelength_windows_protects_a_narrow_line_core_from_the_cleanup_clip_inflate():
    """The motivating real-data bug (dev_notes/real_data_sigma/FINDINGS.md):
    `compute_local_inflation`'s own median-filtered local scale is
    deliberately robust to a narrow (1-2 pixel) residual spike -- exactly
    what a real CaII line core looks like when the model doesn't
    perfectly reproduce its depth. Without protection, neither the
    preliminary nor the final cleanup clip can tell that apart from a
    cosmic ray, and removes it. Confirm directly: an injected single-pixel
    deficit at a CaII center is clipped without `excluded_wavelength_windows`,
    and survives with it -- and with the default (`CAII_PROTECTED_WINDOWS`)."""
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(sig=140.0, snr=60.0, seed=3)
    ca_idx = int(np.argmin(np.abs(wave_fit - CA_CENTERS[0])))
    flux_deficit = flux.copy()
    flux_deficit[ca_idx] -= 20.0 * errors[ca_idx]  # a narrow, severe single-pixel deviation

    unprotected = fit_losvd_robust(A_stack, flux_deficit, errors, cfg, wave_fit=wave_fit, cleaning_method="inflate",
                                   excluded_wavelength_windows=None)
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
    default = fit_losvd_robust(A_stack, flux_deficit, errors, cfg, wave_fit=wave_fit)
    assert not default["clipped_mask"][ca_idx]


@pytest.mark.parametrize("cleaning_method", [None, "inflate"])
def test_continuum_knots_can_be_combined_with_cleaning_method(cleaning_method):
    """Confirms continuum-cofitting runs end to end under both cleaning_method
    modes and returns a sane, near-flat continuum for this mock's true
    continuum of exactly 1.0."""
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0, continuum_knots=4)
    result = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit, cleaning_method=cleaning_method)
    # Not `result["success"]`: with continuum-cofitting the flag often ends up False on an already-converged fit
    # (the C-mean/w-scale direction is flat to rounding level -- see continuum.fit_continuum_block), and whether the
    # trust region stops just inside or just outside its tolerance varies with the numpy/scipy build. Check the answer.
    assert "C" in result and "continuum_coeffs" in result
    m = compute_losvd_moments(v_grid, result["b"])
    assert abs(m["sherm"] - sig) < 10.0
    C_full = reconstruct_continuum(result, wave_fit, cfg)
    assert C_full.shape == wave_fit.shape
    np.testing.assert_allclose(C_full, np.ones_like(C_full), atol=0.2)


def test_continuum_knots_without_wave_fit_raises_a_clear_error():
    A_stack, flux, errors, wave_fit, cfg, v_grid, sig = _mock_case(seed=0, continuum_knots=4)
    with pytest.raises(ValueError, match="wave_fit"):
        fit_losvd_robust(A_stack, flux, errors, cfg, cleaning_method=None)


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
    Parametrized over `cleaning_method`."""

    @staticmethod
    def _bins():
        return [_mock_case(sig=s, seed=seed) for s, seed in [(80.0, 0), (140.0, 1), (60.0, 2)]]

    @pytest.mark.parametrize("cleaning_method", [None, "inflate"])
    def test_n_jobs_1_matches_a_serial_loop(self, cleaning_method):
        bins = self._bins()
        A_stack_list = [b[0] for b in bins]
        flux_list = [b[1] for b in bins]
        errors_list = [b[2] for b in bins]
        wave_fit_list = [b[3] for b in bins]
        cfg_list = [b[4] for b in bins]

        kwargs = dict(cleaning_method=cleaning_method)
        serial = [fit_losvd_robust(A, f, e, cfg, wave_fit=w, **kwargs)
                  for A, f, e, w, cfg in zip(A_stack_list, flux_list, errors_list, wave_fit_list, cfg_list)]
        parallel = fit_losvd_robust_parallel(A_stack_list, flux_list, errors_list, cfg_list,
                                              wave_fit_list=wave_fit_list, n_jobs=1, **kwargs)

        assert len(parallel) == len(serial) == 3
        for s, p in zip(serial, parallel):
            np.testing.assert_array_equal(s["b"], p["b"])
            np.testing.assert_array_equal(s["clipped_mask"], p["clipped_mask"])
            assert s["n_clipped"] == p["n_clipped"]

    @pytest.mark.parametrize("cleaning_method", [None, "inflate"])
    def test_a_real_process_pool_is_bit_identical_to_serial(self, cleaning_method):
        """The "serial" reference is run through a single-worker
        `ProcessPoolExecutor` (`conftest.run_jobs_in_one_worker_process`),
        not called directly in this test process. A direct call compares
        against a different JAX/XLA process state (already-initialized,
        multi-threaded) than what a real `n_jobs=2` worker sees (freshly
        spawned, single-threaded) -- a real few-ULP difference that this
        iterative inflate fit can occasionally amplify, through a
        sigma-clip threshold crossing, into a large, run-dependent
        mismatch unrelated to what this test exists to catch (dispatch/
        ordering bugs). With both sides run through matching fresh worker
        processes, this goes back to exact equality."""
        bins = self._bins()
        A_stack_list = [b[0] for b in bins]
        flux_list = [b[1] for b in bins]
        errors_list = [b[2] for b in bins]
        wave_fit_list = [b[3] for b in bins]
        cfg_list = [b[4] for b in bins]

        kwargs = dict(cleaning_method=cleaning_method)
        jobs = [(A, f, e, cfg, w, kwargs)
                for A, f, e, w, cfg in zip(A_stack_list, flux_list, errors_list, wave_fit_list, cfg_list)]
        serial = run_jobs_in_one_worker_process(_fit_losvd_robust_worker, jobs)
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
                          wave_fit=wave_fit[:tiny], cleaning_method="inflate")


def test_fit_losvd_robust_parallel_returns_a_failed_result_instead_of_crashing_the_batch():
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
            wave_fit_list=[wave_fit_good, wave_fit_bad], n_jobs=1)

    assert results[0]["success"] is True
    assert np.all(np.isfinite(results[0]["b"]))
    assert results[1]["success"] is False
    assert np.all(np.isnan(results[1]["b"]))
    assert np.all(np.isnan(results[1]["w"]))
