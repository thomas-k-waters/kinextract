"""Tests for kinextract.cleaning.fit_losvd_clean -- see that
module's own docstring for the method and its relationship to
`mask_wavelength_windows`. Works for any `n_templates >= 1`; most tests
below use a single template, with one parametrized over `n_templates`
to cover the genuine multi-template (weight-collapsed residual) path too.

Also tests `detect_emission_lines`, the pre-fit emission-line detector.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from kinextract import (
    FitConfig, build_template_stack, build_spline_truth, compute_losvd_moments,
    fit_losvd_clean, fit_losvd_clean_parallel, detect_emission_lines, DEFAULT_CAT_EMISSION_LINES,
    detect_emission_from_residual, compute_fit_rms,
)
from kinextract.cleaning import CAII_TRIPLET_CENTERS_A
from kinextract.multi_template import fit_losvd

CA_CENTERS = [8498.02, 8542.09, 8662.14]


def _make_templates(wavelength, n_templates):
    if n_templates == 1:
        depths_list, widths = [[0.55, 0.70, 0.65]], [5.0]
    else:
        depths_list = [[0.30, 0.65, 0.35], [0.60, 0.25, 0.55], [0.45, 0.45, 0.20]][:n_templates]
        widths = [5.0, 3.0, 7.0][:n_templates]
    templates = []
    for depths, width in zip(depths_list, widths):
        t = np.ones_like(wavelength)
        for cen, d in zip(CA_CENTERS, depths):
            t -= d * np.exp(-0.5 * ((wavelength - cen) / width) ** 2)
        templates.append(t)
    return templates


def _mock_case(vel=0.0, sig=140.0, snr=100.0, seed=0, n_templates=1):
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]
    templates = _make_templates(wavelength, n_templates)
    true_w = np.ones(n_templates) / n_templates

    cfg = FitConfig(vmin=vel - 7 * sig, vmax=vel + 7 * sig)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=vel, sigma=sig, wiggle_scale=0.0, seed=seed)
    A_stack = build_template_stack(wave_fit, wavelength, templates, v_grid)
    galaxy_noiseless = sum(true_w[k] * (A_stack[k] @ b_truth) for k in range(n_templates))
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors
    return A_stack_w, y_w, cfg, v_grid, sig


def test_runs_and_produces_sane_output_with_the_expected_extra_fields():
    A_stack_w, y_w, cfg, v_grid, sig = _mock_case(seed=0)
    result = fit_losvd_clean(A_stack_w, y_w, cfg)
    assert np.all(np.isfinite(result["b"])) and np.all(result["b"] > 0)
    assert "clipped_mask" in result and result["clipped_mask"].shape == (A_stack_w.shape[1],)
    assert result["clipped_mask"].dtype == bool
    assert result["n_clipped"] == int(result["clipped_mask"].sum())
    assert 1 <= result["n_iterations"] <= 3


def test_clean_data_gets_barely_clipped_and_matches_a_plain_fit():
    """On genuinely clean (Gaussian-noise-only) data, sigma-clipping at
    the conventional 3-sigma threshold should clip a tiny, statistically
    expected fraction of pixels (roughly the Gaussian tail beyond 3
    sigma, ~0.3% per pixel) and barely change the result -- confirms this
    doesn't silently distort well-behaved data.
    """
    A_stack_w, y_w, cfg, v_grid, sig = _mock_case(seed=1)
    plain = fit_losvd(A_stack_w, y_w, cfg)
    cleaned = fit_losvd_clean(A_stack_w, y_w, cfg)

    assert cleaned["n_clipped"] <= max(3, int(0.02 * A_stack_w.shape[1])), (
        f"expected only a handful of pixels clipped on clean data, got {cleaned['n_clipped']}"
    )
    m_plain = compute_losvd_moments(v_grid, plain["b"])
    m_cleaned = compute_losvd_moments(v_grid, cleaned["b"])
    assert abs(m_plain["sherm"] - m_cleaned["sherm"]) < 3.0


@pytest.mark.parametrize("n_templates", [1, 3])
def test_localized_contaminant_is_identified_and_clipped(n_templates):
    """The actual motivating scenario: a cluster of severely deviant
    pixels (mimicking an unmodeled sky-subtraction residual or cosmic
    ray) get correctly flagged and excluded, and the cleaned fit is
    dramatically closer to truth than the uncleaned one on average.
    Parametrized over `n_templates` to cover both a single template and a
    genuine multi-template mixture with the same contamination scenario.

    Uses S/N=30 (not the module default 100) and a 10-pixel-wide
    contaminant: a heavily curvature-regularized fit is already fairly
    robust to a tiny contaminant at high S/N, so this uses a scenario
    that actually threatens fit accuracy.
    """
    A_stack_w0, y_w0, cfg, v_grid, true_sig = _mock_case(sig=140.0, snr=30.0, seed=0, n_templates=n_templates)
    contaminated_idx = np.zeros(A_stack_w0.shape[1], dtype=bool)
    contaminated_idx[50:60] = True  # 10 pixels, arbitrary location

    uncleaned_errs, cleaned_errs, all_clipped = [], [], True
    for seed in range(8 if n_templates == 1 else 5):
        A_stack_w, y_w_seed, _, _, _ = _mock_case(sig=140.0, snr=30.0, seed=seed, n_templates=n_templates)
        y_w_contaminated = y_w_seed.copy()
        y_w_contaminated[contaminated_idx] += 50.0  # severe outlier in whitened units

        uncleaned = fit_losvd(A_stack_w, y_w_contaminated, cfg)
        cleaned = fit_losvd_clean(A_stack_w, y_w_contaminated, cfg, clip_sigma=3.0)
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


def test_excluded_wavelength_windows_protect_a_contaminant_inside_them():
    """A severe outlier landing inside a protected window must survive
    every iteration regardless of `clip_sigma`, while an equally severe
    outlier outside any protected window is clipped normally."""
    A_stack_w, y_w, cfg, v_grid, sig = _mock_case(sig=140.0, snr=30.0, seed=0)
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= 8400.0) & (wavelength <= 8800.0)
    wave_fit = wavelength[mask]

    protected_idx = np.argmin(np.abs(wave_fit - CAII_TRIPLET_CENTERS_A[0]))
    unprotected_idx = np.argmin(np.abs(wave_fit - 8600.0))  # between the 2nd/3rd CaII lines

    y_w_contaminated = y_w.copy()
    y_w_contaminated[protected_idx] += 50.0
    y_w_contaminated[unprotected_idx] += 50.0

    windows = tuple((c - 8.0, c + 8.0) for c in CAII_TRIPLET_CENTERS_A)
    result = fit_losvd_clean(A_stack_w, y_w_contaminated, cfg, clip_sigma=3.0,
                              wave_fit=wave_fit, excluded_wavelength_windows=windows)

    assert not result["clipped_mask"][protected_idx], (
        "a contaminant inside an excluded window must never be clipped"
    )
    assert result["clipped_mask"][unprotected_idx], (
        "a contaminant outside any excluded window should still be clipped normally"
    )


def test_excluded_wavelength_windows_without_wave_fit_raises_a_clear_error():
    A_stack_w, y_w, cfg, v_grid, sig = _mock_case(seed=0)
    with pytest.raises(ValueError, match="wave_fit"):
        fit_losvd_clean(A_stack_w, y_w, cfg, excluded_wavelength_windows=((8490.0, 8506.0),))


def test_max_clip_fraction_warns_on_a_large_systematic_mismatch():
    """A wildly wrong velocity grid (mimicking a large redshift error or
    similar systematic mismatch) should clip a large fraction of pixels
    and trigger the sanity warning -- distinguishing "a few bad pixels"
    from "something fundamental is wrong," which iterative clipping alone
    cannot fix and shouldn't be allowed to quietly paper over.
    """
    A_stack_w, y_w, cfg, v_grid, true_sig = _mock_case(seed=3)
    rng = np.random.default_rng(4)
    y_w_bad = y_w.copy()
    bad_idx = rng.choice(len(y_w), size=int(0.5 * len(y_w)), replace=False)
    y_w_bad[bad_idx] += rng.normal(0.0, 20.0, len(bad_idx))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fit_losvd_clean(A_stack_w, y_w_bad, cfg, max_clip_fraction=0.1)
        assert any("above the" in str(x.message) and "sanity threshold" in str(x.message) for x in w)
    assert result["n_clipped"] / A_stack_w.shape[1] > 0.1


class TestRobustScale:
    """`robust_scale=True` -- found necessary on real MUSE data (`08_full_
    galaxy_analysis.ipynb`) where the default (raw whitened residual,
    assuming a correctly-calibrated noise vector) clipped 90%+ of pixels
    and collapsed to a biased, unconstrained baseline, because real
    chi2_red there runs 11-28: a nominal 3-sigma cut was actually well
    under 1 true sigma almost everywhere, not just at the genuine
    contamination.
    """

    def test_matches_plain_on_well_calibrated_noise(self):
        """When the input noise vector IS correctly calibrated (this
        module's own mock convention), the whitened residual's robust
        (MAD) scale should already be close to 1 -- so robust_scale
        shouldn't change the clip count much from the plain version."""
        A_stack_w, y_w, cfg, v_grid, sig = _mock_case(seed=2)
        plain = fit_losvd_clean(A_stack_w, y_w, cfg, robust_scale=False)
        robust = fit_losvd_clean(A_stack_w, y_w, cfg, robust_scale=True)
        assert abs(plain["n_clipped"] - robust["n_clipped"]) <= 3

    def test_avoids_runaway_under_global_noise_miscalibration(self):
        """The actual motivating failure mode: the whole spectrum's noise
        is under-estimated (not a few localized bad pixels), mimicking
        real chi2_red >> 1. The plain version should clip a large,
        systematic-mismatch-sized fraction (its own sanity warning should
        fire); the robust version, correctly recognizing there's no
        genuine *outlier* relative to this spectrum's own (uniformly
        elevated) typical scatter, should clip almost nothing instead --
        the right answer, since nothing here is actually anomalous
        relative to the rest of the spectrum.
        """
        A_stack_w, y_w, cfg, v_grid, sig = _mock_case(seed=3, snr=100.0)
        rng = np.random.default_rng(3)
        y_w_mismatched = y_w + rng.normal(0.0, 4.0, len(y_w))  # uniform excess noise, not localized

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plain = fit_losvd_clean(A_stack_w, y_w_mismatched, cfg)  # default max_clip_fraction=0.1
            assert any("sanity threshold" in str(x.message) for x in w)
        robust = fit_losvd_clean(A_stack_w, y_w_mismatched, cfg, robust_scale=True, max_clip_fraction=0.9)

        assert plain["n_clipped"] > 0.3 * A_stack_w.shape[1]
        assert robust["n_clipped"] < 0.05 * A_stack_w.shape[1]

    def test_still_catches_a_real_localized_contaminant_when_robust(self):
        """robust_scale=True must not become so permissive it stops
        catching genuine, localized contamination -- rerun the actual
        motivating scenario (a severe localized outlier cluster) with
        robust_scale=True and confirm it's still identified and clipped.
        """
        A_stack_w, y_w, cfg, v_grid, true_sig = _mock_case(sig=140.0, snr=30.0, seed=0)
        contaminated_idx = np.zeros(A_stack_w.shape[1], dtype=bool)
        contaminated_idx[50:60] = True
        y_w_contaminated = y_w.copy()
        y_w_contaminated[contaminated_idx] += 50.0

        robust = fit_losvd_clean(A_stack_w, y_w_contaminated, cfg, robust_scale=True)
        assert np.all(robust["clipped_mask"][contaminated_idx])


def test_continuum_knots_can_be_combined_with_cleaning():
    """Found via code review: `fit_losvd_clean` had no `wave_fit` parameter
    at all, so `cfg.continuum_knots` could never be combined with sigma-
    clipping -- exactly the combination needed for real data with both
    unmodeled emission lines and an imperfect continuum. Confirms the fix:
    `wave_fit` is required and sliced by the same `keep` mask as
    `A_stack_w`/`y_w` each iteration, a contaminant is still identified
    and clipped, and the returned `C` is a sane, near-flat continuum
    (this mock's true continuum is exactly 1.0 everywhere).
    """
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]
    templates = _make_templates(wavelength, 1)

    sig = 140.0
    cfg = FitConfig(vmin=-7 * sig, vmax=7 * sig, continuum_knots=4)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=0.0, sigma=sig, wiggle_scale=0.0, seed=0)
    A_stack = build_template_stack(wave_fit, wavelength, templates, v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / 60.0
    rng = np.random.default_rng(0)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    contaminated_idx = np.zeros(len(wave_fit), dtype=bool)
    contaminated_idx[50:55] = True
    galaxy[contaminated_idx] += 30.0 * noise_level
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors

    result = fit_losvd_clean(A_stack_w, y_w, cfg, wave_fit=wave_fit, clip_sigma=4.0)
    assert result["success"]
    assert "C" in result and "continuum_coeffs" in result
    assert np.all(result["clipped_mask"][contaminated_idx])
    np.testing.assert_allclose(result["C"], np.ones_like(result["C"]), atol=0.1)


def test_continuum_knots_without_wave_fit_raises_a_clear_error():
    A_stack_w, y_w, cfg, v_grid, sig = _mock_case(seed=0)
    cfg_continuum = FitConfig(vmin=cfg.vmin, vmax=cfg.vmax, continuum_knots=4)
    with pytest.raises(ValueError, match="wave_fit"):
        fit_losvd_clean(A_stack_w, y_w, cfg_continuum)


def _flat_spectrum(seed=0, snr=50.0):
    wave = np.arange(8400.0, 8800.0, 1.25)
    rng = np.random.default_rng(seed)
    err = np.full_like(wave, 100.0 / snr)
    flux = 100.0 + rng.normal(0.0, err[0], size=wave.size)
    return wave, flux, err


def _inject_gaussian_line(wave, flux, center, amplitude, width=1.0):
    return flux + amplitude * np.exp(-0.5 * ((wave - center) / width) ** 2)


class TestDetectEmissionLines:
    """`detect_emission_lines` -- pre-fit emission-line detection ported
    from the original kinextract's `masking.build_emission_line_mask`
    (excess-S/N over the local flanking continuum, not raw continuum S/N).
    """

    def test_clean_absorption_only_spectrum_flags_nothing(self):
        wave, flux, err = _flat_spectrum(seed=0)
        assert detect_emission_lines(wave, flux, err) == []

    @pytest.mark.parametrize("name,center", DEFAULT_CAT_EMISSION_LINES)
    def test_a_strong_injected_line_at_each_default_center_is_detected(self, name, center):
        wave, flux, err = _flat_spectrum(seed=1)
        flux_with_line = _inject_gaussian_line(wave, flux, center, amplitude=40.0)
        windows = detect_emission_lines(wave, flux_with_line, err)
        assert any(lo <= center <= hi for lo, hi in windows), (
            f"expected {name} at {center} to be flagged, got windows={windows}"
        )

    def test_a_deep_caii_absorption_trough_does_not_falsely_flag_nearby_n_i_lines(self):
        """[N I] 8680.28/8683.40's default +/-20 A context window overlaps
        the CaT's own 8662.14 A line -- without excluding the triplet from
        the flanking-continuum estimate, a deep absorption trough there
        would drag cont_est down and produce a false-positive detection.
        """
        wave, flux, err = _flat_spectrum(seed=2)
        trough_center = CAII_TRIPLET_CENTERS_A[2]
        assert abs(trough_center - 8662.14) < 1e-6
        flux_with_trough = _inject_gaussian_line(wave, flux, trough_center, amplitude=-30.0, width=3.0)
        assert detect_emission_lines(wave, flux_with_trough, err) == []

    def test_snr_threshold_controls_sensitivity(self):
        wave, flux, err = _flat_spectrum(seed=3)
        center = DEFAULT_CAT_EMISSION_LINES[2][1]
        flux_with_weak_line = _inject_gaussian_line(wave, flux, center, amplitude=6.0)
        assert detect_emission_lines(wave, flux_with_weak_line, err, snr_threshold=100.0) == []
        assert detect_emission_lines(wave, flux_with_weak_line, err, snr_threshold=1.0) != []

    def test_returned_windows_are_usable_directly_by_mask_wavelength_windows(self):
        from kinextract import mask_wavelength_windows

        wave, flux, err = _flat_spectrum(seed=4)
        center = DEFAULT_CAT_EMISSION_LINES[3][1]
        flux_with_line = _inject_gaussian_line(wave, flux, center, amplitude=40.0)
        windows = detect_emission_lines(wave, flux_with_line, err)
        assert windows
        wave_masked, flux_masked = mask_wavelength_windows(wave, flux_with_line, windows=windows)
        assert not np.any((wave_masked >= center - 4.0) & (wave_masked <= center + 4.0))
        assert len(wave_masked) < len(wave)


def _flat_residual(seed=0, n=320):
    rng = np.random.default_rng(seed)
    wave = np.arange(8400.0, 8400.0 + 1.25 * n, 1.25)
    residual = rng.normal(0.0, 1.0, size=n)
    return wave, residual


def _inject_gaussian_residual_peak(wave, residual, center, amplitude, width):
    return residual + amplitude * np.exp(-0.5 * ((wave - center) / width) ** 2)


class TestDetectEmissionFromResidual:
    """`detect_emission_from_residual` -- post-fit, Gaussian-peak-shaped
    detection directly against a fit's own whitened residual (pPXF-style
    fit-then-clean, refined to characterize each peak's shape rather than
    clip individual pixels). No line list involved -- purely residual-driven.
    """

    def test_clean_gaussian_noise_residual_rarely_flags_anything(self):
        """At nsigma=3 on pure N(0,1) noise, false positives should be rare
        (Gaussian one-sided 3-sigma tail ~0.13% per pixel) -- not zero, but
        not routine either, across many independent noise draws.
        """
        n_flagged = 0
        n_seeds = 20
        for seed in range(n_seeds):
            wave, residual = _flat_residual(seed=seed)
            n_flagged += len(detect_emission_from_residual(wave, residual))
        rate = n_flagged / (n_seeds * len(wave))
        assert rate < 0.02, f"expected a rare false-positive rate, got {rate:.4f} ({n_flagged} of {n_seeds * len(wave)} pixels)"

    def test_a_strong_injected_residual_peak_is_detected_at_the_right_center(self):
        wave, residual = _flat_residual(seed=1)
        center, true_width = 8600.0, 1.5
        residual_with_peak = _inject_gaussian_residual_peak(wave, residual, center, amplitude=15.0, width=true_width)
        windows = detect_emission_from_residual(wave, residual_with_peak)
        assert windows
        centers = [(lo + hi) / 2.0 for lo, hi in windows]
        assert any(abs(c - center) < 1.0 for c in centers)

    def test_mask_sigma_controls_window_width(self):
        wave, residual = _flat_residual(seed=2)
        center = 8600.0
        residual_with_peak = _inject_gaussian_residual_peak(wave, residual, center, amplitude=15.0, width=1.5)
        narrow = detect_emission_from_residual(wave, residual_with_peak, mask_sigma=1.0)
        wide = detect_emission_from_residual(wave, residual_with_peak, mask_sigma=3.0)
        assert narrow and wide
        narrow_width = narrow[0][1] - narrow[0][0]
        wide_width = wide[0][1] - wide[0][0]
        assert wide_width > narrow_width

    def test_a_broad_feature_wider_than_max_line_width_is_rejected(self):
        """A residual bump far broader than any real emission line (e.g. a
        template-mismatch/continuum residual, not a narrow spectral
        feature) should not be masked as if it were one -- and, without the
        isolation check, gets fragmented into many spurious narrow
        detections instead (the original bug this test caught).
        """
        wave, residual = _flat_residual(seed=3)
        center = 8600.0
        broad_bump = _inject_gaussian_residual_peak(wave, residual, center, amplitude=15.0, width=20.0)
        windows = detect_emission_from_residual(wave, broad_bump, max_line_width_A=5.0)
        near_bump = [(lo, hi) for lo, hi in windows if lo < center + 30 and hi > center - 30]
        assert near_bump == [], f"expected no detections near the broad bump, got {near_bump}"

    def test_negative_dips_are_never_flagged(self):
        """Only emission-like (positive) residual excursions are in scope --
        a negative dip (e.g. an oversubtracted sky line) is a different
        problem handled elsewhere (fixed sky windows, fit_losvd_clean).
        """
        wave, residual = _flat_residual(seed=4)
        center = 8600.0
        residual_with_dip = _inject_gaussian_residual_peak(wave, residual, center, amplitude=-15.0, width=1.5)
        assert detect_emission_from_residual(wave, residual_with_dip) == []

    def test_returned_windows_are_usable_directly_by_mask_wavelength_windows(self):
        from kinextract import mask_wavelength_windows

        wave, residual = _flat_residual(seed=5)
        center = 8600.0
        residual_with_peak = _inject_gaussian_residual_peak(wave, residual, center, amplitude=15.0, width=1.5)
        windows = detect_emission_from_residual(wave, residual_with_peak)
        assert windows
        flux = np.ones_like(wave)
        wave_masked, flux_masked = mask_wavelength_windows(wave, flux, windows=windows)
        assert len(wave_masked) < len(wave)
        assert not np.any((wave_masked >= center - 2.0) & (wave_masked <= center + 2.0))

    def test_a_globally_poor_fit_triggers_the_max_mask_fraction_warning(self):
        """Found via real-data testing: on a fit whose chi2_red is far from
        1 (unmodeled continuum shape/template mismatch/mis-scaled noise,
        not narrow emission), this method can find many separate peaks
        across a large fraction of the spectrum -- the diagnostic signal
        that its own core assumption (residual = noise + rare isolated
        contaminant) doesn't hold here, not evidence of many real lines.
        Simulate that by injecting several widely-spaced narrow peaks
        covering more than the default 15% mask-fraction threshold.
        """
        wave, residual = _flat_residual(seed=6, n=320)
        many_peaks = residual.copy()
        for center in np.linspace(wave[20], wave[-20], 8):
            many_peaks = many_peaks + 15.0 * np.exp(-0.5 * ((wave - center) / 1.5) ** 2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            windows = detect_emission_from_residual(wave, many_peaks, max_mask_fraction=0.05)
            assert any("sanity threshold" in str(x.message) for x in w)
        assert len(windows) >= 4


class TestComputeFitRms:
    def test_perfect_fit_gives_zero(self):
        flux = np.array([1.0, 2.0, 3.0, 4.0])
        errors = np.full(4, 0.5)
        assert compute_fit_rms(flux, flux.copy(), errors) == 0.0

    def test_matches_hand_computed_value(self):
        flux = np.array([1.0, 2.0, 3.0])
        model = np.array([1.1, 1.9, 3.2])
        errors = np.array([0.1, 0.1, 0.1])
        expected = np.sqrt(np.mean(((flux - model) / errors) ** 2))
        assert compute_fit_rms(flux, model, errors) == pytest.approx(expected)

    def test_mask_restricts_to_kept_pixels_only(self):
        # a single huge outlier at index 2 dominates the unmasked RMS;
        # masking it out should match the RMS of the other two pixels alone
        flux = np.array([1.0, 2.0, 100.0])
        model = np.array([1.1, 1.9, 3.2])
        errors = np.array([0.1, 0.1, 0.1])
        mask = np.array([True, True, False])
        expected = np.sqrt(np.mean((((flux - model) / errors)[:2]) ** 2))
        assert compute_fit_rms(flux, model, errors, mask=mask) == pytest.approx(expected)
        assert compute_fit_rms(flux, model, errors) > compute_fit_rms(flux, model, errors, mask=mask)


class TestFitLosvdCleanParallel:
    """`fit_losvd_clean_parallel`'s own core promise, mirroring
    `test_multi_template.py`'s `fit_losvd_parallel` tests: fitting bins
    across worker processes must be bit-identical to fitting them one at a
    time in a serial loop -- pure wall-clock orchestration, zero accuracy
    cost. Needed once cleaning is the default per-bin step rather than an
    occasional add-on for a handful of flagged bins."""

    @staticmethod
    def _bins():
        return [_mock_case(sig=s, seed=seed) for s, seed in [(80.0, 0), (140.0, 1), (60.0, 2)]]

    def test_n_jobs_1_matches_a_serial_loop(self):
        bins = self._bins()
        A_stack_w_list = [b[0] for b in bins]
        y_w_list = [b[1] for b in bins]
        cfg_list = [b[2] for b in bins]

        serial = [fit_losvd_clean(A, y, cfg, robust_scale=True)
                  for A, y, cfg in zip(A_stack_w_list, y_w_list, cfg_list)]
        parallel = fit_losvd_clean_parallel(A_stack_w_list, y_w_list, cfg_list,
                                             robust_scale=True, n_jobs=1)

        assert len(parallel) == len(serial) == 3
        for s, p in zip(serial, parallel):
            np.testing.assert_array_equal(s["b"], p["b"])
            np.testing.assert_array_equal(s["clipped_mask"], p["clipped_mask"])
            assert s["n_clipped"] == p["n_clipped"]

    def test_a_real_process_pool_is_bit_identical_to_serial(self):
        bins = self._bins()
        A_stack_w_list = [b[0] for b in bins]
        y_w_list = [b[1] for b in bins]
        cfg_list = [b[2] for b in bins]

        serial = [fit_losvd_clean(A, y, cfg, robust_scale=True)
                  for A, y, cfg in zip(A_stack_w_list, y_w_list, cfg_list)]
        parallel = fit_losvd_clean_parallel(A_stack_w_list, y_w_list, cfg_list,
                                             robust_scale=True, n_jobs=2)

        for i, (s, p) in enumerate(zip(serial, parallel)):
            np.testing.assert_array_equal(s["b"], p["b"], err_msg=f"bin {i} mismatch")
            np.testing.assert_array_equal(s["clipped_mask"], p["clipped_mask"], err_msg=f"bin {i} mismatch")

    def test_mismatched_list_lengths_raise_a_clear_error(self):
        bins = self._bins()
        A_stack_w_list = [b[0] for b in bins]
        y_w_list = [b[1] for b in bins]
        cfg_list = [b[2] for b in bins]

        with pytest.raises(ValueError, match="same length"):
            fit_losvd_clean_parallel(A_stack_w_list, y_w_list[:-1], cfg_list, n_jobs=1)
        with pytest.raises(ValueError, match="one entry per bin"):
            fit_losvd_clean_parallel(A_stack_w_list, y_w_list, cfg_list[:-1], n_jobs=1)


def test_fit_losvd_clean_raises_a_clear_error_when_too_few_pixels_survive():
    """A spectrum with fewer pixels than free LOSVD coefficients must fail
    loudly and specifically, not silently 'succeed' on a fit driven
    entirely by the roughness penalty with no real data behind it."""
    A_stack_w, y_w, cfg, v_grid, sig = _mock_case(seed=0)
    n_coef = cfg.resolved_n_interior_knots() + cfg.degree + 1
    tiny = max(1, n_coef - 5)
    A_stack_w_tiny = A_stack_w[:, :tiny, :]
    y_w_tiny = y_w[:tiny]
    with pytest.raises(RuntimeError, match="pixels survived clipping"):
        fit_losvd_clean(A_stack_w_tiny, y_w_tiny, cfg)


def test_fit_losvd_clean_parallel_returns_a_failed_result_instead_of_crashing_the_batch():
    """A per-bin failure (here: too few pixels, matching the guard above)
    must produce a NaN-filled, success=False result for that one bin, not
    crash the entire parallel batch -- confirms fit_losvd_clean_parallel's
    own worker-level try/except actually works end to end, not just the
    exception it's designed to catch."""
    good = _mock_case(seed=1)
    A_stack_w_good, y_w_good, cfg, v_grid, sig = good
    n_coef = cfg.resolved_n_interior_knots() + cfg.degree + 1
    tiny = max(1, n_coef - 5)
    A_stack_w_bad = A_stack_w_good[:, :tiny, :]
    y_w_bad = y_w_good[:tiny]

    with pytest.warns(UserWarning, match="returning a NaN-filled"):
        results = fit_losvd_clean_parallel(
            [A_stack_w_good, A_stack_w_bad], [y_w_good, y_w_bad], cfg, n_jobs=1)

    assert results[0]["success"] is True
    assert np.all(np.isfinite(results[0]["b"]))
    assert results[1]["success"] is False
    assert np.all(np.isnan(results[1]["b"]))
    assert np.all(np.isnan(results[1]["w"]))
