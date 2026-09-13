"""Tests for kinextract.cleaning's pre-fit/post-fit emission-line
detection (`detect_emission_lines`, `detect_emission_from_residual`) and
`compute_fit_rms`. The hard-clip fit loop itself (`_fit_losvd_clip`, the
private engine behind `fit_losvd_robust`'s `cleaning_method="clip"` mode) is
tested in `test_robust_fit.py` via the public `fit_losvd_robust` API.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from kinextract import DEFAULT_CAT_EMISSION_LINES, compute_fit_rms, detect_emission_from_residual, detect_emission_lines
from kinextract.cleaning import CAII_TRIPLET_CENTERS_A


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
        problem handled elsewhere (fixed sky windows, fit_losvd_robust's
        clip/inflate modes)."""
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
