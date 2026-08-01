"""Targeted edge-case/robustness checks -- not a broad statistical grid
(see comprehensive_grid.py for that), but a discrete set of specific
"does this break" questions worth answering once, for the comprehensive
status report: extreme sigma/S-N, a very large template mixture, the
wing_lam+continuum_knots combination, and bootstrap at extreme n_boot.

Run directly (not via ProcessPoolExecutor -- these are one-off checks,
not a statistical sweep, so sequential + verbose is more useful here than
fast + parallel). Each check prints PASS/FAIL/NOTE and a short reason;
failures are the report's real content, not incidental.
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
import numpy as np

from kinextract import (
    FitConfig, build_template_stack, gauss_hermite_model, compute_losvd_moments,
    load_emiles_templates, bootstrap_losvd, mask_wavelength_windows,
)
from kinextract.multi_template import fit_losvd

EMILES_GRID = os.path.join(os.path.dirname(__file__), "..", "..", "src", "kinextract",
                            "template_data", "emiles_muse_grid.npz")
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
NSIG = 7.0

RESULTS = []


def _record(name, ok, note=""):
    RESULTS.append((name, ok, note))
    status = "PASS" if ok is True else ("FAIL" if ok is False else "NOTE")
    print(f"[{status}] {name}{': ' + note if note else ''}", flush=True)


def _build_mock(templates, template_wave, weights, true_v, true_sigma, snr, h3=0.0, h4=0.0, seed=0):
    mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
    wave_fit = template_wave[mask]
    cfg = FitConfig(vmin=true_v - NSIG * true_sigma, vmax=true_v + NSIG * true_sigma)
    v_grid = cfg.build_velocity_grid()
    b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, true_v, true_sigma, h3, h4), 0.0)
    b_truth /= b_truth.sum()
    A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
    galaxy_noiseless = sum(weights[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors
    return cfg, v_grid, A_stack_w, y_w, wave_fit


def check_extreme_narrow_sigma():
    """sigma=5 km/s -- far below any validated regime (PARAMETER_GUIDE.md's
    grid starts at 20). Does the fit at least not crash/NaN?"""
    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
    try:
        cfg, v_grid, A_stack_w, y_w, _ = _build_mock(templates, template_wave, np.array([1.0]),
                                                       true_v=0.0, true_sigma=5.0, snr=100.0)
        result = fit_losvd(A_stack_w, y_w, cfg)
        moments = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)
        ok = result["success"] and np.isfinite(moments["sherm"]) and moments["sherm"] > 0
        _record("extreme_narrow_sigma(5 km/s)", ok,
                 f"sigma_rec={moments['sherm']:.2f} (true 5.0), success={result['success']}")
    except Exception as e:
        _record("extreme_narrow_sigma(5 km/s)", False, f"EXCEPTION: {e}")


def check_extreme_broad_sigma():
    """sigma=800 km/s -- far above the validated grid's own top (400)."""
    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
    try:
        cfg, v_grid, A_stack_w, y_w, _ = _build_mock(templates, template_wave, np.array([1.0]),
                                                       true_v=0.0, true_sigma=800.0, snr=100.0)
        result = fit_losvd(A_stack_w, y_w, cfg)
        moments = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)
        bias_pct = 100 * (moments["sherm"] - 800.0) / 800.0
        ok = result["success"] and np.isfinite(moments["sherm"])
        _record("extreme_broad_sigma(800 km/s)", ok,
                 f"sigma_rec={moments['sherm']:.1f} (true 800), bias={bias_pct:+.1f}%")
    except Exception as e:
        _record("extreme_broad_sigma(800 km/s)", False, f"EXCEPTION: {e}")


def check_extreme_low_snr():
    """S/N=5 -- far below the validated grid's own bottom (~15-20)."""
    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
    try:
        cfg, v_grid, A_stack_w, y_w, _ = _build_mock(templates, template_wave, np.array([1.0]),
                                                       true_v=0.0, true_sigma=100.0, snr=5.0)
        result = fit_losvd(A_stack_w, y_w, cfg)
        moments = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)
        ok = result["success"] and np.isfinite(moments["sherm"])
        _record("extreme_low_snr(S/N=5)", ok,
                 f"sigma_rec={moments['sherm']:.1f} (true 100), success={result['success']}")
    except Exception as e:
        _record("extreme_low_snr(S/N=5)", False, f"EXCEPTION: {e}")


def check_extreme_high_snr():
    """S/N=10000 -- essentially noiseless; does the fit converge to the
    true shape with no floating-point pathology at tiny residuals?"""
    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
    try:
        cfg, v_grid, A_stack_w, y_w, _ = _build_mock(templates, template_wave, np.array([1.0]),
                                                       true_v=0.0, true_sigma=100.0, snr=10000.0)
        result = fit_losvd(A_stack_w, y_w, cfg)
        moments = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)
        bias_pct = 100 * (moments["sherm"] - 100.0) / 100.0
        ok = result["success"] and abs(bias_pct) < 5.0
        _record("extreme_high_snr(S/N=10000)", ok,
                 f"sigma_rec={moments['sherm']:.2f} (true 100), bias={bias_pct:+.2f}%")
    except Exception as e:
        _record("extreme_high_snr(S/N=10000)", False, f"EXCEPTION: {e}")


def check_full_636_template_grid_single_fit():
    """Fit against the ENTIRE 636-template E-MILES grid at once (not
    NNLS-thresholded/reduced) -- the worst-case identifiability/runtime
    stress case for a single fit call."""
    templates, template_wave, names, ages, metals = load_emiles_templates(EMILES_GRID)
    weights = np.zeros(len(templates))
    weights[100] = 0.5
    weights[300] = 0.3
    weights[500] = 0.2
    try:
        t0 = time.perf_counter()
        cfg, v_grid, A_stack_w, y_w, _ = _build_mock(templates, template_wave, weights,
                                                       true_v=20.0, true_sigma=100.0, snr=60.0)
        result = fit_losvd(A_stack_w, y_w, cfg)
        runtime = time.perf_counter() - t0
        moments = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)
        n_nonzero = int(np.sum(result["w"] > 1e-6 * result["w"].sum()))
        ok = result["success"] and np.isfinite(moments["sherm"])
        _record("full_636_template_single_fit", ok,
                 f"runtime={runtime:.1f}s, sigma_rec={moments['sherm']:.1f} (true 100), "
                 f"n_templates_with_nonzero_weight={n_nonzero}/636")
    except Exception as e:
        _record("full_636_template_single_fit", False, f"EXCEPTION: {e}\n{traceback.format_exc()}")


def check_wing_lam_plus_continuum_combo():
    """wing_lam>0 combined with continuum_knots -- both are individually
    validated, but is the *combination* (lam_selection must be 'fixed',
    per fit_losvd's own fallback) actually exercised/safe together?"""
    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(1.0, -0.66), (5.0, 0.06), (14.0, 0.40)])
    weights = np.array([0.3, 0.4, 0.3])
    try:
        mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
        wave_fit = template_wave[mask]
        true_v, true_sigma, snr = 20.0, 50.0, 60.0
        cfg_base = FitConfig(vmin=true_v - NSIG * true_sigma, vmax=true_v + NSIG * true_sigma,
                              wing_lam=3e6, continuum_knots=4, lam_selection="fixed", lam=2.0)
        v_grid = cfg_base.build_velocity_grid()
        b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, true_v, true_sigma, 0.0, 0.0), 0.0)
        b_truth /= b_truth.sum()
        A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
        galaxy_noiseless = sum(weights[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))
        true_C = 1.0 + 0.1 * np.sin(np.pi * (wave_fit - wave_fit.min()) / (wave_fit.max() - wave_fit.min()))
        galaxy_distorted = galaxy_noiseless * true_C
        noise_level = np.median(galaxy_distorted) / snr
        rng = np.random.default_rng(11)
        galaxy = galaxy_distorted + rng.normal(0.0, noise_level, len(wave_fit))
        errors = np.full(len(wave_fit), noise_level)
        A_stack_w = A_stack / errors[None, :, None]
        y_w = galaxy / errors

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit_losvd(A_stack_w, y_w, cfg_base, wave_fit=wave_fit)
        moments = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)
        ok = result["success"] and np.isfinite(moments["sherm"]) and result.get("C") is not None
        _record("wing_lam_plus_continuum_combo", ok,
                 f"sigma_rec={moments['sherm']:.1f} (true 50), success={result['success']}, "
                 f"warnings={[str(w.message)[:60] for w in caught]}")
    except Exception as e:
        _record("wing_lam_plus_continuum_combo", False, f"EXCEPTION: {e}\n{traceback.format_exc()}")


def check_bootstrap_n_boot_1():
    """bootstrap_losvd with the minimum useful n_boot=1 -- std should be
    computable (or gracefully degenerate), not crash on ddof=1 with a
    single sample."""
    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
    try:
        cfg, v_grid, A_stack_w, y_w, _ = _build_mock(templates, template_wave, np.array([1.0]),
                                                       true_v=0.0, true_sigma=100.0, snr=60.0)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=1, seed=0)
        ok = np.isfinite(boot["sherm_std"]) or boot["sherm_std"] == 0.0
        _record("bootstrap_n_boot_1", True if not np.isnan(boot["sherm_std"]) else "NOTE",
                 f"sherm_std={boot['sherm_std']} (n_boot=1, expect 0.0 or nan, not a crash)")
    except Exception as e:
        _record("bootstrap_n_boot_1", False, f"EXCEPTION: {e}")


def check_bootstrap_large_n_boot():
    """bootstrap_losvd with n_boot=200 (much larger than any notebook's
    default 30-50) -- pure runtime/stability check, not a new statistical
    claim."""
    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(1.0, -0.66), (5.0, 0.06), (14.0, 0.40)])
    weights = np.array([0.3, 0.4, 0.3])
    try:
        cfg, v_grid, A_stack_w, y_w, _ = _build_mock(templates, template_wave, weights,
                                                       true_v=20.0, true_sigma=80.0, snr=60.0)
        t0 = time.perf_counter()
        boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=200, seed=0)
        runtime = time.perf_counter() - t0
        ok = np.isfinite(boot["sherm_std"]) and boot["sherm_std"] > 0
        _record("bootstrap_n_boot_200", ok, f"runtime={runtime:.1f}s, sherm_std={boot['sherm_std']:.2f}")
    except Exception as e:
        _record("bootstrap_n_boot_200", False, f"EXCEPTION: {e}")


def check_mask_wavelength_windows_all_masked():
    """mask_wavelength_windows with a window covering the ENTIRE fit
    range -- should this raise, or silently return empty arrays that
    then crash something downstream less clearly?"""
    wave = np.linspace(8400.0, 8800.0, 320)
    flux = np.ones_like(wave)
    try:
        wave_m, flux_m = mask_wavelength_windows(wave, flux, windows=[(8400.0, 8800.0)])
        _record("mask_all_wavelengths", "NOTE",
                 f"returned {len(wave_m)} pixels (of {len(wave)}) with a window covering the whole range "
                 f"-- {'empty result, as expected' if len(wave_m) == 0 else 'UNEXPECTED: non-empty result'}")
    except Exception as e:
        _record("mask_all_wavelengths", "NOTE", f"raised {type(e).__name__}: {e}")


def check_single_pixel_fit():
    """An absurdly small fit window (5 pixels) -- does this fail loudly
    and clearly, or silently produce a nonsense answer?"""
    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
    try:
        mask = (template_wave >= 8400.0) & (template_wave <= 8400.0 + 5 * 1.25)
        wave_fit = template_wave[mask]
        cfg = FitConfig(vmin=-700.0, vmax=700.0)
        v_grid = cfg.build_velocity_grid()
        b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, 0.0, 100.0, 0.0, 0.0), 0.0)
        b_truth /= b_truth.sum()
        A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
        galaxy_noiseless = A_stack[0] @ b_truth
        noise_level = np.median(galaxy_noiseless) / 60.0
        rng = np.random.default_rng(0)
        galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
        errors = np.full(len(wave_fit), noise_level)
        A_stack_w = A_stack / errors[None, :, None]
        y_w = galaxy / errors
        result = fit_losvd(A_stack_w, y_w, cfg)
        moments = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)
        _record("single_5_pixel_fit", "NOTE",
                 f"n_pix=5, n_coef={result['n_coef']}, success={result['success']}, "
                 f"sigma_rec={moments['sherm']:.1f} (true 100) -- massively underdetermined, "
                 f"reported as a NOTE not PASS/FAIL since there's no 'correct' behavior to check "
                 f"against, only whether it fails loudly (it doesn't) or silently (it does)")
    except Exception as e:
        _record("single_5_pixel_fit", "NOTE", f"raised {type(e).__name__}: {e} (loud failure, arguably better)")


if __name__ == "__main__":
    checks = [
        check_extreme_narrow_sigma, check_extreme_broad_sigma,
        check_extreme_low_snr, check_extreme_high_snr,
        check_full_636_template_grid_single_fit,
        check_wing_lam_plus_continuum_combo,
        check_bootstrap_n_boot_1, check_bootstrap_large_n_boot,
        check_mask_wavelength_windows_all_masked, check_single_pixel_fit,
    ]
    for fn in checks:
        print(f"\n=== {fn.__name__} ===", flush=True)
        fn()

    print("\n\n=== SUMMARY ===")
    n_pass = sum(1 for _, ok, _ in RESULTS if ok is True)
    n_fail = sum(1 for _, ok, _ in RESULTS if ok is False)
    n_note = sum(1 for _, ok, _ in RESULTS if ok not in (True, False))
    print(f"{n_pass} pass, {n_fail} fail, {n_note} note (of {len(RESULTS)} checks)")
    for name, ok, note in RESULTS:
        status = "PASS" if ok is True else ("FAIL" if ok is False else "NOTE")
        print(f"  [{status}] {name}: {note}")
