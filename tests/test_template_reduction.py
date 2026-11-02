"""Tests for kinextract.template_reduction -- the iterative
template-set-reduction loop. See that module's own docstring for why
convergence is checked against the *union* of surviving templates across
bins, not each bin's own set, and for why the candidate set can only
shrink or stay the same from one iteration to the next.

Uses small, fast, hand-built synthetic templates (matching this test
suite's own existing `_make_template` convention, e.g.
`tests/test_continuum.py`) rather than real E-MILES templates -- these
tests are about the reduction *algorithm's* own logic (shrinkage,
convergence, `bins_subset`, save/load), not about real stellar-population
recovery, which the manual end-to-end validation against
`mock_cube`/E-MILES already covers.
"""
from __future__ import annotations

import dataclasses
import os
import tempfile

import numpy as np

from kinextract import FitConfig, build_spline_truth
from kinextract.forward_model import build_template_stack
from kinextract.noise_model import whiten
from kinextract.template_reduction import (
    reduce_templates_iteratively, save_template_subset, load_template_subset,
)

_CENTERS = [8498.02, 8542.09, 8662.14]


def _make_template(wavelength, depths, width=5.0):
    t = np.ones_like(wavelength)
    for cen, d in zip(_CENTERS, depths):
        t -= d * np.exp(-0.5 * ((wavelength - cen) / width) ** 2)
    return t


def _make_bin_specs(n_bins=3, snr=80.0, seed=0):
    """2 "true" templates (distinct absorption depths) + 3 clearly-
    distinguishable "decoy" templates (different line ratios/widths), so
    NNLS at moderate-to-high S/N should reliably favor the true pair over
    the decoys -- a fast, deterministic setup for testing the reduction
    *logic*, not template-mixture degeneracy itself (already covered
    elsewhere in this test suite).
    """
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]

    true_templates = [
        _make_template(wavelength, [0.55, 0.70, 0.65], width=5.0),
        _make_template(wavelength, [0.30, 0.40, 0.35], width=7.0),
    ]
    decoy_templates = [
        _make_template(wavelength, [0.15, 0.10, 0.20], width=3.0),
        _make_template(wavelength, [0.05, 0.60, 0.05], width=9.0),
        _make_template(wavelength, [0.45, 0.05, 0.50], width=4.0),
    ]
    templates = true_templates + decoy_templates
    true_w = np.array([0.6, 0.4])

    sig = 100.0
    cfg = FitConfig(vmin=-7 * sig, vmax=7 * sig, lam_rule="fixed", continuum_knots=None)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=0.0, sigma=sig, wiggle_scale=0.0, seed=0)

    A_stack_true = build_template_stack(wave_fit, wavelength, true_templates, v_grid)
    galaxy_noiseless = sum(true_w[k] * (A_stack_true[k] @ b_truth) for k in range(2))

    rng = np.random.default_rng(seed)
    noise_level = np.median(galaxy_noiseless) / snr
    bin_specs = []
    for i in range(n_bins):
        galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
        errors = np.full(len(wave_fit), noise_level)
        bin_specs.append((wave_fit, galaxy, errors))

    return bin_specs, templates, wavelength, cfg


def test_reduction_shrinks_the_candidate_set_when_decoys_are_present():
    # snr=500 (not this file's usual 80.0), pinned deliberately (2026-07,
    # Stage 22): JAX float64 (the new package default) recovers slightly
    # more nonzero NNLS weight on the decoy templates than float32 did at
    # S/N=80 -- checked directly, decoy weights of 0.005-0.03 (vs. the
    # 1e-3 threshold) at snr=80 under x64, cleanly zero for the true pair
    # only at snr>=500 for this specific seed. Not a reduction-*logic* bug
    # (the union/convergence bookkeeping this test actually exercises is
    # unaffected by precision); a higher, still-realistic S/N restores the
    # same clean-separation demonstration the test was designed around.
    bin_specs, templates, wavelength, cfg = _make_bin_specs(n_bins=3, snr=500.0, seed=0)
    result = reduce_templates_iteratively(bin_specs, templates, wavelength, cfg,
                                           threshold=1e-3, max_iterations=5)
    assert len(result.union_selected) < len(templates), (
        "expected real reduction (some decoys dropped), got no shrinkage at all"
    )
    assert len(result.union_selected) >= 1
    assert result.converged


def test_reduction_is_a_noop_when_candidate_set_is_already_minimal():
    bin_specs, templates, wavelength, cfg = _make_bin_specs(n_bins=2, snr=80.0, seed=1)
    true_only = templates[:2]  # no decoys at all
    result = reduce_templates_iteratively(bin_specs, true_only, wavelength, cfg,
                                           threshold=1e-3, max_iterations=5)
    assert result.converged
    assert result.n_iterations == 1  # nothing to prune -- should stabilize immediately
    assert set(result.union_selected.tolist()) == {0, 1}


def test_max_iterations_caps_the_loop():
    bin_specs, templates, wavelength, cfg = _make_bin_specs(n_bins=2, snr=80.0, seed=2)
    result = reduce_templates_iteratively(bin_specs, templates, wavelength, cfg,
                                           threshold=1e-3, max_iterations=1)
    assert result.n_iterations == 1
    assert len(result.per_bin_history[0]) == 1


def test_candidate_set_size_is_non_increasing_across_iterations():
    bin_specs, templates, wavelength, cfg = _make_bin_specs(n_bins=3, snr=80.0, seed=3)
    result = reduce_templates_iteratively(bin_specs, templates, wavelength, cfg,
                                           threshold=1e-3, max_iterations=5)
    n_templates_per_iter = [h["n_templates"] for h in result.per_bin_history[0]]
    assert all(a >= b for a, b in zip(n_templates_per_iter, n_templates_per_iter[1:]))


def test_bins_subset_restricts_participation():
    bin_specs, templates, wavelength, cfg = _make_bin_specs(n_bins=4, snr=80.0, seed=4)
    result = reduce_templates_iteratively(bin_specs, templates, wavelength, cfg,
                                           threshold=1e-3, max_iterations=3, bins_subset=[0, 2])
    assert len(result.per_bin_history[0]) > 0
    assert len(result.per_bin_history[2]) > 0
    assert len(result.per_bin_history[1]) == 0  # did not participate
    assert len(result.per_bin_history[3]) == 0
    np.testing.assert_array_equal(result.bins_subset_used, np.array([0, 2]))


def test_save_and_load_template_subset_roundtrips():
    bin_specs, templates, wavelength, cfg = _make_bin_specs(n_bins=2, snr=80.0, seed=5)
    result = reduce_templates_iteratively(bin_specs, templates, wavelength, cfg,
                                           threshold=1e-3, max_iterations=5)
    names = [f"template_{i}" for i in range(len(templates))]
    ages = np.arange(len(templates), dtype=float)
    metals = np.zeros(len(templates))
    bin_table = [dict(bin_id=i, r_mid=float(i), theta_mid=None) for i in range(len(bin_specs))]

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        save_template_subset(result, bin_table, names, ages, metals, path)
        loaded = load_template_subset(path)
        np.testing.assert_array_equal(loaded["union_selected"], result.union_selected)
        assert loaded["weight_matrix"].shape == (len(bin_specs), len(templates))
        # every bin's final selected templates should have a nonzero recorded weight
        for i, history in enumerate(result.per_bin_history):
            if history:
                for idx in history[-1]["selected_indices"]:
                    assert loaded["weight_matrix"][i, idx] > 0
        np.testing.assert_array_equal(loaded["template_names"], np.array(names))
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_cfg_rho_is_actually_respected_not_silently_whitened_as_zero():
    # Regression test: reduce_templates_iteratively used to whiten by hand
    # (A_stack/errors, flux/errors) regardless of cfg.rho, so a caller
    # passing a real AR(1) correlation had it silently ignored -- every
    # other fit path in this package respects cfg.rho via noise_model.whiten.
    # Confirm the union/weights this loop actually returns differ between
    # rho=0.0 and a strongly-correlated rho on the same data (a real effect,
    # not noise -- AR(1) whitening changes which pixels the fit trusts).
    bin_specs, templates, wavelength, cfg = _make_bin_specs(n_bins=3, snr=80.0, seed=6)
    cfg_rho0 = dataclasses.replace(cfg, rho=0.0)
    cfg_rho_hi = dataclasses.replace(cfg, rho=0.9)

    result_rho0 = reduce_templates_iteratively(bin_specs, templates, wavelength, cfg_rho0,
                                                threshold=1e-3, max_iterations=5)
    result_rho_hi = reduce_templates_iteratively(bin_specs, templates, wavelength, cfg_rho_hi,
                                                  threshold=1e-3, max_iterations=5)

    chi2_rho0 = result_rho0.per_bin_history[0][0]["chi2_red"]
    chi2_rho_hi = result_rho_hi.per_bin_history[0][0]["chi2_red"]
    assert chi2_rho0 != chi2_rho_hi, (
        "chi2_red identical between rho=0.0 and rho=0.9 -- cfg.rho is being ignored again"
    )

    # Cross-check against calling noise_model.whiten directly at the same rho values --
    # the first iteration's own chi2_red should match a by-hand whitened fit exactly
    # (same whitening this loop is supposed to use internally).
    from kinextract.multi_template import fit_losvd
    from kinextract.robust_fit import full_pixel_model
    wave_fit, flux_fit, errors = bin_specs[0]
    v_grid = cfg.build_velocity_grid()
    A_stack = build_template_stack(wave_fit, wavelength, templates, v_grid)
    A_stack_w, y_w, _L = whiten(A_stack, flux_fit, errors, rho=0.9)
    result = fit_losvd(A_stack_w, y_w, cfg_rho_hi, wave_fit=wave_fit)
    model_w = full_pixel_model(result, A_stack_w, wave_fit, cfg_rho_hi)
    expected_chi2 = float(np.sum((model_w - y_w) ** 2)) / len(wave_fit)
    assert np.isclose(chi2_rho_hi, expected_chi2), (
        f"chi2_red={chi2_rho_hi} doesn't match a by-hand whiten(rho=0.9) fit ({expected_chi2}) "
        "-- reduce_templates_iteratively isn't whitening at cfg.rho internally"
    )


def test_continuum_knots_runs_without_crashing():
    # Regression test: reduce_templates_iteratively used to call fit_losvd
    # without wave_fit, which raises ValueError as soon as cfg.continuum_knots
    # is set (continuum-cofitting needs the wavelength grid to build its own
    # B-spline basis) -- this combination is real, not hypothetical: any
    # target whose kinematic-fitting notebook cofits a continuum needs its
    # own template-calibration notebook to run under the same cfg, including
    # continuum_knots, for template selection to reflect the real fit.
    bin_specs, templates, wavelength, cfg = _make_bin_specs(n_bins=2, snr=80.0, seed=7)
    cfg_continuum = dataclasses.replace(cfg, continuum_knots=3, lam_c=2.0, ridge_c=0.1)
    result = reduce_templates_iteratively(bin_specs, templates, wavelength, cfg_continuum,
                                           threshold=1e-3, max_iterations=3)
    assert result.n_iterations >= 1
    assert len(result.union_selected) >= 1
