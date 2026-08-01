"""Tests for kinextract.ablation.

Cheap smoke tests (tiny grid, few replicates, few templates) -- the real
overnight ablation ladder run is a separate script
(examples/notebooks/ablation_sweep.py), not something a unit test runs.
"""
from __future__ import annotations

import os

import numpy as np

from kinextract import (
    FitConfig, build_template_stack, build_spline_truth,
    AblationConfig, fit_one, run_ablation_sweep, save_ablation_table, load_ablation_table,
)


def _toy_setup(sig_true=80.0, snr=100.0, seed=0):
    wavelength = 4750.0 + np.arange(3681) * 1.25
    mask = (wavelength >= 8400.0) & (wavelength <= 8800.0)
    wave_fit = wavelength[mask]
    template = np.ones(3681) - 0.4 * np.exp(-0.5 * ((wavelength - 8600.0) / 8.0) ** 2)

    cfg = FitConfig(vmin=-7 * sig_true, vmax=7 * sig_true)
    v_grid = cfg.build_velocity_grid()
    # spline truth (not Gauss-Hermite): no generator-vs-moments gap to worry about
    # for a general fit-mechanics mock -- see dev_notes/validation_suite/FINDINGS.md.
    b_truth, _ = build_spline_truth(v_grid, vel=0.0, sigma=sig_true, wiggle_scale=0.0, seed=seed)
    A_stack = build_template_stack(wave_fit, wavelength, [template], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / snr
    galaxy = galaxy_noiseless + np.random.default_rng(seed).normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors
    return cfg, A_stack_w, y_w, wavelength, [template]


def test_fit_one_unregularized_vs_default_differ():
    cfg, A_stack_w, y_w, _, _ = _toy_setup()
    bare = AblationConfig(name="bare", lam=0.0, moment_method="trapz", fit_h3h4=False)
    full = AblationConfig(name="full")  # shipped defaults
    m_bare = fit_one(A_stack_w, y_w, cfg, bare)
    m_full = fit_one(A_stack_w, y_w, cfg, full)
    assert m_bare["success"] and m_full["success"]
    assert np.isfinite(m_bare["sherm"]) and np.isfinite(m_full["sherm"])
    # Different mechanisms active -- results need not (and generally won't) match exactly.
    assert m_bare["h3"] == 0.0 and m_bare["h4"] == 0.0  # fit_h3h4=False -> fixed zero


def test_fit_one_single_start_flag_takes_effect():
    cfg, A_stack_w, y_w, _, _ = _toy_setup(sig_true=140.0, seed=3)
    multi = AblationConfig(name="multi", losvd_single_start=False)
    single = AblationConfig(name="single", losvd_single_start=True)
    m_multi = fit_one(A_stack_w, y_w, cfg, multi)
    m_single = fit_one(A_stack_w, y_w, cfg, single)
    assert m_multi["success"] and m_single["success"]


def test_run_ablation_sweep_produces_one_row_per_config_per_cell():
    wavelength = 4750.0 + np.arange(3681) * 1.25
    template = np.ones(3681) - 0.4 * np.exp(-0.5 * ((wavelength - 8600.0) / 8.0) ** 2)
    configs = [
        AblationConfig(name="unregularized", lam=0.0, moment_method="trapz", fit_h3h4=False),
        AblationConfig(name="full"),
    ]
    rows = run_ablation_sweep(
        configs, [template], wavelength, weights=np.array([1.0]),
        sigma_grid=[80.0], snr_grid=[60.0], n_reps=4, verbose=False,
    )
    names = {r["config_name"] for r in rows}
    assert names == {"unregularized", "full"}
    assert len(rows) == 2  # 1 sigma x 1 snr x 2 configs


def test_run_ablation_sweep_accepts_weight_sampler():
    wavelength = 4750.0 + np.arange(3681) * 1.25
    templates = [np.ones(3681) - 0.4 * np.exp(-0.5 * ((wavelength - 8600.0) / 8.0) ** 2) for _ in range(3)]
    rng = np.random.default_rng(0)

    def sampler():
        return rng.dirichlet(np.ones(3))

    configs = [AblationConfig(name="full")]
    rows = run_ablation_sweep(
        configs, templates, wavelength, weights=sampler,
        sigma_grid=[80.0], snr_grid=[60.0], n_reps=4, verbose=False,
    )
    assert len(rows) == 1 and rows[0]["n"] == 4


def test_run_ablation_sweep_accepts_h4_true():
    wavelength = 4750.0 + np.arange(3681) * 1.25
    template = np.ones(3681) - 0.4 * np.exp(-0.5 * ((wavelength - 8600.0) / 8.0) ** 2)
    configs = [AblationConfig(name="full")]
    rows = run_ablation_sweep(
        configs, [template], wavelength, weights=np.array([1.0]),
        sigma_grid=[80.0], snr_grid=[60.0], n_reps=4, h4_true=0.15, verbose=False,
    )
    assert len(rows) == 1 and np.isfinite(rows[0]["h4_mean"])


def test_save_and_load_ablation_table_roundtrips(tmp_path):
    wavelength = 4750.0 + np.arange(3681) * 1.25
    template = np.ones(3681) - 0.4 * np.exp(-0.5 * ((wavelength - 8600.0) / 8.0) ** 2)
    configs = [
        AblationConfig(name="unregularized", lam=0.0, moment_method="trapz", fit_h3h4=False),
        AblationConfig(name="full"),
    ]
    rows = run_ablation_sweep(
        configs, [template], wavelength, weights=np.array([1.0]),
        sigma_grid=[50.0, 100.0], snr_grid=[60.0], n_reps=4, verbose=False,
    )
    path = os.path.join(tmp_path, "test_ablation.txt")
    save_ablation_table(rows, path, header_comment="test table")
    loaded = load_ablation_table(path)
    assert len(loaded) == len(rows)
    assert {r["config_name"] for r in loaded} == {"unregularized", "full"}
