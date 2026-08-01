"""Multi-seed continuum-cofitting bias/residual sweep -- the check
`continuum.py`'s own docstring and notebook 02's Section 8 both flag as
missing ("substantially de-risked... but not yet re-validated end-to-end
on a fresh multi-seed bias/residual sweep").

Design: inject a genuine multiplicative continuum distortion
(`true_C = 1 + amp*sin(pi*frac)`, matching notebook 02's own convention)
on top of a real 3-E-MILES-template mock, at a few distortion amplitudes
and S/N levels, N seeds each. Fit four ways per case:
  - no continuum-cofitting, package default (penalty_order=3, adaptive lam)
    -- shows what an unmodeled distortion does to recovery on its own.
  - continuum-cofitting at penalty_order=2, lam_selection='fixed', lam=2.0
    -- the one combination notebook 02 already exercises (single-case,
    not swept) and describes as "the documented-working combination."
  - continuum-cofitting at penalty_order=3, lam_selection='fixed', lam=4096.0
    -- the untested combination (adaptive lam doesn't support continuum-
    cofitting, so a representative order-3 lam has to be chosen by hand;
    4096.0 is a commonly-selected value in this package's own adaptive-lam
    runs at moderate sigma/S-N, not independently tuned for this sweep).
  - continuum-cofitting at penalty_order=3, lam_selection='fixed', lam=32768.0
    -- the top of the order-3 default grid, in case 4096 is too weak here.

Scored against the truth curve's own moments (`compute_losvd_moments`),
this project's own established practice -- never nominal truth parameters.
"""
from __future__ import annotations

import csv
import dataclasses
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from kinextract import (
    FitConfig, build_template_stack, compute_losvd_moments, load_emiles_templates,
    build_spline_truth, whiten,
)
from kinextract.multi_template import fit_losvd

EMILES_GRID = os.path.join(os.path.dirname(__file__), "..", "..",
                           "src/kinextract/template_data/emiles_muse_grid.npz")
TEMPLATE_SELECT = [(1.0, -0.66), (5.0, 0.06), (14.0, 0.40)]
TEMPLATE_W = np.array([0.3, 0.4, 0.3])

TRUE_V, TRUE_SIGMA = 20.0, 90.0
NSIG = 7

AMPLITUDES = [0.05, 0.15, 0.30]
SNR_VALUES = [30.0, 60.0, 100.0]
N_SEEDS = 8

FIT_CONFIGS = [
    ("no_continuum", dict(continuum_knots=None)),
    ("continuum_order2_fixed", dict(continuum_knots=4, lam_selection="fixed", lam=2.0, penalty_order=2)),
    ("continuum_order3_fixed_4096", dict(continuum_knots=4, lam_selection="fixed", lam=4096.0, penalty_order=3)),
    ("continuum_order3_fixed_32768", dict(continuum_knots=4, lam_selection="fixed", lam=32768.0, penalty_order=3)),
]


def build_base_mock():
    templates, template_wave, names, ages, metals = load_emiles_templates(EMILES_GRID, select=TEMPLATE_SELECT)
    observed_wave = 4750.0 + np.arange(3681) * 1.25
    wave_fit = observed_wave[(observed_wave >= 8400.0) & (observed_wave <= 8800.0)]

    cfg = FitConfig(vmin=TRUE_V - NSIG * TRUE_SIGMA, vmax=TRUE_V + NSIG * TRUE_SIGMA)
    v_grid = cfg.build_velocity_grid()
    b_true, _ = build_spline_truth(v_grid, vel=TRUE_V, sigma=TRUE_SIGMA, wiggle_scale=0.15, seed=0)
    truth_moments = compute_losvd_moments(v_grid, b_true, fit_h3h4=True)

    A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
    b_true_flux = b_true / b_true.sum()
    galaxy_shape = sum(TEMPLATE_W[k] * (A_stack[k] @ b_true_flux) for k in range(len(TEMPLATE_W)))

    frac = (wave_fit - wave_fit.min()) / (wave_fit.max() - wave_fit.min())
    return cfg, v_grid, wave_fit, A_stack, galaxy_shape, truth_moments, frac


def run_one_case(cfg, v_grid, wave_fit, A_stack, galaxy_shape, frac, amplitude, snr, seed):
    true_C = 1.0 + amplitude * np.sin(np.pi * frac)
    galaxy_distorted_noiseless = galaxy_shape * true_C
    noise_level = np.median(galaxy_distorted_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_distorted_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w, y_w, _ = whiten(A_stack, galaxy, errors, rho=0.0)

    rows = []
    for label, overrides in FIT_CONFIGS:
        cfg_i = dataclasses.replace(cfg, **overrides)
        wave_arg = wave_fit if cfg_i.continuum_knots is not None else None
        result = fit_losvd(A_stack_w, y_w, cfg_i, wave_fit=wave_arg)
        m = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)
        rows.append(dict(config=label, amplitude=amplitude, snr=snr, seed=seed,
                          vherm=m["vherm"], sherm=m["sherm"], h3=m["h3"], h4=m["h4"],
                          success=result["success"]))
    return rows


def main():
    cfg, v_grid, wave_fit, A_stack, galaxy_shape, truth_moments, frac = build_base_mock()
    print(f"truth: sherm={truth_moments['sherm']:.2f} h3={truth_moments['h3']:.4f} h4={truth_moments['h4']:.4f}")

    all_rows = []
    t0 = time.time()
    n_total = len(AMPLITUDES) * len(SNR_VALUES) * N_SEEDS
    n_done = 0
    for amplitude in AMPLITUDES:
        for snr in SNR_VALUES:
            for seed in range(N_SEEDS):
                rows = run_one_case(cfg, v_grid, wave_fit, A_stack, galaxy_shape, frac, amplitude, snr, seed)
                for r in rows:
                    r["truth_sherm"] = truth_moments["sherm"]
                    r["truth_h3"] = truth_moments["h3"]
                    r["truth_h4"] = truth_moments["h4"]
                all_rows.extend(rows)
                n_done += 1
                print(f"[{n_done}/{n_total}] amp={amplitude} snr={snr} seed={seed} "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "results_multiseed_sweep.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
