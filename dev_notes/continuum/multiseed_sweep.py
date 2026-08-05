"""Rebuild of the continuum-cofitting multi-seed validation sweep
(`dev_notes/continuum/FINDINGS.md`, 2026-07-30), extended to a broader
sigma grid -- the original script was lost from disk; this reconstructs
it from that file's own documented design (same template mixture
convention, same injected-distortion form) and extends it to sigma in
{50, 90, 150} instead of sigma=90 only, per
`dev_notes/V1_HARDENING_ROADMAP.md`'s Phase 2d.

**Also folds in this session's own continuum-mean-unidentifiability
finding** (`dev_notes/continuum/FINDINGS.md`, 2026-08-01): reports the
recovered continuum's RMS both at its raw absolute level (which the
2026-07-30 sweep used, and which is NOT expected to match the truth's own
level -- see that entry) and after subtracting each curve's own mean
(the number that actually reflects fit quality, isolating shape error
from the known-unidentifiable level offset).

Compares two configurations only (not the original four): no
continuum-cofitting (shows what an unmodeled distortion does on its own)
and continuum-cofitting at `penalty_order=3, lam=4096.0` -- the
2026-07-30 sweep's own winner, validated here across sigma rather than
re-litigating the penalty_order=2 vs 3 question it already settled.

Usage: python multiseed_sweep.py
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from kinextract import (
    FitConfig, build_template_stack, build_spline_truth, compute_losvd_moments,
    load_emiles_templates, build_bspline_basis, difference_matrix, fit_losvd_with_continuum,
)
from kinextract.multi_template import fit_losvd

EMILES_GRID = os.path.join(os.path.dirname(__file__), "..", "..",
                           "src", "kinextract", "template_data", "emiles_muse_grid.npz")
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
TEMPLATE_SELECT = [(1.0, -0.66), (5.0, 0.06), (14.0, 0.40)]
TRUE_W = np.array([0.3, 0.4, 0.3])

SIGMA_GRID = [50.0, 90.0, 150.0]
AMP_GRID = [0.05, 0.15, 0.30]
SNR_GRID = [30.0, 60.0, 100.0]
N_SEEDS = 8
CONTINUUM_LAM = 4096.0


def run_one(sigma, amp, snr, seed, templates, template_wave, wave_fit):
    cfg = FitConfig(vmin=-7 * sigma, vmax=7 * sigma)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=0.0, sigma=sigma, wiggle_scale=0.15, seed=seed)
    A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
    galaxy_noiseless = sum(TRUE_W[k] * (A_stack[k] @ b_truth) for k in range(3))

    frac = (wave_fit - wave_fit.min()) / (wave_fit.max() - wave_fit.min())
    true_C = 1.0 + amp * np.sin(np.pi * frac)
    galaxy_distorted_noiseless = galaxy_noiseless * true_C

    noise_level = np.median(galaxy_distorted_noiseless) / snr
    rng = np.random.default_rng(1000 * seed + int(sigma) + int(100 * amp) + int(snr))
    galaxy = galaxy_distorted_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors

    true_moments = compute_losvd_moments(v_grid, b_truth / np.trapezoid(b_truth, v_grid), fit_h3h4=True)

    result_noC = fit_losvd(A_stack_w, y_w, cfg)
    m_noC = compute_losvd_moments(v_grid, result_noC["b"], fit_h3h4=True)

    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D2 = difference_matrix(n_coef, order=cfg.penalty_order, dv=dv, dv_ref=cfg.knot_spacing_kms)
    result_C = fit_losvd_with_continuum(A_stack_w, y_w, wave_fit, basis, D2,
                                        n_continuum_knots=4, lam=CONTINUUM_LAM)
    m_C = compute_losvd_moments(v_grid, result_C["b"], fit_h3h4=True)
    C_rms_absolute = float(np.sqrt(np.mean((result_C["C"] - true_C) ** 2)))
    C_rms_shape = float(np.sqrt(np.mean(((result_C["C"] - result_C["C"].mean())
                                          - (true_C - true_C.mean())) ** 2)))

    return dict(
        sigma=sigma, amp=amp, snr=snr, seed=seed,
        true_sherm=true_moments["sherm"], true_h4=true_moments["h4"],
        noC_sig_bias=m_noC["sherm"] - true_moments["sherm"], noC_h4_bias=m_noC["h4"] - true_moments["h4"],
        C_sig_bias=m_C["sherm"] - true_moments["sherm"], C_h4_bias=m_C["h4"] - true_moments["h4"],
        C_rms_absolute=C_rms_absolute, C_rms_shape=C_rms_shape,
    )


def main():
    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=TEMPLATE_SELECT)
    observed_wavelength = 4750.0 + np.arange(3681) * 1.25
    wave_fit = observed_wavelength[(observed_wavelength >= WAVEFITMIN) & (observed_wavelength <= WAVEFITMAX)]

    cells = [(s, a, n) for s in SIGMA_GRID for a in AMP_GRID for n in SNR_GRID]
    total = len(cells) * N_SEEDS
    out_path = os.path.join(os.path.dirname(__file__), "results_multiseed_sweep_v2.csv")
    fieldnames = None
    done = 0
    t0 = time.time()
    with open(out_path, "w", newline="") as fh:
        writer = None
        for sigma, amp, snr in cells:
            for seed in range(N_SEEDS):
                row = run_one(sigma, amp, snr, seed, templates, template_wave, wave_fit)
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                fh.flush()
                done += 1
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] sigma={sigma:g} amp={amp:g} snr={snr:g} seed={seed} "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)
    print(f"\nwrote {done} rows to {out_path}")


if __name__ == "__main__":
    main()
