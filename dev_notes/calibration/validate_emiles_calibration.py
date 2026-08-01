"""Validates the new E-MILES-based regularization-bias calibration
(`build_emiles_calibration.py`) against KNOWN synthetic truth -- not
against the legacy Fortran pipeline, which was found (see
`VALIDATION_CAMPAIGN.md` Stage 9) to carry its own real biases and is
therefore not a trustworthy ground truth.

Two separate checks, not one, because "does it generalize" has two
different meanings here:

1. **Self-consistency**: a FRESH random Dirichlet mixture over the SAME
   E-MILES template subset the calibration was built from (disjoint seed
   range from the calibration's own `WEIGHT_SEED=30000`) -- does the
   correction generalize across different population mixtures drawn from
   its own training template library?
2. **Cross-domain**: a fresh random Dirichlet mixture over the REAL 33
   MUSE stellar templates instead -- does an E-MILES-trained correction
   still transfer reasonably to real stellar spectra, the templates real
   MUSE fits have historically used? This is the check that matters most
   for real-data trustworthiness, and the one the old MUSE-star-specific
   calibration didn't need to pass (it *was* the real templates).

Usage: python validate_emiles_calibration.py [--regime emiles|real|both]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import (
    FitConfig, build_template_stack, gauss_hermite_model, compute_losvd_moments,
    estimate_snr_whitened, correct_moments, load_calibration_table, load_emiles_templates,
)
from kinextract.multi_template import fit_losvd

EMILES_GRID = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "kinextract",
    "template_data", "emiles_muse_grid.npz",
)
REAL_TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
NEW_CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "kinextract",
    "calibration_data", "regularization_bias_emiles_multitemplate.txt",
)

AGES = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 14.0]
METALS = [-1.49, -0.66, 0.06, 0.40]
WEIGHT_SEED_BASE = 90000  # disjoint from the calibration's own WEIGHT_SEED=30000
SIGMA_GRID = [30, 40, 50, 60, 80, 100, 140]
SNR_GRID = [15, 25, 40, 60, 80, 100, 150]
N_REPS = 25
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
NSIG = 7


def true_moments(vel, sig, n=200_001):
    vv = np.linspace(vel - 15 * sig, vel + 15 * sig, n)
    bb = np.maximum(gauss_hermite_model(vv, 1.0, vel, sig, 0.0, 0.0), 0.0)
    tot = np.trapezoid(bb, vv)
    m1 = np.trapezoid(bb * vv, vv) / tot
    m2 = np.sqrt(np.trapezoid(bb * (vv - m1) ** 2, vv) / tot)
    return m1, m2


def _load_emiles_regime():
    select = [(a, z) for a in AGES for z in METALS]
    templates, wavelength, names, ages, metals = load_emiles_templates(EMILES_GRID, select=select)
    return templates, wavelength


def _load_real_regime():
    with open(os.path.join(REAL_TEMPLATE_DIR, "Tlist")) as f:
        names = [line.strip() for line in f if line.strip()]
    wavelength = 4750.0 + np.arange(3681) * 1.25
    templates = [np.interp(wavelength, *np.loadtxt(os.path.join(REAL_TEMPLATE_DIR, n)).T) for n in names]
    return templates, wavelength


def run_regime(label, templates, wavelength, calibration, outfile):
    n_templates = len(templates)
    mask = (wavelength >= WAVEFITMIN) & (wavelength <= WAVEFITMAX)
    wave_fit = wavelength[mask]
    print(f"\n=== regime: {label} ({n_templates} templates) ===", flush=True)

    rows = []
    t0 = time.time()
    with open(outfile, "w") as f:
        header = (f"{'sigma':>6} {'snr':>6} {'n':>4}   "
                   f"{'raw_err_med':>12} {'corr_err_med':>13}   "
                   f"{'raw_err_mean':>13} {'corr_err_mean':>14}   "
                   f"{'frac_corr_better':>16}\n")
        f.write(header)
        print(header, end="")

        for sig in SIGMA_GRID:
            vmin, vmax = -NSIG * sig, NSIG * sig
            cfg = FitConfig(vmin=vmin, vmax=vmax)
            v_grid = cfg.build_velocity_grid()
            b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, 0.0, sig, 0.0, 0.0), 0.0)
            b_truth /= b_truth.sum()
            A_stack = build_template_stack(wave_fit, wavelength, templates, v_grid)
            true_v, true_s = true_moments(0.0, sig)

            for snr in SNR_GRID:
                raw_errs, corr_errs = [], []
                for rep in range(N_REPS):
                    seed = WEIGHT_SEED_BASE + rep + int(sig * 10) + int(snr * 100)
                    rng_w = np.random.default_rng(seed)
                    w_true = rng_w.dirichlet(np.ones(n_templates))
                    galaxy_noiseless = sum(w_true[k] * (A_stack[k] @ b_truth) for k in range(n_templates))
                    noise_level = np.median(galaxy_noiseless) / snr
                    rng_n = np.random.default_rng(seed + 500000)
                    galaxy = galaxy_noiseless + rng_n.normal(0.0, noise_level, len(wave_fit))
                    errors = np.full(len(wave_fit), noise_level)
                    A_stack_w = A_stack / errors[None, :, None]
                    y_w = galaxy / errors

                    result = fit_losvd(A_stack_w, y_w, cfg)
                    m = compute_losvd_moments(v_grid, result["b"])
                    if not (np.isfinite(m["vherm"]) and np.isfinite(m["sherm"])):
                        continue
                    model_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0])) @ result["b"]
                    snr_est = estimate_snr_whitened(model_w)
                    corr = correct_moments(m["vherm"], m["sherm"], m["h4"], snr_est, calibration=calibration)

                    raw_errs.append(abs(m["sherm"] - true_s))
                    corr_errs.append(abs(corr["sherm_corrected"] - true_s))

                n = len(raw_errs)
                if n < 3:
                    continue
                raw_errs, corr_errs = np.array(raw_errs), np.array(corr_errs)
                frac_better = float(np.mean(corr_errs < raw_errs))
                row = dict(sigma=sig, snr=snr, n=n,
                           raw_err_med=float(np.median(raw_errs)), corr_err_med=float(np.median(corr_errs)),
                           raw_err_mean=float(np.mean(raw_errs)), corr_err_mean=float(np.mean(corr_errs)),
                           frac_corr_better=frac_better)
                rows.append(row)
                line = (f"{sig:6.0f} {snr:6.0f} {n:4d}   "
                        f"{row['raw_err_med']:12.2f} {row['corr_err_med']:13.2f}   "
                        f"{row['raw_err_mean']:13.2f} {row['corr_err_mean']:14.2f}   "
                        f"{frac_better:16.2f}\n")
                f.write(line)
                f.flush()
                print(line, end="", flush=True)

        footer = f"\n# time: {time.time()-t0:.0f}s\n"
        f.write(footer)
        print(footer, end="")

    overall_raw = np.array([r["raw_err_med"] for r in rows])
    overall_corr = np.array([r["corr_err_med"] for r in rows])
    print(f"\n[{label}] Overall: {len(rows)} cells. Correction reduced median sigma error in "
          f"{np.mean(overall_corr < overall_raw)*100:.0f}% of cells.")
    print(f"[{label}] Median raw_err across cells: {np.median(overall_raw):.2f} km/s")
    print(f"[{label}] Median corr_err across cells: {np.median(overall_corr):.2f} km/s")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["emiles", "real", "both"], default="both")
    args = ap.parse_args()

    calibration = load_calibration_table(NEW_CALIBRATION_PATH)
    print(f"loaded new calibration: sigma_range={calibration.sigma_range}, snr_range={calibration.snr_range}")

    if args.regime in ("emiles", "both"):
        templates, wavelength = _load_emiles_regime()
        run_regime("emiles-self-consistency", templates, wavelength, calibration,
                   "validate_emiles_calibration_emiles_regime.txt")

    if args.regime in ("real", "both"):
        templates, wavelength = _load_real_regime()
        run_regime("real-MUSE-star cross-domain", templates, wavelength, calibration,
                   "validate_emiles_calibration_real_regime.txt")
