"""Coverage-calibration study: does a reported bootstrap confidence
interval actually contain the truth as often as it claims to?

This is the natural closing step for tonight's `reselect_lam` fix
(`dev_notes/uncertainty/FINDINGS.md`) -- that work fixed a real, measured
gap between the pinned-lam bootstrap's spread and the true sampling
distribution's own spread, but never checked the thing that actually
matters end-to-end: does `point0['sherm'] +/- sherm_std` (or the
bias-corrected equivalent) contain the true sigma at the rate a "1-sigma"
interval should (~68% for a properly calibrated Gaussian interval)?

Uses the SAME randomized-truth machinery as the rest of
`dev_notes/validation_suite/` (mocks.py's GH/spline generators, always
scored against the truth curve's OWN moments -- never nominal parameters,
per this project's own established practice) rather than a fixed mock, so
this measures genuine coverage across a population of targets, not one
lucky/unlucky case.

For each (sigma, S/N) grid cell: draw N_TRUTHS random cases (sigma/S-N
fixed at the cell, everything else -- h3/h4, templates, weights --
randomized), and for each truth run ONE bootstrap_losvd call (n_boot
replicates, reselect_lam=True, the new default). Record whether the true
sigma/h4 falls within the reported 1-sigma interval, both raw and
(optionally) bias-corrected.

Usage:
    python coverage_sweep.py --sigma-values 50,90,150 --snr-values 40,60,100 \\
        --n-truths 8 --n-boot 15 --out results/coverage_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mocks import EMILES_GRID_DEFAULT, RANGE_PRESETS, observed_wavelength_grid, draw_random_params, build_mock
from kinextract import whiten
from kinextract.uncertainty import bootstrap_losvd


def run_one_coverage_case(rng, noise_seed, sigma, snr, preset, n_boot, n_templates,
                          emiles_ages, emiles_metals, emiles_grid_path, wave_fit,
                          apply_bias_correction):
    params = draw_random_params(rng, fixed={"sigma": sigma, "snr": snr}, preset=preset,
                                gaussian_only=False, n_templates=n_templates,
                                emiles_ages=emiles_ages, emiles_metals=emiles_metals)
    mock = build_mock(params, emiles_grid_path=emiles_grid_path, wave_fit=wave_fit)
    tm = mock.truth_moments

    noise_level = np.median(mock.galaxy_noiseless) / params["snr"]
    nrng = np.random.default_rng(noise_seed)
    galaxy = mock.galaxy_noiseless + nrng.normal(0.0, noise_level, len(mock.wave_fit))
    errors = np.full(len(mock.wave_fit), noise_level)
    A_w, y_w, _ = whiten(mock.A_stack, galaxy, errors, rho=0.0)

    boot = bootstrap_losvd(A_w, y_w, mock.cfg, n_boot=n_boot, seed=noise_seed,
                           apply_bias_correction=apply_bias_correction)

    sherm_raw, sherm_std_raw = boot["point0"]["sherm"], boot["sherm_std"]
    h4_raw, h4_std_raw = boot["point0"]["h4"], boot["h4_std"]
    sherm_corr, sherm_std_corr = boot["point0_corrected"]["sherm"], boot["sherm_std_corrected"]
    h4_corr, h4_std_corr = boot["point0_corrected"]["h4"], boot["h4_std_corrected"]

    def covered(center, half_width, truth):
        return bool(abs(center - truth) <= half_width)

    return dict(
        nominal_sigma=sigma, nominal_snr=snr, truth_sigma=tm["sherm"], truth_h4=tm["h4"],
        truth_kind=params["truth_kind"], gaussian_only=params["gaussian_only"],
        sherm_raw=sherm_raw, sherm_std_raw=sherm_std_raw,
        h4_raw=h4_raw, h4_std_raw=h4_std_raw,
        sherm_corrected=sherm_corr, sherm_std_corrected=sherm_std_corr,
        h4_corrected=h4_corr, h4_std_corrected=h4_std_corr,
        sigma_covered_raw_1s=covered(sherm_raw, sherm_std_raw, tm["sherm"]),
        h4_covered_raw_1s=covered(h4_raw, h4_std_raw, tm["h4"]),
        sigma_covered_corrected_1s=covered(sherm_corr, sherm_std_corr, tm["sherm"]),
        h4_covered_corrected_1s=covered(h4_corr, h4_std_corr, tm["h4"]),
        lam_used=boot["result0"]["lam_used"],
        n_boot=n_boot,
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sigma-values", required=True, help="comma-separated true-sigma grid values")
    p.add_argument("--snr-values", required=True, help="comma-separated S/N grid values")
    p.add_argument("--n-truths", type=int, default=8, help="random truths per (sigma, S/N) cell")
    p.add_argument("--n-boot", type=int, default=15, help="bootstrap replicates per case")
    p.add_argument("--preset", default="safe", choices=list(RANGE_PRESETS))
    p.add_argument("--n-templates", type=int, default=4)
    p.add_argument("--apply-bias-correction", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--emiles-grid", default=EMILES_GRID_DEFAULT)
    return p.parse_args()


def main():
    args = parse_args()
    sigma_values = [float(v) for v in args.sigma_values.split(",")]
    snr_values = [float(v) for v in args.snr_values.split(",")]
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    emiles_path = args.emiles_grid if os.path.isabs(args.emiles_grid) else os.path.join(root, args.emiles_grid)

    grid_file = np.load(emiles_path, allow_pickle=True)
    emiles_ages, emiles_metals = grid_file["ages"], grid_file["metals"]
    wave_fit = observed_wavelength_grid()

    out_path = args.out if os.path.isabs(args.out) else os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_header = not (os.path.exists(out_path) and os.path.getsize(out_path) > 0)

    cells = [(s, n) for s in sigma_values for n in snr_values]
    total = len(cells) * args.n_truths
    done = 0
    t_start = time.time()
    fieldnames = None
    with open(out_path, "a", newline="") as fh:
        writer = None
        for si, (sigma, snr) in enumerate(cells):
            for ti in range(args.n_truths):
                sub_seed = args.seed * 1_000_003 + si * 10_007 + ti
                rng = np.random.default_rng(sub_seed)
                row = run_one_coverage_case(
                    rng, noise_seed=sub_seed, sigma=sigma, snr=snr, preset=args.preset,
                    n_boot=args.n_boot, n_templates=args.n_templates,
                    emiles_ages=emiles_ages, emiles_metals=emiles_metals,
                    emiles_grid_path=emiles_path, wave_fit=wave_fit,
                    apply_bias_correction=args.apply_bias_correction,
                )
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                writer.writerow(row)
                fh.flush()
                done += 1
                elapsed = time.time() - t_start
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] sigma={sigma:g} snr={snr:g} truth#{ti} "
                      f"sigma_cov_raw={row['sigma_covered_raw_1s']} "
                      f"h4_cov_raw={row['h4_covered_raw_1s']}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)
    print(f"\nwrote {done} rows to {out_path}")

    import csv as _csv
    with open(out_path) as f:
        rows = list(_csv.DictReader(f))
    n = len(rows)
    if n:
        sig_cov = sum(r["sigma_covered_raw_1s"] == "True" for r in rows) / n
        h4_cov = sum(r["h4_covered_raw_1s"] == "True" for r in rows) / n
        print(f"empirical 1-sigma coverage (raw): sigma={sig_cov*100:.0f}%  h4={h4_cov*100:.0f}%  "
              f"(nominal target for a 1-sigma interval: ~68%)")


if __name__ == "__main__":
    main()
