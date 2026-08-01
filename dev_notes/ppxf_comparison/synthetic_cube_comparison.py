"""pPXF vs kinextract, head-to-head, on synthetic cubes with known truth.

`dev_notes/scratch_notebooks/ppxf_vs_kinextract_comparison.ipynb` already runs
both engines on the same dartboard-binned cube, but its own synthetic-cube
path uses one fixed `build_mock_cube` config. Here we build a *few* different
mock cubes -- varying sigma, S/N, and LOSVD shape -- and ask, with actual
ground truth available (unlike the real NGC 5102 cube), which regimes each
method wins in and why.

`build_mock_cube` uses a single, spatially-constant true (V, sigma, h3, h4)
LOSVD at every spaxel (see its own docstring), modulated by a Sersic surface-
brightness profile against a constant noise floor -- so S/N declines with
radius automatically within ONE cube. Binning radially (the same reflected
dartboard scheme as the real-data notebook, PA irrelevant here since there is
no true rotation) turns each single cube into a whole S/N sweep for free, with
the same true answer in every bin -- the actual point of using a synthetic
cube instead of a single mock spectrum.

Usage: python synthetic_cube_comparison.py [--quick]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scratch_notebooks"))
from dartboard_pa import build_pa_dartboard

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as putil

from kinextract import (
    FitConfig, build_mock_cube, build_template_stack, compute_losvd_moments,
)
from kinextract.multi_template import fit_losvd

EMILES_GRID = os.path.join(os.path.dirname(__file__), "..", "..",
                           "src/kinextract/template_data/emiles_muse_grid.npz")
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# A handful of distinct regimes, chosen to probe open questions from
# dev_notes/FULL_REPO_AUDIT_2026-07-29.md / AUDIT_BRIEFING.md rather than
# arbitrary settings: a baseline, the high-sigma "sloppy manifold" regime,
# genuine non-Gaussian wings (h3 suppression is a known open issue at
# low-moderate sigma), and a faint/low-S/N cube.
CUBE_CONFIGS = [
    dict(name="baseline_sig50", true_sigma=50.0, truth_kind="spline", wiggle_scale=0.15,
         amplitude=3.0, noise_floor=2.0),
    dict(name="high_sigma150", true_sigma=150.0, truth_kind="spline", wiggle_scale=0.15,
         amplitude=3.0, noise_floor=2.0),
    dict(name="genuine_wings_h3h4", true_sigma=80.0, truth_kind="gh", true_h3=0.08, true_h4=0.12,
         amplitude=3.0, noise_floor=2.0),
    dict(name="faint_sig80", true_sigma=80.0, truth_kind="spline", wiggle_scale=0.15,
         amplitude=0.8, noise_floor=2.0),
]

RMAX_ARCSEC = 5.0
N_THETA_PER_QUADRANT = 2
REFERENCE_R_EDGES_ARCSEC = np.array([0.2, 0.6, 1.3, 2.4, 3.8, 5.5])
NSIG = 7


def build_signal_noise_maps(flux_cube, error_cube):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        signal_map = np.nanmedian(flux_cube, axis=0)
        noise_map = np.nanmedian(error_cube, axis=0)
    good = np.isfinite(signal_map) & np.isfinite(noise_map) & (signal_map > 0) & (noise_map > 0)
    return signal_map, noise_map, good


def build_bin_spectra(flux_cube, error_cube, bin_map, nbins):
    n_pix = flux_cube.shape[0]
    spectra = np.full((n_pix, nbins), np.nan)
    variance = np.full((n_pix, nbins), np.nan)
    for j in range(nbins):
        mask = bin_map == j
        n = int(mask.sum())
        if n == 0:
            continue
        spectra[:, j] = np.nanmean(flux_cube[:, mask], axis=1)
        variance[:, j] = np.nansum(error_cube[:, mask] ** 2, axis=1) / n ** 2
    return spectra, variance


def fit_bin_kinextract(wave_fit, template_wave, templates, flux, errors, approx_v, approx_sigma, nsig=NSIG):
    """Single-stage fit, centered on a rough prior (V, sigma) range -- the
    same role pPXF's own `start=[0, 100]` guess plays, not a blind wide
    search. Fair since both methods get *a* starting guess, not the exact
    truth (`approx_v`/`approx_sigma` here are the cube's nominal config
    values, not its actual truth-curve moments, which can differ -- see
    `build_spline_truth`'s own docstring on that gap)."""
    y_w = flux / errors
    cfg = FitConfig(vmin=approx_v - nsig * approx_sigma, vmax=approx_v + nsig * approx_sigma)
    v_grid = cfg.build_velocity_grid()
    A = build_template_stack(wave_fit, template_wave, templates, v_grid)
    A_w = A / errors[None, :, None]
    result = fit_losvd(A_w, y_w, cfg)
    moments = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)
    return moments


def fit_bin_ppxf(templates_log, galaxy, noise, velscale, lam_gal_log, lam_temp,
                  moments=4, degree=6, mdegree=2):
    start = [0.0, 100.0]
    pp = ppxf(templates_log, galaxy, noise, velscale, start,
              moments=moments, degree=degree, mdegree=mdegree,
              lam=lam_gal_log, lam_temp=lam_temp, quiet=True)
    return dict(vherm=pp.sol[0], sherm=pp.sol[1], h3=pp.sol[2], h4=pp.sol[3])


def run_one_cube(cfg_dict, use_cache=True):
    name = cfg_dict["name"]
    print(f"\n=== cube: {name} ===")
    build_kwargs = {k: v for k, v in cfg_dict.items() if k != "name"}
    cube = build_mock_cube(EMILES_GRID, ny=40, nx=40, seed=0, **build_kwargs)
    truth = cube.truth
    tm = truth["truth_moments"]
    print(f"truth: sherm={tm['sherm']:.2f} h3={tm['h3']:.4f} h4={tm['h4']:.4f} "
          f"(nominal sigma={truth['true_sigma']}, truth_kind={truth['truth_kind']})")

    select = truth["template_select"]
    from kinextract import load_emiles_templates
    templates, template_wave, names, ages, metals = load_emiles_templates(EMILES_GRID, select=select)

    wave_fit = cube.wave
    flux_cube, error_cube = cube.flux, cube.error
    ny, nx = cube.shape
    signal_map, noise_map, good0 = build_signal_noise_maps(flux_cube, error_cube)

    yy, xx = np.indices((ny, nx))
    x_arcsec = (xx - (nx - 1) / 2.0) * cube.pixel_scale
    y_arcsec = (yy - (ny - 1) / 2.0) * cube.pixel_scale
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r_arcsec = np.hypot(x_arcsec - x_arcsec[int(cy), int(cx)], y_arcsec - y_arcsec[int(cy), int(cx)])
    good = good0 & (r_arcsec <= RMAX_ARCSEC)

    result = build_pa_dartboard(
        good, signal_map, noise_map, x_arcsec, y_arcsec, cube.pixel_scale,
        pa_deg=0.0, rmax_arcsec=RMAX_ARCSEC, reference_r_edges_arcsec=REFERENCE_R_EDGES_ARCSEC,
        n_theta_per_quadrant=N_THETA_PER_QUADRANT,
    )
    bin_map, nbins, bin_sn = result.bin_map, result.nbins, result.bin_sn
    print(f"{nbins} dartboard bins, S/N range {np.nanmin(bin_sn):.1f}-{np.nanmax(bin_sn):.1f}")

    bin_spectra, bin_variance = build_bin_spectra(flux_cube, error_cube, bin_map, nbins)
    bin_errors = np.sqrt(np.clip(bin_variance, 1e-12, None))

    cache_path = CACHE_DIR / f"{name}.csv"
    if use_cache and cache_path.exists():
        print(f"loaded from cache: {cache_path.name}")
        with open(cache_path) as f:
            return list(csv.DictReader(f))

    c_kms = 299792.458
    velscale = c_kms * np.min(np.diff(np.log(wave_fit)))
    lam_range_gal = [wave_fit.min(), wave_fit.max()]
    bin_spectra_log, ln_lam_gal, velscale = putil.log_rebin(lam_range_gal, bin_spectra, velscale=velscale)
    bin_variance_log, _, _ = putil.log_rebin(lam_range_gal, bin_variance, velscale=velscale)
    bin_noise_log = np.sqrt(np.clip(bin_variance_log, 1e-12, None))
    lam_gal_log = np.exp(ln_lam_gal)

    templates_lin = np.column_stack(templates)
    lam_range_temp = [template_wave.min(), template_wave.max()]
    templates_log, ln_lam_temp, _ = putil.log_rebin(lam_range_temp, templates_lin, velscale=velscale)
    lam_temp = np.exp(ln_lam_temp)

    rows = []
    t0 = time.time()
    for j in range(nbins):
        if np.any(np.isnan(bin_spectra[:, j])):
            continue
        kin_m = fit_bin_kinextract(wave_fit, template_wave, templates, bin_spectra[:, j], bin_errors[:, j],
                                    approx_v=truth["true_v"], approx_sigma=truth["true_sigma"])
        pp_m = fit_bin_ppxf(templates_log, bin_spectra_log[:, j], bin_noise_log[:, j], velscale,
                             lam_gal_log, lam_temp)
        rows.append(dict(
            cube=name, bin=j, sn=float(bin_sn[j]),
            truth_sherm=tm["sherm"], truth_h3=tm["h3"], truth_h4=tm["h4"], truth_vherm=tm["vherm"],
            kin_sherm=kin_m["sherm"], kin_h3=kin_m["h3"], kin_h4=kin_m["h4"], kin_vherm=kin_m["vherm"],
            ppxf_sherm=pp_m["sherm"], ppxf_h3=pp_m["h3"], ppxf_h4=pp_m["h4"], ppxf_vherm=pp_m["vherm"],
        ))
        if (j + 1) % 10 == 0 or j == nbins - 1:
            print(f"  {j + 1}/{nbins} bins done ({time.time() - t0:.0f}s elapsed)")

    with open(cache_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} rows to {cache_path.name}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only run the first cube config")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    configs = CUBE_CONFIGS[:1] if args.quick else CUBE_CONFIGS
    all_rows = []
    for cfg_dict in configs:
        all_rows.extend(run_one_cube(cfg_dict, use_cache=not args.no_cache))

    out_path = Path(__file__).parent / "results_all_cubes.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} total rows to {out_path}")


if __name__ == "__main__":
    main()
