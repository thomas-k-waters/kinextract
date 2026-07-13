"""Comprehensive kinextract vs pPXF comparison across the validated sigma grid.

Reuses:
- The user's own validated pPXF setup from temporary_pPXF.ipynb: ppxf.sps_util.sps_lib
  for E-MILES loading (full 150-template grid, norm_range=[8000,9000], standard pPXF
  practice), and the same two-pass fit-then-clip-outliers approach as their
  ppxf_fit_and_clean().
- kinextract's own validated setup from notebook 02 / the stress-test session: a
  hand-picked 20-SSP old/moderate-metallicity E-MILES subgrid, 1000A (8000-9000A) fit
  window.
- harness_emiles.build_emiles_mock for the actual synthetic spectra (same generating
  population/continuum/noise convention as the rest of this project's validation).

This is deliberately "each package's own best-practice setup", not artificially forcing
identical templates between them -- pPXF users conventionally use the full SPS grid;
kinextract's own stress test found a curated subgrid works better for it specifically.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import warnings
from importlib import resources
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from harness_emiles import build_emiles_mock, write_spec_file, WAVELENGTH, WAVEMIN, STEP, N_PIX  # noqa: E402

from kinextract import FitConfig, run_spectral_fit, set_verbose  # noqa: E402
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite  # noqa: E402

import ppxf.ppxf_util as util  # noqa: E402
import ppxf.sps_util as lib  # noqa: E402
from ppxf.ppxf import ppxf, robust_sigma  # noqa: E402

EMILES_TLIST = "dev_notes/stress_test/emiles_templates/Tlist"
EMILES_DIR = "dev_notes/stress_test/emiles_templates"
CEE = 299792.458

# ---------------------------------------------------------------------------
# kinextract fit (matches notebook 02 / stress-test-validated config exactly)
# ---------------------------------------------------------------------------
def fit_kinextract(spec_path: Path, sigma_guess: float) -> dict:
    vrange = max(300.0, 3.5 * sigma_guess)
    cfg = FitConfig(
        template_list_file=EMILES_TLIST, template_dir=EMILES_DIR,
        wavemin_full=WAVEMIN, step=STEP, wavefitmin=8000.0, wavefitmax=9000.0,
        zgal=0.0, fit_continuum=True, use_spectrum_errors=True, xlam_auto=True,
        losvd_vmin=-vrange, losvd_vmax=vrange, sigl=sigma_guess, clean=False,
        map_maxiter=10000, print_every=999999,
    )
    fit = run_spectral_fit(cfg, gal_file=str(spec_path))
    gh = fit_losvd_gauss_hermite(fit["state"].xl, fit["outputs"]["b"], fit_h3h4=True)
    return dict(v=gh["vherm"], sigma=gh["sherm"], h3=gh["h3"], h4=gh["h4"],
                chi2_red=fit["outputs"]["chi2_red"])


# ---------------------------------------------------------------------------
# pPXF fit (matches the user's own validated temporary_pPXF.ipynb setup)
# ---------------------------------------------------------------------------
_emiles_sps_cache = {}


def _clip_outliers(galaxy, bestfit, mask, sigma_clip=2.2):
    while True:
        scale = galaxy[mask] @ bestfit[mask] / np.sum(bestfit[mask] ** 2)
        resid = scale * bestfit[mask] - galaxy[mask]
        err = robust_sigma(resid, zero=1)
        ok_old = mask
        mask = np.abs(bestfit - galaxy) < sigma_clip * err
        if np.array_equal(mask, ok_old):
            break
    return mask


def _get_emiles_sps(velscale):
    key = round(velscale, 6)
    if key in _emiles_sps_cache:
        return _emiles_sps_cache[key]
    ppxf_dir = resources.files("ppxf")
    filename = ppxf_dir / "sps_models" / "spectra_emiles_9.0.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sps = lib.sps_lib(str(filename), velscale, None, norm_range=[8000, 9000])
    npix, *reg_dim = sps.templates.shape
    sps.templates = sps.templates.reshape(npix, -1)
    sps.templates /= np.median(sps.templates)
    _emiles_sps_cache[key] = (sps, npix, reg_dim)
    return _emiles_sps_cache[key]


def fit_ppxf(wavelength: np.ndarray, gal: np.ndarray, errs: np.ndarray,
             wavefitmin: float, wavefitmax: float, sigma_guess: float) -> dict:
    mask_w = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    lam_gal = wavelength[mask_w]
    galaxy_raw = gal[mask_w]
    noise_raw = errs[mask_w]

    velscale_guess = np.median(CEE * np.diff(np.log(lam_gal)))
    galaxy, ln_lam_gal, velscale = util.log_rebin(
        [lam_gal[0], lam_gal[-1]], galaxy_raw, velscale=velscale_guess)
    variance, _, _ = util.log_rebin(
        [lam_gal[0], lam_gal[-1]], noise_raw ** 2, velscale=velscale)
    noise = np.sqrt(np.clip(variance, 1e-12, None))

    lam_gal_log = np.exp(ln_lam_gal)  # wavelength grid matching the log-rebinned `galaxy`

    sps, npix, reg_dim = _get_emiles_sps(velscale)
    lam_range_temp = np.exp(sps.ln_lam_temp[[0, -1]])
    mask0 = util.determine_mask(ln_lam_gal, lam_range_temp, width=1000)

    start = [0.0, sigma_guess]
    degree, mdegree, sigma_clip = 6, 2, 2.2  # matches the user's own "red window" settings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = ppxf(sps.templates, galaxy, noise, velscale, start,
                  moments=2, degree=degree, mdegree=mdegree, lam=lam_gal_log,
                  lam_temp=sps.lam_temp, mask=mask0, quiet=True)
        mask = _clip_outliers(galaxy, pp.bestfit, mask0.copy(), sigma_clip=sigma_clip)
        mask &= mask0
        start4 = [pp.sol[0], pp.sol[1], 0.0, 0.0]
        pp = ppxf(sps.templates, galaxy, noise, velscale, start4,
                  moments=4, degree=degree, mdegree=mdegree, lam=lam_gal_log,
                  lam_temp=sps.lam_temp, mask=mask, quiet=True)

    return dict(v=pp.sol[0], sigma=pp.sol[1], h3=pp.sol[2], h4=pp.sol[3],
                chi2=getattr(pp, "chi2", np.nan))


if __name__ == "__main__":
    sigmas = [30, 50, 70, 90, 120, 160, 200, 250, 300, 350]
    seeds = [42, 1, 7, 99, 123]
    true_v = 80.0

    tmpdir = Path(tempfile.mkdtemp(prefix="kinextract_vs_ppxf_"))
    results = []
    t0 = time.time()
    print(f"{'sigma':>6s} {'seed':>5s} {'kin_V':>8s} {'kin_s':>7s} {'ppxf_V':>8s} {'ppxf_s':>7s}")
    for sigma in sigmas:
        for seed in seeds:
            gal, errs = build_emiles_mock(true_v, sigma, seed)
            spec_path = tmpdir / f"mock_{sigma}_{seed}.spec"
            write_spec_file(gal, errs, spec_path)

            kin = fit_kinextract(spec_path, sigma)
            ppx = fit_ppxf(WAVELENGTH, gal, errs, 8000.0, 9000.0, sigma)

            row = dict(sigma=sigma, seed=seed, true_v=true_v,
                       kin_v=kin["v"], kin_sigma=kin["sigma"], kin_h3=kin["h3"], kin_h4=kin["h4"],
                       ppxf_v=ppx["v"], ppxf_sigma=ppx["sigma"], ppxf_h3=ppx["h3"], ppxf_h4=ppx["h4"])
            results.append(row)
            print(f"{sigma:6.0f} {seed:5d} {kin['v']:8.2f} {kin['sigma']:7.2f} "
                  f"{ppx['v']:8.2f} {ppx['sigma']:7.2f}")

    print(f"\nTotal: {time.time()-t0:.0f}s")

    print(f"\n{'sigma':>6s} {'kinV_bias':>10s} {'kinV_std':>9s} {'kinS_bias':>10s} {'kinS_std':>9s} "
          f"{'ppxfV_bias':>11s} {'ppxfV_std':>10s} {'ppxfS_bias':>11s} {'ppxfS_std':>10s}")
    for sigma in sigmas:
        rows = [r for r in results if r["sigma"] == sigma]
        kv = np.array([r["kin_v"] for r in rows]) - true_v
        ks = np.array([r["kin_sigma"] for r in rows]) - sigma
        pv = np.array([r["ppxf_v"] for r in rows]) - true_v
        ps = np.array([r["ppxf_sigma"] for r in rows]) - sigma
        print(f"{sigma:6.0f} {kv.mean():+10.2f} {kv.std():9.2f} {ks.mean():+10.2f} {ks.std():9.2f} "
              f"{pv.mean():+11.2f} {pv.std():10.2f} {ps.mean():+11.2f} {ps.std():10.2f}")

    with open("dev_notes/stress_test/results_kinextract_vs_ppxf.json", "w") as f:
        json.dump(results, f, indent=2)
