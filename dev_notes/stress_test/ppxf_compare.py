"""kinextract vs pPXF head-to-head comparison on identical synthetic spectra."""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from harness_emiles import (  # noqa: E402
    build_emiles_mock, write_spec_file, WAVELENGTH, WAVEMIN, STEP, N_PIX,
    CEE, LAM_CENTER, GEN_AGE_IDX, GEN_METAL_IDX, GEN_WEIGHTS,
)
from prepare_emiles import EMILES_NPZ  # noqa: E402

from kinextract import FitConfig, run_spectral_fit, set_verbose  # noqa: E402
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite  # noqa: E402

from ppxf.ppxf import ppxf  # noqa: E402
from ppxf.ppxf_util import log_rebin  # noqa: E402

WAVEFITMIN, WAVEFITMAX = 8000.0, 9000.0  # "medium" window, validated to help broad sigma
EMILES_TLIST = "dev_notes/stress_test/emiles_templates/Tlist"
EMILES_DIR = "dev_notes/stress_test/emiles_templates"


def fit_kinextract(spec_path, sigma_guess, vrange=None):
    vrange = vrange or max(300.0, 3.5 * sigma_guess)
    cfg = FitConfig(
        template_list_file=EMILES_TLIST, template_dir=EMILES_DIR,
        wavemin_full=WAVEMIN, step=STEP, wavefitmin=WAVEFITMIN, wavefitmax=WAVEFITMAX,
        zgal=0.0, fit_continuum=True, use_spectrum_errors=True, xlam_auto=True,
        losvd_vmin=-vrange, losvd_vmax=vrange, sigl=sigma_guess, clean=False,
        map_maxiter=10000, print_every=999999,
    )
    fit = run_spectral_fit(cfg, gal_file=str(spec_path))
    gh = fit_losvd_gauss_hermite(fit["state"].xl, fit["outputs"]["b"], fit_h3h4=True)
    return dict(v=gh["vherm"], sigma=gh["sherm"], h3=gh["h3"], h4=gh["h4"],
                chi2_red=fit["outputs"]["chi2_red"])


def fit_ppxf(gal, errs, sigma_guess):
    """Standard pPXF workflow: log-rebin galaxy + full E-MILES subgrid at a
    matched velscale, fit with moments=4 (V, sigma, h3, h4) and an additive
    polynomial continuum (pPXF's own standard convention -- no equivalent to
    kinextract's P-spline co-fit, but the closest standard pPXF practice)."""
    d = np.load(EMILES_NPZ)
    lam_temp_full = d["lam"]
    templates_full = d["templates"]

    mask = (WAVELENGTH >= WAVEFITMIN) & (WAVELENGTH <= WAVEFITMAX)
    lam_gal = WAVELENGTH[mask]
    gal_masked = gal[mask]
    err_masked = errs[mask]

    gal_log, ln_lam_gal, velscale = log_rebin(
        [lam_gal[0], lam_gal[-1]], gal_masked)
    err_log, _, _ = log_rebin([lam_gal[0], lam_gal[-1]], err_masked)
    err_log = np.maximum(err_log, 1e-6)

    tmask = (lam_temp_full >= WAVEFITMIN - 50) & (lam_temp_full <= WAVEFITMAX + 50)
    lam_temp = lam_temp_full[tmask]

    tlist_names = [ln.strip() for ln in open(EMILES_TLIST) if ln.strip()]
    # Reconstruct (age_idx, metal_idx) from the ages/metals arrays used to build
    # the fitting subgrid (same indices as prepare_emiles.build_emiles_template_set's
    # default: old ages [16,18,20,22,24], moderate metals [2,3,4,5]).
    age_idx_list = [16, 18, 20, 22, 24]
    metal_idx_list = [2, 3, 4, 5]

    templates_log = []
    for ai in age_idx_list:
        for mi in metal_idx_list:
            flux = templates_full[tmask, ai, mi]
            t_log, ln_lam_temp, _ = log_rebin([lam_temp[0], lam_temp[-1]], flux, velscale=velscale)
            templates_log.append(t_log)
    templates_log = np.column_stack(templates_log)
    templates_log /= np.median(templates_log)

    # pPXF's velocity zero-point offset between template and galaxy log-lambda grids.
    dv = (ln_lam_temp[0] - ln_lam_gal[0]) * CEE

    start = [0.0, sigma_guess]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = ppxf(templates_log, gal_log, err_log, velscale, start,
                  moments=4, degree=4, vsyst=dv, quiet=True)
    return dict(v=pp.sol[0], sigma=pp.sol[1], h3=pp.sol[2], h4=pp.sol[3],
                chi2=pp.chi2)


if __name__ == "__main__":
    import json
    import time

    sigmas = [40, 70, 100, 140, 250, 350]
    seeds = [42, 1, 7]
    true_v = 80.0

    tmpdir = Path(tempfile.mkdtemp(prefix="kinextract_ppxf_"))
    results = []
    t0 = time.time()
    print(f"{'sigma':>6s} {'seed':>5s} {'kin_V':>8s} {'kin_sig':>8s} {'ppxf_V':>8s} {'ppxf_sig':>8s}")
    for sigma in sigmas:
        for seed in seeds:
            gal, errs = build_emiles_mock(true_v, sigma, seed)
            spec_path = tmpdir / f"mock_{sigma}_{seed}.spec"
            write_spec_file(gal, errs, spec_path)

            kin = fit_kinextract(spec_path, sigma)
            ppx = fit_ppxf(gal, errs, sigma)
            row = dict(sigma=sigma, seed=seed, true_v=true_v,
                       kin_v=kin["v"], kin_sigma=kin["sigma"],
                       ppxf_v=ppx["v"], ppxf_sigma=ppx["sigma"])
            results.append(row)
            print(f"{sigma:6.0f} {seed:5d} {kin['v']:8.2f} {kin['sigma']:8.2f} "
                  f"{ppx['v']:8.2f} {ppx['sigma']:8.2f}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    with open("dev_notes/stress_test/results_ppxf_compare.json", "w") as f:
        json.dump(results, f, indent=2)
