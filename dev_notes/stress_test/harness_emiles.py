"""E-MILES variant of the stress-test harness: generate mocks from a known
subset of E-MILES SSPs, fit back with the full E-MILES subgrid (same
generate-subset-of-fit-library paradigm as the G/K MUSE-star test)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndimage_shift

sys.path.insert(0, str(Path(__file__).parent))
from harness import (  # noqa: E402
    WAVELENGTH, WAVEMIN, STEP, N_PIX, WAVEFITMIN, WAVEFITMAX, CEE, LAM_CENTER,
    write_spec_file,
)
from prepare_emiles import EMILES_NPZ  # noqa: E402

from kinextract import FitConfig, run_spectral_fit, set_verbose  # noqa: E402
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite  # noqa: E402
from kinextract.errors import _losvd_moments  # noqa: E402


def load_emiles_flux(age_idx: int, metal_idx: int) -> np.ndarray:
    d = np.load(EMILES_NPZ)
    flux = d["templates"][:, age_idx, metal_idx]
    lam = d["lam"]
    on_grid = np.interp(WAVELENGTH, lam, flux)
    pos = on_grid > 0
    med = np.nanmedian(on_grid[pos]) if pos.any() else 1.0
    return on_grid / med if med > 0 else on_grid


# Generating population: 2 old, moderate-metallicity SSPs (a subset of the
# fitting subgrid built by prepare_emiles.build_emiles_template_set).
GEN_AGE_IDX = [20, 24]     # ~6.3 and ~15.8 Gyr
GEN_METAL_IDX = [3, 5]     # -0.4 and +0.22
GEN_WEIGHTS = np.array([0.6, 0.4])


def build_emiles_mock(true_v: float, true_sigma: float, seed: int,
                       noise: float = 250.0, cont_level: float = 12_000.0):
    templates = np.array([load_emiles_flux(a, m) for a, m in zip(GEN_AGE_IDX, GEN_METAL_IDX)])
    pop_template = np.tensordot(GEN_WEIGHTS, templates, axes=1)

    sigma_pix = true_sigma * LAM_CENTER / (CEE * STEP)
    shift_pix = true_v * LAM_CENTER / (CEE * STEP)
    gal_norm = ndimage_shift(gaussian_filter(pop_template, sigma_pix), +shift_pix)

    x_norm = (WAVELENGTH - WAVELENGTH.mean()) / (0.5 * (WAVELENGTH[-1] - WAVELENGTH[0]))
    slope = 1.0 + 0.80 * x_norm + 0.5 * x_norm**2 - 0.10 * x_norm**3
    hump = 0.50 * np.exp(-0.5 * ((x_norm - 0.20) / 0.45) ** 2)
    fringes = 0.01 * np.sin(2 * np.pi * (WAVELENGTH - WAVELENGTH[0]) / 50.0)
    cont = cont_level * (slope + hump + fringes)

    RNG = np.random.default_rng(seed)
    gal = gal_norm * cont + RNG.normal(0.0, noise, N_PIX)
    errs = np.full(N_PIX, noise)
    return gal, errs


def fit_one_emiles(true_v, true_sigma, seed, template_list_file, template_dir,
                    vrange, sigl, noise=250.0, tmpdir=None, data_fwhm_A=None,
                    template_fwhm_A=None):
    gal, errs = build_emiles_mock(true_v, true_sigma, seed, noise)
    tmpdir = tmpdir or Path(tempfile.mkdtemp(prefix="kinextract_emiles_stress_"))
    spec_path = tmpdir / f"mock_emiles_s{true_sigma}_{seed}.spec"
    write_spec_file(gal, errs, spec_path)

    kwargs = dict(
        template_list_file=template_list_file, template_dir=template_dir,
        wavemin_full=WAVEMIN, step=STEP, wavefitmin=WAVEFITMIN, wavefitmax=WAVEFITMAX,
        zgal=0.0, fit_continuum=True, use_spectrum_errors=True, xlam_auto=True,
        losvd_vmin=-vrange, losvd_vmax=vrange, sigl=sigl, clean=False,
        map_maxiter=10000, print_every=999999,
    )
    if data_fwhm_A is not None and template_fwhm_A is not None:
        kwargs["data_fwhm_A"] = data_fwhm_A
        kwargs["template_fwhm_A"] = template_fwhm_A
    cfg = FitConfig(**kwargs)
    try:
        fit = run_spectral_fit(cfg, gal_file=str(spec_path))
    except Exception as exc:  # noqa: BLE001
        return dict(seed=seed, true_sigma=true_sigma, ok=False, error=str(exc))
    st = fit["state"]
    out = fit["outputs"]
    b = out["b"]
    gh = fit_losvd_gauss_hermite(st.xl, b, fit_h3h4=True)
    mv, ms = _losvd_moments(st.xl, b)
    return dict(
        seed=seed, true_sigma=true_sigma, true_v=true_v, ok=True,
        gh_v=float(gh["vherm"]), gh_sigma=float(gh["sherm"]),
        raw_v=float(mv), raw_sigma=float(ms),
        chi2_red=float(out["chi2_red"]), xlam=float(st.xlam),
        success=bool(fit["result"].success),
    )


if __name__ == "__main__":
    import json
    import time

    EMILES_TLIST = "dev_notes/stress_test/emiles_templates/Tlist"
    EMILES_DIR = "dev_notes/stress_test/emiles_templates"

    sigmas = [30, 50, 70, 90, 120, 160, 200, 250, 300, 350]
    seeds = [42, 1, 7, 99, 123]
    t0 = time.time()
    results = []
    tmpdir = Path(tempfile.mkdtemp(prefix="kinextract_emiles_sweep_"))
    print(f"{'='*90}\nSWEEP: E-MILES (20 SSP templates, MUSE-native FWHM=2.51A, no LSF correction needed)\n{'='*90}")
    for sigma in sigmas:
        vrange = max(300.0, 3.5 * sigma)
        sigl = sigma
        rows = []
        for seed in seeds:
            r = fit_one_emiles(80.0, sigma, seed, EMILES_TLIST, EMILES_DIR, vrange, sigl, tmpdir=tmpdir)
            rows.append(r)
            results.append(r)
        ok_rows = [r for r in rows if r["ok"]]
        if not ok_rows:
            print(f"  sigma={sigma:6.1f}  ALL FITS FAILED: {rows[0].get('error')}")
            continue
        v_vals = np.array([r["gh_v"] for r in ok_rows])
        s_vals = np.array([r["gh_sigma"] for r in ok_rows])
        chi2_vals = np.array([r["chi2_red"] for r in ok_rows])
        print(f"  sigma={sigma:6.1f}  V: mean={v_vals.mean():+7.2f} std={v_vals.std():5.2f} "
              f"range=[{v_vals.min():+7.2f},{v_vals.max():+7.2f}]  "
              f"sigma: mean={s_vals.mean():7.2f} std={s_vals.std():5.2f} "
              f"range=[{s_vals.min():7.2f},{s_vals.max():7.2f}]  "
              f"chi2_red: {chi2_vals.mean():.2f}")
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    with open("dev_notes/stress_test/results_emiles.json", "w") as f:
        json.dump(results, f, indent=2)
