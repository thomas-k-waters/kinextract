"""Reusable stress-test harness for LOSVD recovery across sigma regimes.

Not a permanent part of the package -- lives in dev_notes/ as a working tool
for the autonomous stress-test session (see broad_sigma_stress_test_log.md).
Builds synthetic MUSE-like galaxy spectra from real stellar templates (known
mixture, known LOSVD, known continuum), fits them back with kinextract, and
reports recovery quality across multiple noise seeds so genuine multi-modal
instability (which a single seed can't reveal) shows up directly.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndimage_shift
from scipy.integrate import trapezoid

from kinextract import FitConfig, run_spectral_fit, set_verbose
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite
from kinextract.errors import _losvd_moments

DATA_DIR = Path("/Users/waterstk/Documents/Gultekin_Astrophysics/kinextract/examples/data/muse")

WAVEMIN, STEP, N_PIX = 4750.0, 1.25, 3681
WAVELENGTH = WAVEMIN + np.arange(N_PIX) * STEP
WAVEFITMIN, WAVEFITMAX = 8415.0, 8750.0
CEE, LAM_CENTER = 299792.458, 8580.0

# G/K giants + supergiants only, from the MUSE library's own spectral-type
# table (excludes G3V/K3V dwarfs and every non-G/K type) -- matches the
# user's own real template-selection practice (e.g. NGC 3258).
GK_GIANTS = [
    "HD200081_av.dat",  # G0
    "HD179821_av.dat",  # G5Ia
    "HD193896_av.dat",  # G5IIIa
    "HD204155_av.dat",  # G5
    "HD099648_av.dat",  # G8Iab
    "HD170820_av.dat",  # K0III
    "HD173158_av.dat",  # K0
    "HD232078_av.dat",  # K3IIp
    "HD099998_av.dat",  # K3.5III
    "HD114960_av.dat",  # K5III
]

# Default "true" generating population: 3 of the G/K giants, dominant + 2 minor.
DEFAULT_GEN_STARS = ["HD193896_av.dat", "HD170820_av.dat", "HD114960_av.dat"]
DEFAULT_TRUE_WEIGHTS = np.array([0.55, 0.30, 0.15])


def load_normalized_template(name: str) -> np.ndarray:
    d = np.loadtxt(DATA_DIR / name)
    flux = np.interp(WAVELENGTH, d[:, 0], d[:, 1])
    return flux / gaussian_filter(flux, sigma=200)


def build_mock_spectrum(
    true_v: float, true_sigma: float, seed: int,
    gen_stars: list[str] = DEFAULT_GEN_STARS,
    true_weights: np.ndarray = DEFAULT_TRUE_WEIGHTS,
    noise: float = 250.0, cont_level: float = 12_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (gal_flux, gal_err) on the standard MUSE grid."""
    templates = np.array([load_normalized_template(n) for n in gen_stars])
    pop_template = np.tensordot(true_weights, templates, axes=1)

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


def write_spec_file(gal: np.ndarray, errs: np.ndarray, out_path: Path) -> None:
    np.savetxt(out_path, np.column_stack([np.arange(1, N_PIX + 1), gal, errs]),
               fmt="%6d  %14.4f  %14.4f")


def fit_one(
    true_v: float, true_sigma: float, seed: int,
    template_list_file: str, template_dir: str,
    vrange: float, sigl: float,
    gen_stars: list[str] = DEFAULT_GEN_STARS,
    true_weights: np.ndarray = DEFAULT_TRUE_WEIGHTS,
    noise: float = 250.0,
    extra_cfg_kwargs: dict | None = None,
    tmpdir: Path | None = None,
) -> dict:
    """Build one mock + fit it once. Returns a dict of recovery diagnostics."""
    gal, errs = build_mock_spectrum(true_v, true_sigma, seed, gen_stars, true_weights, noise)
    tmpdir = tmpdir or Path(tempfile.mkdtemp(prefix="kinextract_stress_"))
    spec_path = tmpdir / f"mock_s{true_sigma}_{seed}.spec"
    write_spec_file(gal, errs, spec_path)

    kwargs = dict(
        template_list_file=template_list_file, template_dir=template_dir,
        wavemin_full=WAVEMIN, step=STEP, wavefitmin=WAVEFITMIN, wavefitmax=WAVEFITMAX,
        zgal=0.0, fit_continuum=True, use_spectrum_errors=True, xlam_auto=True,
        losvd_vmin=-vrange, losvd_vmax=vrange, sigl=sigl, clean=False,
        map_maxiter=10000, print_every=999999,
    )
    if extra_cfg_kwargs:
        kwargs.update(extra_cfg_kwargs)
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
        v_center=float(getattr(st, "v_center", np.nan)),
        success=bool(fit["result"].success),
    )


def sweep(
    sigmas: list[float], seeds: list[int],
    template_list_file: str, template_dir: str,
    vrange_fn=None, sigl_fn=None,
    true_v: float = 80.0,
    gen_stars: list[str] = DEFAULT_GEN_STARS,
    true_weights: np.ndarray = DEFAULT_TRUE_WEIGHTS,
    noise: float = 250.0,
    extra_cfg_kwargs: dict | None = None,
    label: str = "",
) -> list[dict]:
    """Multi-sigma x multi-seed sweep. vrange_fn/sigl_fn: sigma -> value
    (default: vrange = max(300, 3.5*sigma), sigl = sigma)."""
    if vrange_fn is None:
        vrange_fn = lambda s: max(300.0, 3.5 * s)
    if sigl_fn is None:
        sigl_fn = lambda s: s

    tmpdir = Path(tempfile.mkdtemp(prefix="kinextract_sweep_"))
    results = []
    print(f"\n{'='*90}\nSWEEP: {label}\n{'='*90}")
    for sigma in sigmas:
        vrange = vrange_fn(sigma)
        sigl = sigl_fn(sigma)
        rows = []
        for seed in seeds:
            r = fit_one(true_v, sigma, seed, template_list_file, template_dir,
                        vrange, sigl, gen_stars, true_weights, noise,
                        extra_cfg_kwargs, tmpdir)
            rows.append(r)
            results.append(r)
        ok_rows = [r for r in rows if r["ok"]]
        if not ok_rows:
            print(f"  sigma={sigma:6.1f}  ALL FITS FAILED")
            continue
        v_vals = np.array([r["gh_v"] for r in ok_rows])
        s_vals = np.array([r["gh_sigma"] for r in ok_rows])
        chi2_vals = np.array([r["chi2_red"] for r in ok_rows])
        print(f"  sigma={sigma:6.1f}  V: mean={v_vals.mean():+7.2f} std={v_vals.std():5.2f} "
              f"range=[{v_vals.min():+7.2f},{v_vals.max():+7.2f}]  "
              f"sigma: mean={s_vals.mean():7.2f} std={s_vals.std():5.2f} "
              f"range=[{s_vals.min():7.2f},{s_vals.max():7.2f}]  "
              f"chi2_red: {chi2_vals.mean():.2f}")
    return results


def summarize_sweep(results: list[dict], true_v: float = 80.0) -> None:
    """Print a compact pass/fail table: does the seed-to-seed *scatter* stay
    tight enough that a properly-calibrated error bar could plausibly cover
    truth? (proxy check -- not a substitute for an actual bootstrap CI, but
    much cheaper and directly measures the multi-modality that's been the
    dominant failure mode so far)."""
    import collections
    by_sigma = collections.defaultdict(list)
    for r in results:
        if r["ok"]:
            by_sigma[r["true_sigma"]].append(r)
    print(f"\n{'sigma':>8s} {'V bias':>8s} {'V std':>7s} {'sig bias':>9s} {'sig std':>8s} {'verdict':>10s}")
    for sigma, rows in sorted(by_sigma.items()):
        v = np.array([r["gh_v"] for r in rows])
        s = np.array([r["gh_sigma"] for r in rows])
        v_bias = v.mean() - true_v
        s_bias = s.mean() - sigma
        v_std = v.std()
        s_std = s.std()
        # Rough verdict: bias within ~1.5x the seed-to-seed std, and std itself
        # not absurd relative to sigma (a real, usable error bar).
        ok = (abs(v_bias) < max(3 * v_std, 5) and v_std < 0.3 * sigma + 10
              and abs(s_bias) < 0.25 * sigma)
        print(f"{sigma:8.1f} {v_bias:+8.2f} {v_std:7.2f} {s_bias:+9.2f} {s_std:8.2f} "
              f"{'OK' if ok else 'FAIL':>10s}")
