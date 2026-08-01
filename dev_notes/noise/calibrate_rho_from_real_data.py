"""A1: calibrates the AR(1) correlation parameter (`noise_model.py`'s
`rho`) against real, continuum-detrended residuals -- not the raw
(non-continuum) residual, which `nima_detrend_vs_noise_correlation.py`
already showed is dominated by a much larger, unrelated continuum-
mismatch trend (raw lag-1 ACF ~0.99 vs ~0.46 once continuum is fit).

Repeats that same real-star-vs-other-real-stars mismatch fit (with
continuum on) across several stars, not just the one used in the initial
finding, and averages `estimate_correlation_length`'s own estimate --
a single-star number is not a defensible package-wide default on its own.

Usage: python calibrate_rho_from_real_data.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import FitConfig, build_bspline_basis, difference_matrix, build_template_stack, fit_losvd
from kinextract.continuum import fit_losvd_with_continuum
from kinextract.noise_model import estimate_correlation_length

MUSE_STAR_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/kinextract/examples/data/muse"
MUSE_TEMPLATE_LIST_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0

GALAXY_STAR_FILES = [
    "HD057060_av.dat", "HD064332_av.dat", "HD067507_av.dat", "HD085405_av.dat",
    "HD096446_av.dat", "HD099648_av.dat",
]


def _load_dat(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1], d[:, 2]


def _load_other_templates(exclude_name, template_dir=MUSE_TEMPLATE_LIST_DIR):
    with open(os.path.join(template_dir, "Tlist")) as f:
        names = [line.strip() for line in f if line.strip()]
    names = [n for n in names if exclude_name not in n]
    templates, wave_ref = [], None
    for name in names:
        d = np.loadtxt(os.path.join(template_dir, name))
        if wave_ref is None:
            wave_ref = d[:, 0]
        templates.append(np.interp(wave_ref, d[:, 0], d[:, 1]))
    return templates, wave_ref


def rho_for_one_star(galaxy_file):
    gwave, gflux, gerr = _load_dat(os.path.join(MUSE_STAR_DIR, galaxy_file))
    templates, twave = _load_other_templates(galaxy_file)

    mask = (gwave >= WAVEFITMIN) & (gwave <= WAVEFITMAX)
    wave_fit = gwave[mask]
    flux_fit = gflux[mask]
    err_fit = gerr[mask]

    cfg = FitConfig(vmin=-300.0, vmax=300.0, continuum_knots=4)
    v_grid = cfg.build_velocity_grid()
    A_stack = build_template_stack(wave_fit, twave, templates, v_grid)
    A_stack_w = A_stack / err_fit[None, :, None]
    y_w = flux_fit / err_fit

    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D2 = difference_matrix(n_coef, order=2, dv=dv, dv_ref=cfg.knot_spacing_kms)

    result = fit_losvd_with_continuum(A_stack_w, y_w, wave_fit, basis, D2, n_continuum_knots=4, lam=cfg.lam)
    b, w, C = result["b"], result["w"], result["C"]
    model = C * sum(w[k] * (A_stack[k] @ b) for k in range(len(templates)))
    resid = flux_fit - model
    return estimate_correlation_length(resid), result["success"]


def run():
    rhos = []
    for star in GALAXY_STAR_FILES:
        rho, success = rho_for_one_star(star)
        rhos.append(rho)
        print(f"{star:>20}  success={success}  rho={rho:.4f}", flush=True)

    rhos = np.array(rhos)
    print(f"\nmean rho={rhos.mean():.4f}  median={np.median(rhos):.4f}  std={rhos.std(ddof=1):.4f}")
    print(f"range=[{rhos.min():.4f}, {rhos.max():.4f}]")


if __name__ == "__main__":
    run()
