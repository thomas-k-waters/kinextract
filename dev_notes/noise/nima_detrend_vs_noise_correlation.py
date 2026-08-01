"""Tests the sharpest, most novel, checkable prediction from Nima's
statistical framing (see conversation record): that some of the
"instrumental" pixel correlation found in `noise_correlation_audit.py`
might actually be *leftover template-incompleteness trend* masquerading
as noise, not (only) real drizzle-resampling correlation -- because that
audit never let the model detrend for what the template library can't
capture (no continuum term).

Nima's recipe, in his own order: (1) signal is built from an admittedly
incomplete reference basis, (2) that incompleteness leaves a *trend*
which must be detrended with a generic basis (his "M") before (3) the
residual can be trusted as actual, testable noise. Applied here: fit a
real MUSE star spectrum (real per-pixel noise, real measured data -- not
a synthetic re-injection) using *other* real MUSE stars as templates
(genuine incompleteness/mismatch, since no other star is an exact
duplicate), with the shared multiplicative continuum block
(`continuum.fit_losvd_with_continuum`) either off or on, and compare the
fit residual's autocorrelation in both cases using the exact same
diagnostic as `noise_correlation_audit.py` (local-detrend + first-
difference lag-1 ACF vs. the white-noise null).

If Nima's ordering is right: residual correlation should be measurably
*smaller* with the continuum on, since it's absorbing the systematic
mismatch trend the no-continuum fit has nowhere else to put except into
what looks like "structured noise." If the correlation is essentially
unchanged either way, that's evidence the audit's original finding is
really instrumental (drizzle-resampling), not a detrending gap -- an
equally important, and equally honest, thing to know.

Usage: python nima_detrend_vs_noise_correlation.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import (
    FitConfig, build_bspline_basis, difference_matrix, build_template_stack, fit_losvd,
)
from kinextract.continuum import fit_losvd_with_continuum

MUSE_STAR_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/kinextract/examples/data/muse"
MUSE_TEMPLATE_LIST_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
MAX_LAG = 10

# the "galaxy": a real star's own raw (not continuum-normalized) spectrum + its own error column
GALAXY_STAR_FILE = "HD057060_av.dat"


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
    return templates, wave_ref, names


def _acf(x, max_lag):
    x = x - np.mean(x)
    n = len(x)
    var = np.sum(x ** 2)
    return np.array([np.sum(x[:n - lag] * x[lag:]) / var if lag > 0 else 1.0 for lag in range(max_lag + 1)])


def run():
    gwave, gflux, gerr = _load_dat(os.path.join(MUSE_STAR_DIR, GALAXY_STAR_FILE))
    templates, twave, tnames = _load_other_templates(GALAXY_STAR_FILE)
    print(f"'galaxy' = {GALAXY_STAR_FILE} (excluded from its own {len(templates)}-template fitting library)")

    mask = (gwave >= WAVEFITMIN) & (gwave <= WAVEFITMAX)
    wave_fit = gwave[mask]
    flux_fit = gflux[mask]
    err_fit = gerr[mask]

    # modest fixed velocity grid -- a real star has no meaningful broadening of its own;
    # this just gives the LOSVD spline room to find a narrow peak near v=0
    cfg = FitConfig(vmin=-300.0, vmax=300.0)
    v_grid = cfg.build_velocity_grid()
    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D2 = difference_matrix(n_coef, order=2, dv=dv, dv_ref=cfg.knot_spacing_kms)

    A_stack = build_template_stack(wave_fit, twave, templates, v_grid)
    A_stack_w = A_stack / err_fit[None, :, None]
    y_w = flux_fit / err_fit

    print("fitting WITHOUT continuum...")
    result_no_c = fit_losvd(A_stack_w, y_w, cfg)
    w_no_c = result_no_c["w"]
    model_no_c = sum(w_no_c[k] * (A_stack[k] @ result_no_c["b"]) for k in range(len(templates)))
    resid_no_c = flux_fit - model_no_c

    print("fitting WITH continuum...")
    result_c = fit_losvd_with_continuum(A_stack_w, y_w, wave_fit, basis, D2, n_continuum_knots=4, lam=cfg.lam)
    b_c, w_c, C_c = result_c["b"], result_c["w"], result_c["C"]
    model_c = C_c * sum(w_c[k] * (A_stack[k] @ b_c) for k in range(len(templates)))
    resid_c = flux_fit - model_c

    for label, resid in (("WITHOUT continuum", resid_no_c), ("WITH continuum", resid_c)):
        n = len(resid)
        ci95 = 1.96 / np.sqrt(n)
        acf = _acf(resid, MAX_LAG)
        acf_diff = _acf(np.diff(resid), MAX_LAG)
        rms = np.std(resid)
        print(f"\n[{label}]  resid RMS={rms:.4e}  95% white-noise bound={ci95:.4f}")
        print(f"  lag1 ACF (raw resid)  = {acf[1]:+.4f}   lag2={acf[2]:+.4f}   lag3={acf[3]:+.4f}")
        print(f"  lag1 ACF (1st diff)   = {acf_diff[1]:+.4f}   lag2={acf_diff[2]:+.4f}   lag3={acf_diff[3]:+.4f}")


if __name__ == "__main__":
    run()
