"""LOSVD-recovery benchmark for the regularization / noise-model work of 2026-10-31 (NGC 4751 MUSE regime).

Why a new benchmark: dev_notes/validation_suite generates and fits with the same template library under white noise,
which is the regime where the current method already works. The real NGC 4751 fits fail where the spectral model is
wrong (reduced chi2 before inflation 2 at S/N 50 -> 44 at S/N 250), so the mocks here
  * are built from XSL SSPs and fitted with the NGC 4751 NFM E-MILES subset (real template mismatch), with a
    mismatch-free control built from the fitting templates themselves,
  * use the extraction's own window (8300-8850 A rest), MUSE sampling and LSF, AR(1) noise (rho 0.7), a random
    multiplicative continuum, and the extraction's robust fit (inflate, CaT-protected clipping, measured rho,
    continuum knots), followed by the Laplace uncertainty, exactly as kinematics/simple_extraction.py does,
  * use non-parametric truths: Gaussian, Gauss-Hermite, core + broad wings, double-peaked, random spline,
  * are scored where the dynamical models look: the 13 model bins (300 km/s, -1950..+1950), after the spmc_in
    transform the extraction writes (transvd_bounds_laplace).

Scores per realization (all on unit-sum model-binned LOSVDs, truth vs recovered):
  shape_rms    rms of (S_rec - S_true) over the bins with light        -- accuracy, independent of the errors
  chi2_spmc    sum (S_rec - S_true)^2 / E_spmc^2 / n                   -- what model.x would see (ideal ~1)
  chi2_post    same with the binned Laplace posterior scatter          -- whether the posterior itself is honest
  cover_spmc   fraction of bins with |S_rec - S_true| <= E_spmc        -- ideal ~0.68
  dV, dsig_frac, dh3, dh4 of GH fits to the recovered vs true LOSVD
  lam_used, chi2_dof_prelim, n_clip

    python bench.py "<method>" <out.csv> [--reps N] [--jobs J] [--grid screen|full|quick]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import zlib
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                           # one thread per worker: the pool supplies the parallelism
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np

SRC = "/Users/waterstk/Documents/Gultekin_Astrophysics/kinextract/src"
sys.path.insert(0, SRC)
HERE = Path(__file__).resolve().parent
N4751_SUBSET = "/Users/waterstk/Documents/Nagar_Astrophysics/EHT_Project/notebooks/N4751_notebooks/notebook_data/muse/ngc4751_template_subset.npz"
EMILES = Path(SRC) / "kinextract/template_data/emiles_muse_grid.npz"
C_KMS = 299792.458
Z = 0.006985
MODEL_VBINS = np.linspace(-1800, 1800, 13)

SHAPES = ("gauss", "gh", "wings", "double", "spline")
SIGMAS = (200.0, 350.0, 550.0)
SNRS = (30.0, 80.0, 200.0, 400.0)
TRUTHS = ("xsl", "emiles")           # xsl: template mismatch; emiles: built from the fitting templates


# ------------------------------------------------------------------------------------------------ templates
_CACHE = {}


def fit_templates():
    if "fit" not in _CACHE:
        from kinextract import load_emiles_templates
        from kinextract.template_reduction import load_template_subset
        s = load_template_subset(N4751_SUBSET)
        u = s["union_selected"]
        sel = list(zip(s["template_ages"][u].tolist(), s["template_metals"][u].tolist()))
        t, w, names, ages, metals = load_emiles_templates(str(EMILES), select=sel)
        _CACHE["fit"] = (np.array(t), w, np.asarray(ages), np.asarray(metals))
    return _CACHE["fit"]


def xsl_templates():
    """XSL SSPs (ppxf's spectra_xsl_9.0), ages >= 3 Gyr and [M/H] -0.6..+0.2, LSF-matched to MUSE (0.25 A grid,
    shifted -2 km/s onto the E-MILES wavelength scale as measured for NGC 4751)."""
    if "xsl" not in _CACHE:
        import ppxf
        from kinextract.lsf import match_gaussian_lsf, muse_lsf_fwhm
        s = np.load(os.path.join(os.path.dirname(ppxf.__file__), "sps_models", "spectra_xsl_9.0.npz"))
        lam, fwhm = s["lam"], s["fwhm"]
        w = np.arange(7900.0, 9300.0, 0.25)
        T, meta = [], []
        for i, a in enumerate(s["ages"]):
            if a < 3:
                continue
            for j, m in enumerate(s["metals"]):
                if -0.65 <= m <= 0.25:
                    f, _ = match_gaussian_lsf(w, np.interp(w, lam, s["templates"][:, i, j]),
                                              lambda x: np.interp(x, lam, fwhm), muse_lsf_fwhm)
                    T.append(f / np.median(f)); meta.append((float(a), float(m)))
        _CACHE["xsl"] = (np.array(T), w * (1 - 2.0 / C_KMS), meta)
    return _CACHE["xsl"]


MUSE_LIB = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates"


def alt_templates():
    """[(name, T, wave)]: E-MILES 1-14 Gyr, [M/H] >= -0.96 (72), and the MUSE stellar library stars with a CaT
    cross-correlation >= 0.9 and S/N >= 40 (20), each shifted to rest by its CaT cross-correlation velocity."""
    if "alt" not in _CACHE:
        import glob
        g = np.load(EMILES)
        ga, gm = g["ages"], g["metals"]
        ages = sorted({float(ga[np.argmin(abs(ga - a))]) for a in (1, 2, 3, 4.5, 6.5, 8, 10, 12.5, 14)})
        idx = [i for i in range(len(ga)) if float(ga[i]) in ages and gm[i] >= -1.0]
        em = (g["flux"][:, idx].T, g["wavelength"])
        ref = g["flux"][:, np.argmin(abs(ga - 10) + 10 * abs(gm - 0.06))]
        w = np.arange(8420, 8720, 0.25); m = (w > 8450) & (w < 8700)
        y = np.interp(w, g["wavelength"], ref); y = y / np.polyval(np.polyfit(w, y, 2), w)
        vs = np.arange(-450, 450.1, 1.0); T = []
        tw = np.arange(4800.0, 9300.0, 0.25)
        for f in sorted(glob.glob(MUSE_LIB + "/*_av.dat")):
            d = np.loadtxt(f); x = np.interp(w, d[:, 0], d[:, 1]); x = x / np.polyval(np.polyfit(w, x, 2), w)
            cc = np.array([np.corrcoef(np.interp(w[m], w * (1 + v / C_KMS), x), y[m])[0, 1] for v in vs])
            i = int(np.argmax(cc)); sel = (d[:, 0] > 8400) & (d[:, 0] < 8800)
            if cc[i] < 0.9 or np.nanmedian(d[sel, 1] / np.maximum(d[sel, 2], 1e-30)) < 40:
                continue
            fl = np.interp(tw, d[:, 0] * (1 + vs[i] / C_KMS), d[:, 1]); T.append(fl / np.median(fl[(tw > 8400) & (tw < 8800)]))
        _CACHE["alt"] = [("emiles_old", em[0], em[1]), ("muse_stars", np.array(T), tw)]
    return _CACHE["alt"]


def wave_fit():
    obs = 4749.64 + np.arange(3681) * 1.25                   # the NGC 4751 cubes' observed grid
    r = obs / (1 + Z)
    return r[(r >= 8300.0) & (r <= 8850.0)]


# ------------------------------------------------------------------------------------------------ truths
def truth_losvd(shape, sigma, v_true, rng, v):
    g = lambda mu, s: np.exp(-0.5 * ((v - mu) / s) ** 2)  # noqa: E731
    if shape == "gauss":
        b = g(v_true, sigma)
    elif shape == "gh":
        from kinextract import gauss_hermite_model
        h3, h4 = rng.choice([-1, 1]) * rng.uniform(0.03, 0.08), rng.uniform(-0.05, 0.08)
        b = np.maximum(gauss_hermite_model(v, amp=1.0, vel=v_true, sig=sigma, h3=h3, h4=h4), 0)
    elif shape == "wings":                                   # unresolved-center / PSF wings (Mehrgan+23)
        f = rng.uniform(0.05, 0.15)
        b = (1 - f) * g(v_true, sigma) / sigma + f * g(v_true, sigma * rng.uniform(2.0, 3.0)) / (sigma * 2.5)
    elif shape == "double":                                  # two streams, e.g. a counter-rotating disc
        d = rng.uniform(1.0, 1.6) * sigma
        f = rng.uniform(0.25, 0.45)
        s2 = 0.6 * sigma
        b = (1 - f) * g(v_true - f * d, s2) + f * g(v_true + (1 - f) * d, s2)
    elif shape == "spline":
        from kinextract.losvd import build_spline_truth
        b, _ = build_spline_truth(v, vel=v_true, sigma=sigma, wiggle_scale=0.15, seed=int(rng.integers(2 ** 31)))
    else:
        raise ValueError(shape)
    return b / np.trapezoid(b, v)


def model_binned(v, b):
    """Unit-sum integrals of b over the 13 model bins (the truth's analog of spmc_binned)."""
    fine = np.arange(-1950.0, 1950.0, 1.0)
    bf = np.interp(fine, v, b, left=0, right=0)
    S = np.array([bf[(fine >= c - 150) & (fine < c + 150)].sum() for c in MODEL_VBINS])
    return S / S.sum()


def spmc_binned(vel, a, lo, hi):
    err = np.maximum(hi - a, (hi - lo) / 2)
    mb = lambda q: np.array([q[(vel >= c - 150) & (vel < c + 150)].sum() for c in MODEL_VBINS])  # noqa: E731
    return mb(a) / a.sum(), mb(err) / a.sum()


# ------------------------------------------------------------------------------------------------ methods
LAM_GRID = (1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0)


DEFAULTS = dict(lam="fixed4096", sky="none", grid="wide", add="none", noise="none", terr="0")


def parse_method(method: str) -> dict:
    """'lam=aic;sky=mask40' -> settings (unspecified keys take DEFAULTS).
    lam   fixed<N> | aic
    sky   none | mask<T>   (mask pixels where the LSF-blended MUSE sky-line flux >= T, dilated +-1 px)
    grid  wide (+-3000 km/s for every sigma, the NFM extraction's grid) | ring (+-max(1500, 5 sigma + 300))
    add   none | feat3 (three featureless positive pseudo-templates: a positive additive continuum via NNLS)
    noise none | sky (per-pixel variance err^2 + (kappa s)^2 + (f model)^2, s = the MUSE sky-line profile, kappa and f
          fitted by maximum likelihood to the residuals of a preliminary fit; the robust fit then starts from it)
    terr  0 | 1 (template error: refit at the adopted lam with E-MILES >=1 Gyr/[M/H]>=-1 and with 20 MUSE-library
          stars -- neither contains the XSL truth -- and add the spread of the model-binned LOSVDs in quadrature)"""
    out = dict(DEFAULTS)
    for part in filter(None, method.split(";")):
        k, v = part.split("=")
        out[k] = v
    return out


def make_cfg(m: dict, vmax, rho=0.0):
    from kinextract import FitConfig
    kw = dict(vmin=-vmax, vmax=vmax, rho=rho, continuum_knots=4, lam_c=2.0, ridge_c=0.1, penalty_order=3)
    if m["lam"].startswith("fixed"):
        return FitConfig(lam_selection="fixed", lam=float(m["lam"][5:]), **kw), {}
    if m["lam"] == "aic":
        return FitConfig(lam_selection="aic_local", lam_grid=LAM_GRID, max_edge_amplitude_frac=0.01, **kw), {}
    raise ValueError(m)


def sky_profile(wf):
    P = np.load(HERE / "muse_sky_profile_rest_z0.006985.npy")
    s = np.interp(wf, P[:, 0], P[:, 1])
    return s / np.percentile(P[:, 1], 99)


def noise_model_errors(A, flux, err, wf, cfg):
    """Per-pixel errors with a sky-residual and a template-mismatch term, sqrt(err^2 + (kappa s)^2 + (f m)^2): a
    fixed-lam fit without cleaning gives the model m; kappa, f maximize the Gaussian likelihood of its residuals
    (10-sigma outliers excluded). Returns (errors, kappa, f)."""
    import dataclasses
    from scipy.optimize import minimize
    from kinextract import fit_losvd_robust
    from kinextract.robust_fit import full_pixel_model
    c0 = dataclasses.replace(cfg, lam_selection="fixed", lam=4096.0, rho=0.0)
    r0 = fit_losvd_robust(A, flux, err, c0, wave_fit=wf, cleaning_method=None)
    m = full_pixel_model(r0, A, wf, c0)
    res = flux - m
    s = sky_profile(wf)
    ok = np.abs(res) < 10 * np.sqrt(err ** 2 + (0.3 * s * np.median(m)) ** 2 + (0.03 * m) ** 2)

    def nll(p):
        V = err[ok] ** 2 + (np.exp(p[0]) * s[ok]) ** 2 + (np.exp(p[1]) * m[ok]) ** 2
        return np.sum(res[ok] ** 2 / V + np.log(V))
    best = min((minimize(nll, x0, method="Nelder-Mead") for x0 in ([np.log(1e-3 * np.median(m)), np.log(1e-3)],
                                                                    [np.log(0.05 * np.median(m)), np.log(0.01)])),
               key=lambda r: r.fun)
    kappa, f = np.exp(best.x)
    return np.sqrt(err ** 2 + (kappa * s) ** 2 + (f * m) ** 2), float(kappa / np.median(m)), float(f)


def sky_mask(wf, threshold):
    P = np.load(HERE / "muse_sky_profile_rest_z0.006985.npy")
    m = np.interp(wf, P[:, 0], (P[:, 1] >= threshold).astype(float)) > 0.5
    return np.convolve(m, np.ones(3), "same") > 0


CONTAMS = ("clean", "skyadd")
_SKY = np.load(HERE / "n4751_nfm_sky_residuals_rest.npy")       # fractional residuals of 12 S/N~300 NFM bins, 8300-8849 A


def contaminate(gal, wf, contam, rng):
    """skyadd: the real NFM sky-residual pattern of a random high-S/N bin (x U(0.5, 1.5)) plus an additive smooth
    continuum of 1-4 % of the median (sky continuum / scattered light)."""
    if contam == "clean":
        return gal
    r = np.interp(wf, np.arange(8300, 8850, 1.0), _SKY[rng.integers(len(_SKY))])
    x = (wf - wf.mean()) / np.ptp(wf) * 2
    add = rng.uniform(0.01, 0.04) * (1 + 0.5 * np.polynomial.legendre.legval(x, rng.normal(0, 1, 3)))
    return gal * (1 + rng.uniform(0.5, 1.5) * r) + add * np.median(gal)


# ------------------------------------------------------------------------------------------------ one realization
def run_one(task):
    warnings.filterwarnings("ignore")
    from kinextract import (CAII_PROTECTED_WINDOWS, build_template_stack, compute_losvd_moments, edge_amplitude_frac,
                            fit_losvd_robust, laplace_losvd_uncertainty_parallel)
    from kinextract.legacy_io import transvd_bounds_laplace
    from kinextract.noise_model import whiten
    method, shape, sigma, snr, truth, contam, rep, seed = task
    M = parse_method(method)
    rng = np.random.default_rng(seed)
    wf = wave_fit()
    vmax = 3000.0 if M["grid"] == "wide" else max(1500.0, 5.0 * sigma + 300.0)
    v_true = rng.uniform(-150, 150)
    vt = np.arange(-4500.0, 4500.0, 4.0)
    bt = truth_losvd(shape, sigma, v_true, rng, vt)
    # galaxy: 3 templates, Dirichlet weights; times a smooth continuum (+-3 %)
    if truth == "xsl":
        T, tw, _ = xsl_templates()
    else:
        T, tw, ages, metals = fit_templates()
        old = np.flatnonzero((ages >= 3) & (metals >= -0.7))
        T = T[old]
    k = rng.choice(len(T), size=3, replace=False)
    wts = rng.dirichlet(np.ones(3))
    A_t = build_template_stack(wf, tw, list(T[k]), vt)
    gal = sum(wts[i] * (A_t[i] @ bt) for i in range(3))
    x = (wf - wf.mean()) / np.ptp(wf) * 2
    cont = 1 + 0.03 * np.polynomial.legendre.legval(x, rng.normal(0, 1, 4) / np.arange(1, 5))
    gal = gal * cont
    gal = gal / np.median(gal)
    gal = contaminate(gal, wf, contam, np.random.default_rng(seed + 7))
    err = np.sqrt(gal / np.median(gal)) / snr                # ~Poisson-shaped per-pixel error at the requested S/N
    rho = 0.7
    n = len(wf)
    ar = np.empty(n); e = rng.standard_normal(n); ar[0] = e[0]
    for i in range(1, n):
        ar[i] = rho * ar[i - 1] + np.sqrt(1 - rho ** 2) * e[i]
    flux = gal + err * ar
    if M["sky"].startswith("mask"):
        keep = ~sky_mask(wf, float(M["sky"][4:]))
        wf, flux, err = wf[keep], flux[keep], err[keep]
    # fit, as the extraction does
    Tf, twf, _, _ = fit_templates()
    if M["add"] == "feat3":
        xx = np.clip((twf - 8575.0) / 300.0, -1, 1)
        Tf = np.vstack([Tf, np.ones_like(twf), (1 + xx) / 2 + 0.01, (1 - xx) / 2 + 0.01])
    cfg, robust_extra = make_cfg(M, vmax, rho=0.0)
    vg = cfg.build_velocity_grid()
    A = build_template_stack(wf, twf, list(Tf), vg)
    t0 = time.time()
    kappa = fmis = np.nan
    if M["noise"] == "sky":
        err, kappa, fmis = noise_model_errors(A, flux, err, wf, cfg)
    r = fit_losvd_robust(A, flux, err, cfg, wave_fit=wf, cleaning_method="inflate", smooth_window_A=3.0,
                         inflation_cap=100.0, excluded_wavelength_windows=CAII_PROTECTED_WINDOWS, measure_rho=True,
                         min_inflation_to_apply=1.1, **robust_extra)
    import dataclasses
    cfg_f = dataclasses.replace(cfg, lam_selection="fixed", lam=float(r["lam_used"]), rho=float(r["rho_used"]))
    A_w, y_w, _ = whiten(A, flux, r["errors_inflated"], rho=cfg_f.rho)
    lap = laplace_losvd_uncertainty_parallel([A_w], [y_w], [cfg_f], n_samples=300, seed=seed, n_jobs=1, wave_fit=[wf],
                                             n_continuum_knots=4, lam_c=2.0, ridge_c=0.1,
                                             clipped_mask=[r["clipped_mask"]])[0]
    E_tmpl = None
    if M["terr"] == "1":
        Sb = [model_binned(vg, r["b"])]
        for _, Ta, twa in alt_templates():
            Aa = build_template_stack(wf, twa, list(Ta), vg)
            ra = fit_losvd_robust(Aa, flux, err, cfg_f, wave_fit=wf, cleaning_method="inflate", smooth_window_A=3.0,
                                  inflation_cap=100.0, excluded_wavelength_windows=CAII_PROTECTED_WINDOWS,
                                  measure_rho=False, min_inflation_to_apply=1.1)
            if edge_amplitude_frac(ra["b"]) <= 0.01:
                Sb.append(model_binned(vg, ra["b"]))
        E_tmpl = np.std(Sb, axis=0, ddof=1) if len(Sb) > 1 else np.zeros(13)
    t_fit = time.time() - t0
    # score in model space
    S_true = model_binned(vt, bt)
    vel, a, lo, hi = transvd_bounds_laplace(vg, lap["b_median"], lap["b_p16"], lap["b_p84"], invert=False,
                                            vmax_out=1950.0)
    S_rec, E_spmc = spmc_binned(vel, a, lo, hi)
    samp = np.array([model_binned(vg, s) for s in lap["b_samples"][:200]])
    E_post = samp.std(axis=0)
    ok = (S_true > 1e-3) | (S_rec > 1e-3)
    d = (S_rec - S_true)[ok]
    E_tot = np.sqrt(E_post ** 2 + (E_tmpl ** 2 if E_tmpl is not None else 0))
    mt = compute_losvd_moments(vt, bt, fit_h3h4=True)
    mr = compute_losvd_moments(vg, lap["b_median"], fit_h3h4=True)
    return dict(method=method, shape=shape, sigma=sigma, snr=snr, truth=truth, contam=contam, rep=rep, seed=seed,
                shape_rms=float(np.sqrt(np.mean(d ** 2))), chi2_spmc=float(np.mean(d ** 2 / E_spmc[ok] ** 2)),
                chi2_post=float(np.mean(d ** 2 / np.maximum(E_post[ok], 1e-12) ** 2)),
                cover_spmc=float(np.mean(np.abs(d) <= E_spmc[ok])), cover_post=float(np.mean(np.abs(d) <= E_post[ok])),
                espmc_over_epost=float(np.median(E_spmc[ok] / np.maximum(E_post[ok], 1e-12))),
                dV=float(mr["vherm"] - mt["vherm"]), dsig_frac=float((mr["sherm"] - mt["sherm"]) / mt["sherm"]),
                dh3=float(mr["h3"] - mt["h3"]), dh4=float(mr["h4"] - mt["h4"]), h4_true=float(mt["h4"]),
                lam_used=float(r["lam_used"]), rho_used=float(r["rho_used"]), chi2_dof_prelim=float(r.get("chi2_dof_prelim", np.nan)),
                n_clip=int(r["n_clipped"]), kappa=kappa, f_mismatch=fmis,
                chi2_tot=float(np.mean(d ** 2 / np.maximum(E_tot[ok], 1e-12) ** 2)),
                cover_tot=float(np.mean(np.abs(d) <= E_tot[ok])), edge=float(edge_amplitude_frac(r["b"])),
                t_fit=t_fit)


GRIDS = {  # name: (shapes, sigmas, snrs, truths, contams)
    "full": (SHAPES, SIGMAS, SNRS, TRUTHS, CONTAMS),
    "screen": (SHAPES, (250.0, 550.0), (40.0, 120.0, 300.0), ("xsl",), CONTAMS),
    "quick": (("gauss", "wings", "double"), (350.0,), (40.0, 300.0), ("xsl",), CONTAMS),
}


def tasks(method, reps, grid="screen"):
    shapes, sigmas, snrs, truths, contams = GRIDS[grid]
    out = []
    for shape, sigma, snr, truth, contam in itertools.product(shapes, sigmas, snrs, truths, contams):
        for rep in range(reps):
            seed = zlib.crc32(f"{shape}|{sigma}|{snr}|{truth}|{contam}|{rep}".encode())  # same mock for every method
            out.append((method, shape, sigma, snr, truth, contam, rep, seed))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method"); ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=12); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--grid", default="screen", choices=list(GRIDS))
    a = ap.parse_args()
    T = tasks(a.method, a.reps, a.grid)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {(r["shape"], float(r["sigma"]), float(r["snr"]), r["truth"], r["contam"], int(r["rep"]))
                    for r in csv.DictReader(f)}
    T = [t for t in T if (t[1], t[2], t[3], t[4], t[5], t[6]) not in done]
    print(f"{a.method}: {len(T)} realizations to run ({len(done)} already in {a.out})", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex, open(a.out, "a", newline="") as f:
        w = None
        futs = {ex.submit(run_one, t): t for t in T}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                row = fu.result()
            except Exception as e:  # keep going; record the failure (it counts against the method)
                print("FAILED:", futs[fu][1:7], repr(e)[:300], flush=True)
                with open(a.out.replace(".csv", ".failed.txt"), "a") as ff:
                    ff.write("|".join(map(str, futs[fu][1:7])) + "\t" + repr(e)[:300] + "\n")
                continue
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row)); (f.tell() == 0) and w.writeheader()
            w.writerow(row); f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(T)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
