"""Priority-1 follow-up: does OG kinextract's LINEAR-SPACE D2 penalty cure the
real-data sigma inflation that no log-space fix could touch?

Motivation (2026-07-28). Reading OG kinextract's `_compute_smoothness`
(sco_framework_updated/bin/trash/kinextract.py:2930) shows it is a different
method from ours in the one place that matters:

    smooth += lam * (b[j+1] - 2*b[j] + b[j-1])**2      # D2 on b ITSELF
    if abs(xl[j]) > sfac*sigl0:                        # quartic wing taper
        lam = xlam * (abs(xl[j]/sigl0)/sfac)**4

We penalize `log b` through spline coefficients; OG penalizes `b` directly.
Consequences, all of which point at the pedestal:

  * Null space. Ours (D3 on log b) is quadratic-in-log-b, which CONTAINS
    exp(+a v^2) -- the divergent edge pedestal. OG's (D2 on b) is linear in
    b, a bounded ramp that cannot diverge.
  * Edge rows. OG's j=0 and j=nl-1 terms are (b[1]-2b[0]) and
    (b[nl-2]-2b[nl-1]), i.e. an implicit b=0 boundary condition that
    directly penalizes nonzero edge amplitude. Our clamped spline has none.
  * Taper. The quartic wing taper is meaningful in linear space. Ported into
    log space it was measured INERT at the selected lam (see FINDINGS.md).

Because the model is linear in b given template weights, and the penalty is
quadratic in b, the b-step is an exactly-solvable nonnegative least squares
problem on the augmented system [A_eff; sqrt(lam*taper) D] vs [y; 0]. No
optimizer, no local minima, no float32 stalls -- which also removes a whole
class of failure we have been fighting.

Scored on BOTH the real bins (must approach legacy 54.3 / 40-45) and the NB02
mock (must NOT be destroyed -- non-portability is what killed the wing-mass
and wing_lam fixes).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import nnls

from kinextract import (FitConfig, build_template_stack, compute_losvd_moments,
                             load_emiles_templates, gauss_hermite_model)
from kinextract.losvd import count_losvd_peaks
from kinextract.template_reduction import load_template_subset

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
EM = os.path.join(ROOT, "src/kinextract/template_data/emiles_muse_grid.npz")
SFAC = 1.8


def d2_matrix(nl):
    """OG's second-difference operator, including its implicit b=0 edge rows."""
    D = np.zeros((nl, nl))
    for j in range(nl):
        if j == 0:
            D[j, 0] = -2.0; D[j, 1] = 1.0
        elif j == nl - 1:
            D[j, nl - 1] = -2.0; D[j, nl - 2] = 1.0
        else:
            D[j, j - 1] = 1.0; D[j, j] = -2.0; D[j, j + 1] = 1.0
    return D


def taper_weights(v, sigl0, sfac=SFAC, power=4.0):
    """OG's quartic wing up-weighting, in linear space where it bites."""
    d = np.abs(v)
    sigl0 = max(float(sigl0), 1e-6)
    return np.where(d > sfac * sigl0, (d / sigl0 / sfac) ** power, 1.0)


def fit_linear_d2(A, y, err, v, lam, sigl0, n_iter=25, tol=1e-8):
    """Alternating NNLS: template weights <-> nonnegative penalized LOSVD.

    A : (K, npix, nv) template design stack
    Returns b normalized to unit sum, plus the template weights.
    """
    K, npix, nv = A.shape
    Aw = A / err[None, :, None]
    yw = y / err
    D = d2_matrix(nv)
    W = np.sqrt(lam * taper_weights(v, sigl0))[:, None] * D   # sqrt of the penalty

    # start from a Gaussian at the current width guess
    b = np.exp(-0.5 * (v / max(sigl0, 1.0)) ** 2); b /= b.sum()
    w = np.full(K, 1.0 / K)
    prev = None
    for _ in range(n_iter):
        # --- template weights given b ---
        design = np.stack([Aw[k] @ b for k in range(K)], axis=1)
        w, _ = nnls(design, yw)
        if not np.any(w > 0):
            w = np.full(K, 1.0 / K)
        # --- LOSVD given weights: augmented NNLS, exact solve ---
        A_eff = np.tensordot(w, Aw, axes=(0, 0))            # (npix, nv)
        Aug = np.vstack([A_eff, W])
        rhs = np.concatenate([yw, np.zeros(nv)])
        b, _ = nnls(Aug, rhs)
        s = b.sum()
        if s <= 0:
            break
        b = b / s
        chi2 = float(np.sum((A_eff @ b * s - yw) ** 2))
        if prev is not None and abs(prev - chi2) < tol * max(1.0, abs(prev)):
            break
        prev = chi2
    return b, w


def fit_with_sigl0_loop(A, y, err, v, lam, sig_init=80.0, n_outer=4):
    """Taper onset depends on sigma, which depends on the fit -> iterate."""
    sigl0 = float(sig_init)
    b = None
    for _ in range(n_outer):
        b, _ = fit_linear_d2(A, y, err, v, lam, sigl0)
        m = compute_losvd_moments(v, b, fit_h3h4=True)
        new = float(m["sherm"])
        if not np.isfinite(new) or new <= 0:
            break
        if abs(new - sigl0) < 0.5:
            sigl0 = new
            break
        sigl0 = 0.5 * sigl0 + 0.5 * new
    return b, sigl0


def roughness(b):
    """OG's jaggedness measure, normalized to the peak."""
    bp = np.maximum(b, 0.0)
    pk = bp.max()
    if pk <= 0:
        return np.inf
    return float(np.max(np.abs(np.diff(np.diff(bp))))) / pk


def og_select(A, y, err, v, grid, sig_init=80.0,
              max_peaks=1, smooth_thresh=0.25):
    """OG's xlam_auto rule: the SMALLEST lam giving a unimodal, non-jagged
    LOSVD. Note this is a feasibility rule, not an optimality rule -- it
    never trades chi2 against dof, which is exactly why AIC's failure mode
    on systematics-dominated residuals cannot occur."""
    rows = []
    chosen = None
    for lam in grid:
        b, sigl0 = fit_with_sigl0_loop(A, y, err, v, lam, sig_init=sig_init)
        if b is None:
            continue
        m = compute_losvd_moments(v, b, fit_h3h4=True)
        npk = count_losvd_peaks(b, min_prominence=0.1)
        rg = roughness(b)
        ok = (npk <= max_peaks) and (rg < smooth_thresh)
        rows.append((lam, m, npk, rg, ok, b))
        if ok and chosen is None:
            chosen = rows[-1]
    return rows, chosen


# ----------------------------------------------------------------- real data
def run_real():
    d = np.load(os.path.join(ROOT, "examples/data/muse/ngc5102_representative_bins.npz"))
    Z = float(d["z"])
    subset = load_template_subset(os.path.join(ROOT, "examples/data/muse/ngc5102_template_subset.npz"))
    ui = subset["union_selected"]
    select = list(zip(subset["template_ages"][ui].tolist(),
                      subset["template_metals"][ui].tolist()))
    templates, template_wave, *_ = load_emiles_templates(EM, select=select)

    LEGACY = {"high": 54.3, "mid": "40-45"}
    for label in ("high", "mid"):
        wr = d[f"wave_obs_{label}"] / (1.0 + Z)
        msk = (wr >= 8400) & (wr <= 8800)
        wave, flux, err = wr[msk], d[f"flux_obs_{label}"][msk], d[f"error_obs_{label}"][msk]
        cfg = FitConfig(vmin=-400.0, vmax=400.0)
        v = cfg.build_velocity_grid()
        A = build_template_stack(wave, template_wave, templates, v)
        grid = (1.0, 8.0, 64.0, 512.0, 4096.0, 32768.0)
        rows, chosen = og_select(A, flux, err, v, grid)
        print(f"\n--- {label} bin (legacy Fortran sigma ~ {LEGACY[label]}) ---")
        print(f"{'lam':>9s} {'V':>8s} {'sigma':>8s} {'h3':>7s} {'h4':>7s} "
              f"{'peaks':>6s} {'rough':>7s} {'OK':>4s}")
        for lam, m, npk, rg, ok, _ in rows:
            print(f"{lam:9.4g} {m['vherm']:+8.1f} {m['sherm']:8.1f} {m['h3']:+7.3f} "
                  f"{m['h4']:+7.3f} {npk:6d} {rg:7.3f} {'yes' if ok else 'no':>4s}")
        if chosen:
            lam, m, npk, rg, ok, b = chosen
            edge = max(b[:5].max(), b[-5:].max()) / b.max() * 100
            print(f"  OG rule selects lam={lam:g}  ->  sigma = {m['sherm']:.1f} km/s"
                  f"   (edge amplitude {edge:.2f}% of peak)")
        else:
            print("  OG rule selected nothing on this grid")


# ----------------------------------------------------------------- NB02 mock
def run_mock():
    sel = [(1.0, -0.66), (5.0, 0.06), (14.0, 0.06), (14.0, 0.40)]
    TW = np.array([.15, .35, .35, .15])
    V, S, H3, H4, SNR = 30., 90., 0.03, 0.05, 60.
    tmpl, tw, *_ = load_emiles_templates(EM, select=sel)
    obs = 4750. + np.arange(3681) * 1.25
    wave = obs[(obs >= 8400) & (obs <= 8800)]
    cfg = FitConfig(vmin=V - 7 * S, vmax=V + 7 * S)
    vg = cfg.build_velocity_grid()
    bt = np.maximum(gauss_hermite_model(vg, 1., V, S, H3, H4), 0.); bt /= bt.sum()
    A = build_template_stack(wave, tw, tmpl, vg)
    g0 = sum(TW[k] * (A[k] @ bt) for k in range(4))
    nl = np.median(g0) / SNR
    btn = bt / np.trapezoid(bt, vg); pk = btn.max(); core = btn > 0.01 * pk
    print("\n--- NB02 mock (portability check; truth V=+34.6 sigma=98.4 "
          "h3=+0.036 h4=+0.050) ---")
    print("    benchmarks: order-3 log-space 7.73%/h4=0.000 | order-2 best 6.19%/h4=0.033")
    print(f"{'lam':>9s} {'V':>8s} {'sigma':>8s} {'h3':>7s} {'h4':>7s} {'maxres%':>9s} {'OK':>4s}")
    grid = (1.0, 8.0, 64.0, 512.0, 4096.0, 32768.0)
    for lam in grid:
        Ms, MR, OKs = [], [], []
        for sd in range(3):
            gal = g0 + np.random.default_rng(sd).normal(0, nl, len(wave))
            e = np.full(len(wave), nl)
            b, _ = fit_with_sigl0_loop(A, gal, e, vg, lam, sig_init=90.0)
            m = compute_losvd_moments(vg, b, fit_h3h4=True)
            bn = b / np.trapezoid(np.maximum(b, 0), vg)
            MR.append((np.abs(bn - btn) / pk * 100)[core].max())
            Ms.append([m['vherm'], m['sherm'], m['h3'], m['h4']])
            OKs.append(count_losvd_peaks(b, 0.1) <= 1 and roughness(b) < 0.25)
        a = np.mean(Ms, axis=0)
        print(f"{lam:9.4g} {a[0]:+8.1f} {a[1]:8.1f} {a[2]:+7.3f} {a[3]:+7.3f} "
              f"{np.mean(MR):9.2f} {'yes' if all(OKs) else 'no':>4s}")


if __name__ == "__main__":
    run_real()
    run_mock()
