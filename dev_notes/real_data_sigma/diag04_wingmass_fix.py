"""Priority-1, step 4: candidate fix.

Diagnosis so far: the D3 penalty's null space contains exp(+a v^2) as
cheaply as exp(-a v^2) (verified: identical ||D3 c||^2), so nothing stops
the fit adding a divergent edge pedestal. On mocks the perfect forward
model suppresses it; on real data, model mismatch actively rewards it
(it acts like a veiling/continuum term), and sigma inflates 42 -> 81.

Rejected already: a one-sided convexity penalty relu(D2 c)^2 -- dominated
by clamped-spline boundary artifacts (a true Gaussian scores WORSE than
the pathological shape).

Tested here: penalize the LOSVD's own WING MASS FRACTION,
    P = sum(taper * b) / sum(b)
which is exactly scale-invariant under (w,b)->(cw,b/c) -- the degeneracy
that made the existing raw-amplitude `wing_lam` need per-target retuning --
is linear in b, and is directly the pathological quantity (16-63% of peak
at the grid edge, vs <0.2% for any real LOSVD).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize as scipy_minimize, nnls

from kinextract import (FitConfig, build_template_stack, compute_losvd_moments,
                             mask_wavelength_windows, load_emiles_templates, gauss_hermite_model)
from kinextract.losvd import build_bspline_basis, difference_matrix, build_wing_taper, default_initial_coeffs
from kinextract.template_reduction import load_template_subset

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE,"..","..")
SKY = [(8414.,8419.),(8449.,8454.),(8711.,8714.),(8748.5,8753.)]

def fit_wingmass(A_stack_w, y_w, cfg, wing_frac_lam, lam, n_iter=12):
    """Alternating (b, w) fit with the wing-mass-fraction penalty added."""
    v = cfg.build_velocity_grid(); nk = cfg.resolved_n_interior_knots()
    basis, nc = build_bspline_basis(v, n_interior_knots=nk, degree=cfg.degree)
    dv = (cfg.vmax-cfg.vmin)/(nk+1)
    D = difference_matrix(nc, order=cfg.penalty_order, dv=dv, dv_ref=cfg.knot_spacing_kms)
    taper = build_wing_taper(v, onset_frac=cfg.wing_onset_frac)
    bj, Dj, tj = jnp.array(basis), jnp.array(D), jnp.array(taper)
    nT = A_stack_w.shape[0]
    w = np.ones(nT)
    c = None
    for it in range(n_iter):
        A_eff = np.tensordot(w, A_stack_w, axes=([0],[0]))
        Aj, yj = jnp.array(A_eff), jnp.array(y_w)
        def obj(cc, lamj, wfj):
            b = jnp.exp(bj@cc)
            r = Aj@b - yj
            wing = jnp.sum(tj*b)/jnp.sum(b)
            return 0.5*jnp.sum(r**2) + 0.5*lamj*jnp.sum((Dj@cc)**2) + 0.5*wfj*wing
        vg = jax.jit(jax.value_and_grad(obj))
        lamj, wfj = jnp.asarray(float(lam)), jnp.asarray(float(wing_frac_lam))
        starts = [c] if c is not None else [default_initial_coeffs(A_eff, y_w, basis),
                                            np.full(nc, np.log(0.01))]
        best=None
        for c0 in starts:
            f = lambda x: (lambda t:(float(t[0]), np.array(t[1])))(vg(jnp.array(x), lamj, wfj))
            r_ = scipy_minimize(f, c0, jac=True, method='L-BFGS-B')
            if best is None or r_.fun < best.fun: best = r_
        c = best.x
        b = np.exp(basis@c)
        design = np.stack([A_stack_w[k]@b for k in range(nT)], axis=1)
        w, _ = nnls(design, y_w)
        if not np.any(w>0): break
    return v, np.exp(basis@c), w

# ---------- real data ----------
d = np.load(os.path.join(ROOT,"examples/data/muse/ngc5102_representative_bins.npz")); Z=float(d["z"])
sub = load_template_subset(os.path.join(ROOT,"examples/data/muse/ngc5102_template_subset.npz"))
ui = sub["union_selected"]
sel = list(zip(sub["template_ages"][ui].tolist(), sub["template_metals"][ui].tolist()))
tmpl, tw, *_ = load_emiles_templates(os.path.join(ROOT,"src/kinextract/template_data/emiles_muse_grid.npz"), select=sel)

def real(label):
    wr = d[f"wave_obs_{label}"]/(1+Z); m=(wr>=8400)&(wr<=8800)
    wave, flux = mask_wavelength_windows(wr[m], d[f"flux_obs_{label}"][m], windows=SKY)
    _, err = mask_wavelength_windows(wr[m], d[f"error_obs_{label}"][m], windows=SKY)
    return wave, flux, err

print("=== REAL DATA (legacy Fortran: nucleus ~54, mid-radius ~40-45 km/s) ===")
print(f"{'bin':6s} {'wing_frac_lam':>14s} {'V':>8s} {'sigma':>8s} {'h4':>8s} {'wingmass%':>10s} {'chi2red':>9s}")
for label in ["high","med"]:
    wave, flux, err = real(label)
    A = None
    for wfl in [0.0, 1e2, 1e3, 1e4, 1e5]:
        cfg = FitConfig(vmin=-400., vmax=400.)
        vgrid = cfg.build_velocity_grid()
        if A is None:
            A = build_template_stack(wave, tw, tmpl, vgrid)
            Aw = A/err[None,:,None]; yw = flux/err
            taper = build_wing_taper(vgrid, onset_frac=cfg.wing_onset_frac)
        v, b, w = fit_wingmass(Aw, yw, cfg, wfl, lam=512.0)
        mom = compute_losvd_moments(v, b, fit_h3h4=True)
        Aeff = np.tensordot(w, Aw, axes=([0],[0]))
        chi2red = float(np.sum((Aeff@b - yw)**2))/len(wave)
        print(f"{label:6s} {wfl:14.4g} {mom['vherm']:+8.1f} {mom['sherm']:8.1f} {mom['h4']:+8.3f} "
              f"{100*np.sum(taper*b)/np.sum(b):10.2f} {chi2red:9.1f}")

# ---------- mock control: must NOT break the validated sigma=50 recovery ----------
print("\n=== MOCK CONTROL (single E-MILES template, V=20, sigma=50, S/N=60, seed 0) ===")
t1, tw1, *_ = load_emiles_templates(os.path.join(ROOT,"src/kinextract/template_data/emiles_muse_grid.npz"), select=[(5.0,0.06)])
mk=(tw1>=8400)&(tw1<=8800); wf=tw1[mk]
cfgm = FitConfig(vmin=20-7*50., vmax=20+7*50.)
vm = cfgm.build_velocity_grid()
bt = np.maximum(gauss_hermite_model(vm,1.,20.,50.,0.,0.),0.); bt/=bt.sum()
Am = build_template_stack(wf, tw1, t1, vm)
g0 = Am[0]@bt; nl=np.median(g0)/60.
gal = g0 + np.random.default_rng(0).normal(0,nl,len(wf)); errm=np.full(len(wf),nl)
Awm = Am/errm[None,:,None]; ywm = gal/errm
btn = bt/np.trapezoid(bt,vm); pk=btn.max(); core=btn>0.01*pk
taperm = build_wing_taper(vm, onset_frac=cfgm.wing_onset_frac)
print(f"{'wing_frac_lam':>14s} {'V':>8s} {'sigma':>8s} {'maxresid%':>10s} {'wingmass%':>10s}")
for wfl in [0.0, 1e2, 1e3, 1e4, 1e5]:
    v, b, w = fit_wingmass(Awm, ywm, cfgm, wfl, lam=512.0)
    mom = compute_losvd_moments(v, b, fit_h3h4=False)
    bn = b/np.trapezoid(np.maximum(b,0),v)
    res = (np.abs(bn-btn)/pk*100)[core].max()
    print(f"{wfl:14.4g} {mom['vherm']:+8.1f} {mom['sherm']:8.1f} {res:10.2f} {100*np.sum(taperm*b)/np.sum(b):10.3f}")
