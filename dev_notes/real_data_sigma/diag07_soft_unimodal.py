"""Priority-1, step 7: unimodality as a SOFT penalty on the standard
coefficient parameterization.

diag06 imposed unimodality by reparameterizing c as a cumulative sum with
box bounds. That is exact in principle but wrecked the conditioning (L is
lower-triangular ones), and it FAILED ITS OWN CONTROL: with the constraint
switched off it gave sigma=41.1 on the mock where the real code path gives
49.0 at the same lam. Its numbers are therefore not usable and are not
reported anywhere as results.

Here instead: keep the exact parameterization the package already uses,
and add one extra term to the objective --
    kappa * sum_j relu(-s_j * (c_{j+1}-c_j))^2 ,  s_j = +1 before the peak,
                                                        -1 after
which is zero for any unimodal coefficient sequence (hence, by the
variation-diminishing property of B-splines, any unimodal b) and grows
quadratically with the violation. It is scale-invariant (differences of
log b are unchanged by b -> c*b), so unlike `wing_lam` it needs no
per-target amplitude tuning.

The harness is validated against the real code path at kappa=0 BEFORE any
constrained number is believed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize as scipy_minimize, nnls

from kinextract import (FitConfig, build_template_stack, compute_losvd_moments,
                             mask_wavelength_windows, load_emiles_templates, gauss_hermite_model)
from kinextract.losvd import (build_bspline_basis, difference_matrix, build_wing_taper,
                                   default_initial_coeffs, _default_multi_starts)
from kinextract.multi_template import fit_losvd_basis
from kinextract.template_reduction import load_template_subset

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.join(HERE,"..","..")
SKY=[(8414.,8419.),(8449.,8454.),(8711.,8714.),(8748.5,8753.)]
EM=os.path.join(ROOT,"src/kinextract/template_data/emiles_muse_grid.npz")

def greville(vmin, vmax, n_interior, degree=3):
    interior=np.linspace(vmin,vmax,n_interior+2)[1:-1]
    knots=np.concatenate([[vmin]*(degree+1),interior,[vmax]*(degree+1)])
    n=len(knots)-degree-1
    return np.array([knots[i+1:i+1+degree].mean() for i in range(n)])

def fit_soft_unimodal(A_stack_w, y_w, cfg, lam, kappa, n_iter=20, tol=1e-4, patience=2):
    """Mirrors _alternate_once (multi-start on iter 0, warm-start after,
    best-seen tracking) with the unimodality term added."""
    v=cfg.build_velocity_grid(); nk=cfg.resolved_n_interior_knots()
    basis,nc=build_bspline_basis(v,n_interior_knots=nk,degree=cfg.degree)
    dv=(cfg.vmax-cfg.vmin)/(nk+1)
    D=difference_matrix(nc,order=cfg.penalty_order,dv=dv,dv_ref=cfg.knot_spacing_kms)
    gv=greville(cfg.vmin,cfg.vmax,nk)
    bj,Dj=jnp.array(basis),jnp.array(D)
    nT=A_stack_w.shape[0]; w=np.ones(nT)
    warm=None; best=None; best_obj=np.inf; stale=0
    for it in range(n_iter):
        A_eff=np.tensordot(w,A_stack_w,axes=([0],[0]))
        Aj,yj=jnp.array(A_eff),jnp.array(y_w)
        # peak coefficient index from the current curve (Greville abscissae)
        if warm is None:
            p_idx=nc//2
        else:
            p_idx=int(np.argmin(np.abs(gv - v[int(np.argmax(np.exp(basis@warm)))])))
        s=np.ones(nc-1); s[p_idx:]=-1.0
        sj=jnp.array(s); kj=jnp.asarray(float(kappa)); lj=jnp.asarray(float(lam))
        def obj(c):
            b=jnp.exp(bj@c); r=Aj@b-yj
            dif=c[1:]-c[:-1]
            viol=jnp.maximum(-sj*dif, 0.0)
            return (0.5*jnp.sum(r**2) + 0.5*lj*jnp.sum((Dj@c)**2)
                    + 0.5*kj*jnp.sum(viol**2))
        vg=jax.jit(jax.value_and_grad(obj))
        f=lambda x:(lambda t:(float(t[0]),np.array(t[1])))(vg(jnp.array(x)))
        starts=[warm] if warm is not None else _default_multi_starts(A_eff,y_w,basis)
        rbest=None
        for c0 in starts:
            r_=scipy_minimize(f,c0,jac=True,method='L-BFGS-B')
            if rbest is None or r_.fun<rbest.fun: rbest=r_
        warm=rbest.x
        b=np.exp(basis@warm)
        design=np.stack([A_stack_w[k]@b for k in range(nT)],axis=1)
        w,_=nnls(design,y_w)
        if not np.any(w>0): break
        A_fin=np.tensordot(w,A_stack_w,axes=([0],[0]))
        objective=0.5*np.sum((A_fin@b-y_w)**2)+0.5*lam*np.sum((D@warm)**2)
        if best is None or objective<best_obj-tol*max(abs(best_obj),1.0):
            best_obj=objective; best=(b.copy(), w.copy()); stale=0
        else:
            stale+=1
            if stale>=patience: break
    return v, best[0], best[1]

# ---------------- HARNESS VALIDATION: kappa=0 must reproduce the real code path
print("--- harness check: kappa=0 vs the package's own fit_losvd_basis (must agree) ---")
t1,tw1,*_=load_emiles_templates(EM,select=[(5.0,0.06)])
mk=(tw1>=8400)&(tw1<=8800); wf=tw1[mk]
cfgm=FitConfig(vmin=20-350.,vmax=20+350.); vm=cfgm.build_velocity_grid()
nkm=cfgm.resolved_n_interior_knots()
basism,ncm=build_bspline_basis(vm,n_interior_knots=nkm,degree=3)
dvm=(cfgm.vmax-cfgm.vmin)/(nkm+1)
Dm=difference_matrix(ncm,order=3,dv=dvm,dv_ref=cfgm.knot_spacing_kms)
bt=np.maximum(gauss_hermite_model(vm,1.,20.,50.,0.,0.),0.); bt/=bt.sum()
Am=build_template_stack(wf,tw1,t1,vm); g0=Am[0]@bt; nl=np.median(g0)/60.
gal=g0+np.random.default_rng(0).normal(0,nl,len(wf)); errm=np.full(len(wf),nl)
Awm=Am/errm[None,:,None]; ywm=gal/errm
for lam in [512.0, 32768.0]:
    rpkg=fit_losvd_basis(Awm,ywm,basism,Dm,lam=lam)
    mp=compute_losvd_moments(vm,rpkg['b'],fit_h3h4=False)
    v_,b_,w_=fit_soft_unimodal(Awm,ywm,cfgm,lam,kappa=0.0)
    mh=compute_losvd_moments(v_,b_,fit_h3h4=False)
    ok = abs(mp['sherm']-mh['sherm'])<0.3
    print(f"   lam={lam:8.0f}  package sigma={mp['sherm']:7.2f}   harness sigma={mh['sherm']:7.2f}   "
          f"{'OK' if ok else '*** HARNESS DISAGREES ***'}")

# ---------------- real data + mock, kappa swept
d=np.load(os.path.join(ROOT,"examples/data/muse/ngc5102_representative_bins.npz")); Z=float(d["z"])
sub=load_template_subset(os.path.join(ROOT,"examples/data/muse/ngc5102_template_subset.npz")); ui=sub["union_selected"]
sel=list(zip(sub["template_ages"][ui].tolist(),sub["template_metals"][ui].tolist()))
tmpl,tw,*_=load_emiles_templates(EM,select=sel)

print("\n=== REAL DATA at lam=1 (what aic_local actually picks) ===")
print(f"{'bin':5s} {'kappa':>10s} {'V':>8s} {'sigma':>8s} {'h4':>8s} {'wing%':>7s} {'chi2red':>8s}   legacy")
for label,leg in [("high","~54.3"),("med","~40-45")]:
    wr=d[f"wave_obs_{label}"]/(1+Z); m=(wr>=8400)&(wr<=8800)
    wave,flux=mask_wavelength_windows(wr[m],d[f"flux_obs_{label}"][m],windows=SKY)
    _,err=mask_wavelength_windows(wr[m],d[f"error_obs_{label}"][m],windows=SKY)
    cfg=FitConfig(vmin=-400.,vmax=400.); vg_=cfg.build_velocity_grid()
    A=build_template_stack(wave,tw,tmpl,vg_); Aw=A/err[None,:,None]; yw=flux/err
    taper=build_wing_taper(vg_,onset_frac=cfg.wing_onset_frac)
    for kap in [0.0, 1e2, 1e4, 1e6]:
        v_,b_,w_=fit_soft_unimodal(Aw,yw,cfg,1.0,kappa=kap)
        mo=compute_losvd_moments(v_,b_,fit_h3h4=True)
        Aeff=np.tensordot(w_,Aw,axes=([0],[0]))
        c2=float(np.sum((Aeff@b_-yw)**2))/len(wave)
        print(f"{label:5s} {kap:10.4g} {mo['vherm']:+8.1f} {mo['sherm']:8.1f} {mo['h4']:+8.3f} "
              f"{100*np.sum(taper*b_)/np.sum(b_):7.2f} {c2:8.1f}   {leg}")

print("\n=== MOCK CONTROL (V=20 sigma=50 S/N=60, 5 seeds, lam=32768 = what aic picks) ===")
btn=bt/np.trapezoid(bt,vm); pk=btn.max(); core=btn>0.01*pk
truth=compute_losvd_moments(vm,btn,fit_h3h4=False)
print(f"{'kappa':>10s} {'sigma':>8s} {'|dsig|':>8s} {'maxresid%':>10s}   (truth {truth['sherm']:.2f})")
for kap in [0.0, 1e2, 1e4, 1e6]:
    ss,rr=[],[]
    for seed in range(5):
        gal=g0+np.random.default_rng(seed).normal(0,nl,len(wf))
        v_,b_,w_=fit_soft_unimodal(Am/errm[None,:,None],gal/errm,cfgm,32768.0,kappa=kap)
        mo=compute_losvd_moments(v_,b_,fit_h3h4=False)
        bn=b_/np.trapezoid(np.maximum(b_,0),v_)
        ss.append(mo['sherm']); rr.append((np.abs(bn-btn)/pk*100)[core].max())
    print(f"{kap:10.4g} {np.mean(ss):8.2f} {abs(np.mean(ss)-truth['sherm']):8.2f} {np.mean(rr):10.2f}")
