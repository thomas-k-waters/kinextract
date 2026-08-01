"""Priority-1, step 6: the structural fix.

Everything upstream is established:
  * the pedestal, not the core, is the entire sigma error;
  * D3's null space cannot forbid it (a Gaussian and its divergent mirror
    image have identical ||D3 c||^2);
  * NO lam-selection rule can fix it -- QAIC and the discrepancy principle
    both still land on lam=8 (sigma=84) because the pedestal genuinely
    buys ~1200 units of chi2 by absorbing template mismatch.

So constrain the LOSVD structurally. A single-component LOSVD is
unimodal, and B-splines are *variation-diminishing*: a unimodal
coefficient sequence gives a unimodal b(v). So unimodality can be imposed
exactly, with no new tuning parameter, as plain box bounds on the
coefficient differences:
    c_j = c_0 + sum_{i<=j} d_i,   d_i >= 0 for i <= p,  d_i <= 0 for i > p
with the peak index p re-read from the current b each outer iteration.
This forbids the U-shape at ANY lam, rather than hoping smoothing hides it.
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

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.join(HERE,"..","..")
SKY=[(8414.,8419.),(8449.,8454.),(8711.,8714.),(8748.5,8753.)]

def fit_unimodal(A_stack_w, y_w, cfg, lam, unimodal=True, n_iter=10):
    v=cfg.build_velocity_grid(); nk=cfg.resolved_n_interior_knots()
    basis,nc=build_bspline_basis(v,n_interior_knots=nk,degree=cfg.degree)
    dv=(cfg.vmax-cfg.vmin)/(nk+1)
    D=difference_matrix(nc,order=cfg.penalty_order,dv=dv,dv_ref=cfg.knot_spacing_kms)
    bj,Dj=jnp.array(basis),jnp.array(D)
    # cumulative-sum parameterization: par = [c0, d_1..d_{nc-1}]
    L=np.tril(np.ones((nc,nc)))            # c = L @ par  with par[0]=c0, par[i]=d_i
    Lj=jnp.array(L)
    nT=A_stack_w.shape[0]; w=np.ones(nT); par=None
    for it in range(n_iter):
        A_eff=np.tensordot(w,A_stack_w,axes=([0],[0]))
        Aj,yj=jnp.array(A_eff),jnp.array(y_w)
        def obj(p):
            c=Lj@p
            b=jnp.exp(bj@c); r=Aj@b-yj
            return 0.5*jnp.sum(r**2)+0.5*lam*jnp.sum((Dj@c)**2)
        vg=jax.jit(jax.value_and_grad(obj))
        if par is None:
            c0=default_initial_coeffs(A_eff,y_w,basis)
            par=np.concatenate([[c0[0]], np.diff(c0)])
        # peak index from the current curve
        c_cur=L@par; b_cur=np.exp(basis@c_cur)
        pk_v=v[int(np.argmax(b_cur))]
        # coefficient j sits near knot-average velocity; approximate by even spacing
        knot_v=np.linspace(cfg.vmin,cfg.vmax,nc)
        p_idx=int(np.argmin(np.abs(knot_v-pk_v)))
        if unimodal:
            bounds=[(None,None)]+[((0.0,None) if i<=p_idx else (None,0.0)) for i in range(1,nc)]
            # project current par into the feasible set
            par=np.array(par)
            for i in range(1,nc):
                if i<=p_idx: par[i]=max(par[i],0.0)
                else:        par[i]=min(par[i],0.0)
        else:
            bounds=[(None,None)]*nc
        f=lambda x:(lambda t:(float(t[0]),np.array(t[1])))(vg(jnp.array(x)))
        r_=scipy_minimize(f,par,jac=True,method='L-BFGS-B',bounds=bounds)
        par=r_.x
        b=np.exp(basis@(L@par))
        design=np.stack([A_stack_w[k]@b for k in range(nT)],axis=1)
        w,_=nnls(design,y_w)
        if not np.any(w>0): break
    return v, np.exp(basis@(L@par)), w

d=np.load(os.path.join(ROOT,"examples/data/muse/ngc5102_representative_bins.npz")); Z=float(d["z"])
sub=load_template_subset(os.path.join(ROOT,"examples/data/muse/ngc5102_template_subset.npz")); ui=sub["union_selected"]
sel=list(zip(sub["template_ages"][ui].tolist(),sub["template_metals"][ui].tolist()))
tmpl,tw,*_=load_emiles_templates(os.path.join(ROOT,"src/kinextract/template_data/emiles_muse_grid.npz"),select=sel)

print("=== REAL DATA (legacy: nucleus ~54.3, mid-radius ~40-45) ===")
print(f"{'bin':5s} {'lam':>8s} {'unimodal':>9s} {'V':>8s} {'sigma':>8s} {'h4':>8s} {'wing%':>7s} {'chi2red':>8s}")
for label in ["high","med"]:
    wr=d[f"wave_obs_{label}"]/(1+Z); m=(wr>=8400)&(wr<=8800)
    wave,flux=mask_wavelength_windows(wr[m],d[f"flux_obs_{label}"][m],windows=SKY)
    _,err=mask_wavelength_windows(wr[m],d[f"error_obs_{label}"][m],windows=SKY)
    cfg=FitConfig(vmin=-400.,vmax=400.); vg_=cfg.build_velocity_grid()
    A=build_template_stack(wave,tw,tmpl,vg_); Aw=A/err[None,:,None]; yw=flux/err
    taper=build_wing_taper(vg_,onset_frac=cfg.wing_onset_frac)
    for lam in [1.0, 8.0, 512.0]:
        for uni in [False, True]:
            v,b,w=fit_unimodal(Aw,yw,cfg,lam,unimodal=uni)
            mom=compute_losvd_moments(v,b,fit_h3h4=True)
            Aeff=np.tensordot(w,Aw,axes=([0],[0]))
            chi2red=float(np.sum((Aeff@b-yw)**2))/len(wave)
            print(f"{label:5s} {lam:8.4g} {str(uni):>9s} {mom['vherm']:+8.1f} {mom['sherm']:8.1f} "
                  f"{mom['h4']:+8.3f} {100*np.sum(taper*b)/np.sum(b):7.2f} {chi2red:8.1f}")

print("\n=== MOCK CONTROL (single E-MILES, V=20 sigma=50 S/N=60, 5 seeds) ===")
t1,tw1,*_=load_emiles_templates(os.path.join(ROOT,"src/kinextract/template_data/emiles_muse_grid.npz"),select=[(5.0,0.06)])
mk=(tw1>=8400)&(tw1<=8800); wf=tw1[mk]
cfgm=FitConfig(vmin=20-350.,vmax=20+350.); vm=cfgm.build_velocity_grid()
bt=np.maximum(gauss_hermite_model(vm,1.,20.,50.,0.,0.),0.); bt/=bt.sum()
Am=build_template_stack(wf,tw1,t1,vm); g0=Am[0]@bt; nl=np.median(g0)/60.
btn=bt/np.trapezoid(bt,vm); pk=btn.max(); core=btn>0.01*pk
truth=compute_losvd_moments(vm,btn,fit_h3h4=False)
print(f"{'lam':>8s} {'unimodal':>9s} {'sigma(mean)':>12s} {'|dsig|':>8s} {'maxresid%':>10s}   (truth sigma={truth['sherm']:.2f})")
for lam in [512.0, 32768.0]:
    for uni in [False, True]:
        ss,rr=[],[]
        for seed in range(5):
            gal=g0+np.random.default_rng(seed).normal(0,nl,len(wf)); errm=np.full(len(wf),nl)
            v,b,w=fit_unimodal(Am/errm[None,:,None],gal/errm,cfgm,lam,unimodal=uni)
            mo=compute_losvd_moments(v,b,fit_h3h4=False)
            bn=b/np.trapezoid(np.maximum(b,0),v)
            ss.append(mo['sherm']); rr.append((np.abs(bn-btn)/pk*100)[core].max())
        print(f"{lam:8.4g} {str(uni):>9s} {np.mean(ss):12.2f} {abs(np.mean(ss)-truth['sherm']):8.2f} {np.mean(rr):10.2f}")
