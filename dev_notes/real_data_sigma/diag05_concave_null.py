"""Priority-1, step 5: the proposed fix.

Root cause (established): D3's null space is *any quadratic* in log-b, so
exp(+a v^2) -- which diverges at the grid edges -- is exactly as free as a
Gaussian. Real-data systematics (chi2_red 11-16) reward that divergent
pedestal, sigma inflates ~2x, and aic_local then picks the least-
regularized lam because AIC's chi2-vs-m_eff trade assumes residuals are
noise, not systematics.

FIX TESTED HERE: keep the D3 penalty, but forbid the divergent half of its
own null space. Write the coefficient vector as
    c = M @ [a0, a1, a2] + K @ z
where M's columns are the blossom sequences of {1, v, v^2} (the exact
null space of the third-difference operator on a clamped cubic spline)
and K spans its orthogonal complement. Then simply BOUND a2 <= 0.
a2 is the quadratic coefficient of log b: a Gaussian has a2 = -1/(2 sig^2)
< 0; the pathological blow-up has a2 > 0. So this admits every Gaussian
and every perturbation on top of one, and excludes only shapes that
diverge -- with no new tuning parameter at all, and (unlike a row-wise
relu(D2 c) penalty, already ruled out) no clamped-spline boundary
artifacts, since a2 is a single global mode.
"""
import sys, os
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

def blossom_M(vmin, vmax, n_interior, degree=3):
    interior=np.linspace(vmin,vmax,n_interior+2)[1:-1]
    knots=np.concatenate([[vmin]*(degree+1),interior,[vmax]*(degree+1)])
    n=len(knots)-degree-1
    t=[knots[i+1:i+1+degree] for i in range(n)]
    ones=np.ones(n)
    xi=np.array([tt.mean() for tt in t])
    pi2=np.array([(tt[0]*tt[1]+tt[0]*tt[2]+tt[1]*tt[2])/3 for tt in t])
    return np.stack([ones,xi,pi2],axis=1)

def fit_concave(A_stack_w, y_w, cfg, lam, n_iter=12, enforce=True):
    v=cfg.build_velocity_grid(); nk=cfg.resolved_n_interior_knots()
    basis,nc=build_bspline_basis(v,n_interior_knots=nk,degree=cfg.degree)
    dv=(cfg.vmax-cfg.vmin)/(nk+1)
    D=difference_matrix(nc,order=cfg.penalty_order,dv=dv,dv_ref=cfg.knot_spacing_kms)
    M=blossom_M(cfg.vmin,cfg.vmax,nk)
    # scale columns so the three null-space params are comparably conditioned
    scale=np.array([1.0, 1.0/max(abs(M[:,1]).max(),1e-9), 1.0/max(abs(M[:,2]).max(),1e-9)])
    Ms=M*scale
    Q,_=np.linalg.qr(Ms)
    K=np.linalg.svd(Ms.T)[2][3:].T if nc>3 else np.zeros((nc,0))
    # orthogonal complement of col(M)
    U,S,Vt=np.linalg.svd(Ms, full_matrices=True)
    K=U[:,3:]
    bj,Dj=jnp.array(basis),jnp.array(D)
    Msj,Kj=jnp.array(Ms),jnp.array(K)
    nT=A_stack_w.shape[0]; w=np.ones(nT); p=None
    for it in range(n_iter):
        A_eff=np.tensordot(w,A_stack_w,axes=([0],[0]))
        Aj,yj=jnp.array(A_eff),jnp.array(y_w)
        def obj(par, lamj):
            a=par[:3]; z=par[3:]
            c=Msj@a + Kj@z
            b=jnp.exp(bj@c); r=Aj@b-yj
            return 0.5*jnp.sum(r**2)+0.5*lamj*jnp.sum((Dj@c)**2)
        vg=jax.jit(jax.value_and_grad(obj)); lamj=jnp.asarray(float(lam))
        if p is None:
            c0=default_initial_coeffs(A_eff,y_w,basis)
            a0=np.linalg.lstsq(Ms,c0,rcond=None)[0]
            if enforce: a0[2]=min(a0[2],-1e-12)
            z0=K.T@(c0-Ms@a0)
            p=np.concatenate([a0,z0])
        bounds=[(None,None),(None,None),((None,0.0) if enforce else (None,None))]+[(None,None)]*(nc-3)
        f=lambda x:(lambda t:(float(t[0]),np.array(t[1])))(vg(jnp.array(x),lamj))
        r_=scipy_minimize(f,p,jac=True,method='L-BFGS-B',bounds=bounds)
        p=r_.x
        c=Ms@p[:3]+K@p[3:]
        b=np.exp(basis@c)
        design=np.stack([A_stack_w[k]@b for k in range(nT)],axis=1)
        w,_=nnls(design,y_w)
        if not np.any(w>0): break
    return v, np.exp(basis@(Ms@p[:3]+K@p[3:])), w, float(p[2])

d=np.load(os.path.join(ROOT,"examples/data/muse/ngc5102_representative_bins.npz")); Z=float(d["z"])
sub=load_template_subset(os.path.join(ROOT,"examples/data/muse/ngc5102_template_subset.npz")); ui=sub["union_selected"]
sel=list(zip(sub["template_ages"][ui].tolist(),sub["template_metals"][ui].tolist()))
tmpl,tw,*_=load_emiles_templates(os.path.join(ROOT,"src/kinextract/template_data/emiles_muse_grid.npz"),select=sel)

print("=== REAL DATA (legacy: nucleus ~54, mid-radius ~40-45) ===")
print(f"{'bin':5s} {'lam':>8s} {'a2<=0?':>7s} {'V':>8s} {'sigma':>8s} {'h4':>8s} {'wing%':>7s} {'chi2red':>8s}")
for label in ["high","med"]:
    wr=d[f"wave_obs_{label}"]/(1+Z); m=(wr>=8400)&(wr<=8800)
    wave,flux=mask_wavelength_windows(wr[m],d[f"flux_obs_{label}"][m],windows=SKY)
    _,err=mask_wavelength_windows(wr[m],d[f"error_obs_{label}"][m],windows=SKY)
    cfg=FitConfig(vmin=-400.,vmax=400.)
    vgr=cfg.build_velocity_grid()
    A=build_template_stack(wave,tw,tmpl,vgr); Aw=A/err[None,:,None]; yw=flux/err
    taper=build_wing_taper(vgr,onset_frac=cfg.wing_onset_frac)
    for lam in [1.0, 8.0, 512.0]:
        for enf in [False, True]:
            v,b,w,a2=fit_concave(Aw,yw,cfg,lam,enforce=enf)
            mom=compute_losvd_moments(v,b,fit_h3h4=True)
            Aeff=np.tensordot(w,Aw,axes=([0],[0]))
            chi2red=float(np.sum((Aeff@b-yw)**2))/len(wave)
            print(f"{label:5s} {lam:8.4g} {str(enf):>7s} {mom['vherm']:+8.1f} {mom['sherm']:8.1f} "
                  f"{mom['h4']:+8.3f} {100*np.sum(taper*b)/np.sum(b):7.2f} {chi2red:8.1f}")

print("\n=== MOCK CONTROL (single E-MILES, V=20 sigma=50 S/N=60 seed 0) ===")
t1,tw1,*_=load_emiles_templates(os.path.join(ROOT,"src/kinextract/template_data/emiles_muse_grid.npz"),select=[(5.0,0.06)])
mk=(tw1>=8400)&(tw1<=8800); wf=tw1[mk]
cfgm=FitConfig(vmin=20-350.,vmax=20+350.); vm=cfgm.build_velocity_grid()
bt=np.maximum(gauss_hermite_model(vm,1.,20.,50.,0.,0.),0.); bt/=bt.sum()
Am=build_template_stack(wf,tw1,t1,vm); g0=Am[0]@bt; nl=np.median(g0)/60.
gal=g0+np.random.default_rng(0).normal(0,nl,len(wf)); errm=np.full(len(wf),nl)
Awm=Am/errm[None,:,None]; ywm=gal/errm
btn=bt/np.trapezoid(bt,vm); pk=btn.max(); core=btn>0.01*pk
print(f"{'lam':>8s} {'a2<=0?':>7s} {'V':>8s} {'sigma':>8s} {'maxresid%':>10s}")
for lam in [8.0, 512.0, 32768.0]:
    for enf in [False, True]:
        v,b,w,a2=fit_concave(Awm,ywm,cfgm,lam,enforce=enf)
        mom=compute_losvd_moments(v,b,fit_h3h4=False)
        bn=b/np.trapezoid(np.maximum(b,0),v)
        print(f"{lam:8.4g} {str(enf):>7s} {mom['vherm']:+8.1f} {mom['sherm']:8.1f} "
              f"{(np.abs(bn-btn)/pk*100)[core].max():10.2f}")
