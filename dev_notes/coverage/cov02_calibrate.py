"""How much too narrow is the LOSVD uncertainty band, and what widening
restores nominal coverage?

cov01 established: the bootstrap bias estimate is ~1.2-1.7% of peak while
the real max residual is 7.5-9%, so bias correction from it overcorrects
and does not help. The band is simply too narrow. This measures, against
known truth, the multiplier k on the 16-84 half-width that restores 68%
coverage -- an empirical calibration, honest about being one.

One pass of fits; multipliers scanned offline from the stored curves.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from kinextract import (FitConfig, build_template_stack, gauss_hermite_model,
                             load_emiles_templates, bootstrap_losvd)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
EM = os.path.join(ROOT, "src/kinextract/template_data/emiles_muse_grid.npz")
N_BOOT, N_SEED = 12, 4
CASES = {
 "NB01 Gaussian s=50": dict(sel=[(1.0,-0.66),(5.0,0.06),(14.0,0.40)], w=[.3,.4,.3],
                            V=20., S=50., h3=0., h4=0., snr=60.),
 "NB02 h3/h4 s=90":    dict(sel=[(1.0,-0.66),(5.0,0.06),(14.0,0.06),(14.0,0.40)],
                            w=[.15,.35,.35,.15], V=30., S=90., h3=.03, h4=.05, snr=60.),
}
def norm(b, v): return b/np.trapezoid(np.maximum(b,0.0), v)
KS = [1,2,3,4,5,6,8,10]

for name, C in CASES.items():
    tmpl, tw, *_ = load_emiles_templates(EM, select=C["sel"])
    W=np.asarray(C["w"],float); W/=W.sum()
    obs=4750.+np.arange(3681)*1.25; wave=obs[(obs>=8400)&(obs<=8800)]
    cfg=FitConfig(vmin=C["V"]-7*C["S"], vmax=C["V"]+7*C["S"]); v=cfg.build_velocity_grid()
    bt=np.maximum(gauss_hermite_model(v,1.,C["V"],C["S"],C["h3"],C["h4"]),0.); bt/=bt.sum()
    A=build_template_stack(wave,tw,tmpl,v)
    g0=sum(W[k]*(A[k]@bt) for k in range(len(tmpl))); nl=np.median(g0)/C["snr"]
    btn=norm(bt,v); pk=btn.max(); core=btn>0.01*pk

    store=[]
    for sd in range(N_SEED):
        gal=g0+np.random.default_rng(sd).normal(0,nl,len(wave)); e=np.full(len(wave),nl)
        bo=bootstrap_losvd(A/e[None,:,None], gal/e, cfg, n_boot=N_BOOT, seed=100+sd)
        b0=norm(bo["result0"]["b"],v)
        bb=np.array([norm(x,v) for x in bo["b_boot"]])
        store.append((b0, np.percentile(bb,16,axis=0), np.percentile(bb,84,axis=0)))

    print(f"\n=== {name} === ({N_SEED} seeds x {N_BOOT} boot)")
    print(f"  {'k':>4s} {'coverage of truth (core bins)':>32s}")
    best=None
    for k in KS:
        cov=[]
        for b0,lo,hi in store:
            l=b0-k*(b0-lo); h=b0+k*(hi-b0)
            cov.append(np.mean((btn[core]>=l[core])&(btn[core]<=h[core])))
        c=100*np.mean(cov)
        print(f"  {k:4d} {c:29.1f}%")
        if best is None or abs(c-68.)<abs(best[1]-68.): best=(k,c)
    print(f"  -> k={best[0]} gives {best[1]:.1f}% (nominal 68%)")
