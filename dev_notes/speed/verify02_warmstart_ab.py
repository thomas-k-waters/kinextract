"""Paired A/B of the warm-start optimization: does it change ANSWER
QUALITY, or only runtime?

Warm-starting the LOSVD sub-fit across alternating iterations (instead of
a cold 3-way multi-start every iteration) cut a default fit from 34.1 s to
5.0 s. It also shifts results slightly, because it changes which optimum
the alternation settles into -- within the basin ambiguity this package
already documents. That is a real behaviour change and has to be measured,
not assumed benign.

Paired design: identical noise draws, both arms, N seeds x 3 sigma values.
Scores against ground truth, so "different" can be judged as better/worse
rather than just different.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from kinextract import (FitConfig, build_template_stack, gauss_hermite_model,
                             compute_losvd_moments, load_emiles_templates)
from kinextract.multi_template import fit_losvd_basis
from kinextract.losvd import build_bspline_basis, difference_matrix
from kinextract.aic_regularization import select_lambda_aic_local

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
EM = os.path.join(ROOT, "src/kinextract/template_data/emiles_muse_grid.npz")
templates, tw, *_ = load_emiles_templates(EM, select=[(1.0,-0.66),(5.0,0.06),(14.0,0.40)])
W = np.array([0.3,0.4,0.3])
obs = 4750.0+np.arange(3681)*1.25; wave = obs[(obs>=8400)&(obs<=8800)]
SNR, TRUE_V, NSEED = 60.0, 20.0, 12

print(f"{'sigma':>6} {'arm':>5} {'|dV|':>7} {'|dsig|':>8} {'maxresid%':>10} {'time/fit':>9}")
summary = {}
for SIG in [50.0, 140.0, 300.0]:
    cfg = FitConfig(vmin=TRUE_V-7*SIG, vmax=TRUE_V+7*SIG)
    v = cfg.build_velocity_grid()
    nk = cfg.resolved_n_interior_knots()
    basis, nc = build_bspline_basis(v, n_interior_knots=nk, degree=3)
    dv = (cfg.vmax-cfg.vmin)/(nk+1)
    D = difference_matrix(nc, order=3, dv=dv, dv_ref=cfg.knot_spacing_kms)
    bt = np.maximum(gauss_hermite_model(v,1.,TRUE_V,SIG,0.,0.),0.); bt/=bt.sum()
    A = build_template_stack(wave, tw, templates, v)
    g0 = sum(W[k]*(A[k]@bt) for k in range(3)); nl = np.median(g0)/SNR
    btn = bt/np.trapezoid(bt,v); pk=btn.max(); core=btn>0.01*pk
    truth = compute_losvd_moments(v, btn, fit_h3h4=False)
    for warm in (True, False):
        dV, dS, res, tt = [], [], [], []
        for seed in range(NSEED):
            gal = g0 + np.random.default_rng(seed).normal(0,nl,len(wave))
            err = np.full(len(wave), nl); Aw = A/err[None,:,None]; yw = gal/err
            t0 = time.perf_counter()
            selr = select_lambda_aic_local(Aw, yw, cfg, np.asarray(cfg.resolved_lam_grid()),
                                           parsimony_delta=2.0, min_resolvable_sigma_frac=0.5)
            r = fit_losvd_basis(Aw, yw, basis, D, lam=selr.best_lam, losvd_warm_start=warm)
            tt.append(time.perf_counter()-t0)
            m = compute_losvd_moments(v, r['b'], fit_h3h4=False)
            bn = r['b']/np.trapezoid(np.maximum(r['b'],0),v)
            dV.append(abs(m['vherm']-truth['vherm'])); dS.append(abs(m['sherm']-truth['sherm']))
            res.append((np.abs(bn-btn)/pk*100)[core].max())
        summary[(SIG,warm)] = (np.mean(dV), np.mean(dS), np.mean(res), np.mean(tt))
        print(f"{SIG:6.0f} {'warm' if warm else 'cold':>5} {np.mean(dV):7.2f} {np.mean(dS):8.2f} "
              f"{np.mean(res):10.2f} {np.mean(tt):8.1f}s")

print("\n--- warm vs cold (negative = warm is better) ---")
for SIG in [50.0,140.0,300.0]:
    w_,c_ = summary[(SIG,True)], summary[(SIG,False)]
    print(f"sigma={SIG:5.0f}:  d|dV|={w_[0]-c_[0]:+6.2f}  d|dsig|={w_[1]-c_[1]:+6.2f}  "
          f"dmaxresid={w_[2]-c_[2]:+6.2f}%  speedup={c_[3]/w_[3]:.1f}x")
