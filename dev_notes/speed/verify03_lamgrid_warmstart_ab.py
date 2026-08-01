"""Paired A/B of `select_lambda_aic_local`'s new `warm_start_lam_grid` option:
does chaining each lam-grid point's fit from its neighbour's converged
coefficients (instead of cold-starting `fit_losvd`'s full multi-start at
every one of the 6 default grid points) change ANSWER QUALITY, or only
runtime?

Same paired-design methodology as `verify02_warmstart_ab.py` (identical
noise draws both arms, N seeds x 3 sigma values, scored against ground
truth) -- this is the analogous check for the *other* half of
`select_lambda_aic_local`'s own cost (dev_notes/speed/RESULTS.md measured
the un-warm-started 6-point scan at ~9.5x a single fixed-lam fit; only
the within-lam alternating-iteration warm-start, not the across-lam one,
was previously validated).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from kinextract import (FitConfig, build_template_stack, gauss_hermite_model,
                             compute_losvd_moments, load_emiles_templates)
from kinextract.aic_regularization import select_lambda_aic_local

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
EM = os.path.join(ROOT, "src/kinextract/template_data/emiles_muse_grid.npz")
templates, tw, *_ = load_emiles_templates(EM, select=[(1.0,-0.66),(5.0,0.06),(14.0,0.40)])
W = np.array([0.3,0.4,0.3])
obs = 4750.0+np.arange(3681)*1.25; wave = obs[(obs>=8400)&(obs<=8800)]
SNR, TRUE_V, NSEED = 60.0, 20.0, 12

print(f"{'sigma':>6} {'arm':>5} {'|dV|':>7} {'|dsig|':>8} {'maxresid%':>10} {'best_lam mismatches':>20} {'time/scan':>10}")
summary = {}
for SIG in [50.0, 140.0, 300.0]:
    cfg = FitConfig(vmin=TRUE_V-7*SIG, vmax=TRUE_V+7*SIG)
    v = cfg.build_velocity_grid()
    bt = np.maximum(gauss_hermite_model(v,1.,TRUE_V,SIG,0.,0.),0.); bt/=bt.sum()
    A = build_template_stack(wave, tw, templates, v)
    g0 = sum(W[k]*(A[k]@bt) for k in range(3)); nl = np.median(g0)/SNR
    btn = bt/np.trapezoid(bt,v); pk=btn.max(); core=btn>0.01*pk
    truth = compute_losvd_moments(v, btn, fit_h3h4=False)
    is_d3 = cfg.penalty_order == 3
    lam_grid = np.asarray(cfg.resolved_lam_grid())

    lam_choices = {}
    for warm in (True, False):
        dV, dS, res, tt, lams = [], [], [], [], []
        for seed in range(NSEED):
            gal = g0 + np.random.default_rng(seed).normal(0,nl,len(wave))
            err = np.full(len(wave), nl); Aw = A/err[None,:,None]; yw = gal/err
            t0 = time.perf_counter()
            selr = select_lambda_aic_local(Aw, yw, cfg, lam_grid,
                                           parsimony_delta=2.0 if is_d3 else 0.0,
                                           min_resolvable_sigma_frac=0.5 if is_d3 else 0.0,
                                           max_edge_amplitude_frac=cfg.max_edge_amplitude_frac,
                                           warm_start_lam_grid=warm)
            tt.append(time.perf_counter()-t0)
            r = selr.best_fit
            lams.append(selr.best_lam)
            m = compute_losvd_moments(v, r['b'], fit_h3h4=False)
            bn = r['b']/np.trapezoid(np.maximum(r['b'],0),v)
            dV.append(abs(m['vherm']-truth['vherm'])); dS.append(abs(m['sherm']-truth['sherm']))
            res.append((np.abs(bn-btn)/pk*100)[core].max())
        lam_choices[warm] = lams
        summary[(SIG,warm)] = (np.mean(dV), np.mean(dS), np.mean(res), np.mean(tt))
    mismatches = sum(1 for a, b in zip(lam_choices[True], lam_choices[False]) if a != b)
    for warm in (True, False):
        r_ = summary[(SIG,warm)]
        print(f"{SIG:6.0f} {'warm' if warm else 'cold':>5} {r_[0]:7.2f} {r_[1]:8.2f} "
              f"{r_[2]:10.2f} {mismatches if warm else '--':>20} {r_[3]:10.3f}s")

print("\n--- warm vs cold (negative = warm is better) ---")
for SIG in [50.0,140.0,300.0]:
    w_,c_ = summary[(SIG,True)], summary[(SIG,False)]
    print(f"sigma={SIG:5.0f}:  d|dV|={w_[0]-c_[0]:+6.2f}  d|dsig|={w_[1]-c_[1]:+6.2f}  "
          f"dmaxresid={w_[2]-c_[2]:+6.2f}%  speedup={c_[3]/w_[3]:.1f}x")
