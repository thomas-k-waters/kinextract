"""(c) Does a curve-level bias estimate make the uncertainty band honest?

The parametric bootstrap resamples noise around the *fitted* model, so
every replicate is pulled toward the same regularized optimum: it measures
variance and is blind to the penalty's shrinkage. Measured consequence
(notebook 02): the truth's peak sits ~3x the band width above the
recovered peak -- the band is not wrong about noise, it is silent about
bias.

Standard fix for a penalized estimator, and it needs no new fits because
`bootstrap_losvd` already produces the replicates:

    bias_hat(v) = median_k b_k(v) - b0(v)        (bootstrap bias estimate)
    b_debiased  = b0 - bias_hat                   (first-order correction)
    band        = sqrt(stat_width^2 + bias_hat^2) (bias folded in)

All curves unit-area normalized first, since only shape is identifiable.

**This is tested, not assumed.** The same construction at moment level is
already documented in this package as prone to overcorrection, so the
metric here is COVERAGE (does truth actually fall inside the band across
the profile), reported before and after, over several seeds.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from kinextract import (FitConfig, build_template_stack, gauss_hermite_model,
                             compute_losvd_moments, load_emiles_templates, bootstrap_losvd)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
EM = os.path.join(ROOT, "src/kinextract/template_data/emiles_muse_grid.npz")
N_BOOT, N_SEED = 15, 5

CASES = {
 "NB01 Gaussian s=50": dict(sel=[(1.0,-0.66),(5.0,0.06),(14.0,0.40)], w=[.3,.4,.3],
                            V=20., S=50., h3=0., h4=0., snr=60.),
 "NB02 h3/h4 s=90":    dict(sel=[(1.0,-0.66),(5.0,0.06),(14.0,0.06),(14.0,0.40)],
                            w=[.15,.35,.35,.15], V=30., S=90., h3=.03, h4=.05, snr=60.),
}

def norm(b, v):
    return b / np.trapezoid(np.maximum(b, 0.0), v)

for name, C in CASES.items():
    tmpl, tw, *_ = load_emiles_templates(EM, select=C["sel"])
    W = np.asarray(C["w"], float); W /= W.sum()
    obs = 4750. + np.arange(3681)*1.25
    wave = obs[(obs >= 8400) & (obs <= 8800)]
    cfg = FitConfig(vmin=C["V"]-7*C["S"], vmax=C["V"]+7*C["S"])
    v = cfg.build_velocity_grid()
    bt = np.maximum(gauss_hermite_model(v, 1., C["V"], C["S"], C["h3"], C["h4"]), 0.); bt /= bt.sum()
    A = build_template_stack(wave, tw, tmpl, v)
    g0 = sum(W[k]*(A[k] @ bt) for k in range(len(tmpl)))
    nl = np.median(g0)/C["snr"]
    btn = norm(bt, v); pk = btn.max(); core = btn > 0.01*pk

    cov_raw, cov_deb, bias_pk, resid_raw, resid_deb = [], [], [], [], []
    for sd in range(N_SEED):
        gal = g0 + np.random.default_rng(sd).normal(0, nl, len(wave))
        e = np.full(len(wave), nl)
        bo = bootstrap_losvd(A/e[None,:,None], gal/e, cfg, n_boot=N_BOOT, seed=100+sd)
        b0 = norm(bo["result0"]["b"], v)
        bb = np.array([norm(x, v) for x in bo["b_boot"]])
        lo_s, hi_s = np.percentile(bb, 16, axis=0), np.percentile(bb, 84, axis=0)

        # raw band, centred on the point estimate
        cov_raw.append(np.mean((btn[core] >= lo_s[core]) & (btn[core] <= hi_s[core])))
        resid_raw.append((np.abs(b0-btn)/pk*100)[core].max())

        # bias-corrected + widened
        bias = np.median(bb, axis=0) - b0
        b_deb = b0 - bias
        half_lo = np.sqrt((b0-lo_s)**2 + bias**2)
        half_hi = np.sqrt((hi_s-b0)**2 + bias**2)
        lo_t, hi_t = b_deb-half_lo, b_deb+half_hi
        cov_deb.append(np.mean((btn[core] >= lo_t[core]) & (btn[core] <= hi_t[core])))
        resid_deb.append((np.abs(b_deb-btn)/pk*100)[core].max())
        bias_pk.append(abs(bias[np.argmax(btn)])/pk*100)

    print(f"\n=== {name} ===  ({N_SEED} seeds x {N_BOOT} bootstrap)")
    print(f"  bootstrap bias estimate at the peak : {np.mean(bias_pk):.2f}% of peak")
    print(f"  max |resid|   raw point estimate    : {np.mean(resid_raw):6.2f}%")
    print(f"  max |resid|   bias-corrected        : {np.mean(resid_deb):6.2f}%   "
          f"({'better' if np.mean(resid_deb)<np.mean(resid_raw) else 'WORSE -- overcorrects'})")
    print(f"  COVERAGE of truth, raw 16-84 band   : {100*np.mean(cov_raw):5.1f}% of core bins")
    print(f"  COVERAGE, debiased + widened band   : {100*np.mean(cov_deb):5.1f}% of core bins")
