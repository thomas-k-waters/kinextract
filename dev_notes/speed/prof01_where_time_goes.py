"""Speed regression: profile a single default fit_losvd on the
01_basic_mock_fit.ipynb case (3 E-MILES templates, S/N=60, sigma=50),
which the user measures at ~38 s -- far slower than pre-Stage-22.
Counts actual work done, not just wall time, so the fix targets the
right thing.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np

import kinextract
from kinextract import FitConfig, build_template_stack, gauss_hermite_model, load_emiles_templates
from kinextract.multi_template import fit_losvd
import kinextract.losvd as losvd_mod
import kinextract.multi_template as mt_mod
import kinextract.aic_regularization as aic_mod

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
EM = os.path.join(ROOT, "src/kinextract/template_data/emiles_muse_grid.npz")

templates, tw, *_ = load_emiles_templates(EM, select=[(1.0,-0.66),(5.0,0.06),(14.0,0.40)])
W = np.array([0.3,0.4,0.3])
obs = 4750.0 + np.arange(3681)*1.25
wave = obs[(obs>=8400)&(obs<=8800)]
TRUE_V, SIG, SNR = 20.0, 50.0, 60.0
cfg = FitConfig(vmin=TRUE_V-7*SIG, vmax=TRUE_V+7*SIG)
v = cfg.build_velocity_grid()
bt = np.maximum(gauss_hermite_model(v,1.,TRUE_V,SIG,0.,0.),0.); bt/=bt.sum()
A = build_template_stack(wave, tw, templates, v)
g0 = sum(W[k]*(A[k]@bt) for k in range(3))
nl = np.median(g0)/SNR
gal = g0 + np.random.default_rng(42).normal(0,nl,len(wave))
err = np.full(len(wave), nl)
Aw = A/err[None,:,None]; yw = gal/err

import jax
print(f"jax x64 enabled: {jax.config.jax_enable_x64}")
print(f"problem size: n_pix={len(wave)}  n_vbins={len(v)}  n_coef={cfg.resolved_n_interior_knots()+3}  "
      f"n_templates=3  lam_grid={cfg.resolved_lam_grid()}")

# --- instrument: count calls and time spent in each layer ---
counters = dict(fit_log_spline_map=0, lbfgs_runs=0, alternate_once=0, fit_losvd_basis=0,
                m_eff_local=0, lbfgs_time=0.0, objgrad_calls=0)

orig_flsm = losvd_mod.fit_log_spline_map
def counting_flsm(A_w, y_w, basis, D2, **kw):
    counters['fit_log_spline_map'] += 1
    counters['lbfgs_runs'] += 1 if kw.get('coeffs0') is not None else 3
    t0=time.perf_counter(); r = orig_flsm(A_w, y_w, basis, D2, **kw)
    counters['lbfgs_time'] += time.perf_counter()-t0
    return r
losvd_mod.fit_log_spline_map = counting_flsm
mt_mod.fit_log_spline_map = counting_flsm

orig_alt = mt_mod._alternate_once
def counting_alt(*a, **kw):
    counters['alternate_once'] += 1
    return orig_alt(*a, **kw)
mt_mod._alternate_once = counting_alt

orig_flb = mt_mod.fit_losvd_basis
def counting_flb(*a, **kw):
    counters['fit_losvd_basis'] += 1
    return orig_flb(*a, **kw)
mt_mod.fit_losvd_basis = counting_flb
aic_mod.fit_losvd_basis = getattr(aic_mod, 'fit_losvd_basis', counting_flb)

orig_meff = aic_mod.estimate_m_eff_local
def counting_meff(*a, **kw):
    counters['m_eff_local'] += 1
    return orig_meff(*a, **kw)
aic_mod.estimate_m_eff_local = counting_meff

t0 = time.perf_counter()
r = fit_losvd(Aw, yw, cfg)
total = time.perf_counter()-t0

print(f"\n--- ONE default fit_losvd: {total:.1f} s  (lam_used={r['lam_used']:g}) ---")
for k,val in counters.items():
    print(f"   {k:22s} {val:10.4g}")
print(f"\n   time inside fit_log_spline_map : {counters['lbfgs_time']:.1f} s "
      f"({100*counters['lbfgs_time']/total:.0f}% of total)")
print(f"   mean per fit_log_spline_map    : {counters['lbfgs_time']/max(counters['fit_log_spline_map'],1):.2f} s")
print(f"   => {counters['fit_losvd_basis']} full alternating fits "
      f"({len(cfg.resolved_lam_grid())} lam values x m_eff + 1 final refit)")
print(f"   => {counters['lbfgs_runs']} total L-BFGS-B optimizations")
