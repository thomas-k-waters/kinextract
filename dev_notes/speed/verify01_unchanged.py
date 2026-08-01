"""Speed work must not change any answer. Compare against values recorded
BEFORE the warm-start / refit-reuse changes, on the same seeds and data.

Reference values (measured earlier this session, pre-change):
  mock 01 (3 E-MILES tmpl, V=20 sigma=50 S/N=60 seed 42):
        V=+18.690  sigma=50.929  lam_used=32768
  real NGC5102 nuclear (high), stage-1 grid, order 3:
        V=-7.0     sigma=81.4    lam_used=1
  real NGC5102 mid-radius (med):
        V=+16.2    sigma=97.5    lam_used=1
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from kinextract import (FitConfig, build_template_stack, gauss_hermite_model,
                             compute_losvd_moments, load_emiles_templates, mask_wavelength_windows)
from kinextract.multi_template import fit_losvd
from kinextract.template_reduction import load_template_subset

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
EM = os.path.join(ROOT, "src/kinextract/template_data/emiles_muse_grid.npz")
SKY = [(8414.,8419.),(8449.,8454.),(8711.,8714.),(8748.5,8753.)]
ok = True
def check(name, got, ref, tol):
    global ok
    good = abs(got-ref) <= tol
    ok &= good
    print(f"   {name:12s} got {got:+9.3f}   ref {ref:+9.3f}   {'OK' if good else '*** CHANGED ***'}")

# --- mock ---
templates, tw, *_ = load_emiles_templates(EM, select=[(1.0,-0.66),(5.0,0.06),(14.0,0.40)])
W = np.array([0.3,0.4,0.3])
obs = 4750.0+np.arange(3681)*1.25; wave = obs[(obs>=8400)&(obs<=8800)]
cfg = FitConfig(vmin=20-350., vmax=20+350.)
v = cfg.build_velocity_grid()
bt = np.maximum(gauss_hermite_model(v,1.,20.,50.,0.,0.),0.); bt/=bt.sum()
A = build_template_stack(wave, tw, templates, v)
g0 = sum(W[k]*(A[k]@bt) for k in range(3)); nl=np.median(g0)/60.
gal = g0+np.random.default_rng(42).normal(0,nl,len(wave)); err=np.full(len(wave),nl)
t0=time.perf_counter(); r = fit_losvd(A/err[None,:,None], gal/err, cfg); dt=time.perf_counter()-t0
m = compute_losvd_moments(v, r['b'], fit_h3h4=True)
btn = bt/np.trapezoid(bt,v); pk=btn.max(); core=btn>0.01*pk
bn = r['b']/np.trapezoid(np.maximum(r['b'],0),v)
print(f"mock 01  [{dt:.1f}s]  lam_used={r['lam_used']:g}  maxresid={(np.abs(bn-btn)/pk*100)[core].max():.2f}%")
check("V", m['vherm'], 18.690, 0.05); check("sigma", m['sherm'], 50.929, 0.05)

# --- real data ---
d = np.load(os.path.join(ROOT,"examples/data/muse/ngc5102_representative_bins.npz")); Z=float(d["z"])
sub = load_template_subset(os.path.join(ROOT,"examples/data/muse/ngc5102_template_subset.npz")); ui=sub["union_selected"]
sel = list(zip(sub["template_ages"][ui].tolist(), sub["template_metals"][ui].tolist()))
tmpl, tw2, *_ = load_emiles_templates(EM, select=sel)
for label, refV, refS in [("high",-7.0,81.4), ("med",16.2,97.5)]:
    wr = d[f"wave_obs_{label}"]/(1+Z); mk=(wr>=8400)&(wr<=8800)
    wv, fl = mask_wavelength_windows(wr[mk], d[f"flux_obs_{label}"][mk], windows=SKY)
    _, er = mask_wavelength_windows(wr[mk], d[f"error_obs_{label}"][mk], windows=SKY)
    cfgr = FitConfig(vmin=-400., vmax=400.)
    vr = cfgr.build_velocity_grid()
    Ar = build_template_stack(wv, tw2, tmpl, vr)
    t0=time.perf_counter(); rr = fit_losvd(Ar/er[None,:,None], fl/er, cfgr); dt=time.perf_counter()-t0
    mm = compute_losvd_moments(vr, rr['b'], fit_h3h4=True)
    print(f"real {label:5s} [{dt:.1f}s]  lam_used={rr['lam_used']:g}")
    check("V", mm['vherm'], refV, 0.15); check("sigma", mm['sherm'], refS, 0.15)

print("\nALL UNCHANGED" if ok else "\n*** REGRESSION: results differ ***")
