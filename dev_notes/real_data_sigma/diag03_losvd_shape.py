"""Priority-1, step 3: diag02 showed a parametric GH fit gets sigma=42
on the SAME data/templates where the non-parametric spline gets 81, at
comparable chi2. So look at the recovered curve itself (this project's
own rule) -- what shape is the spline landing on?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, nnls

from kinextract import (FitConfig, build_template_stack, compute_losvd_moments,
                             mask_wavelength_windows, load_emiles_templates, gauss_hermite_model)
from kinextract.multi_template import fit_losvd
from kinextract.gh_moments import compute_direct_moments
from kinextract.template_reduction import load_template_subset

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
SKY = [(8414.0,8419.0),(8449.0,8454.0),(8711.0,8714.0),(8748.5,8753.0)]
d = np.load(os.path.join(ROOT,"examples/data/muse/ngc5102_representative_bins.npz"))
Z = float(d["z"])
subset = load_template_subset(os.path.join(ROOT,"examples/data/muse/ngc5102_template_subset.npz"))
ui = subset["union_selected"]
select = list(zip(subset["template_ages"][ui].tolist(), subset["template_metals"][ui].tolist()))
templates, template_wave, *_ = load_emiles_templates(
    os.path.join(ROOT,"src/kinextract/template_data/emiles_muse_grid.npz"), select=select)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, label in zip(axes, ["high","med","low"]):
    wr = d[f"wave_obs_{label}"]/(1.0+Z); m=(wr>=8400)&(wr<=8800)
    wave, flux, err = wr[m], d[f"flux_obs_{label}"][m], d[f"error_obs_{label}"][m]
    wave, flux = mask_wavelength_windows(wave, flux, windows=SKY)
    _, err = mask_wavelength_windows(wr[m], err, windows=SKY)

    cfg = FitConfig(vmin=-400., vmax=400.)
    vg = cfg.build_velocity_grid()
    A = build_template_stack(wave, template_wave, templates, vg)
    Aw = A/err[None,:,None]; yw = flux/err
    r = fit_losvd(Aw, yw, cfg)
    b = r['b']/np.trapezoid(np.maximum(r['b'],0), vg)

    # parametric comparison on the same data
    def resid(p):
        V, logs, h3, h4 = p
        bb = np.maximum(gauss_hermite_model(vg,1.,V,np.exp(logs),h3,h4),0.); s=bb.sum()
        if s<=0: return np.full(len(yw),1e6)
        bb/=s
        design = np.stack([Aw[k]@bb for k in range(len(templates))],axis=1)
        w,_ = nnls(design, yw); return design@w - yw
    best=None
    for V0,s0 in [(0.,60.),(0.,120.),(-50.,80.)]:
        rr = least_squares(resid,[V0,np.log(s0),0.,0.],
                           bounds=([-400.,np.log(10.),-.3,-.3],[400.,np.log(400.),.3,.3]),max_nfev=4000)
        if best is None or rr.cost<best.cost: best=rr
    Vp, logsp, h3p, h4p = best.x
    bp = np.maximum(gauss_hermite_model(vg,1.,Vp,np.exp(logsp),h3p,h4p),0.)
    bp /= np.trapezoid(bp, vg)

    mom = compute_direct_moments(vg, r['b'])
    # how much LOSVD mass sits outside |v-V| > 3*sigma_parametric ?
    far = np.abs(vg - Vp) > 3*np.exp(logsp)
    ped = np.trapezoid(np.where(far, b, 0.), vg)
    peak = b.max()
    print(f"{label}: spline sigma={mom['sherm']:.1f}  parametric sigma={np.exp(logsp):.1f}  "
          f"| mass beyond 3sig_param = {100*ped:.1f}%  | b at +-300 km/s = {b[np.abs(vg-Vp)>290].max()/peak*100:.2f}% of peak")

    ax.plot(vg, b, color='seagreen', lw=2.0, label=f"spline (sigma={mom['sherm']:.0f})")
    ax.plot(vg, bp, color='crimson', lw=1.8, ls='--', label=f"parametric GH (sigma={np.exp(logsp):.0f})")
    ax.set_yscale('log'); ax.set_ylim(peak*1e-5, peak*2)
    ax.axvspan(Vp-3*np.exp(logsp), Vp+3*np.exp(logsp), color='0.85', zorder=0, label='+-3 sigma_param')
    ax.set_title(f"{label} S/N", fontsize=11); ax.set_xlabel('v (km/s)')
    ax.legend(fontsize=8)
axes[0].set_ylabel('b(v), log scale')
plt.tight_layout()
out = os.path.join(HERE,"diag03_losvd_shapes.png"); plt.savefig(out, dpi=130, bbox_inches='tight')
print("wrote", out)
