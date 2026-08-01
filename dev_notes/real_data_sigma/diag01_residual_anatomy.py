"""Priority-1 diagnostic: why does the real MUSE nuclear bin recover
sigma ~80-95 km/s when the legacy Fortran pipeline gets ~54 km/s?

First established fact (diag run, 2026-07-27): penalty_order 2 and 3 give
the SAME answer (79.7 vs 81.4), so this is NOT the new D3 penalty. And
chi2_red ~ 16 -- the forward model does not reproduce the data at all.
This script plots the fit and residuals to see WHAT the model is failing
to explain, rather than guessing.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kinextract import (FitConfig, build_template_stack, compute_losvd_moments,
                             mask_wavelength_windows, load_emiles_templates)
from kinextract.multi_template import fit_losvd
from kinextract.template_reduction import load_template_subset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
SKY = [(8414.0,8419.0),(8449.0,8454.0),(8711.0,8714.0),(8748.5,8753.0)]
CA = [8498.02, 8542.09, 8662.14]

d = np.load(os.path.join(ROOT,"examples/data/muse/ngc5102_representative_bins.npz"))
Z = float(d["z"])
subset = load_template_subset(os.path.join(ROOT,"examples/data/muse/ngc5102_template_subset.npz"))
ui = subset["union_selected"]
select = list(zip(subset["template_ages"][ui].tolist(), subset["template_metals"][ui].tolist()))
templates, template_wave, *_ = load_emiles_templates(
    os.path.join(ROOT,"src/kinextract/template_data/emiles_muse_grid.npz"), select=select)

def get(label, sky=True):
    wr = d[f"wave_obs_{label}"]/(1.0+Z)
    m = (wr>=8400)&(wr<=8800)
    wave, flux, err = wr[m], d[f"flux_obs_{label}"][m], d[f"error_obs_{label}"][m]
    if sky:
        wave2, flux2 = mask_wavelength_windows(wave, flux, windows=SKY)
        _, err2 = mask_wavelength_windows(wave, err, windows=SKY)
        return wave2, flux2, err2
    return wave, flux, err

fig, axes = plt.subplots(3, 1, figsize=(13, 10))
for ax, label in zip(axes, ["high","med","low"]):
    wave, flux, err = get(label)
    cfg = FitConfig(vmin=-400., vmax=400.)
    vg = cfg.build_velocity_grid()
    A = build_template_stack(wave, template_wave, templates, vg)
    Aw = A/err[None,:,None]; yw = flux/err
    r = fit_losvd(Aw, yw, cfg)
    mom = compute_losvd_moments(vg, r['b'], fit_h3h4=True)
    Aeff = np.tensordot(r['w'], Aw, axes=([0],[0]))
    model = Aeff @ r['b']          # whitened model
    resid = model - yw             # in sigma units
    chi2red = float(np.sum(resid**2))/len(wave)

    ax.plot(wave, flux, lw=0.9, color='black', label='data')
    ax.plot(wave, model*err, lw=1.3, color='tomato', label='model')
    for c in CA: ax.axvline(c, lw=0.7, color='green', ls=':', alpha=0.6)
    ax2 = ax.twinx()
    ax2.plot(wave, resid, lw=0.6, color='steelblue', alpha=0.6)
    ax2.axhline(0, lw=0.5, color='k'); ax2.set_ylabel('resid (sigma)', fontsize=8, color='steelblue')
    ax.set_title(f"{label}: V={mom['vherm']:+.1f} sigma={mom['sherm']:.1f} h3={mom['h3']:+.3f} "
                 f"h4={mom['h4']:+.3f}  chi2red={chi2red:.1f}  |resid|med={np.median(np.abs(resid)):.1f}sig",
                 fontsize=10)
    ax.set_ylabel('flux'); ax.legend(fontsize=8, loc='lower left')
    print(f"{label}: V={mom['vherm']:+7.1f} sig={mom['sherm']:6.1f} h4={mom['h4']:+.3f} "
          f"chi2red={chi2red:8.1f}  median|resid|={np.median(np.abs(resid)):.2f} sigma  "
          f"max|resid|={np.abs(resid).max():.0f} sigma")
plt.xlabel('wavelength (A, rest)')
plt.tight_layout()
out = os.path.join(HERE, "diag01_residuals.png")
plt.savefig(out, dpi=130, bbox_inches='tight')
print("wrote", out)
