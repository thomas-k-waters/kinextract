"""Priority-1, step 2: test each candidate cause of the real-data sigma
inflation on the nuclear (high-S/N) NGC 5102 bin, where legacy Fortran
gets ~54 km/s and kinextract gets ~80-95.

Established already (diag01): penalty_order 2 vs 3 makes no difference
(79.7 vs 81.4), so this is not the new penalty. chi2_red ~ 16 with
smoothly-structured residuals on 50-100 A scales -- a continuum/template
mismatch signature, not noise.

Candidates tested here, each in isolation and on identical data:
  A  baseline (as the notebook does it today)
  B  + continuum cofitting (FitConfig.continuum_knots)
  C  + continuum-normalized data AND templates (divide both by a smooth
       spline through their own continua -- what OG kinextract's `.norm`
       file convention does before fitting at all)
  D  + aggressive sky masking (auto-detected positive outliers)
  E  C + D together
  F  parametric 5-param Gauss-Hermite fit (no LOSVD spline at all) --
     isolates "is this the non-parametric machinery, or the data/model?"
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import median_filter

from kinextract import (FitConfig, build_template_stack, compute_losvd_moments,
                             mask_wavelength_windows, load_emiles_templates, gauss_hermite_model)
from kinextract.multi_template import fit_losvd
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

LABEL = sys.argv[1] if len(sys.argv) > 1 else "high"

def base_data(label):
    wr = d[f"wave_obs_{label}"]/(1.0+Z)
    m = (wr>=8400)&(wr<=8800)
    return wr[m], d[f"flux_obs_{label}"][m], d[f"error_obs_{label}"][m]

def smooth_continuum(wave, flux, frac=0.12):
    """Robust smooth continuum: wide running max-ish (upper envelope) via
    a large median filter of the upper half, then a low-order polynomial.
    Absorption-line spectra sit below their continuum, so a plain median
    filter biases low -- use a high percentile filter instead."""
    n = max(9, int(frac*len(wave))//2*2+1)
    from scipy.ndimage import percentile_filter
    env = percentile_filter(flux, 85, size=n, mode='nearest')
    c = np.polyfit(wave - wave.mean(), env, 5)
    return np.polyval(c, wave - wave.mean())

def auto_sky_mask(wave, flux, nsig=3.0):
    """Flag positive outliers vs a local continuum -- sky emission residuals."""
    cont = smooth_continuum(wave, flux)
    r = flux - cont
    s = 1.4826*np.median(np.abs(r - np.median(r)))
    bad = r > nsig*s
    # grow by 1 pixel each side
    bad = bad | np.roll(bad,1) | np.roll(bad,-1)
    return ~bad

def run_fit(wave, flux, err, tmpl, tw, label, penalty_order=3, continuum_knots=None):
    kw = dict(vmin=-400., vmax=400., penalty_order=penalty_order)
    if continuum_knots is not None:
        kw.update(continuum_knots=continuum_knots, lam_selection='fixed', lam=2.0, penalty_order=2)
    cfg = FitConfig(**kw)
    vg = cfg.build_velocity_grid()
    A = build_template_stack(wave, tw, tmpl, vg)
    Aw = A/err[None,:,None]; yw = flux/err
    r = fit_losvd(Aw, yw, cfg, wave_fit=wave) if continuum_knots else fit_losvd(Aw, yw, cfg)
    mom = compute_losvd_moments(vg, r['b'], fit_h3h4=True)
    Aeff = np.tensordot(r['w'], Aw, axes=([0],[0]))
    model = (r['C']*(Aeff@r['b'])) if continuum_knots else (Aeff@r['b'])
    chi2red = float(np.sum((model-yw)**2))/len(wave)
    print(f"  {label:52s} V={mom['vherm']:+7.1f}  sigma={mom['sherm']:6.1f}  "
          f"h3={mom['h3']:+.3f} h4={mom['h4']:+.3f}  chi2red={chi2red:8.1f}  npix={len(wave)}")
    return mom

def parametric_gh(wave, flux, err, tmpl, tw):
    """5-param GH fit through the same design matrix + free template weights
    via NNLS at each step -- the closest thing to a classical pPXF fit."""
    from scipy.optimize import nnls
    cfg = FitConfig(vmin=-400., vmax=400.)
    vg = cfg.build_velocity_grid()
    A = build_template_stack(wave, tw, tmpl, vg)
    Aw = A/err[None,:,None]; yw = flux/err
    def resid(p):
        V, logs, h3, h4 = p
        b = np.maximum(gauss_hermite_model(vg, 1.0, V, np.exp(logs), h3, h4), 0.0)
        s = b.sum()
        if s <= 0: return np.full(len(yw), 1e6)
        b = b/s
        design = np.stack([Aw[k]@b for k in range(len(tmpl))], axis=1)
        w, _ = nnls(design, yw)
        return design@w - yw
    best = None
    for V0, s0 in [(0., 60.), (0., 120.), (-50., 80.)]:
        r = least_squares(resid, [V0, np.log(s0), 0., 0.],
                          bounds=([-400., np.log(10.), -0.3, -0.3],[400., np.log(400.), 0.3, 0.3]),
                          max_nfev=4000)
        if best is None or r.cost < best.cost: best = r
    V, logs, h3, h4 = best.x
    chi2red = 2*best.cost/len(yw)
    print(f"  {'F  parametric GH (5-param, NNLS weights)':52s} V={V:+7.1f}  sigma={np.exp(logs):6.1f}  "
          f"h3={h3:+.3f} h4={h4:+.3f}  chi2red={chi2red:8.1f}  npix={len(yw)}")

wave0, flux0, err0 = base_data(LABEL)
wA, fA = mask_wavelength_windows(wave0, flux0, windows=SKY)
_,  eA = mask_wavelength_windows(wave0, err0, windows=SKY)

print(f"\n===== {LABEL} S/N aperture (legacy Fortran: ~54 km/s for the nucleus) =====")
run_fit(wA, fA, eA, templates, template_wave, "A  baseline (notebook's current recipe)")
run_fit(wA, fA, eA, templates, template_wave, "B  + continuum cofitting (continuum_knots=6)", continuum_knots=6)

# C: continuum-normalize data and templates
cont_d = smooth_continuum(wA, fA)
fC, eC = fA/cont_d, eA/cont_d
tmask = (template_wave>=8300)&(template_wave<=8900)
tw_c = template_wave[tmask]
tmpl_c = [t[tmask]/smooth_continuum(tw_c, t[tmask]) for t in templates]
run_fit(wA, fC, eC, tmpl_c, tw_c, "C  continuum-normalized data AND templates")

# D: aggressive auto sky mask
keep = auto_sky_mask(wA, fA)
print(f"  [auto sky mask keeps {keep.sum()}/{len(keep)} px, drops {(~keep).sum()}]")
run_fit(wA[keep], fA[keep], eA[keep], templates, template_wave, "D  + auto sky mask (3sig positive outliers)")

# E: C + D
keepE = auto_sky_mask(wA, fA)
run_fit(wA[keepE], fC[keepE], eC[keepE], tmpl_c, tw_c, "E  continuum-normalized + auto sky mask")

parametric_gh(wA, fA, eA, templates, template_wave)
