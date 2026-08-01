"""Fits all 100 NGC 5102 bins with kinextract's multi-template LOSVD
recovery against the legacy sco_framework Fortran pipeline's own template
library and .norm spectra -- point estimates only (no bootstrap; ~15-20 min
for all 100 bins), with the Pinkney-style regularization-bias correction
(`kinextract.bias_correction`) applied to sigma/h4 via
`compute_losvd_moments(..., regularization_bias_correction=True)`, using an
S/N estimated from each bin's own best-fit model
(`bias_correction.estimate_snr_whitened`). See 03_real_data_ngc5102.ipynb's
own markdown for the methodology notes (deredshift, noise convention,
template library, velocity grid). Run from this directory:
`python fit_all_bins.py`. Paths below are this machine's actual
external-data locations, not shipped with the repo.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from kinextract import FitConfig, build_template_stack, compute_losvd_moments, mask_wavelength_windows
from kinextract.multi_template import fit_losvd
from kinextract.bias_correction import estimate_snr_whitened

DATA_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/normalizing_flows/data/N5102/kinematics/muse/kinematic_extraction"
TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
OUT_PATH = os.path.join(os.path.dirname(__file__), os.environ.get("FIT_ALL_BINS_OUT", "real_data_all_bins_masked.npz"))
Z = 0.0016181
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
NSIG = 7

# VALIDATION_CAMPAIGN.md Stage 4: four fixed, recurring severely-deviant
# wavelength windows found in every bin checked, growing worse toward
# faint/outer bins, not near CaII/Paschen -- most consistent with
# imperfect sky-subtraction (OH-airglow residuals). N5102-specific, not
# part of the general library -- see mask_wavelength_windows's own
# docstring. +-1 A padding around each originally-identified worst pixel.
SKY_RESIDUAL_WINDOWS = [
    (8414.0, 8419.0), (8449.0, 8454.0), (8711.0, 8714.0), (8748.5, 8753.0),
]

# --- Load pallmc.out reference table ---
with open(os.path.join(DATA_DIR, "pallmc.out")) as f:
    rows = [l.split() for l in f if l.strip()]
bins = []
for r in rows:
    radius, V, Verr, sig, sigerr, h3, h3err, h4, h4err, simname = r[:10]
    binname = simname.replace("spmc.sim", "")
    bins.append(dict(binname=binname, radius=float(radius), V=float(V), Verr=float(Verr),
                      sig=float(sig), sigerr=float(sigerr), h3=float(h3), h3err=float(h3err),
                      h4=float(h4), h4err=float(h4err)))
print(f"{len(bins)} bins loaded from pallmc.out")

# --- Load template library once ---
with open(os.path.join(TEMPLATE_DIR, "Tlist")) as f:
    tnames = [line.strip() for line in f if line.strip()]
raw = [np.loadtxt(os.path.join(TEMPLATE_DIR, name)) for name in tnames]
template_wave = raw[0][:, 0]
template_fluxes = [np.interp(template_wave, td[:, 0], td[:, 1]) for td in raw]
print(f"loaded {len(template_fluxes)} templates")

results = {}
t0 = time.time()
for i, b in enumerate(bins):
    binname = b["binname"]
    norm_path = os.path.join(DATA_DIR, f"{binname}sp.norm")
    rms_path = os.path.join(DATA_DIR, f"{binname}sp.rms")
    try:
        d = np.loadtxt(norm_path)
        wave_obs, flux_obs = d[:, 0], d[:, 1]
        wave_rest = wave_obs / (1.0 + Z)
        mask = (wave_rest >= WAVEFITMIN) & (wave_rest <= WAVEFITMAX)
        wave_fit = wave_rest[mask]
        flux_fit = flux_obs[mask]
        noise_level = float(np.loadtxt(rms_path)[0])
        wave_fit, flux_fit = mask_wavelength_windows(wave_fit, flux_fit, windows=SKY_RESIDUAL_WINDOWS)

        vmin, vmax = b["V"] - NSIG * b["sig"], b["V"] + NSIG * b["sig"]
        cfg = FitConfig(vmin=vmin, vmax=vmax)
        v_grid = cfg.build_velocity_grid()
        A_stack = build_template_stack(wave_fit, template_wave, template_fluxes, v_grid)
        errors = np.full(len(wave_fit), noise_level)
        A_stack_w = A_stack / errors[None, :, None]
        y_w = flux_fit / errors

        result = fit_losvd(A_stack_w, y_w, cfg)
        A_eff_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0]))
        model_w = A_eff_w @ result["b"]
        snr_est = estimate_snr_whitened(model_w)
        moments = compute_losvd_moments(v_grid, result["b"])
        moments_corr = compute_losvd_moments(v_grid, result["b"], regularization_bias_correction=True,
                                              snr=snr_est)
        results[binname] = dict(
            v_grid=v_grid, b=result["b"], w=result["w"],
            vherm=moments["vherm"], sherm=moments["sherm"], h3=moments["h3"], h4=moments["h4"],
            sherm_corrected=moments_corr["sherm"], h4_corrected=moments_corr["h4"],
            sig_uncertainty_inflation=moments_corr["sig_uncertainty_inflation"],
            h4_uncertainty_inflation=moments_corr["h4_uncertainty_inflation"],
            snr_estimated=snr_est,
            success=moments["success"],
            radius=b["radius"], V_legacy=b["V"], Verr_legacy=b["Verr"],
            sig_legacy=b["sig"], sigerr_legacy=b["sigerr"],
            h3_legacy=b["h3"], h3err_legacy=b["h3err"], h4_legacy=b["h4"], h4err_legacy=b["h4err"],
        )
        print(f"[{i+1:3d}/{len(bins)}] {binname:10s} r={b['radius']:+7.2f}  "
              f"V={moments['vherm']:+7.2f} (legacy {b['V']:+7.2f})  "
              f"sig={moments['sherm']:6.2f} corr={moments_corr['sherm']:6.2f} (legacy {b['sig']:6.2f})  "
              f"S/N={snr_est:5.0f}  t={time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"[{i+1:3d}/{len(bins)}] {binname:10s} FAILED: {e}", flush=True)

np.savez(OUT_PATH, results=results, bins=bins)
print(f"\nDone. {len(results)}/{len(bins)} bins fit successfully. Total time: {time.time()-t0:.0f}s")
print(f"Saved to {OUT_PATH}")
