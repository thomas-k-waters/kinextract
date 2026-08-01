"""Quantify shift_template's linear-interpolation error directly, with no
fitting involved -- isolates the interpolation mechanism itself from any
downstream regularization/noise effects.

Mechanism: shift_template evaluates a uniformly-sampled template at a
Doppler-shifted wavelength grid via np.interp (piecewise linear). Linear
interpolation of a band-limited signal is not itself band-limited -- its
error is periodic in the FRACTIONAL pixel shift, with period equal to the
template's own native sampling. At ~8500 Angstrom with a 0.9 Angstrom/pix
E-MILES native grid, one native pixel corresponds to a velocity increment
of c*dlambda/lambda = 3e5*0.9/8500 ~ 31.8 km/s -- matching this project's
own earlier finding of a ~31 km/s-periodic effect.

Ground truth for "the exact shift" is a highly-oversampled cubic-spline
resampling of the template (10x the native sampling), itself then linearly
interpolated at the EXACT same query points -- at 10x oversampling, linear
interpolation error is negligible (<0.1% typically), so this is a valid
proxy for the true continuous signal without needing an analytic template.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
from scipy.interpolate import CubicSpline
from kinextract import load_emiles_templates
from kinextract.forward_model import shift_template, C_KMS

EM = "src/kinextract/template_data/emiles_muse_grid.npz"
T, tw, *_ = load_emiles_templates(EM, select=[(5.0, 0.06)])
template = T[0]

# restrict to the CaT fit window's rest-frame neighborhood for realism
mask = (tw >= 8300) & (tw <= 8900)
tw_w, temp_w = tw[mask], template[mask]

wave_obs = np.arange(8400.0, 8800.0, 1.25)  # MUSE-like observed grid

# oversampled ground truth: cubic spline the template at 20x density, matching
# what an (idealized) continuous, band-limited signal's shift would look like
cs = CubicSpline(tw_w, temp_w)
tw_fine = np.arange(tw_w[0], tw_w[-1], 0.9 / 20)
temp_fine = cs(tw_fine)


def shift_true(v):
    lam_rest = wave_obs / (1.0 + v / C_KMS)
    return np.interp(lam_rest, tw_fine, temp_fine, left=0.0, right=0.0)


print("velocity(km/s)  frac_pixel_phase   max|linear-true|/max(true)  rms|linear-true|/rms(true)")
one_pixel_kms = C_KMS * 0.9 / 8500.0
print(f"(one native pixel ~ {one_pixel_kms:.2f} km/s at 8500 A)\n")
velocities = np.linspace(0.0, 2 * one_pixel_kms, 21)
errs = []
for v in velocities:
    linear = shift_template(wave_obs, tw_w, temp_w, v)
    true = shift_true(v)
    phase = (v % one_pixel_kms) / one_pixel_kms
    denom_max = np.abs(true).max()
    denom_rms = np.sqrt((true ** 2).mean())
    err = linear - true
    max_rel = np.abs(err).max() / denom_max * 100
    rms_rel = np.sqrt((err ** 2).mean()) / denom_rms * 100
    errs.append((v, phase, max_rel, rms_rel))
    print(f"{v:14.2f}  {phase:16.3f}  {max_rel:24.3f}%  {rms_rel:22.3f}%")

max_over_scan = max(e[2] for e in errs)
print(f"\npeak max-relative interpolation error over one full pixel period: {max_over_scan:.2f}%")


# ---------------------------------------------------------------------
# Candidate fix: Lanczos-windowed sinc resampling (band-limited, template_wave
# must be uniformly sampled -- confirmed true for the E-MILES grid).
# ---------------------------------------------------------------------
def _lanczos_kernel(x, a):
    x = np.asarray(x, dtype=float)
    out = np.sinc(x) * np.sinc(x / a)
    out[np.abs(x) >= a] = 0.0
    return out


def shift_template_lanczos(wave_obs, template_wave, template_flux, v_kms, a=3):
    dx = template_wave[1] - template_wave[0]
    lam_rest = wave_obs / (1.0 + v_kms / C_KMS)
    idx = (lam_rest - template_wave[0]) / dx
    idx_floor = np.floor(idx).astype(int)
    n = len(template_wave)
    out = np.zeros_like(lam_rest)
    for t in range(-a + 1, a + 1):
        k = idx_floor + t
        valid = (k >= 0) & (k < n)
        w = _lanczos_kernel(idx - k, a)
        out[valid] += template_flux[k[valid]] * w[valid]
    out_of_range = (lam_rest < template_wave[0]) | (lam_rest > template_wave[-1])
    out[out_of_range] = 0.0
    return out


print("\n\n=== Lanczos (a=3) vs linear, same scan ===")
print("velocity(km/s)  frac_phase   linear max-rel%   lanczos max-rel%   lanczos rms-rel%")
lin_errs, lz_errs = [], []
for v in velocities:
    linear = shift_template(wave_obs, tw_w, temp_w, v)
    lz = shift_template_lanczos(wave_obs, tw_w, temp_w, v, a=3)
    true = shift_true(v)
    phase = (v % one_pixel_kms) / one_pixel_kms
    denom_max = np.abs(true).max()
    denom_rms = np.sqrt((true ** 2).mean())
    lin_max = np.abs(linear - true).max() / denom_max * 100
    lz_max = np.abs(lz - true).max() / denom_max * 100
    lz_rms = np.sqrt(((lz - true) ** 2).mean()) / denom_rms * 100
    lin_errs.append(lin_max); lz_errs.append(lz_max)
    print(f"{v:14.2f}  {phase:10.3f}  {lin_max:16.3f}%  {lz_max:16.3f}%  {lz_rms:16.4f}%")

print(f"\npeak max-rel error: linear={max(lin_errs):.3f}%  lanczos(a=3)={max(lz_errs):.4f}%")
print(f"reduction factor: {max(lin_errs)/max(lz_errs):.1f}x")
