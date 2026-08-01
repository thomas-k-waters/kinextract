"""Reruns the two catastrophically-failing coverage cells from
`coverage_calibration_test.py` (sigma=50, h4_true=0.15, S/N in {40,100}),
using the h4=0.15-matched calibration table built by
`calibrate_bias_h4wing.py`, instead of the shipped h4=0 default.

**Column-semantics fix, found while building the table.** `calibrate_bias.
save_calibration_table` writes raw mean recovered h4 as `h4_mean`; `bias_
correction.load_calibration_table` treats that column directly as the
bias (`h4_bias = h4_mean`) -- correct only when the calibration's own
`h4_true=0`, since bias is `recovered - true`. At `h4_true=0.15`, the raw
`h4_mean` column understates the real bias by exactly 0.15 (e.g.
h4_mean=+0.0509 at sigma=50/S-N=40 is really a bias of 0.0509-0.15=
-0.0991, a real suppression, not the small positive number it looks like
if loaded naively). This script builds the `BiasCalibration` directly
with that correction applied, rather than reusing `load_calibration_table`
unmodified.

Same seeds, same mixture (seed=20000), same mocks as the original
overnight sweep -- this isolates the calibration-table choice as the only
variable.

Usage: python rerun_coverage_with_h4wing_calibration.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import FitConfig, build_template_stack, gauss_hermite_model, bootstrap_losvd
from kinextract.bias_correction import BiasCalibration

MUSE_TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WAVELENGTH = 4750.0 + np.arange(3681) * 1.25
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
MIXTURE_SEED = 20000

SIGMA = 50.0
H4_TRUE = 0.15
SNR_GRID = [40.0, 100.0]
N_OUTER = 30
N_BOOT_INNER = 15

H4WING_TABLE_PATH = os.path.join(os.path.dirname(__file__), "regularization_bias_h4wing_muse.txt")


def _load_h4wing_calibration_corrected(path, h4_true):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.split()[0] == "sigma":
                continue
            parts = line.split()
            sigma_true, snr, _n, _vmean, _vstd, sig_bias_mean, sig_bias_std = (float(parts[i]) for i in range(7))
            h4_mean_raw, h4_std = float(parts[8]), float(parts[9])
            rows.append((sigma_true, snr, sig_bias_mean, sig_bias_std, h4_mean_raw, h4_std))
    arr = np.array(rows)
    sigma_true, snr, sig_bias_mean, sig_bias_std, h4_mean_raw, h4_std = arr.T
    recovered_sigma = sigma_true + sig_bias_mean
    h4_bias_corrected = h4_mean_raw - h4_true  # THE FIX: bias = recovered - true, not recovered alone
    return BiasCalibration(
        recovered_sigma=recovered_sigma, snr=snr,
        sig_bias=sig_bias_mean, sig_bias_std=sig_bias_std,
        h4_bias=h4_bias_corrected, h4_bias_std=h4_std,
        sigma_range=(float(recovered_sigma.min()), float(recovered_sigma.max())),
        snr_range=(float(snr.min()), float(snr.max())),
    )


def _load_muse_templates():
    with open(os.path.join(MUSE_TEMPLATE_DIR, "Tlist")) as f:
        names = [line.strip() for line in f if line.strip()]
    templates = []
    for name in names:
        d = np.loadtxt(os.path.join(MUSE_TEMPLATE_DIR, name))
        templates.append(np.interp(WAVELENGTH, d[:, 0], d[:, 1]))
    return templates


def _true_moments(sigma, h4_true, v_true=0.0):
    vv = np.linspace(v_true - 15 * sigma, v_true + 15 * sigma, 200_001)
    bb = np.maximum(gauss_hermite_model(vv, 1.0, v_true, sigma, 0.0, h4_true), 0.0)
    tot = np.trapezoid(bb, vv)
    true_v = float(np.trapezoid(bb * vv, vv) / tot)
    true_s = float(np.sqrt(np.trapezoid(bb * (vv - true_v) ** 2, vv) / tot))
    return true_v, true_s


def run():
    calib_h4wing = _load_h4wing_calibration_corrected(H4WING_TABLE_PATH, H4_TRUE)
    print("h4wing calibration h4_bias range:", calib_h4wing.h4_bias.min(), "to", calib_h4wing.h4_bias.max())

    templates = _load_muse_templates()
    rng_mix = np.random.default_rng(MIXTURE_SEED)
    true_weights = rng_mix.dirichlet(np.ones(len(templates)))
    mask = (WAVELENGTH >= WAVEFITMIN) & (WAVELENGTH <= WAVEFITMAX)
    wave_fit = WAVELENGTH[mask]

    cfg = FitConfig(vmin=-7 * SIGMA, vmax=7 * SIGMA)
    v_grid = cfg.build_velocity_grid()
    b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, 0.0, SIGMA, 0.0, H4_TRUE), 0.0)
    b_truth /= b_truth.sum()
    A_stack = build_template_stack(wave_fit, WAVELENGTH, templates, v_grid)
    true_v, true_s = _true_moments(SIGMA, H4_TRUE)

    for snr in SNR_GRID:
        for label, calib in (("shipped (h4=0)", None), ("h4wing (h4=0.15, corrected)", calib_h4wing)):
            sig_in_1sig, h4_in_1sig, n_valid = 0, 0, 0
            h4_biases = []
            for rep in range(N_OUTER):
                # IDENTICAL seeds to coverage_calibration_test.py's own h4_true=0.15 cells
                seed = 3_000_000 + rep + int(SIGMA * 10) + int(snr * 100) + int(H4_TRUE * 10000)
                galaxy_noiseless = sum(true_weights[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))
                noise_level = np.median(galaxy_noiseless) / snr
                rng = np.random.default_rng(seed)
                galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
                errors = np.full(len(wave_fit), noise_level)
                A_stack_w = A_stack / errors[None, :, None]
                y_w = galaxy / errors

                boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=N_BOOT_INNER, seed=seed + 1,
                                        apply_bias_correction=True, bias_calibration=calib)
                s_est = boot["point0_corrected"]["sherm"]
                s_std = boot["sherm_std_corrected"]
                h4_est = boot["point0_corrected"]["h4"]
                h4_std = boot["h4_std_corrected"]
                if not (np.isfinite(s_est) and np.isfinite(s_std) and s_std > 0
                        and np.isfinite(h4_est) and np.isfinite(h4_std) and h4_std > 0):
                    continue
                n_valid += 1
                sig_in_1sig += abs(s_est - true_s) < 1.0 * s_std
                h4_in_1sig += abs(h4_est - H4_TRUE) < 1.0 * h4_std
                h4_biases.append(h4_est - H4_TRUE)

            print(f"[snr={snr:5.0f}][{label:>28s}] n={n_valid:3d}  "
                  f"sig_cov68={100*sig_in_1sig/n_valid:5.1f}%  h4_cov68={100*h4_in_1sig/n_valid:5.1f}%  "
                  f"h4_bias_mean={np.mean(h4_biases):+.4f}", flush=True)


if __name__ == "__main__":
    run()
    print("\nreference from the original overnight sweep (shipped h4=0 calibration):")
    print("  snr=40:  sig_cov68=53.3%  h4_cov68=0.0%")
    print("  snr=100: sig_cov68=63.3%  h4_cov68=3.3%")
