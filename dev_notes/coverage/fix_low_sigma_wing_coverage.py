"""Targeted follow-up to `coverage_calibration_test.py`'s worst finding:
at sigma=50, h4_true=0.15, the shipped default (lam=2.0 + Pinkney bias
correction) gave 0.0% and 3.3% empirical coverage of the true h4 at the
nominal 68% level -- not a bias problem, a confidently-wrong problem.

**Hypothesis, informed by this project's own ablation ladder** (the
unregularized configuration beats lam=2.0 dramatically at low sigma,
independent of template mismatch): lam=2.0 is the wrong regularization
strength at sigma=50 regardless of h4, and the shipped bias-correction
table was calibrated *for* lam=2.0's bias pattern -- applying it after
switching to a different lam does not make sense, so this compares raw
(uncorrected) bootstrap output at lam=0.0 against the current shipped
default, on the *same* mocks that just failed.

Usage: python fix_low_sigma_wing_coverage.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import FitConfig, build_template_stack, gauss_hermite_model, bootstrap_losvd

MUSE_TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WAVELENGTH = 4750.0 + np.arange(3681) * 1.25
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
MIXTURE_SEED = 20000

SIGMA = 50.0
H4_TRUE = 0.15
SNR_GRID = [40.0, 100.0]
N_OUTER = 30
N_BOOT_INNER = 15
LAM_CANDIDATES = [0.0, 0.5, 1.0]  # compare against shipped lam=2.0 (already measured: 0%/3.3% coverage)


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
    templates = _load_muse_templates()
    print(f"loaded {len(templates)} MUSE templates")
    rng_mix = np.random.default_rng(MIXTURE_SEED)
    true_weights = rng_mix.dirichlet(np.ones(len(templates)))

    mask = (WAVELENGTH >= WAVEFITMIN) & (WAVELENGTH <= WAVEFITMAX)
    wave_fit = WAVELENGTH[mask]

    true_v, true_s = _true_moments(SIGMA, H4_TRUE)
    v_grid_ref = FitConfig(vmin=-7 * SIGMA, vmax=7 * SIGMA).build_velocity_grid()
    b_truth = np.maximum(gauss_hermite_model(v_grid_ref, 1.0, 0.0, SIGMA, 0.0, H4_TRUE), 0.0)
    b_truth /= b_truth.sum()
    A_stack = build_template_stack(wave_fit, WAVELENGTH, templates, v_grid_ref)

    rows = []
    t0 = time.time()
    for snr in SNR_GRID:
        for lam in LAM_CANDIDATES:
            cfg = FitConfig(vmin=-7 * SIGMA, vmax=7 * SIGMA, lam=lam)
            sig_in_1sig, h4_in_1sig, h4_bias_list, sig_bias_list = 0, 0, [], []
            n_valid = 0
            for rep in range(N_OUTER):
                # identical noise draws to coverage_calibration_test.py's own seeding,
                # so this is a fair, directly comparable re-run of the same mocks
                seed = 3_000_000 + rep + int(SIGMA * 10) + int(snr * 100) + int(H4_TRUE * 10000)
                galaxy_noiseless = sum(true_weights[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))
                noise_level = np.median(galaxy_noiseless) / snr
                rng = np.random.default_rng(seed)
                galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
                errors = np.full(len(wave_fit), noise_level)
                A_stack_w = A_stack / errors[None, :, None]
                y_w = galaxy / errors

                try:
                    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=N_BOOT_INNER, seed=seed + 1,
                                            apply_bias_correction=False)
                except Exception as exc:
                    print(f"  (skip rep {rep}, exception: {exc})", flush=True)
                    continue
                s_est = boot["point0"]["sherm"]
                s_std = boot["sherm_std"]
                h4_est = boot["point0"]["h4"]
                h4_std = boot["h4_std"]
                if not (np.isfinite(s_est) and np.isfinite(s_std) and s_std > 0
                        and np.isfinite(h4_est) and np.isfinite(h4_std) and h4_std > 0):
                    continue
                n_valid += 1
                sig_in_1sig += abs(s_est - true_s) < 1.0 * s_std
                h4_in_1sig += abs(h4_est - H4_TRUE) < 1.0 * h4_std
                h4_bias_list.append(h4_est - H4_TRUE)
                sig_bias_list.append(s_est - true_s)

            if n_valid < 5:
                print(f"  (skip snr={snr} lam={lam}: only {n_valid} valid reps)")
                continue
            row = dict(
                snr=snr, lam=lam, n=n_valid,
                sig_cov68=sig_in_1sig / n_valid, h4_cov68=h4_in_1sig / n_valid,
                h4_bias_mean=float(np.mean(h4_bias_list)), sig_bias_mean=float(np.mean(sig_bias_list)),
            )
            rows.append(row)
            print(f"[snr={snr:5.0f}][lam={lam:4.1f}] n={n_valid:3d}  "
                  f"sig_cov68={row['sig_cov68']*100:5.1f}%  h4_cov68={row['h4_cov68']*100:5.1f}%  "
                  f"sig_bias={row['sig_bias_mean']:+7.2f}  h4_bias={row['h4_bias_mean']:+.4f}  "
                  f"t={time.time()-t0:.0f}s", flush=True)
    return rows


if __name__ == "__main__":
    rows = run()
    print(f"\n{'='*60}\ndone, {len(rows)} rows\n{'='*60}")
    print("reference (from the overnight sweep, lam=2.0 + bias correction):")
    print("  snr=40:  sig_cov68=53.3%  h4_cov68=0.0%")
    print("  snr=100: sig_cov68=63.3%  h4_cov68=3.3%")
