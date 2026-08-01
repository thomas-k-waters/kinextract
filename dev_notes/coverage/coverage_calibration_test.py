"""Tests whether kinextract's reported uncertainty is honest: does
the stated 68%/95% interval actually contain the true value 68%/95% of
the time, in repeated synthetic trials with known truth?

**Why this matters more than another bias number.** Every bias-correction
and regularization-selection effort this project has done asks "is the
point estimate close to truth." This asks a different, arguably more
fundamental question: "when the code says +/-X km/s, is that honest?" A
method can have a well-characterized point-estimate bias (which we
correct for) while *also* reporting a confidence interval that's too
narrow (undercoverage -- overconfident) or too wide (overcoverage --
wasteful) and nobody would know, because this specific check has never
been run. This is a basic frequentist calibration check (does a nominal
X% interval have empirical X% coverage), applied to `bootstrap_losvd`'s
own `sherm_std_corrected`/`h4_std_corrected` (the package's own
"recommended reported uncertainty").

**Design.** For each (sigma, S/N, h4_true) cell: draw `N_OUTER`
independent noisy mocks from known truth (real 33-star MUSE library, one
fixed representative mixture, matching every other synthetic test this
session). For each mock, run `bootstrap_losvd` (with `N_BOOT_INNER`
internal replicates) to get the bias-corrected point estimate and its
reported std. Check whether `|point - truth| < k*std` for k=1 (nominal
68%) and k=2 (nominal 95%), for both sigma and h4. Aggregate the
empirical coverage fraction across the `N_OUTER` trials and compare to
the nominal target -- this is a nested Monte Carlo (`N_OUTER *
(1 + N_BOOT_INNER)` fits per cell), expensive by construction, which is
why it's an overnight job rather than a unit test.

Includes both h4_true=0.0 (Gaussian, matching every prior validation in
this project) and h4_true=0.15 (genuine wing, per this week's finding
that Gaussian-only validation is blind to wing-specific failure modes)
for every sigma/S-N cell, so this doubles as another genuine-wing check.

Usage: python coverage_calibration_test.py [outfile]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import (
    FitConfig, build_template_stack, gauss_hermite_model, bootstrap_losvd,
)

MUSE_TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WAVELENGTH = 4750.0 + np.arange(3681) * 1.25
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0

SIGMA_GRID = [50.0, 150.0, 300.0]
SNR_GRID = [40.0, 100.0]
H4_TRUE_GRID = [0.0, 0.15]
N_OUTER = 30
N_BOOT_INNER = 15
MIXTURE_SEED = 20000  # same fixed mixture used throughout this session's tests


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

    rows = []
    t0 = time.time()
    for h4_true in H4_TRUE_GRID:
        for sig in SIGMA_GRID:
            cfg = FitConfig(vmin=-7 * sig, vmax=7 * sig)
            v_grid = cfg.build_velocity_grid()
            b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, 0.0, sig, 0.0, h4_true), 0.0)
            b_truth /= b_truth.sum()
            A_stack = build_template_stack(wave_fit, WAVELENGTH, templates, v_grid)
            true_v, true_s = _true_moments(sig, h4_true)

            for snr in SNR_GRID:
                sig_in_1sig, sig_in_2sig = 0, 0
                h4_in_1sig, h4_in_2sig = 0, 0
                n_valid = 0
                for rep in range(N_OUTER):
                    seed = 3_000_000 + rep + int(sig * 10) + int(snr * 100) + int(h4_true * 10000)
                    galaxy_noiseless = sum(true_weights[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))
                    noise_level = np.median(galaxy_noiseless) / snr
                    rng = np.random.default_rng(seed)
                    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
                    errors = np.full(len(wave_fit), noise_level)
                    A_stack_w = A_stack / errors[None, :, None]
                    y_w = galaxy / errors

                    try:
                        boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=N_BOOT_INNER, seed=seed + 1)
                    except Exception as exc:
                        print(f"  (skip rep {rep}, exception: {exc})", flush=True)
                        continue
                    s_est = boot["point0_corrected"]["sherm"]
                    s_std = boot["sherm_std_corrected"]
                    h4_est = boot["point0_corrected"]["h4"]
                    h4_std = boot["h4_std_corrected"]
                    if not (np.isfinite(s_est) and np.isfinite(s_std) and s_std > 0
                            and np.isfinite(h4_est) and np.isfinite(h4_std) and h4_std > 0):
                        continue
                    n_valid += 1
                    sig_err = abs(s_est - true_s)
                    h4_err = abs(h4_est - h4_true)
                    sig_in_1sig += sig_err < 1.0 * s_std
                    sig_in_2sig += sig_err < 2.0 * s_std
                    h4_in_1sig += h4_err < 1.0 * h4_std
                    h4_in_2sig += h4_err < 2.0 * h4_std

                if n_valid < 5:
                    print(f"  (skip h4_true={h4_true} sigma={sig} snr={snr}: only {n_valid} valid reps)")
                    continue
                row = dict(
                    h4_true=h4_true, sigma=sig, snr=snr, n=n_valid,
                    sig_cov68=sig_in_1sig / n_valid, sig_cov95=sig_in_2sig / n_valid,
                    h4_cov68=h4_in_1sig / n_valid, h4_cov95=h4_in_2sig / n_valid,
                )
                rows.append(row)
                print(f"[h4_true={h4_true:.2f}][sigma={sig:5.0f}][snr={snr:5.0f}] n={n_valid:3d}  "
                      f"sig_cov: 68%->{row['sig_cov68']*100:5.1f}%  95%->{row['sig_cov95']*100:5.1f}%   "
                      f"h4_cov: 68%->{row['h4_cov68']*100:5.1f}%  95%->{row['h4_cov95']*100:5.1f}%   "
                      f"t={time.time()-t0:.0f}s", flush=True)
    return rows


def save_rows(rows, path):
    with open(path, "w") as f:
        f.write(f"{'h4_true':>8} {'sigma':>6} {'snr':>6} {'n':>4}   "
                f"{'sig_cov68':>10} {'sig_cov95':>10}   {'h4_cov68':>9} {'h4_cov95':>9}\n")
        for r in rows:
            f.write(f"{r['h4_true']:8.2f} {r['sigma']:6.0f} {r['snr']:6.0f} {r['n']:4d}   "
                    f"{r['sig_cov68']*100:9.1f}% {r['sig_cov95']*100:9.1f}%   "
                    f"{r['h4_cov68']*100:8.1f}% {r['h4_cov95']*100:8.1f}%\n")


if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "coverage_calibration_results.txt")
    rows = run()
    save_rows(rows, outfile)
    print(f"wrote {len(rows)} rows to {outfile}")
