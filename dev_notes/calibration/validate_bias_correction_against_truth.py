"""Validates the regularization-bias correction against KNOWN synthetic
truth -- not against the legacy Fortran pipeline (`pallmc.out`), which was
found this session to carry its own real biases and is therefore not a
trustworthy ground truth for real NGC 5102 data. This is the legitimate
scoreboard for "does the correction help": does the corrected sigma land
closer to the true sigma than the raw sigma does, in synthetic conditions
built to resemble real N5102 fitting as closely as possible?

Differs from `calibrate_bias_muse_multitemplate.py` (which built the
calibration table this validates) in the one way that matters for this to
be a fair, non-circular test: a FRESH random Dirichlet population mixture
is drawn per replicate (seed range disjoint from the calibration's own
`WEIGHT_SEED=10000`), not the single fixed mixture the calibration itself
was built from -- this tests whether the correction generalizes across
different real stellar populations, not just whether it can recover the
one mixture it was calibrated on.

Sigma/S-N grid chosen to bracket what's actually observed in the real
N5102 bins (`fit_all_bins.py`'s own `snr_estimated`/`sherm` output ranged
roughly S/N 14-101, raw sigma ~44-65 km/s) with some margin either side.

Usage: python validate_bias_correction_against_truth.py [outfile]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from kinextract import (
    FitConfig, build_template_stack, gauss_hermite_model, compute_losvd_moments,
    estimate_snr_whitened, correct_moments,
)
from kinextract.multi_template import fit_losvd

TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WEIGHT_SEED_BASE = 90000  # disjoint from calibration's WEIGHT_SEED=10000 -- fresh mixtures, not the calibrated one
SIGMA_GRID = [30, 40, 50, 60, 80, 100]
SNR_GRID = [15, 25, 40, 60, 80, 100, 150]
N_REPS = 25
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
NSIG = 7


def true_moments(vel, sig, n=200_001):
    vv = np.linspace(vel - 15 * sig, vel + 15 * sig, n)
    bb = np.maximum(gauss_hermite_model(vv, 1.0, vel, sig, 0.0, 0.0), 0.0)
    tot = np.trapezoid(bb, vv)
    m1 = np.trapezoid(bb * vv, vv) / tot
    m2 = np.sqrt(np.trapezoid(bb * (vv - m1) ** 2, vv) / tot)
    return m1, m2


def main(outfile):
    with open(os.path.join(TEMPLATE_DIR, "Tlist")) as f:
        names = [line.strip() for line in f if line.strip()]
    wavelength = 4750.0 + np.arange(3681) * 1.25
    templates = []
    for name in names:
        d = np.loadtxt(os.path.join(TEMPLATE_DIR, name))
        templates.append(np.interp(wavelength, d[:, 0], d[:, 1]))
    n_templates = len(templates)
    mask = (wavelength >= WAVEFITMIN) & (wavelength <= WAVEFITMAX)
    wave_fit = wavelength[mask]
    print(f"loaded {n_templates} real MUSE templates; validating against KNOWN truth "
          f"(fresh random mixture per replicate, disjoint from the calibration's own seed)")

    rows = []
    t0 = time.time()
    with open(outfile, "w") as f:
        header = (f"{'sigma':>6} {'snr':>6} {'n':>4}   "
                   f"{'raw_err_med':>12} {'corr_err_med':>13}   "
                   f"{'raw_err_mean':>13} {'corr_err_mean':>14}   "
                   f"{'frac_corr_better':>16}\n")
        f.write(header)
        print(header, end="")

        for sig in SIGMA_GRID:
            vmin, vmax = -NSIG * sig, NSIG * sig
            cfg = FitConfig(vmin=vmin, vmax=vmax)
            v_grid = cfg.build_velocity_grid()
            b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, 0.0, sig, 0.0, 0.0), 0.0)
            b_truth /= b_truth.sum()
            A_stack = build_template_stack(wave_fit, wavelength, templates, v_grid)
            true_v, true_s = true_moments(0.0, sig)

            for snr in SNR_GRID:
                raw_errs, corr_errs = [], []
                for rep in range(N_REPS):
                    seed = WEIGHT_SEED_BASE + rep + int(sig * 10) + int(snr * 100)
                    rng_w = np.random.default_rng(seed)
                    w_true = rng_w.dirichlet(np.ones(n_templates))
                    galaxy_noiseless = sum(w_true[k] * (A_stack[k] @ b_truth) for k in range(n_templates))
                    noise_level = np.median(galaxy_noiseless) / snr
                    rng_n = np.random.default_rng(seed + 500000)
                    galaxy = galaxy_noiseless + rng_n.normal(0.0, noise_level, len(wave_fit))
                    errors = np.full(len(wave_fit), noise_level)
                    A_stack_w = A_stack / errors[None, :, None]
                    y_w = galaxy / errors

                    result = fit_losvd(A_stack_w, y_w, cfg)
                    m = compute_losvd_moments(v_grid, result["b"])
                    if not (np.isfinite(m["vherm"]) and np.isfinite(m["sherm"])):
                        continue
                    model_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0])) @ result["b"]
                    snr_est = estimate_snr_whitened(model_w)
                    corr = correct_moments(m["vherm"], m["sherm"], m["h4"], snr_est)

                    raw_errs.append(abs(m["sherm"] - true_s))
                    corr_errs.append(abs(corr["sherm_corrected"] - true_s))

                n = len(raw_errs)
                if n < 3:
                    continue
                raw_errs, corr_errs = np.array(raw_errs), np.array(corr_errs)
                frac_better = float(np.mean(corr_errs < raw_errs))
                row = dict(sigma=sig, snr=snr, n=n,
                           raw_err_med=float(np.median(raw_errs)), corr_err_med=float(np.median(corr_errs)),
                           raw_err_mean=float(np.mean(raw_errs)), corr_err_mean=float(np.mean(corr_errs)),
                           frac_corr_better=frac_better)
                rows.append(row)
                line = (f"{sig:6.0f} {snr:6.0f} {n:4d}   "
                        f"{row['raw_err_med']:12.2f} {row['corr_err_med']:13.2f}   "
                        f"{row['raw_err_mean']:13.2f} {row['corr_err_mean']:14.2f}   "
                        f"{frac_better:16.2f}\n")
                f.write(line)
                f.flush()
                print(line, end="", flush=True)

        footer = f"\n# time: {time.time()-t0:.0f}s\n"
        f.write(footer)
        print(footer, end="")

    overall_raw = np.array([r["raw_err_med"] for r in rows])
    overall_corr = np.array([r["corr_err_med"] for r in rows])
    print(f"\nOverall: {len(rows)} cells. Correction reduced median sigma error in "
          f"{np.mean(overall_corr < overall_raw)*100:.0f}% of cells.")
    print(f"Median raw_err across cells: {np.median(overall_raw):.2f} km/s")
    print(f"Median corr_err across cells: {np.median(overall_corr):.2f} km/s")


if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else "validate_bias_correction_against_truth.txt"
    main(outfile)
