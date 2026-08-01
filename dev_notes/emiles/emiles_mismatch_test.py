"""Tests whether fitting with the LSF-matched E-MILES SSP template set
(`prepare_emiles_templates.py`) instead of the real 33-star MUSE Library
of Stellar Spectra measurably changes recovered LOSVD bias -- the direct
question raised by Mehrgan et al. (2023)'s template-mismatch findings:
does E-MILES's much broader spectral-type/age/metallicity coverage
reduce bias from gaps in the 33-star library, or does fitting with
genuinely different (population-synthesis vs. real-star) templates
introduce its own mismatch bias that offsets or exceeds that benefit?

**Design.** Every mock is generated from a fixed Dirichlet mixture over
the real 33-star MUSE library (playing the role of "the real galaxy's
actual stellar population," the only ground truth this project has ever
used). Each mock is then fit two ways on *identical* noise draws:

  - `muse_control`: fit with the same 33-star library used to generate
    the mock -- the already-characterized bias floor from the ablation
    ladder, here as a same-data control rather than a re-derivation.
  - `emiles_mismatch`: fit with the 25-template E-MILES subgrid instead
    -- genuine template mismatch (different actual spectra, real stars
    vs. synthetic SSPs), the case this script exists to quantify.

Run at both `lam=0.0` and `lam=2.0` (the ablation ladder's shipped
default) per the user's own suggestion: the WINGFIT paper doesn't say
whether its own template-mismatch test used regularization, so this
checks whether regularization changes the mismatch picture rather than
assuming either answer.

Usage: python emiles_mismatch_test.py [outfile]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import (
    FitConfig, build_template_stack, gauss_hermite_model, fit_losvd, compute_losvd_moments,
)

MUSE_TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
EMILES_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "emiles_templates")

WAVELENGTH = 4750.0 + np.arange(3681) * 1.25
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0

SIGMA_GRID = [30, 50, 80, 140, 220, 300]
SNR = 60.0
N_REPS = 15
LAM_GRID = [0.0, 2.0]
V_TRUE = 0.0
MIXTURE_SEED = 20000  # fixed Dirichlet mixture over the 33 MUSE templates, reused every replicate


def _load_templates(template_dir):
    with open(os.path.join(template_dir, "Tlist")) as f:
        names = [line.strip() for line in f if line.strip()]
    templates = []
    for name in names:
        d = np.loadtxt(os.path.join(template_dir, name))
        templates.append(np.interp(WAVELENGTH, d[:, 0], d[:, 1]))
    return templates, names


def run(sigma_grid=SIGMA_GRID, lam_grid=LAM_GRID, wavefitmin=WAVEFITMIN, wavefitmax=WAVEFITMAX):
    muse_templates, muse_names = _load_templates(MUSE_TEMPLATE_DIR)
    emiles_templates, emiles_names = _load_templates(EMILES_TEMPLATE_DIR)
    print(f"loaded {len(muse_templates)} MUSE templates, {len(emiles_templates)} E-MILES templates")
    print(f"fit window: {wavefitmin:.0f}-{wavefitmax:.0f} A ({wavefitmax - wavefitmin:.0f} A wide)")

    rng_mix = np.random.default_rng(MIXTURE_SEED)
    true_weights = rng_mix.dirichlet(np.ones(len(muse_templates)))

    mask = (WAVELENGTH >= wavefitmin) & (WAVELENGTH <= wavefitmax)
    wave_fit = WAVELENGTH[mask]

    rows = []
    t0 = time.time()
    for lam in lam_grid:
        for sig in sigma_grid:
            cfg = FitConfig(vmin=V_TRUE - 7 * sig, vmax=V_TRUE + 7 * sig, lam=lam)
            v_grid = cfg.build_velocity_grid()
            b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, V_TRUE, sig, 0.0, 0.0), 0.0)
            b_truth /= b_truth.sum()

            A_muse = build_template_stack(wave_fit, WAVELENGTH, muse_templates, v_grid)
            A_emiles = build_template_stack(wave_fit, WAVELENGTH, emiles_templates, v_grid)

            vv = np.linspace(V_TRUE - 15 * sig, V_TRUE + 15 * sig, 200_001)
            bb = np.maximum(gauss_hermite_model(vv, 1.0, V_TRUE, sig, 0.0, 0.0), 0.0)
            tot = np.trapezoid(bb, vv)
            true_v = float(np.trapezoid(bb * vv, vv) / tot)
            true_s = float(np.sqrt(np.trapezoid(bb * (vv - true_v) ** 2, vv) / tot))

            errs = {"muse_control": dict(v=[], s=[], h4=[]), "emiles_mismatch": dict(v=[], s=[], h4=[])}
            for rep in range(N_REPS):
                seed = 1_000_000 + rep + int(sig * 10) + int(lam * 100)
                galaxy_noiseless = sum(true_weights[k] * (A_muse[k] @ b_truth) for k in range(len(muse_templates)))
                noise_level = np.median(galaxy_noiseless) / SNR
                rng = np.random.default_rng(seed)
                galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
                errors = np.full(len(wave_fit), noise_level)
                y_w = galaxy / errors

                for label, A_stack in (("muse_control", A_muse), ("emiles_mismatch", A_emiles)):
                    A_stack_w = A_stack / errors[None, :, None]
                    result = fit_losvd(A_stack_w, y_w, cfg)
                    if not np.all(np.isfinite(result["b"])):
                        continue
                    m = compute_losvd_moments(v_grid, result["b"])
                    if not (np.isfinite(m["vherm"]) and np.isfinite(m["sherm"])):
                        continue
                    errs[label]["v"].append(m["vherm"] - true_v)
                    errs[label]["s"].append(m["sherm"] - true_s)
                    errs[label]["h4"].append(m["h4"])

            for label in ("muse_control", "emiles_mismatch"):
                d = errs[label]
                n = len(d["s"])
                if n < 3:
                    continue
                row = dict(
                    label=label, lam=lam, sigma=float(sig), n=n,
                    v_bias_mean=float(np.mean(d["v"])), v_bias_std=float(np.std(d["v"], ddof=1)),
                    sig_bias_mean=float(np.mean(d["s"])), sig_bias_std=float(np.std(d["s"], ddof=1)),
                    h4_mean=float(np.mean(d["h4"])), h4_std=float(np.std(d["h4"], ddof=1)),
                )
                rows.append(row)
                print(f"[lam={lam:4.1f}][{label:>16s}] sigma={sig:5.0f} n={n:3d}  "
                      f"sig_bias={row['sig_bias_mean']:+7.2f} ({100*row['sig_bias_mean']/true_s:+6.1f}%)  "
                      f"h4={row['h4_mean']:+.4f}  t={time.time()-t0:.0f}s", flush=True)
    return rows


def save_rows(rows, path):
    with open(path, "w") as f:
        f.write(f"{'label':>16} {'lam':>5} {'sigma':>6} {'n':>4}   "
                f"{'V_bias_mean':>12} {'V_bias_std':>11}   "
                f"{'sig_bias_mean':>14} {'sig_bias_std':>13} {'sig_bias_pct':>13}   "
                f"{'h4_mean':>9} {'h4_std':>8}\n")
        for r in rows:
            pct = 100 * r["sig_bias_mean"] / r["sigma"]
            f.write(f"{r['label']:>16} {r['lam']:5.1f} {r['sigma']:6.0f} {r['n']:4d}   "
                    f"{r['v_bias_mean']:+12.2f} {r['v_bias_std']:11.2f}   "
                    f"{r['sig_bias_mean']:+14.2f} {r['sig_bias_std']:13.2f} {pct:+12.1f}%   "
                    f"{r['h4_mean']:+9.4f} {r['h4_std']:8.4f}\n")


if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "emiles_mismatch_test_results.txt")
    rows = run()
    save_rows(rows, outfile)
    print(f"wrote {len(rows)} rows to {outfile}")
