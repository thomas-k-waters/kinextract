"""Firms up `aic_vs_fixed_regularization_test.py`'s finding (aic_local
beats fixed lam=0/2 on the h4=0.15 wing mock at n=6 reps/cell) before
committing to the bigger Stage-C recalibration project it motivates --
per the user's own explicit ask: verify more thoroughly given how much
this project has been burned before by declaring something fixed on a
small sample (the h4-calibration-domain bug, the coverage-calibration
catastrophe).

**Two things this adds over the original test**:
1. Many more replicates per cell (N_REPS below vs. the original's 6) --
   the expensive bootstrap `select_lambda_aic` arm is dropped entirely to
   make this affordable (it was never the promotion candidate; aic_local
   is, and this project's own aic_regularization.py already validated
   aic_local tracks the bootstrap version's chi2 closely at a given lam).
2. Both `H4_TRUE=0.0` (the plain-Gaussian truth used everywhere else in
   this project's validation) and `H4_TRUE=0.15` (the genuine-wing case)
   -- checks aic_local's advantage isn't an artifact specific to the one
   wing mock it was first tested against.

Usage: python aic_local_firming_up_test.py [outfile]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import (
    FitConfig, build_template_stack, gauss_hermite_model, fit_losvd, compute_losvd_moments,
)
from kinextract.aic_regularization import select_lambda_aic_local

MUSE_TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WAVELENGTH = 4750.0 + np.arange(3681) * 1.25
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0

SIGMA_GRID = [50.0, 150.0, 300.0]
SNR = 60.0
N_REPS = 25
H4_GRID = [0.0, 0.15]
LAM_GRID = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
MIXTURE_SEED = 20000  # same fixed mixture as the original test, for a shared baseline


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
    for h4_true in H4_GRID:
        for sig in SIGMA_GRID:
            cfg_base = FitConfig(vmin=-7 * sig, vmax=7 * sig)
            v_grid = cfg_base.build_velocity_grid()
            b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, 0.0, sig, 0.0, h4_true), 0.0)
            b_truth /= b_truth.sum()
            A_stack = build_template_stack(wave_fit, WAVELENGTH, templates, v_grid)
            true_v, true_s = _true_moments(sig, h4_true)

            errs = {"lam0": dict(v=[], s=[], h4=[]), "lam2": dict(v=[], s=[], h4=[]),
                    "aic_local": dict(v=[], s=[], h4=[], best_lam=[])}
            for rep in range(N_REPS):
                seed = 4_000_000 + rep + int(sig * 10) + int(h4_true * 1000)
                galaxy_noiseless = sum(true_weights[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))
                noise_level = np.median(galaxy_noiseless) / SNR
                rng = np.random.default_rng(seed)
                galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
                errors = np.full(len(wave_fit), noise_level)
                A_stack_w = A_stack / errors[None, :, None]
                y_w = galaxy / errors

                for label, lam in (("lam0", 0.0), ("lam2", 2.0)):
                    cfg = FitConfig(vmin=-7 * sig, vmax=7 * sig, lam=lam)
                    result = fit_losvd(A_stack_w, y_w, cfg)
                    if not np.all(np.isfinite(result["b"])):
                        continue
                    m = compute_losvd_moments(v_grid, result["b"])
                    if not (np.isfinite(m["vherm"]) and np.isfinite(m["sherm"])):
                        continue
                    errs[label]["v"].append(m["vherm"] - true_v)
                    errs[label]["s"].append(m["sherm"] - true_s)
                    errs[label]["h4"].append(m["h4"] - h4_true)

                aic_local_result = select_lambda_aic_local(A_stack_w, y_w, cfg_base, LAM_GRID)
                cfg_aic_local = FitConfig(vmin=-7 * sig, vmax=7 * sig, lam=aic_local_result.best_lam)
                result = fit_losvd(A_stack_w, y_w, cfg_aic_local)
                if np.all(np.isfinite(result["b"])):
                    m = compute_losvd_moments(v_grid, result["b"])
                    if np.isfinite(m["vherm"]) and np.isfinite(m["sherm"]):
                        errs["aic_local"]["v"].append(m["vherm"] - true_v)
                        errs["aic_local"]["s"].append(m["sherm"] - true_s)
                        errs["aic_local"]["h4"].append(m["h4"] - h4_true)
                        errs["aic_local"]["best_lam"].append(aic_local_result.best_lam)

            for label in ("lam0", "lam2", "aic_local"):
                d = errs[label]
                n = len(d["s"])
                if n < 2:
                    continue
                row = dict(
                    label=label, h4_true=h4_true, sigma=float(sig), n=n,
                    v_bias_mean=float(np.mean(d["v"])),
                    sig_bias_mean=float(np.mean(d["s"])), sig_bias_std=float(np.std(d["s"], ddof=1)),
                    h4_bias_mean=float(np.mean(d["h4"])), h4_bias_std=float(np.std(d["h4"], ddof=1)),
                    mean_best_lam=float(np.mean(d["best_lam"])) if label == "aic_local" else float("nan"),
                )
                rows.append(row)
                print(f"[h4_true={h4_true:.2f}][sigma={sig:5.0f}][{label:>9s}] n={n:2d}  "
                      f"sig_bias={row['sig_bias_mean']:+7.2f} ({100*row['sig_bias_mean']/true_s:+6.1f}%)  "
                      f"h4_bias={row['h4_bias_mean']:+.4f}  "
                      f"{'mean_lam='+format(row['mean_best_lam'],'.2f') if label == 'aic_local' else ''}  "
                      f"t={time.time()-t0:.0f}s", flush=True)
    return rows


def save_rows(rows, path):
    with open(path, "w") as f:
        f.write(f"{'label':>9} {'h4_true':>7} {'sigma':>6} {'n':>4}   {'V_bias_mean':>12}   "
                f"{'sig_bias_mean':>14} {'sig_bias_std':>13} {'sig_bias_pct':>13}   "
                f"{'h4_bias_mean':>13} {'h4_bias_std':>12} {'mean_best_lam':>14}\n")
        for r in rows:
            pct = 100 * r["sig_bias_mean"] / r["sigma"]
            f.write(f"{r['label']:>9} {r['h4_true']:7.2f} {r['sigma']:6.0f} {r['n']:4d}   {r['v_bias_mean']:+12.2f}   "
                    f"{r['sig_bias_mean']:+14.2f} {r['sig_bias_std']:13.2f} {pct:+12.1f}%   "
                    f"{r['h4_bias_mean']:+13.4f} {r['h4_bias_std']:12.4f} {r['mean_best_lam']:14.3f}\n")


if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "aic_local_firming_up_results.txt")
    rows = run()
    save_rows(rows, outfile)
    print(f"wrote {len(rows)} rows to {outfile}")
