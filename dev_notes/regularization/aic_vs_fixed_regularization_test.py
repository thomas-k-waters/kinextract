"""Compares Lipka & Thomas (2021)/Thomas & Lipka (2022)'s bootstrap-AIC
regularization selection (`kinextract.aic_regularization`) against
this package's fixed `lam=2.0` default and the unregularized `lam=0.0`
floor -- on a mock with a *genuine* non-Gaussian LOSVD (h4 > 0, real
high-velocity wings), not the plain-Gaussian (h3=h4=0) truth every other
validation script in this project has used so far.

**Why a genuine-wing mock specifically.** The user's concern: for
black-hole-mass work the high-velocity wings ARE the science (stars on
orbits that reach deep into the black hole's sphere of influence), so a
regularization method must not just have low *aggregate* bias -- it must
not systematically smooth away real wing mass while doing so. Every
sigma/S-N grid used elsewhere in this project's validation (the ablation
ladder, the calibration sweep, the E-MILES mismatch test) generated
mocks from `gauss_hermite_model(..., h3=0, h4=0)` -- a plain Gaussian.
That is blind to wing-suppression by construction: there is no real wing
signal in those mocks for a method to over-smooth away. This script uses
`h4_true = 0.15` (real, moderate wing excess) so "does the fit recover
the true h4 or shrink it toward 0" is a directly answerable question,
not an assumption.

**Templates**: the real 33-star MUSE library, generating and fitting
with the *same* library (per the user's own explicit priority call:
settle regularization before reintroducing template mismatch as a
confound -- see the E-MILES mismatch test for that separate question).

**Design**: same noise draw, four regularization choices per replicate:
`lam=0.0` (unregularized floor), `lam=2.0` (shipped default),
`select_lambda_aic` (bootstrap-AIC, scanning `LAM_GRID`), and
`select_lambda_aic_local` (the cheap closed-form approximation -- added
specifically to close a real gap: the earlier version of this script only
tested the *expensive bootstrap* AIC method against this wing mock, never
the local approximation actually proposed as the new default in the
statistical refactor plan, despite `aic_regularization.py`'s own
docstring flagging this exact test as required first). Compares V/sigma/
h4 bias for all four, with h4 bias reported as *signed* (not just
magnitude) so wing suppression (h4 biased toward 0, i.e. negative when
h4_true > 0) is distinguishable from wing inflation (biased further
positive).

Usage: python aic_vs_fixed_regularization_test.py [outfile]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import (
    FitConfig, build_template_stack, gauss_hermite_model, fit_losvd, compute_losvd_moments,
)
from kinextract.aic_regularization import select_lambda_aic, select_lambda_aic_local

MUSE_TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WAVELENGTH = 4750.0 + np.arange(3681) * 1.25
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0

SIGMA_GRID = [50.0, 150.0, 300.0]
SNR = 60.0
N_REPS = 6
H4_TRUE = 0.15  # genuine, moderate wing excess -- not the h3=h4=0 truth used elsewhere
LAM_GRID = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
N_BOOT = 10
MIXTURE_SEED = 20000  # same fixed mixture as the E-MILES mismatch test, for a shared baseline


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
    for sig in SIGMA_GRID:
        cfg_base = FitConfig(vmin=-7 * sig, vmax=7 * sig)
        v_grid = cfg_base.build_velocity_grid()
        b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, 0.0, sig, 0.0, H4_TRUE), 0.0)
        b_truth /= b_truth.sum()
        A_stack = build_template_stack(wave_fit, WAVELENGTH, templates, v_grid)
        true_v, true_s = _true_moments(sig, H4_TRUE)

        errs = {"lam0": dict(v=[], s=[], h4=[]), "lam2": dict(v=[], s=[], h4=[]),
                "aic": dict(v=[], s=[], h4=[], best_lam=[]),
                "aic_local": dict(v=[], s=[], h4=[], best_lam=[])}
        for rep in range(N_REPS):
            seed = 2_000_000 + rep + int(sig * 10)
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
                errs[label]["h4"].append(m["h4"] - H4_TRUE)

            aic_result = select_lambda_aic(A_stack_w, y_w, cfg_base, LAM_GRID, n_boot=N_BOOT, seed=seed)
            cfg_aic = FitConfig(vmin=-7 * sig, vmax=7 * sig, lam=aic_result.best_lam)
            result = fit_losvd(A_stack_w, y_w, cfg_aic)
            if np.all(np.isfinite(result["b"])):
                m = compute_losvd_moments(v_grid, result["b"])
                if np.isfinite(m["vherm"]) and np.isfinite(m["sherm"]):
                    errs["aic"]["v"].append(m["vherm"] - true_v)
                    errs["aic"]["s"].append(m["sherm"] - true_s)
                    errs["aic"]["h4"].append(m["h4"] - H4_TRUE)
                    errs["aic"]["best_lam"].append(aic_result.best_lam)

            aic_local_result = select_lambda_aic_local(A_stack_w, y_w, cfg_base, LAM_GRID)
            cfg_aic_local = FitConfig(vmin=-7 * sig, vmax=7 * sig, lam=aic_local_result.best_lam)
            result = fit_losvd(A_stack_w, y_w, cfg_aic_local)
            if np.all(np.isfinite(result["b"])):
                m = compute_losvd_moments(v_grid, result["b"])
                if np.isfinite(m["vherm"]) and np.isfinite(m["sherm"]):
                    errs["aic_local"]["v"].append(m["vherm"] - true_v)
                    errs["aic_local"]["s"].append(m["sherm"] - true_s)
                    errs["aic_local"]["h4"].append(m["h4"] - H4_TRUE)
                    errs["aic_local"]["best_lam"].append(aic_local_result.best_lam)

            print(f"  sigma={sig:5.0f} rep={rep:2d}  AIC best_lam={aic_result.best_lam:6.2f}  "
                  f"AIC_local best_lam={aic_local_result.best_lam:6.2f}  "
                  f"t={time.time()-t0:.0f}s", flush=True)

        for label in ("lam0", "lam2", "aic", "aic_local"):
            d = errs[label]
            n = len(d["s"])
            if n < 2:
                continue
            row = dict(
                label=label, sigma=float(sig), n=n,
                v_bias_mean=float(np.mean(d["v"])),
                sig_bias_mean=float(np.mean(d["s"])), sig_bias_std=float(np.std(d["s"], ddof=1)) if n > 1 else 0.0,
                h4_bias_mean=float(np.mean(d["h4"])), h4_bias_std=float(np.std(d["h4"], ddof=1)) if n > 1 else 0.0,
                mean_best_lam=float(np.mean(d["best_lam"])) if label in ("aic", "aic_local") else float("nan"),
            )
            rows.append(row)
            print(f"[sigma={sig:5.0f}][{label:>5s}] n={n:2d}  "
                  f"sig_bias={row['sig_bias_mean']:+7.2f} ({100*row['sig_bias_mean']/true_s:+6.1f}%)  "
                  f"h4_bias={row['h4_bias_mean']:+.4f} (true h4={H4_TRUE})  "
                  f"{'mean_lam='+format(row['mean_best_lam'],'.2f') if label in ('aic', 'aic_local') else ''}",
                  flush=True)
    return rows


def save_rows(rows, path):
    with open(path, "w") as f:
        f.write(f"{'label':>6} {'sigma':>6} {'n':>4}   {'V_bias_mean':>12}   "
                f"{'sig_bias_mean':>14} {'sig_bias_std':>13} {'sig_bias_pct':>13}   "
                f"{'h4_bias_mean':>13} {'h4_bias_std':>12} {'mean_best_lam':>14}\n")
        for r in rows:
            pct = 100 * r["sig_bias_mean"] / r["sigma"]
            f.write(f"{r['label']:>6} {r['sigma']:6.0f} {r['n']:4d}   {r['v_bias_mean']:+12.2f}   "
                    f"{r['sig_bias_mean']:+14.2f} {r['sig_bias_std']:13.2f} {pct:+12.1f}%   "
                    f"{r['h4_bias_mean']:+13.4f} {r['h4_bias_std']:12.4f} {r['mean_best_lam']:14.3f}\n")


if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "aic_vs_fixed_regularization_results.txt")
    rows = run()
    save_rows(rows, outfile)
    print(f"wrote {len(rows)} rows to {outfile}")
