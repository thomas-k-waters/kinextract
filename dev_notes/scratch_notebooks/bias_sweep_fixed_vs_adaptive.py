"""Bias-vs-sigma sweep, fixed-lam vs. adaptive-lam, across all four
Gauss-Hermite moments (V, sigma, h3, h4) -- the sandbox notebook's own
sweep cell, extracted into a real module so it can be parallelized with
`ProcessPoolExecutor` (a Jupyter-notebook-local function can't be pickled
for a spawned worker process on macOS; a module-level function in a real
file can).

**Why this exists**: the notebook's original sweep cell (fixed lam only,
sigma/h4 only) took ~47.5 minutes running sequentially (6 sigma values x
10 reps x 11 fits/rep with bootstrap on). This module (1) adds V/h3
alongside sigma/h4, (2) adds a second "adaptive lam" case
(`select_lambda_aic_local`) run on the *same* noise draw as the fixed
case for a paired comparison, and (3) parallelizes across all (sigma,
rep) combinations via `ProcessPoolExecutor` -- independent computations,
zero risk to any statistical result, matched to the machine's own core
count.

Usage from the notebook: `from bias_sweep_fixed_vs_adaptive import run;
rows = run(...)`. Usage from the command line: `python
bias_sweep_fixed_vs_adaptive.py [outfile]` (uses the module's own
defaults below).
"""
import sys, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

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
MIXTURE_SEED = 20000  # same fixed Dirichlet mixture the notebook and this session's other tests use

SIGMA_GRID = [30, 50, 80, 140, 220, 300]
SNR = 60.0
N_REPS = 15
V_TRUE, H3_TRUE, H4_TRUE = 0.0, 0.0, 0.0
LAM_FIXED = 2.0
LAM_GRID_ADAPTIVE = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
NSIG = 7.0


def _load_templates():
    with open(os.path.join(MUSE_TEMPLATE_DIR, "Tlist")) as f:
        names = [line.strip() for line in f if line.strip()]
    templates = [np.interp(WAVELENGTH, *np.loadtxt(os.path.join(MUSE_TEMPLATE_DIR, n)).T) for n in names]
    rng_mix = np.random.default_rng(MIXTURE_SEED)
    true_weights = rng_mix.dirichlet(np.ones(len(templates)))
    return templates, true_weights


def _true_moments(v_true, sigma, h3_true, h4_true):
    vv = np.linspace(v_true - 15 * sigma, v_true + 15 * sigma, 200_001)
    bb = np.maximum(gauss_hermite_model(vv, 1.0, v_true, sigma, h3_true, h4_true), 0.0)
    tot = np.trapezoid(bb, vv)
    true_v = float(np.trapezoid(bb * vv, vv) / tot)
    true_s = float(np.sqrt(np.trapezoid(bb * (vv - true_v) ** 2, vv) / tot))
    return true_v, true_s


def _run_one(sigma, rep, snr, v_true, h3_true, h4_true, lam_fixed, lam_grid_adaptive,
             wavefitmin, wavefitmax, nsig, seed_base):
    """One (sigma, rep): builds the mock once, fits it BOTH ways (fixed
    lam, adaptive lam) on the *same* noise draw, and returns both cases'
    (V, sigma, h3, h4) bias. Module-level (not notebook-local) so
    `ProcessPoolExecutor` can pickle a reference to it.
    """
    templates, true_weights = _load_templates()
    mask = (WAVELENGTH >= wavefitmin) & (WAVELENGTH <= wavefitmax)
    wave_fit = WAVELENGTH[mask]

    cfg_base = FitConfig(vmin=v_true - nsig * sigma, vmax=v_true + nsig * sigma, lam=lam_fixed)
    v_grid = cfg_base.build_velocity_grid()
    b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, v_true, sigma, h3_true, h4_true), 0.0)
    b_truth /= b_truth.sum()
    A_stack = build_template_stack(wave_fit, WAVELENGTH, templates, v_grid)
    true_v, true_s = _true_moments(v_true, sigma, h3_true, h4_true)

    seed = seed_base + rep + int(sigma * 10)
    galaxy_noiseless = sum(true_weights[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors

    out = dict(sigma=sigma, rep=rep)

    result_fixed = fit_losvd(A_stack_w, y_w, cfg_base)
    m_fixed = compute_losvd_moments(v_grid, result_fixed["b"])
    out["fixed"] = dict(
        v_bias=m_fixed["vherm"] - true_v, sig_bias=m_fixed["sherm"] - true_s,
        h3_bias=m_fixed["h3"] - h3_true, h4_bias=m_fixed["h4"] - h4_true, lam=lam_fixed,
    )

    aic_result = select_lambda_aic_local(A_stack_w, y_w, cfg_base, lam_grid_adaptive)
    cfg_adaptive = replace(cfg_base, lam=aic_result.best_lam)
    result_adaptive = fit_losvd(A_stack_w, y_w, cfg_adaptive)
    m_adaptive = compute_losvd_moments(v_grid, result_adaptive["b"])
    out["adaptive"] = dict(
        v_bias=m_adaptive["vherm"] - true_v, sig_bias=m_adaptive["sherm"] - true_s,
        h3_bias=m_adaptive["h3"] - h3_true, h4_bias=m_adaptive["h4"] - h4_true,
        lam=aic_result.best_lam,
    )
    return out


def run(sigma_grid=SIGMA_GRID, n_reps=N_REPS, snr=SNR, v_true=V_TRUE, h3_true=H3_TRUE, h4_true=H4_TRUE,
        lam_fixed=LAM_FIXED, lam_grid_adaptive=LAM_GRID_ADAPTIVE, wavefitmin=WAVEFITMIN, wavefitmax=WAVEFITMAX,
        nsig=NSIG, seed_base=5_000_000, max_workers=None, verbose=True):
    """Runs the full (sigma x rep) grid in parallel via `ProcessPoolExecutor`
    (default `max_workers=None` uses all available cores). Returns a list
    of per-sigma dicts with mean/std bias for both `fixed` and `adaptive`
    cases across all four moments.
    """
    max_workers = max_workers or os.cpu_count()
    jobs = [(sigma, rep) for sigma in sigma_grid for rep in range(n_reps)]
    results = {sigma: {"fixed": {"v": [], "sig": [], "h3": [], "h4": []},
                        "adaptive": {"v": [], "sig": [], "h3": [], "h4": [], "lam": []}}
               for sigma in sigma_grid}

    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_one, sigma, rep, snr, v_true, h3_true, h4_true, lam_fixed,
                      lam_grid_adaptive, wavefitmin, wavefitmax, nsig, seed_base): (sigma, rep)
            for sigma, rep in jobs
        }
        for fut in as_completed(futures):
            out = fut.result()
            sigma = out["sigma"]
            for case in ("fixed", "adaptive"):
                results[sigma][case]["v"].append(out[case]["v_bias"])
                results[sigma][case]["sig"].append(out[case]["sig_bias"])
                results[sigma][case]["h3"].append(out[case]["h3_bias"])
                results[sigma][case]["h4"].append(out[case]["h4_bias"])
            results[sigma]["adaptive"]["lam"].append(out["adaptive"]["lam"])
            n_done += 1
            if verbose and n_done % max(1, len(jobs) // 20) == 0:
                print(f"  {n_done}/{len(jobs)} done, t={time.time()-t0:.0f}s", flush=True)

    rows = []
    for sigma in sigma_grid:
        for case in ("fixed", "adaptive"):
            d = results[sigma][case]
            row = dict(sigma=float(sigma), case=case,
                       v_mean=float(np.mean(d["v"])), v_std=float(np.std(d["v"], ddof=1)),
                       sig_mean=float(np.mean(d["sig"])), sig_std=float(np.std(d["sig"], ddof=1)),
                       h3_mean=float(np.mean(d["h3"])), h3_std=float(np.std(d["h3"], ddof=1)),
                       h4_mean=float(np.mean(d["h4"])), h4_std=float(np.std(d["h4"], ddof=1)))
            if case == "adaptive":
                row["mean_lam"] = float(np.mean(d["lam"]))
            rows.append(row)
            if verbose:
                print(f"[sigma={sigma:5.0f}][{case:>8s}]  V={row['v_mean']:+6.2f}  "
                      f"sig={row['sig_mean']:+7.2f} ({100*row['sig_mean']/sigma:+6.1f}%)  "
                      f"h3={row['h3_mean']:+.4f}  h4={row['h4_mean']:+.4f}"
                      + (f"  mean_lam={row['mean_lam']:.2f}" if case == "adaptive" else ""), flush=True)
    print(f"total wall time: {time.time()-t0:.0f}s ({max_workers} workers)", flush=True)
    return rows


def save_rows(rows, path):
    with open(path, "w") as f:
        f.write(f"{'sigma':>6} {'case':>9}   {'V_mean':>8} {'V_std':>7}   "
                f"{'sig_mean':>10} {'sig_std':>9}   {'h3_mean':>9} {'h3_std':>8}   "
                f"{'h4_mean':>9} {'h4_std':>8}   {'mean_lam':>9}\n")
        for r in rows:
            f.write(f"{r['sigma']:6.0f} {r['case']:>9}   {r['v_mean']:+8.2f} {r['v_std']:7.2f}   "
                    f"{r['sig_mean']:+10.2f} {r['sig_std']:9.2f}   "
                    f"{r['h3_mean']:+9.4f} {r['h3_std']:8.4f}   "
                    f"{r['h4_mean']:+9.4f} {r['h4_std']:8.4f}   "
                    f"{r.get('mean_lam', float('nan')):9.3f}\n")


if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "bias_sweep_fixed_vs_adaptive_results.txt")
    rows = run()
    save_rows(rows, outfile)
    print(f"wrote {len(rows)} rows to {outfile}")
