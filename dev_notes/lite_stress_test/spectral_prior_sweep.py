"""Exploratory (alpha, beta) sweep for kinextract.spectral_prior,
on the exact two regimes already validated with hand-tuned `wing_lam`
this session (01_basic_mock_fit.ipynb's 3-template sigma=50/S-N=60 case,
wing_lam=3e6 tuned; 02_realistic_mock_fit.ipynb's 4-template-degenerate
sigma=90/S-N=60 case, wing_lam=3e7 tuned) plus one lower-S/N case for a
generalization check -- so the spectral-prior's results are directly,
apples-to-apples comparable against numbers already reported to the user.

Tuning methodology matches PARAMETER_GUIDE.md's own wing_lam recipe
exactly: for each beta, sweep alpha on a log grid, pick the alpha
minimizing |median(recovered sigma) - true sigma| across N_SEEDS seeds,
then compare each beta's own best-alpha result against (a) the
wing_lam=0 baseline and (b) the already-reported hand-tuned wing_lam
result for that same regime.
"""
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import FitConfig, build_template_stack, gauss_hermite_model, compute_losvd_moments, load_emiles_templates
from kinextract.spectral_prior import eigendecompose_curvature, fit_losvd_with_spectral_prior
from kinextract.losvd import build_bspline_basis, difference_matrix

EMILES_GRID = os.path.join(os.path.dirname(__file__), "..", "..", "src", "kinextract",
                            "template_data", "emiles_muse_grid.npz")
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
NSIG = 7.0
N_SEEDS = 10
SEED_BASE = 7_000_000

BETA_GRID = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
ALPHA_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

REGIMES = {
    "nb01_mix3_sigma50_snr60": dict(
        select=[(1.0, -0.66), (5.0, 0.06), (14.0, 0.40)], weights=[0.3, 0.4, 0.3],
        true_v=20.0, true_sigma=50.0, h3_true=0.0, h4_true=0.0, snr=60.0,
        wing_lam_tuned=3e6,
    ),
    "nb02_mix4deg_sigma90_snr60": dict(
        select=[(1.0, -0.66), (5.0, 0.06), (14.0, 0.06), (14.0, 0.40)], weights=[0.15, 0.35, 0.35, 0.15],
        true_v=30.0, true_sigma=90.0, h3_true=0.03, h4_true=0.05, snr=60.0,
        wing_lam_tuned=3e7,
    ),
    "low_snr_mix3_sigma50_snr30": dict(
        select=[(1.0, -0.66), (5.0, 0.06), (14.0, 0.40)], weights=[0.3, 0.4, 0.3],
        true_v=20.0, true_sigma=50.0, h3_true=0.0, h4_true=0.0, snr=30.0,
        wing_lam_tuned=None,  # not previously tuned -- generalization check
    ),
}

_CACHE = {}


def _get_regime_templates(regime_name):
    if regime_name not in _CACHE:
        r = REGIMES[regime_name]
        templates, template_wave, names, ages, metals = load_emiles_templates(EMILES_GRID, select=r["select"])
        _CACHE[regime_name] = (templates, template_wave)
    return _CACHE[regime_name]


def _run_one(regime_name, alpha, beta, seed):
    r = REGIMES[regime_name]
    templates, template_wave = _get_regime_templates(regime_name)
    weights = np.array(r["weights"])
    mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
    wave_fit = template_wave[mask]

    true_v, true_sigma, h3_true, h4_true, snr = r["true_v"], r["true_sigma"], r["h3_true"], r["h4_true"], r["snr"]
    cfg = FitConfig(vmin=true_v - NSIG * true_sigma, vmax=true_v + NSIG * true_sigma)
    v_grid = cfg.build_velocity_grid()
    b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, true_v, true_sigma, h3_true, h4_true), 0.0)
    b_truth /= b_truth.sum()
    A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
    galaxy_noiseless = sum(weights[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))

    vv = np.linspace(true_v - 15 * true_sigma, true_v + 15 * true_sigma, 100_001)
    bb = np.maximum(gauss_hermite_model(vv, 1.0, true_v, true_sigma, h3_true, h4_true), 0.0)
    tot = np.trapezoid(bb, vv)
    true_v_exact = float(np.trapezoid(bb * vv, vv) / tot)
    true_s_exact = float(np.sqrt(np.trapezoid(bb * (vv - true_v_exact) ** 2, vv) / tot))

    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(SEED_BASE + seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors

    t0 = time.perf_counter()
    result = fit_losvd_with_spectral_prior(A_stack_w, y_w, cfg, alpha=alpha, beta=beta)
    runtime_s = time.perf_counter() - t0
    moments = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)

    b_n = result["b"] / np.trapezoid(np.maximum(result["b"], 0.0), v_grid)
    core = b_n > 0.01 * b_n.max()
    wing_mass = float(np.trapezoid(b_n[~core], v_grid[~core])) if (~core).any() else 0.0

    return dict(regime=regime_name, alpha=alpha, beta=beta, seed=seed,
                true_v=true_v_exact, true_s=true_s_exact,
                v_bias=moments["vherm"] - true_v_exact, sig_bias=moments["sherm"] - true_s_exact,
                h3_bias=moments["h3"] - h3_true, h4_bias=moments["h4"] - h4_true,
                success=result["success"], wing_mass=wing_mass, runtime_s=runtime_s)


FIELDS = ["regime", "alpha", "beta", "seed", "true_v", "true_s", "v_bias", "sig_bias",
          "h3_bias", "h4_bias", "success", "wing_mass", "runtime_s"]


def run(outfile, regimes=None, alpha_grid=ALPHA_GRID, beta_grid=BETA_GRID, n_seeds=N_SEEDS, max_workers=None):
    regimes = regimes or list(REGIMES.keys())
    max_workers = max_workers or os.cpu_count()
    jobs = [(regime, alpha, beta, seed)
            for regime in regimes for beta in beta_grid for alpha in alpha_grid for seed in range(n_seeds)]
    print(f"total jobs: {len(jobs)}  workers: {max_workers}", flush=True)

    t0 = time.time()
    n_done = 0
    write_header = not os.path.exists(outfile)
    with open(outfile, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_run_one, regime, alpha, beta, seed) for regime, alpha, beta, seed in jobs]
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"JOB FAILED: {e}", flush=True)
                    continue
                writer.writerow(row)
                f.flush()
                n_done += 1
                if n_done % 50 == 0 or n_done == len(jobs):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    eta = (len(jobs) - n_done) / rate if rate > 0 else float("nan")
                    print(f"  {n_done}/{len(jobs)} done, t={elapsed:.0f}s, rate={rate:.2f}/s, ETA={eta:.0f}s", flush=True)
    print(f"DONE. total wall time: {time.time()-t0:.0f}s -> {outfile}", flush=True)


if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "spectral_prior_sweep_results.csv")
    run(outfile)
