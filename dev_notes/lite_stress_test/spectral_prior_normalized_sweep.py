"""Final validation of the eigenvalue-normalization fix (Stage 21) for
kinextract.spectral_prior, across the *entire* comprehensive_grid.py
cross-product (all 4 recipes, full sigma x S/N grid, 20 seeds -- matching
that grid's own convention).

NOTE on seeding: this uses `zlib.crc32(recipe.encode())`, not Python's
built-in `hash(recipe)` -- an earlier version of this script (and of
`comprehensive_grid.py`/`spectral_prior_broad_sweep.py`) used `hash()`,
which is randomized per-process for security and is NOT reproducible
across separate script invocations. The "identical seeds, true paired
comparison" claim in this module's own earlier version and in
`VALIDATION_CAMPAIGN.md` Stage 20/21's first pass was therefore wrong --
each sweep script actually drew *independent* noise from the same
generative model, not matched noise. The median-over-20-seeds comparisons
remain statistically valid either way (same distribution, just not
paired), but the tighter, lower-variance paired comparison was never
actually happening. Fixed here for real reproducibility going forward.

`(alpha=100.0, beta=1.5, normalize_eigenvalues=True)` -- found via a
properly-powered (20-seed, not the 5-8 seeds an earlier quick scan used
and was misled by) scan across the full sigma range: this pair keeps
sigma bias in a narrow ~2-5.5% band from sigma=20 to sigma=400, at a
fixed S/N=60, single-template recipe -- see `VALIDATION_CAMPAIGN.md`
Stage 21 for the full scan. This run checks whether that holds across
the *entire* grid (all 4 recipes, all S/N), not just the one slice it was
found on.
"""
import csv
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import FitConfig, build_template_stack, gauss_hermite_model, compute_losvd_moments
from kinextract.spectral_prior import fit_losvd_with_spectral_prior
import comprehensive_grid as cg

WAVEFITMIN, WAVEFITMAX = cg.WAVEFITMIN, cg.WAVEFITMAX
NSIG = cg.NSIG
V_TRUE = cg.V_TRUE
SEED_BASE = cg.SEED_BASE  # identical to comprehensive_grid.py -- identical noise draws

SIGMA_GRID = cg.SIGMA_GRID
SNR_GRID = cg.SNR_GRID
RECIPES = cg.RECIPES
ALPHA, BETA = 100.0, 1.5
N_REPS = cg.N_REPS  # 20, matching the baseline grid exactly


def _run_one(sigma, snr, recipe, rep):
    templates, template_wave, true_weights, h3_true, h4_true = cg._get_recipe(recipe)
    mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
    wave_fit = template_wave[mask]

    cfg = FitConfig(vmin=V_TRUE - NSIG * sigma, vmax=V_TRUE + NSIG * sigma)
    v_grid = cfg.build_velocity_grid()
    b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, V_TRUE, sigma, h3_true, h4_true), 0.0)
    b_truth /= b_truth.sum()
    A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)

    vv = np.linspace(V_TRUE - 15 * sigma, V_TRUE + 15 * sigma, 100_001)
    bb = np.maximum(gauss_hermite_model(vv, 1.0, V_TRUE, sigma, h3_true, h4_true), 0.0)
    tot = np.trapezoid(bb, vv)
    true_v = float(np.trapezoid(bb * vv, vv) / tot)
    true_s = float(np.sqrt(np.trapezoid(bb * (vv - true_v) ** 2, vv) / tot))

    seed = SEED_BASE + rep + int(sigma * 100) + int(snr * 7) + (zlib.crc32(recipe.encode()) % 10_000)
    galaxy_noiseless = sum(true_weights[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed % (2**32))
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors

    t0 = time.perf_counter()
    result = fit_losvd_with_spectral_prior(A_stack_w, y_w, cfg, alpha=ALPHA, beta=BETA, normalize_eigenvalues=True)
    runtime_s = time.perf_counter() - t0
    moments = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)

    b_n = result["b"] / np.trapezoid(np.maximum(result["b"], 0.0), v_grid)
    core = b_n > 0.01 * b_n.max()
    wing_mass = float(np.trapezoid(b_n[~core], v_grid[~core])) if (~core).any() else 0.0

    return dict(
        sigma=sigma, snr=snr, recipe=recipe, alpha=ALPHA, beta=BETA, rep=rep, n_templates=len(templates),
        true_v=true_v, true_s=true_s, h3_true=h3_true, h4_true=h4_true,
        v_bias=moments["vherm"] - true_v, sig_bias=moments["sherm"] - true_s,
        h3_bias=moments["h3"] - h3_true, h4_bias=moments["h4"] - h4_true,
        success=result["success"], wing_mass=wing_mass, runtime_s=runtime_s,
    )


FIELDS = ["sigma", "snr", "recipe", "alpha", "beta", "rep", "n_templates", "true_v", "true_s",
          "h3_true", "h4_true", "v_bias", "sig_bias", "h3_bias", "h4_bias", "success", "wing_mass", "runtime_s"]


def run(outfile, sigma_grid=SIGMA_GRID, snr_grid=SNR_GRID, recipes=RECIPES, n_reps=N_REPS, max_workers=None):
    max_workers = max_workers or os.cpu_count()
    jobs = [(sigma, snr, recipe, rep)
            for recipe in recipes for sigma in sigma_grid for snr in snr_grid for rep in range(n_reps)]
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
            futures = [ex.submit(_run_one, sigma, snr, recipe, rep) for sigma, snr, recipe, rep in jobs]
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
    outfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "spectral_prior_normalized_sweep_results.csv")
    run(outfile)
