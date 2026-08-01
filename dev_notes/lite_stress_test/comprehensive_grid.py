"""Comprehensive baseline stress test: sigma x S/N x template-recipe grid,
characterizing kinextract's *current default* behavior
(`lam_selection='aic_local'`, `wing_lam=0.0` -- what a user gets out of
the box, no manual tuning) across a much broader regime than any single
notebook covers. Companion to `../../PARAMETER_GUIDE.md` and
`../../VALIDATION_CAMPAIGN.md` -- this is the empirical backing for a
comprehensive status report, not a one-off notebook demo.

Four template-recipe cases, deliberately spanning the identifiability
spectrum:
  - "single"  : one real E-MILES template, no mixture-weight problem at all.
  - "mix3"    : the well-behaved 3-template recipe from
                `01_basic_mock_fit.ipynb`.
  - "mix4deg" : the genuinely near-collinear 4-template E-MILES recipe from
                `02_realistic_mock_fit.ipynb` (h3=0.03, h4=0.05 truth).
  - "muse33"  : the real 32-star MUSE library with a random Dirichlet
                mixture (same convention as
                `dev_notes/scratch_notebooks/bias_sweep_fixed_vs_adaptive.py`)
                -- the most realistic many-template case.

Writes incrementally to a CSV (one row per (sigma, snr, recipe, rep)) so
progress is inspectable while running and safe against interruption.
Parallelized across all available cores via `ProcessPoolExecutor` --
independent computations, no shared state, no risk to any statistical
result.
"""
import csv
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import (
    FitConfig, build_template_stack, gauss_hermite_model, compute_losvd_moments, load_emiles_templates,
)
from kinextract.multi_template import fit_losvd

EMILES_GRID = os.path.join(os.path.dirname(__file__), "..", "..", "src", "kinextract",
                            "template_data", "emiles_muse_grid.npz")
MUSE_TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
NSIG = 7.0

SIGMA_GRID = [20.0, 30.0, 50.0, 80.0, 140.0, 220.0, 300.0, 400.0]
SNR_GRID = [15.0, 20.0, 30.0, 50.0, 60.0, 100.0, 200.0]
RECIPES = ["single", "mix3", "mix4deg", "muse33"]
N_REPS = 20
V_TRUE = 20.0  # nonzero on purpose -- V-bias-at-zero is a degenerate special case
SEED_BASE = 9_000_000

_CACHE = {}


def _get_recipe(recipe):
    """Loads/builds (templates, template_wave, true_weights, h3_true, h4_true) for
    a named recipe, memoized per worker process (each recipe's templates are
    loaded from disk once per process, not once per job)."""
    if recipe in _CACHE:
        return _CACHE[recipe]

    if recipe == "single":
        templates, template_wave, names, ages, metals = load_emiles_templates(
            EMILES_GRID, select=[(5.0, 0.06)])
        true_weights = np.array([1.0])
        h3_true, h4_true = 0.0, 0.0
    elif recipe == "mix3":
        templates, template_wave, names, ages, metals = load_emiles_templates(
            EMILES_GRID, select=[(1.0, -0.66), (5.0, 0.06), (14.0, 0.40)])
        true_weights = np.array([0.3, 0.4, 0.3])
        h3_true, h4_true = 0.0, 0.0
    elif recipe == "mix4deg":
        templates, template_wave, names, ages, metals = load_emiles_templates(
            EMILES_GRID, select=[(1.0, -0.66), (5.0, 0.06), (14.0, 0.06), (14.0, 0.40)])
        true_weights = np.array([0.15, 0.35, 0.35, 0.15])
        h3_true, h4_true = 0.03, 0.05
    elif recipe == "muse33":
        with open(os.path.join(MUSE_TEMPLATE_DIR, "Tlist")) as f:
            names = [line.strip() for line in f if line.strip()]
        template_wave = 4750.0 + np.arange(3681) * 1.25
        templates = [np.interp(template_wave, *np.loadtxt(os.path.join(MUSE_TEMPLATE_DIR, n)).T) for n in names]
        rng_mix = np.random.default_rng(20000)
        true_weights = rng_mix.dirichlet(np.ones(len(templates)))
        h3_true, h4_true = 0.0, 0.0
    else:
        raise ValueError(recipe)

    out = (templates, template_wave, true_weights, h3_true, h4_true)
    _CACHE[recipe] = out
    return out


def _run_one(sigma, snr, recipe, rep):
    templates, template_wave, true_weights, h3_true, h4_true = _get_recipe(recipe)
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
    result = fit_losvd(A_stack_w, y_w, cfg)
    runtime_s = time.perf_counter() - t0
    moments = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)

    A_eff_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0]))
    model_w = A_eff_w @ result["b"]
    chi2_red = float(np.sum((model_w - y_w) ** 2)) / len(wave_fit)

    b_n = result["b"] / np.trapezoid(np.maximum(result["b"], 0.0), v_grid)
    core = b_n > 0.01 * b_n.max()
    wing_mass = float(np.trapezoid(b_n[~core], v_grid[~core])) if (~core).any() else 0.0

    return dict(
        sigma=sigma, snr=snr, recipe=recipe, rep=rep, n_templates=len(templates),
        true_v=true_v, true_s=true_s, h3_true=h3_true, h4_true=h4_true,
        v_rec=moments["vherm"], sig_rec=moments["sherm"], h3_rec=moments["h3"], h4_rec=moments["h4"],
        v_bias=moments["vherm"] - true_v, sig_bias=moments["sherm"] - true_s,
        h3_bias=moments["h3"] - h3_true, h4_bias=moments["h4"] - h4_true,
        chi2_red=chi2_red, lam_used=result["lam_used"], success=result["success"],
        wing_mass=wing_mass, runtime_s=runtime_s,
    )


FIELDS = ["sigma", "snr", "recipe", "rep", "n_templates", "true_v", "true_s", "h3_true", "h4_true",
          "v_rec", "sig_rec", "h3_rec", "h4_rec", "v_bias", "sig_bias", "h3_bias", "h4_bias",
          "chi2_red", "lam_used", "success", "wing_mass", "runtime_s"]


def run(outfile, sigma_grid=SIGMA_GRID, snr_grid=SNR_GRID, recipes=RECIPES, n_reps=N_REPS,
        max_workers=None):
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
                    print(f"  {n_done}/{len(jobs)} done, t={elapsed:.0f}s, "
                          f"rate={rate:.2f}/s, ETA={eta:.0f}s", flush=True)
    print(f"DONE. total wall time: {time.time()-t0:.0f}s -> {outfile}", flush=True)


if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "comprehensive_grid_results.csv")
    run(outfile)
