"""06c: the FINAL candidate configuration, run through the actual
production code path — `multi_template.fit_losvd` with
`FitConfig(penalty_order=3)` (order-aware lam grid + the delta-AIC<=2
parsimony tiebreak now wired into `select_lambda_aic_local`) under JAX
float64.

Same grid/seeds/data construction as 06/06b; candidate arm only (the
paired default arm is 06b's `default_d2_aic_x64` rows, same draws).
Method name: `candidate_d3_guard_x64`.
"""
import csv
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import jax; jax.config.update("jax_enable_x64", True)
import warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMILES_GRID = os.path.join(HERE, "..", "..", "src", "kinextract", "template_data", "emiles_muse_grid.npz")
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
NSIG = 7.0
TRUE_V = 20.0

SIGMAS = [20.0]
SNR_LIST = [30.0, 100.0]
SHAPES = [(0.0, 0.0), (0.06, 0.06)]
RECIPES = ["single", "mix3"]
SEEDS = list(range(15))

CSV_PATH = os.path.join(HERE, "06d_sigma20_guarded_x64.csv")
FIELDS = ["sigma_true", "snr", "h3_true", "h4_true", "recipe", "seed", "method",
          "lam_used", "max_resid_pct", "frac_over5_pct", "wing_mass_ratio",
          "V", "sigma_mom", "h3_fit", "h4_fit",
          "V_truthcurve", "sherm_truthcurve", "h3_truthcurve", "h4_truthcurve",
          "runtime_s"]

_CACHE = {}


def _get_recipe(recipe):
    from kinextract import load_emiles_templates
    if recipe in _CACHE:
        return _CACHE[recipe]
    if recipe == "single":
        templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
        true_weights = np.array([1.0])
    elif recipe == "mix3":
        templates, template_wave, *_ = load_emiles_templates(
            EMILES_GRID, select=[(1.0, -0.66), (5.0, 0.06), (14.0, 0.40)])
        true_weights = np.array([0.3, 0.4, 0.3])
    else:
        raise ValueError(recipe)
    _CACHE[recipe] = (templates, template_wave, true_weights)
    return _CACHE[recipe]


def run_case(sigma, snr, h3t, h4t, recipe, seed):
    from kinextract import (FitConfig, build_template_stack, gauss_hermite_model,
                                 compute_losvd_moments)
    from kinextract.multi_template import fit_losvd

    templates, template_wave, true_weights = _get_recipe(recipe)
    mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
    wave_fit = template_wave[mask]
    cfg = FitConfig(vmin=TRUE_V - NSIG * sigma, vmax=TRUE_V + NSIG * sigma, penalty_order=3)
    v_grid = cfg.build_velocity_grid()
    b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, TRUE_V, sigma, h3t, h4t), 0.0)
    b_truth /= b_truth.sum()
    A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
    galaxy_noiseless = sum(true_weights[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))

    noise_level = np.median(galaxy_noiseless) / snr
    data_seed = zlib.crc32(f"{sigma}_{snr}_{h3t}_{h4t}_{recipe}_{seed}".encode()) % (2 ** 32)
    rng = np.random.default_rng(data_seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors

    b_truth_n = b_truth / np.trapezoid(b_truth, v_grid)
    peak = b_truth_n.max()
    core = b_truth_n > 0.01 * peak
    dist = np.abs(v_grid - TRUE_V) / sigma
    wing_band = (dist >= 3.0) & (dist <= 6.0)
    truth_wing_mass = np.trapezoid(np.where(wing_band, b_truth_n, 0.0), v_grid)
    mom_t = compute_losvd_moments(v_grid, b_truth_n, fit_h3h4=True)

    t0 = time.perf_counter()
    res = fit_losvd(A_stack_w, y_w, cfg)
    runtime = time.perf_counter() - t0

    b_fit_n = res["b"] / np.trapezoid(np.maximum(res["b"], 0.0), v_grid)
    resid = np.abs(b_fit_n - b_truth_n) / peak * 100
    mom = compute_losvd_moments(v_grid, b_fit_n, fit_h3h4=True)
    fit_wing_mass = np.trapezoid(np.where(wing_band, b_fit_n, 0.0), v_grid)
    return [dict(
        sigma_true=sigma, snr=snr, h3_true=h3t, h4_true=h4t, recipe=recipe,
        seed=seed, method="candidate_d3_guard_x64", lam_used=res["lam_used"],
        max_resid_pct=round(float(resid[core].max()), 3),
        frac_over5_pct=round(float(100 * np.mean(resid[core] > 5)), 2),
        wing_mass_ratio=round(float(fit_wing_mass / truth_wing_mass), 3),
        V=round(float(mom["vherm"]), 3), sigma_mom=round(float(mom["sherm"]), 3),
        h3_fit=round(float(mom["h3"]), 4), h4_fit=round(float(mom["h4"]), 4),
        V_truthcurve=round(float(mom_t["vherm"]), 3),
        sherm_truthcurve=round(float(mom_t["sherm"]), 3),
        h3_truthcurve=round(float(mom_t["h3"]), 4), h4_truthcurve=round(float(mom_t["h4"]), 4),
        runtime_s=round(runtime, 2),
    )]


def main():
    t0 = time.perf_counter()
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            futs = {ex.submit(run_case, sg, snr, h3, h4, rec, seed): (sg, snr, h3, h4, rec, seed)
                    for sg in SIGMAS for snr in SNR_LIST for (h3, h4) in SHAPES
                    for rec in RECIPES for seed in SEEDS}
            done = 0
            for fut in as_completed(futs):
                for row in fut.result():
                    writer.writerow(row)
                fh.flush()
                done += 1
                print(f"[{time.perf_counter()-t0:8.1f}s] {done}/{len(futs)} {futs[fut]}", flush=True)
    print(f"finished in {time.perf_counter()-t0:.1f}s -> {CSV_PATH}")


if __name__ == "__main__":
    main()
