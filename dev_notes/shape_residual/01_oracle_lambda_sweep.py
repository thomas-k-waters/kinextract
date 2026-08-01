"""Oracle-lambda diagnostic for the shape-residual problem
(REGULARIZATION_SHAPE_RESIDUAL_HANDOFF.md).

Question this answers, per (S/N, seed), single-template sigma=50 case:
for every fixed `lam` on an extended grid, what is the core shape
residual, and what does the AIC-local criterion (chi2 + 2*m_eff) look
like there? From that we get, per realization:

  - `lam_oracle`  : the fixed lam minimizing max core shape residual
                    (uses the truth -- not available in production, this
                    is a diagnostic bound).
  - `lam_aic`     : argmin of chi2 + 2*m_eff over the *default*
                    `FitConfig.lam_grid` subset -- exactly what
                    `lam_selection='aic_local'` would pick.
  - residuals at both, so the total defect splits into
    "penalty-form floor" (residual at oracle lam) vs
    "selection gap" (residual at AIC lam minus residual at oracle lam).

Also records, per (snr, seed, lam): m_eff, chi2, objective_gap, V/sigma
moments, wing-mass ratio (recovered/true mass at 3-6 sigma from center),
and the curvature-penalty value sum((D2@coeffs)^2) vs. the truth's own
projected value -- the handoff's suggested direct curvature test.

Writes one CSV row per (snr, seed, lam) incrementally; saves all
recovered curves to an NPZ at the end for direct plotting.
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

HERE = os.path.dirname(os.path.abspath(__file__))
EMILES_GRID = os.path.join(HERE, "..", "..", "src", "kinextract", "template_data", "emiles_muse_grid.npz")
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
NSIG = 7.0
TRUE_V, SIGMA = 20.0, 50.0

LAM_GRID_EXT = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
DEFAULT_LAM_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)  # FitConfig default
SNR_LIST = [30.0, 60.0, 100.0, 200.0]
SEEDS = list(range(15))

CSV_PATH = os.path.join(HERE, "01_oracle_lambda_sweep.csv")
NPZ_PATH = os.path.join(HERE, "01_oracle_lambda_curves.npz")

FIELDS = ["snr", "seed", "lam", "max_resid_pct", "frac_over5_pct",
          "signed_peak_resid_pct", "wing_mass_ratio", "m_eff", "chi2",
          "criterion", "objective_gap", "objective", "curv_fit", "curv_truth",
          "V", "sigma_mom"]


def run_case(snr, seed):
    # imports inside worker for ProcessPoolExecutor friendliness
    from kinextract import (FitConfig, build_template_stack, gauss_hermite_model,
                                 load_emiles_templates, compute_losvd_moments)
    from kinextract.aic_regularization import estimate_m_eff_local
    from kinextract.losvd import build_bspline_basis, difference_matrix

    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
    mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
    wave_fit = template_wave[mask]
    cfg = FitConfig(vmin=TRUE_V - NSIG * SIGMA, vmax=TRUE_V + NSIG * SIGMA)
    v_grid = cfg.build_velocity_grid()
    b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, TRUE_V, SIGMA, 0.0, 0.0), 0.0)
    b_truth /= b_truth.sum()
    A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth

    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)  # same convention as the handoff appendix
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors

    b_truth_n = b_truth / np.trapezoid(b_truth, v_grid)
    peak = b_truth_n.max()
    core = b_truth_n > 0.01 * peak
    peak_idx = int(np.argmax(b_truth_n))
    dist = np.abs(v_grid - TRUE_V) / SIGMA
    wing_band = (dist >= 3.0) & (dist <= 6.0)
    truth_wing_mass = np.trapezoid(np.where(wing_band, b_truth_n, 0.0), v_grid)

    # truth coefficients projected onto the same basis (for the curvature test)
    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D2 = difference_matrix(n_coef, order=2, dv=dv, dv_ref=cfg.knot_spacing_kms)
    log_truth = np.log(np.clip(b_truth_n, 1e-10 * peak, None))
    coeffs_truth, *_ = np.linalg.lstsq(basis, log_truth, rcond=None)
    curv_truth = float(np.sum((D2 @ coeffs_truth) ** 2))

    rows, curves = [], {}
    for lam in LAM_GRID_EXT:
        m_eff, chi2, result = estimate_m_eff_local(A_stack_w, y_w, cfg, lam)
        b_fit_n = result["b"] / np.trapezoid(np.maximum(result["b"], 0.0), v_grid)
        resid_signed = (b_fit_n - b_truth_n) / peak * 100
        resid = np.abs(resid_signed)
        mom = compute_losvd_moments(v_grid, b_fit_n, fit_h3h4=False)
        fit_wing_mass = np.trapezoid(np.where(wing_band, b_fit_n, 0.0), v_grid)
        rows.append(dict(
            snr=snr, seed=seed, lam=lam,
            max_resid_pct=round(float(resid[core].max()), 3),
            frac_over5_pct=round(float(100 * np.mean(resid[core] > 5)), 2),
            signed_peak_resid_pct=round(float(resid_signed[peak_idx]), 3),
            wing_mass_ratio=round(float(fit_wing_mass / truth_wing_mass), 3),
            m_eff=round(float(m_eff), 3), chi2=round(float(chi2), 3),
            criterion=round(float(chi2 + 2.0 * m_eff), 3),
            objective_gap=result.get("objective_gap"), objective=round(float(result["objective"]), 3),
            curv_fit=round(float(np.sum((D2 @ result["coeffs"]) ** 2)), 4),
            curv_truth=round(curv_truth, 4),
            V=round(float(mom["vherm"]), 3), sigma_mom=round(float(mom["sherm"]), 3),
        ))
        curves[f"b_{snr:.0f}_{seed}_{lam}"] = b_fit_n
    curves[f"truth_{snr:.0f}_{seed}"] = b_truth_n
    return rows, curves, v_grid


def main():
    t0 = time.perf_counter()
    write_header = not os.path.exists(CSV_PATH)
    all_curves = {}
    v_grid_out = None
    with open(CSV_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            futs = {ex.submit(run_case, snr, seed): (snr, seed)
                    for snr in SNR_LIST for seed in SEEDS}
            done = 0
            for fut in as_completed(futs):
                snr, seed = futs[fut]
                rows, curves, v_grid = fut.result()
                v_grid_out = v_grid
                all_curves.update(curves)
                for row in rows:
                    writer.writerow(row)
                fh.flush()
                done += 1
                print(f"[{time.perf_counter()-t0:7.1f}s] done snr={snr:.0f} seed={seed} "
                      f"({done}/{len(futs)})", flush=True)
    all_curves["v_grid"] = v_grid_out
    np.savez(NPZ_PATH, **all_curves)
    print(f"finished in {time.perf_counter()-t0:.1f}s -> {CSV_PATH}, {NPZ_PATH}")


if __name__ == "__main__":
    main()
