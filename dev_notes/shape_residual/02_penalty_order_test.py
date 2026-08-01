"""Penalty-order test: does a third-difference (D3) penalty on the
log-spline coefficients remove the shape-residual floor that the
standard D2 penalty has (01_oracle_lambda_sweep)?

Rationale (see REGULARIZATION_INVESTIGATION_LOG.md): D2's null space is
*linear* log-b, so heavy smoothing drives the recovered LOSVD toward
exponential tails (the documented wing-inflation mechanism), which is
why no single lam can serve core and wings at once. D3's null space is
*quadratic* log-b -- any Gaussian (any V, sigma, amplitude) is
penalty-free -- so heavy smoothing shrinks toward the best-fitting
Gaussian instead. If right, the oracle-lam residual under D3 should
drop well below D2's floor at low S/N, with large lam values usable.

Same setup/seeds as 01 (single E-MILES template, V=20, sigma=50).
Writes one CSV row per (snr, seed, lam) incrementally.
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

# D3 is a weaker constraint per unit lam than D2, so the usable grid
# extends much higher.
LAM_GRID_D3 = [0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0]
SNR_LIST = [30.0, 60.0, 100.0, 200.0]
SEEDS = list(range(15))

CSV_PATH = os.path.join(HERE, "02_penalty_order_d3.csv")
NPZ_PATH = os.path.join(HERE, "02_penalty_order_d3_curves.npz")

FIELDS = ["snr", "seed", "lam", "max_resid_pct", "frac_over5_pct",
          "signed_peak_resid_pct", "wing_mass_ratio", "m_eff", "chi2",
          "criterion", "objective_gap", "V", "sigma_mom"]


def run_case(snr, seed):
    from kinextract import (FitConfig, build_template_stack, gauss_hermite_model,
                                 load_emiles_templates, compute_losvd_moments)
    from kinextract.losvd import build_bspline_basis, difference_matrix
    from kinextract.multi_template import fit_losvd_basis

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
    rng = np.random.default_rng(seed)
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

    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D3 = difference_matrix(n_coef, order=3, dv=dv, dv_ref=cfg.knot_spacing_kms)

    rows, curves = [], {}
    for lam in LAM_GRID_D3:
        result = fit_losvd_basis(A_stack_w, y_w, basis, D3, lam=lam)
        b = result["b"]
        w = result["w"]
        A_eff = sum(w[k] * A_stack_w[k] for k in range(len(w)))
        chi2 = float(np.sum((y_w - A_eff @ b) ** 2))
        # local m_eff with the D3 operator (same linearization as
        # aic_regularization.estimate_m_eff_local)
        J_b = b[:, None] * basis
        X = A_eff @ J_b
        XtX = X.T @ X
        M = XtX + lam * (D3.T @ D3)
        m_eff = float(np.trace(np.linalg.lstsq(M, XtX, rcond=None)[0]))

        b_fit_n = b / np.trapezoid(np.maximum(b, 0.0), v_grid)
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
            m_eff=round(m_eff, 3), chi2=round(chi2, 3),
            criterion=round(chi2 + 2.0 * m_eff, 3),
            objective_gap=result.get("losvd_objective_gap"),
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
