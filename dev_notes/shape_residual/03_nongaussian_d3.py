"""Non-Gaussian-truth test for the D3 penalty: does shrink-toward-
Gaussian erase real h3/h4 structure at the lam a data-driven rule picks?

Truth shapes: (h3,h4) in {(0.10,0.00), (0.00,0.10), (0.06,-0.06)} --
within the package's own validation-grid range. Same single-template
sigma=50 setup and seeds as 01/02. Both penalty orders on each noise
draw (paired): D2 with its own lam grid, D3 with the extended grid.

Records the same core-residual metrics plus recovered h3/h4 (via
compute_losvd_moments) and the truth curve's own h3/h4 through the
same moment pipeline (so GH-parameter vs moment-definition differences
don't contaminate the bias readout).
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

SHAPES = [(0.10, 0.00), (0.00, 0.10), (0.06, -0.06)]
LAMS = {2: [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        3: [0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0]}
SNR_LIST = [30.0, 60.0, 100.0]
SEEDS = list(range(15))

CSV_PATH = os.path.join(HERE, "03_nongaussian_d3.csv")

FIELDS = ["h3_true", "h4_true", "snr", "seed", "order", "lam",
          "max_resid_pct", "frac_over5_pct", "wing_mass_ratio",
          "m_eff", "chi2", "V", "sigma_mom", "h3_fit", "h4_fit",
          "h3_truthcurve", "h4_truthcurve"]


def run_case(h3, h4, snr, seed):
    from kinextract import (FitConfig, build_template_stack, gauss_hermite_model,
                                 load_emiles_templates, compute_losvd_moments)
    from kinextract.losvd import build_bspline_basis, difference_matrix
    from kinextract.multi_template import fit_losvd_basis

    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
    mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
    wave_fit = template_wave[mask]
    cfg = FitConfig(vmin=TRUE_V - NSIG * SIGMA, vmax=TRUE_V + NSIG * SIGMA)
    v_grid = cfg.build_velocity_grid()
    b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, TRUE_V, SIGMA, h3, h4), 0.0)
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
    dist = np.abs(v_grid - TRUE_V) / SIGMA
    wing_band = (dist >= 3.0) & (dist <= 6.0)
    truth_wing_mass = np.trapezoid(np.where(wing_band, b_truth_n, 0.0), v_grid)
    mom_t = compute_losvd_moments(v_grid, b_truth_n, fit_h3h4=True)

    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)

    rows = []
    for order, lam_grid in LAMS.items():
        D = difference_matrix(n_coef, order=order, dv=dv, dv_ref=cfg.knot_spacing_kms)
        DtD = D.T @ D
        for lam in lam_grid:
            result = fit_losvd_basis(A_stack_w, y_w, basis, D, lam=lam)
            b, w = result["b"], result["w"]
            A_eff = sum(w[k] * A_stack_w[k] for k in range(len(w)))
            chi2 = float(np.sum((y_w - A_eff @ b) ** 2))
            J_b = b[:, None] * basis
            X = A_eff @ J_b
            XtX = X.T @ X
            m_eff = float(np.trace(np.linalg.lstsq(XtX + lam * DtD, XtX, rcond=None)[0]))

            b_fit_n = b / np.trapezoid(np.maximum(b, 0.0), v_grid)
            resid = np.abs(b_fit_n - b_truth_n) / peak * 100
            mom = compute_losvd_moments(v_grid, b_fit_n, fit_h3h4=True)
            fit_wing_mass = np.trapezoid(np.where(wing_band, b_fit_n, 0.0), v_grid)
            rows.append(dict(
                h3_true=h3, h4_true=h4, snr=snr, seed=seed, order=order, lam=lam,
                max_resid_pct=round(float(resid[core].max()), 3),
                frac_over5_pct=round(float(100 * np.mean(resid[core] > 5)), 2),
                wing_mass_ratio=round(float(fit_wing_mass / truth_wing_mass), 3),
                m_eff=round(m_eff, 3), chi2=round(chi2, 3),
                V=round(float(mom["vherm"]), 3), sigma_mom=round(float(mom["sherm"]), 3),
                h3_fit=round(float(mom["h3"]), 4), h4_fit=round(float(mom["h4"]), 4),
                h3_truthcurve=round(float(mom_t["h3"]), 4), h4_truthcurve=round(float(mom_t["h4"]), 4),
            ))
    return rows


def main():
    t0 = time.perf_counter()
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            futs = {ex.submit(run_case, h3, h4, snr, seed): (h3, h4, snr, seed)
                    for (h3, h4) in SHAPES for snr in SNR_LIST for seed in SEEDS}
            done = 0
            for fut in as_completed(futs):
                for row in fut.result():
                    writer.writerow(row)
                fh.flush()
                done += 1
                print(f"[{time.perf_counter()-t0:7.1f}s] {done}/{len(futs)} {futs[fut]}", flush=True)
    print(f"finished in {time.perf_counter()-t0:.1f}s -> {CSV_PATH}")


if __name__ == "__main__":
    main()
