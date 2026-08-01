"""Sigma-range portability test for the D3 penalty: does one lam (or one
selection rule) work from sigma=20 to sigma=300, and which knot-spacing
scaling exponent makes lam portable across grid widths?

The package's difference_matrix scales D by (dv_ref/dv)^1 regardless of
order -- calibrated empirically for D2. For a third-difference operator
the physically natural exponent is 3 (a fixed smooth log-curve's third
differences scale as dv^3). Test both: for each sigma, sweep lam and
find the oracle; if the oracle lam drifts systematically with sigma
under exponent 1 but stays put under exponent 3 (or vice versa), that
decides the scaling.

Gaussian truth, single template, V=20. sigma in {20, 50, 140, 300},
S/N in {30, 100}, seeds 0..14, scale_pow in {1, 3}.
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
TRUE_V = 20.0

SIGMAS = [20.0, 50.0, 140.0, 300.0]
SNR_LIST = [30.0, 100.0]
SEEDS = list(range(15))
SCALE_POWS = [1, 3]
LAM_GRID = [0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0]

CSV_PATH = os.path.join(HERE, "04_sigma_range_d3.csv")
FIELDS = ["sigma_true", "snr", "seed", "scale_pow", "lam", "max_resid_pct",
          "frac_over5_pct", "wing_mass_ratio", "m_eff", "chi2", "V", "sigma_mom",
          "n_knots", "dv_knot"]


def run_case(sigma, snr, seed, scale_pow):
    from kinextract import (FitConfig, build_template_stack, gauss_hermite_model,
                                 load_emiles_templates, compute_losvd_moments)
    from kinextract.losvd import build_bspline_basis
    from kinextract.multi_template import fit_losvd_basis

    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
    mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
    wave_fit = template_wave[mask]
    cfg = FitConfig(vmin=TRUE_V - NSIG * sigma, vmax=TRUE_V + NSIG * sigma)
    v_grid = cfg.build_velocity_grid()
    b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, TRUE_V, sigma, 0.0, 0.0), 0.0)
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
    dist = np.abs(v_grid - TRUE_V) / sigma
    wing_band = (dist >= 3.0) & (dist <= 6.0)
    truth_wing_mass = np.trapezoid(np.where(wing_band, b_truth_n, 0.0), v_grid)

    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D3 = np.eye(n_coef)
    for _ in range(3):
        D3 = np.diff(D3, axis=0)
    D3 = D3 * (cfg.knot_spacing_kms / dv) ** scale_pow
    DtD = D3.T @ D3

    rows = []
    for lam in LAM_GRID:
        result = fit_losvd_basis(A_stack_w, y_w, basis, D3, lam=lam)
        b, w = result["b"], result["w"]
        A_eff = sum(w[k] * A_stack_w[k] for k in range(len(w)))
        chi2 = float(np.sum((y_w - A_eff @ b) ** 2))
        J_b = b[:, None] * basis
        X = A_eff @ J_b
        XtX = X.T @ X
        m_eff = float(np.trace(np.linalg.lstsq(XtX + lam * DtD, XtX, rcond=None)[0]))
        b_fit_n = b / np.trapezoid(np.maximum(b, 0.0), v_grid)
        resid = np.abs(b_fit_n - b_truth_n) / peak * 100
        mom = compute_losvd_moments(v_grid, b_fit_n, fit_h3h4=False)
        fit_wing_mass = np.trapezoid(np.where(wing_band, b_fit_n, 0.0), v_grid)
        rows.append(dict(
            sigma_true=sigma, snr=snr, seed=seed, scale_pow=scale_pow, lam=lam,
            max_resid_pct=round(float(resid[core].max()), 3),
            frac_over5_pct=round(float(100 * np.mean(resid[core] > 5)), 2),
            wing_mass_ratio=round(float(fit_wing_mass / truth_wing_mass), 3),
            m_eff=round(m_eff, 3), chi2=round(chi2, 3),
            V=round(float(mom["vherm"]), 3), sigma_mom=round(float(mom["sherm"]), 3),
            n_knots=n_knots, dv_knot=round(dv, 2),
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
            futs = {ex.submit(run_case, sg, snr, seed, sp): (sg, snr, seed, sp)
                    for sg in SIGMAS for snr in SNR_LIST for seed in SEEDS for sp in SCALE_POWS}
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
