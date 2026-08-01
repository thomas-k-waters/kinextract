"""Paired A/B check of `FitConfig.self_normalize_lam_grid` at the specific
low-sigma regime (sigma 25-55, NSIG=7 grid) where `min_interior_knots`
floors the knot count and the fixed absolute `lam_grid` becomes uniformly
inflated in effective strength -- see `select_lambda_aic_local`'s
`self_normalize_lam_grid` docstring and `FitConfig`'s own field help text.

Multiple seeds per sigma, same mock/template/grid for both arms, only
`self_normalize_lam_grid` differs. Reports recovered sigma bias (against
the known spline-truth curve's own moments, not nominal params -- see
dev_notes/validation_suite/FINDINGS.md) with and without the fix.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")
from kinextract import FitConfig, build_template_stack, build_spline_truth, compute_losvd_moments


def _mock(sigma, snr, seed):
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]

    template = np.ones(n_pix)
    for cen, depth in zip([8498.02, 8542.09, 8662.14], [0.55, 0.70, 0.65]):
        template -= depth * np.exp(-0.5 * ((wavelength - cen) / 5.0) ** 2)

    cfg_base = FitConfig(vmin=-7 * sigma, vmax=7 * sigma)
    v_grid = cfg_base.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=0.0, sigma=sigma, wiggle_scale=0.15, seed=seed)
    truth_moments = compute_losvd_moments(v_grid, b_truth)
    A_stack = build_template_stack(wave_fit, wavelength, [template], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed + 10000)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors
    return A_stack_w, y_w, truth_moments


def main():
    from kinextract.multi_template import fit_losvd
    from kinextract import compute_direct_moments

    sigmas = [25.0, 35.0, 45.0, 55.0]
    n_seeds = 6
    snr = 60.0

    print(f"{'sigma':>6} {'seed':>4} {'truth':>8} {'off':>8} {'on':>8} "
          f"{'bias_off%':>10} {'bias_on%':>9} {'lam_off':>10} {'lam_on':>10}")
    summary = {}
    for sigma in sigmas:
        biases_off, biases_on = [], []
        for seed in range(n_seeds):
            A_stack_w, y_w, truth_moments = _mock(sigma, snr, seed)
            truth_sigma = truth_moments["sherm"]

            cfg_off = FitConfig(vmin=-7 * sigma, vmax=7 * sigma, self_normalize_lam_grid=False)
            cfg_on = FitConfig(vmin=-7 * sigma, vmax=7 * sigma, self_normalize_lam_grid=True)

            r_off = fit_losvd(A_stack_w, y_w, cfg_off)
            r_on = fit_losvd(A_stack_w, y_w, cfg_on)

            v_grid = cfg_off.build_velocity_grid()
            s_off = compute_direct_moments(v_grid, r_off["b"])["sherm"]
            s_on = compute_direct_moments(v_grid, r_on["b"])["sherm"]

            bias_off = 100.0 * (s_off - truth_sigma) / truth_sigma
            bias_on = 100.0 * (s_on - truth_sigma) / truth_sigma
            biases_off.append(bias_off)
            biases_on.append(bias_on)

            print(f"{sigma:6.0f} {seed:4d} {truth_sigma:8.2f} {s_off:8.2f} {s_on:8.2f} "
                  f"{bias_off:10.1f} {bias_on:9.1f} {r_off['lam_used']:10.4g} {r_on['lam_used']:10.4g}")

        summary[sigma] = (np.median(biases_off), np.median(biases_on),
                           np.median(np.abs(biases_off)), np.median(np.abs(biases_on)))

    print("\n=== summary (median signed / median |bias|, %) ===")
    print(f"{'sigma':>6} {'median_off':>11} {'median_on':>10} {'medabs_off':>11} {'medabs_on':>10}")
    for sigma, (med_off, med_on, medabs_off, medabs_on) in summary.items():
        print(f"{sigma:6.0f} {med_off:11.1f} {med_on:10.1f} {medabs_off:11.1f} {medabs_on:10.1f}")


if __name__ == "__main__":
    main()
