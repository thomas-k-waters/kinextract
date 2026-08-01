"""Reproduce the handoff report's core finding exactly (single E-MILES
template, V=20, sigma=50, S/N=60, seed=0): default adaptive-lam fit
shows ~10.9% peak-region shape residual, noiseless fit ~3.8%.

Also times each fit so later sweeps can be sized realistically, and
saves the recovered/true curves to an NPZ for direct plotting (per the
project rule: look at the curve, not just the aggregate number).
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import FitConfig, build_template_stack, gauss_hermite_model, load_emiles_templates
from kinextract.multi_template import fit_losvd

HERE = os.path.dirname(os.path.abspath(__file__))
EMILES_GRID = os.path.join(HERE, "..", "..", "src", "kinextract", "template_data", "emiles_muse_grid.npz")
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
NSIG = 7.0
TRUE_V, SIGMA, SNR = 20.0, 50.0, 60.0

templates, template_wave, names, ages, metals = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
wave_fit = template_wave[mask]
cfg = FitConfig(vmin=TRUE_V - NSIG * SIGMA, vmax=TRUE_V + NSIG * SIGMA)
v_grid = cfg.build_velocity_grid()
b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, TRUE_V, SIGMA, 0.0, 0.0), 0.0)
b_truth /= b_truth.sum()
A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
galaxy_noiseless = A_stack[0] @ b_truth

b_truth_n = b_truth / np.trapezoid(np.maximum(b_truth, 0.0), v_grid)
peak = b_truth_n.max()
core = b_truth_n > 0.01 * peak

def metrics(result):
    b_fit_n = result["b"] / np.trapezoid(np.maximum(result["b"], 0.0), v_grid)
    resid = np.abs(b_fit_n - b_truth_n) / peak * 100
    return b_fit_n, resid[core].max(), 100 * np.mean(resid[core] > 5)

# --- noisy case (S/N=60, seed=0) ---
noise_level = np.median(galaxy_noiseless) / SNR
rng = np.random.default_rng(0)
galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
errors = np.full(len(wave_fit), noise_level)
A_stack_w = A_stack / errors[None, :, None]
y_w = galaxy / errors

t0 = time.perf_counter()
result_base = fit_losvd(A_stack_w, y_w, cfg)  # adaptive lam (default)
t_adaptive = time.perf_counter() - t0
b_base_n, maxres_base, frac_base = metrics(result_base)
print(f"default (adaptive lam={result_base.get('lam_used')}): max resid (core) = {maxres_base:.1f}%  "
      f"frac(core)>5% = {frac_base:.1f}%   [{t_adaptive:.1f}s]")

cfg_fixed = FitConfig(vmin=cfg.vmin, vmax=cfg.vmax, lam_selection="fixed", lam=2.0)
t0 = time.perf_counter()
result_fixed = fit_losvd(A_stack_w, y_w, cfg_fixed)
t_fixed = time.perf_counter() - t0
b_fixed_n, maxres_fixed, frac_fixed = metrics(result_fixed)
print(f"fixed lam=2:  max resid (core) = {maxres_fixed:.1f}%  frac(core)>5% = {frac_fixed:.1f}%   [{t_fixed:.1f}s]")

# --- noiseless case ---
errors_nl = np.full(len(wave_fit), np.median(galaxy_noiseless) * 1e-6)
result_nl = fit_losvd(A_stack / errors_nl[None, :, None], galaxy_noiseless / errors_nl, cfg)
b_nl_n, maxres_nl, frac_nl = metrics(result_nl)
print(f"noiseless (adaptive lam={result_nl.get('lam_used')}): max resid (core) = {maxres_nl:.1f}%  "
      f"frac(core)>5% = {frac_nl:.1f}%")

np.savez(os.path.join(HERE, "00_curves.npz"),
         v_grid=v_grid, b_truth_n=b_truth_n, b_base_n=b_base_n,
         b_fixed_n=b_fixed_n, b_nl_n=b_nl_n, core=core)
print("curves saved to 00_curves.npz")
