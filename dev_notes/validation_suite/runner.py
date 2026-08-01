"""Fit one randomized mock case and return a standardized, flat result record.

Every record reports the nominal generator parameters, the truth-curve's own
moments (the only valid bias baseline -- see `mocks.py`'s module docstring),
the generator-mismatch diagnostic, and the recovered moments/bias, plus enough
fit diagnostics (lam_used, success, clipped_mass_frac, n_local_maxima) that a
downstream analysis can stratify or exclude pathological cases rather than
average over them silently.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from kinextract import compute_losvd_moments, whiten
from kinextract.multi_template import fit_losvd

from mocks import EMILES_GRID_DEFAULT, MockCase, build_mock, draw_random_params


def fit_case(mock: MockCase, noise_seed: int, rho: float = 0.0) -> dict:
    """Add noise (seeded) to `mock`'s noiseless spectrum, fit it, and return
    a single flat dict -- one row of a results table.
    """
    p = mock.params
    noise_level = np.median(mock.galaxy_noiseless) / p["snr"]
    errors = np.full(len(mock.wave_fit), noise_level)
    rng = np.random.default_rng(noise_seed)
    galaxy = mock.galaxy_noiseless + rng.normal(0.0, noise_level, len(mock.wave_fit))

    A_stack_w, y_w, _L = whiten(mock.A_stack, galaxy, errors, rho=rho)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_losvd(A_stack_w, y_w, mock.cfg)
    wall_s = time.time() - t0

    b = np.asarray(result["b"])
    moments = compute_losvd_moments(mock.v_grid, b, fit_h3h4=True)
    w = np.asarray(result["w"], dtype=float)
    w_frac = w / w.sum() if w.sum() > 0 else w

    tm = mock.truth_moments
    gm = mock.generator_mismatch
    row = {
        # provenance / nominal inputs
        "noise_seed": noise_seed,
        "truth_kind": p.get("truth_kind", "gh"),
        "nominal_v": p["v"], "nominal_sigma": p["sigma"], "nominal_h3": p["h3"], "nominal_h4": p["h4"],
        "nominal_wiggle_scale": p.get("wiggle_scale", np.nan),
        "snr": p["snr"], "n_templates": p["n_templates"], "preset": p["preset"],
        "gaussian_only": p["gaussian_only"],
        # the only valid "truth" -- the truth curve's OWN moments
        "truth_v": tm["vherm"], "truth_sigma": tm["sherm"], "truth_h3": tm["h3"], "truth_h4": tm["h4"],
        # generator diagnostics: how far the truth curve is from its own nominal
        # generator parameters, and how much it needed clipping to stay >= 0
        "gen_mismatch_dv": gm["dv"], "gen_mismatch_dsigma": gm["dsigma"],
        "gen_mismatch_dsigma_frac": gm["dsigma_frac"], "gen_mismatch_dh3": gm["dh3"],
        "gen_mismatch_dh4": gm["dh4"],
        "clipped_mass_frac": mock.clipped_mass_frac, "n_local_maxima": mock.n_local_maxima,
        # recovered
        "fit_v": moments["vherm"], "fit_sigma": moments["sherm"],
        "fit_h3": moments["h3"], "fit_h4": moments["h4"],
        # bias relative to the truth curve -- the ONLY correct comparison
        "bias_v": moments["vherm"] - tm["vherm"],
        "bias_sigma": moments["sherm"] - tm["sherm"],
        "bias_sigma_frac": (moments["sherm"] - tm["sherm"]) / tm["sherm"] if tm["sherm"] else np.nan,
        "bias_h3": moments["h3"] - tm["h3"],
        "bias_h4": moments["h4"] - tm["h4"],
        # fit mechanics
        "lam_used": result.get("lam_used", np.nan),
        "success": bool(result.get("success", True)),
        "wall_s": wall_s,
        "max_template_weight_frac": float(w_frac.max()) if w_frac.size else np.nan,
    }
    return row


def run_one(rng: np.random.Generator, noise_seed: int, fixed: dict | None = None,
            preset: str = "safe", gaussian_only: bool = False, n_templates: int = 4,
            emiles_ages=None, emiles_metals=None, emiles_grid_path: str = EMILES_GRID_DEFAULT,
            wave_fit=None, truth_kind: str = "gh") -> dict:
    """Draw one randomized case, build it, fit it, and return its result row.

    `rng` drives the PARAMETER draw (sigma/h3/h4 or wiggle_scale/templates/
    weights not in `fixed`); `noise_seed` separately and independently drives
    the NOISE draw, so parameter randomization and noise randomization are
    decoupled -- you can refit the same drawn case at several noise seeds, or
    draw fresh cases for every seed, depending on what the sweep needs.

    `truth_kind`: "gh" (default, the truncated Gauss-Hermite generator) or
    "spline" (a GH-free truth defined directly as exp(spline) -- see
    `mocks.build_spline_truth`). See `mocks.build_mock`'s docstring for how
    `generator_mismatch` differs in meaning between the two.
    """
    params = draw_random_params(rng, fixed=fixed, preset=preset, gaussian_only=gaussian_only,
                                 n_templates=n_templates, emiles_ages=emiles_ages, emiles_metals=emiles_metals,
                                 truth_kind=truth_kind)
    mock = build_mock(params, emiles_grid_path=emiles_grid_path, wave_fit=wave_fit)
    return fit_case(mock, noise_seed=noise_seed)
