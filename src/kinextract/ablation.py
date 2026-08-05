"""Ablation-testing framework: quantify how much each individual fitting-
pipeline mechanism contributes to recovered sigma/h4 bias, in isolation
and cumulatively, on the *same* underlying synthetic data (real templates,
known truth, free NNLS weight estimation) -- extensible by design, so a
newly-added package feature gets a new `AblationConfig` field and a new
ladder entry, not a bespoke test written from scratch.

**Why this exists.** Layering a synthetic-calibrated regularization-bias
correction (`kinextract.bias_correction`) on top of the *whole*
fitting pipeline at once made it impossible to tell whether the
correction itself was wrong or some other already-present mechanism
(multi-start, the moment-extraction taper, h3/h4 estimation,
`shape_aware_correction`, ...) was interacting with it -- see
`bias_correction`'s own module docstring for that history. This module
answers "how much bias does each piece contribute on its own, and
cumulatively" first, before any correction is layered back on.

**The ladder is a list of `AblationConfig`s, each adding one thing to the
previous.** A typical ladder (see
`examples/notebooks/ablation_sweep.py`, the driver script that runs this
against the real MUSE template library):

    unregularized       -> lam=0.0, everything else off/plain
    + regularization     -> lam=2.0 (current shipped default)
    + multi-start         -> losvd_single_start=False
    + taper moment        -> moment_method="taper"
    + h3/h4               -> fit_h3h4=True
    + shape_aware_correction (branch, off by default in the real package)
    full stack (shipped defaults)

Run `run_ablation_sweep` with this list against a sigma/S-N grid; each
config's own bias table is directly comparable to the others since
everything else (templates, mixture-per-replicate, noise) is identical
across configs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from .config import FitConfig
from .forward_model import build_template_stack
from .gh_moments import (
    compute_direct_moments,
    fit_h3_h4,
    gauss_hermite_model,
    shape_aware_taper_correction,
)
from .multi_template import fit_losvd


@dataclass
class AblationConfig:
    """One point in the ablation ladder.

    Add a new field (with a default matching current shipped package
    behavior) to extend this to a new feature without changing any
    existing config's meaning -- e.g. a future penalty axis, a future
    adaptive-regularization axis, etc.
    """
    name: str
    lam: float = 2.0                  # 0.0 = unregularized
    losvd_single_start: bool = False  # True = bypass fit_log_spline_map's multi-start
    moment_method: str = "taper"      # "taper" (compute_direct_moments) or "trapz" (plain, unwindowed)
    fit_h3h4: bool = True
    shape_aware_correction: bool = False


def _plain_trapz_moments(v: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Full-domain, unwindowed first/second moment of `b(v)` -- the "bare
    bones" counterpart to `compute_direct_moments`'s smooth logistic
    taper. Not exposed as a general-purpose estimator (it has no wing-
    noise protection at all) -- exists here specifically to isolate the
    taper's own marginal bias contribution.
    """
    bpos = np.maximum(np.asarray(b, float), 0.0)
    v = np.asarray(v, float)
    total = bpos.sum()
    if total <= 0:
        return float("nan"), float("nan")
    vel = float(np.sum(v * bpos) / total)
    sig = float(np.sqrt(np.sum(bpos * (v - vel) ** 2) / total))
    return vel, sig


def fit_one(A_stack_w: np.ndarray, y_w: np.ndarray, cfg_base: FitConfig,
            ablation: AblationConfig) -> dict:
    """Fit once under a given `AblationConfig`, applying exactly the
    mechanisms that config turns on.

    Returns a dict with `vherm`, `sherm`, `h3`, `h4`, `success`, `b`
    (the raw recovered LOSVD, for anything needing the curve itself).
    """
    # lam_selection="fixed" forced -- this tool exists to test specific, deliberately-chosen
    # lam values (ablation.lam); letting fit_losvd's own default adaptive selection silently
    # override that would defeat the entire point of an ablation sweep.
    cfg = replace(cfg_base, lam=ablation.lam, lam_selection="fixed",
                  losvd_single_start=ablation.losvd_single_start,
                  fit_h3h4=ablation.fit_h3h4)
    v_grid = cfg.build_velocity_grid()
    result = fit_losvd(A_stack_w, y_w, cfg)
    b = result["b"]
    if not np.all(np.isfinite(b)):
        return dict(vherm=np.nan, sherm=np.nan, h3=0.0, h4=0.0, success=False, b=b)

    if ablation.moment_method == "trapz":
        vel, sig = _plain_trapz_moments(v_grid, b)
    else:
        direct = compute_direct_moments(v_grid, b)
        vel, sig = direct["vherm"], direct["sherm"]

    if not (np.isfinite(vel) and np.isfinite(sig) and sig > 0):
        return dict(vherm=np.nan, sherm=np.nan, h3=0.0, h4=0.0, success=False, b=b)

    if ablation.fit_h3h4:
        h34 = fit_h3_h4(v_grid, b, vel, sig)
        h3, h4 = h34["h3"], h34["h4"]
    else:
        h3, h4 = 0.0, 0.0

    if ablation.shape_aware_correction:
        factor, v_shift = shape_aware_taper_correction(vel, sig, h3, h4)
        sig = sig * factor
        vel = vel + v_shift

    return dict(vherm=vel, sherm=sig, h3=h3, h4=h4, success=bool(result["success"]), b=b)


def run_ablation_sweep(configs: list[AblationConfig], templates, template_wave: np.ndarray,
                        weights, sigma_grid, snr_grid, v_true: float = 0.0, n_reps: int = 20,
                        wavefitmin: float = 8400.0, wavefitmax: float = 8800.0,
                        wave_obs: np.ndarray | None = None, h4_true: float = 0.0, nsig: int = 7,
                        seed_base: int = 0, verbose: bool = True) -> list[dict]:
    """Run the same sigma/S-N/noise sweep under every `AblationConfig` in
    `configs`, on identical mock data per (sigma, snr, replicate) cell
    across all configs -- so any difference in bias between configs is
    attributable to the config's own toggles, not a different noise draw.

    Parameters mirror `kinextract.calibrate_bias.run_calibration_sweep`
    (see its own docstring) with one addition: `configs`, the ablation
    ladder to sweep. `weights` -- a fixed array or a no-argument callable
    -- is resolved once per (sigma, snr, replicate) cell and reused across
    every config in `configs` for that cell (so mocks are identical across
    the ladder, not just the same generating distribution).

    `h4_true` (default 0.0): set > 0 for a genuine non-Gaussian truth --
    every ladder run before this parameter existed used h4_true=0
    exclusively (see `calibrate_bias.run_calibration_sweep`'s own
    docstring for why that's a real, previously-unexamined validation gap
    for anything claiming to preserve real wing structure).

    Returns a list of dicts, one per (config, sigma, snr) cell, with keys
    `config_name`, `sigma`, `snr`, `n`, `v_bias_mean`, `v_bias_std`,
    `sig_bias_mean`, `sig_bias_std`, `h4_mean`, `h4_std` -- same schema as
    `calibrate_bias.run_calibration_sweep`'s rows plus the leading
    `config_name` column, so per-config subsets can be fed straight into
    `bias_correction.save_calibration_table`/`load_calibration_table`.
    """
    if wave_obs is None:
        wave_obs = 4750.0 + np.arange(3681) * 1.25
    mask = (wave_obs >= wavefitmin) & (wave_obs <= wavefitmax)
    wave_fit = wave_obs[mask]
    n_templates = len(templates)
    weights_is_fixed = not callable(weights)
    if weights_is_fixed:
        weights = np.asarray(weights, float)

    rows = []
    t0 = time.time()
    for sig in sigma_grid:
        cfg_base = FitConfig(vmin=v_true - nsig * sig, vmax=v_true + nsig * sig)
        v_grid = cfg_base.build_velocity_grid()
        b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, v_true, sig, 0.0, h4_true), 0.0)
        b_truth /= b_truth.sum()
        A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
        vv = np.linspace(v_true - 15 * sig, v_true + 15 * sig, 200_001)
        bb = np.maximum(gauss_hermite_model(vv, 1.0, v_true, sig, 0.0, h4_true), 0.0)
        tot = np.trapezoid(bb, vv)
        true_v = float(np.trapezoid(bb * vv, vv) / tot)
        true_s = float(np.sqrt(np.trapezoid(bb * (vv - true_v) ** 2, vv) / tot))

        for snr in snr_grid:
            per_config = {c.name: dict(v_errs=[], s_errs=[], h4_vals=[]) for c in configs}
            for rep in range(n_reps):
                seed = seed_base + rep + int(sig * 10) + int(snr * 100)
                w = weights if weights_is_fixed else np.asarray(weights(), float)
                galaxy_noiseless = sum(w[k] * (A_stack[k] @ b_truth) for k in range(n_templates))
                noise_level = np.median(galaxy_noiseless) / snr
                rng = np.random.default_rng(seed)
                galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
                errors = np.full(len(wave_fit), noise_level)
                A_stack_w = A_stack / errors[None, :, None]
                y_w = galaxy / errors

                for ablation in configs:
                    m = fit_one(A_stack_w, y_w, cfg_base, ablation)
                    if np.isfinite(m["vherm"]) and np.isfinite(m["sherm"]):
                        per_config[ablation.name]["v_errs"].append(m["vherm"] - true_v)
                        per_config[ablation.name]["s_errs"].append(m["sherm"] - true_s)
                        per_config[ablation.name]["h4_vals"].append(m["h4"])

            for ablation in configs:
                d = per_config[ablation.name]
                n = len(d["v_errs"])
                if n < 3:
                    continue
                sig_bias_mean = float(np.mean(d["s_errs"]))
                row = dict(
                    config_name=ablation.name, sigma=float(sig), snr=float(snr), n=n,
                    v_bias_mean=float(np.mean(d["v_errs"])),
                    v_bias_std=float(np.std(d["v_errs"], ddof=1)) if n > 1 else 0.0,
                    sig_bias_mean=sig_bias_mean,
                    sig_bias_std=float(np.std(d["s_errs"], ddof=1)) if n > 1 else 0.0,
                    h4_mean=float(np.mean(d["h4_vals"])),
                    h4_std=float(np.std(d["h4_vals"], ddof=1)) if n > 1 else 0.0,
                )
                rows.append(row)
                if verbose:
                    print(f"[{ablation.name:>20s}] sigma={sig:6.0f} S/N={snr:6.0f} n={n:3d}  "
                          f"sig_bias={sig_bias_mean:+7.2f} ({100*sig_bias_mean/true_s:+6.1f}%)  "
                          f"h4={row['h4_mean']:+.4f}  t={time.time()-t0:.0f}s", flush=True)
    return rows


def save_ablation_table(rows: list[dict], path: str, header_comment: str = "") -> None:
    """Write `run_ablation_sweep`'s output to a plain whitespace-delimited
    file -- one `config_name` column plus the same columns
    `bias_correction.save_calibration_table` writes, so a per-config
    subset can be sliced out and loaded with
    `bias_correction.load_calibration_table` directly.
    """
    with open(path, "w") as f:
        if header_comment:
            f.write(f"# {header_comment}\n")
        f.write(f"{'config_name':>20} {'sigma':>6} {'S/N':>6} {'n':>4}   "
                f"{'V_bias_mean':>12} {'V_bias_std':>11}   "
                f"{'sig_bias_mean':>14} {'sig_bias_std':>13}   "
                f"{'sig_bias_pct':>13}   "
                f"{'h4_mean':>9} {'h4_std':>8}\n")
        for r in rows:
            pct = 100 * r["sig_bias_mean"] / r["sigma"] if r["sigma"] else float("nan")
            f.write(f"{r['config_name']:>20} {r['sigma']:6.0f} {r['snr']:6.0f} {r['n']:4d}   "
                    f"{r['v_bias_mean']:+12.2f} {r['v_bias_std']:11.2f}   "
                    f"{r['sig_bias_mean']:+14.2f} {r['sig_bias_std']:13.2f}   "
                    f"{pct:+12.1f}%   "
                    f"{r['h4_mean']:+9.4f} {r['h4_std']:8.4f}\n")


def load_ablation_table(path: str) -> list[dict]:
    """Read back a file written by `save_ablation_table`."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.split()[0] == "config_name":
                continue
            parts = line.split()
            name = parts[0]
            sigma, snr, n, v_mean, v_std, sig_mean, sig_std = (float(parts[i]) for i in range(1, 8))
            h4_mean, h4_std = float(parts[9]), float(parts[10])
            rows.append(dict(config_name=name, sigma=sigma, snr=snr, n=int(n),
                              v_bias_mean=v_mean, v_bias_std=v_std,
                              sig_bias_mean=sig_mean, sig_bias_std=sig_std,
                              h4_mean=h4_mean, h4_std=h4_std))
    return rows
