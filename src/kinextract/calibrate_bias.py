"""Build a `bias_correction.BiasCalibration` table from your own templates
and fitting setup, rather than trusting the shipped default.

**Why this exists.** Regularization bias is a real, unavoidable property
of this MPL/Tikhonov fitting approach (Pinkney et al. 2003), and the
correction for it is calibration-specific, not a universal constant --
a calibration built from one template shape/library does not transfer
cleanly to a different one (see `dev_notes/VALIDATION_CAMPAIGN.md` Stages 9-10 for
what was tried and rejected before landing on the current default). The
fix is not to stop correcting for it, but to calibrate the correction
against synthetic data that actually matches your real setup, and to make
that recalibration path easy and well-documented rather than assuming the
shipped default transfers everywhere.

**The current shipped default is built from the E-MILES SSP grid**
(`regularization_bias_emiles_multitemplate.txt`, see
`dev_notes/calibration/build_emiles_calibration.py`): a synthetic,
parametrized (age x metallicity), portable, LSF-matchable grid
(`kinextract.emiles`) -- any instrument's LSF can be matched to it,
so it is a reproducible, general-purpose basis rather than tied to one
specific real observation.

**Usage -- this is the general recalibration framework, not a one-off
script.** `run_calibration_sweep` itself has never been tied to any
specific template library (it takes plain `templates`/`template_wave`
arrays) -- what changes between the shipped default and your own
calibration is only which templates you load. To calibrate against your
own template library (a different E-MILES age/metallicity/IMF subset, a
different star catalog, an HST/STIS-LSF-matched grid, anything
`build_template_stack` accepts): load your own template library (the same
one you fit real data against), build a representative population-mixture
weight vector (or a sampler drawing a fresh one per replicate, to also
average over weight-recovery noise), and run `run_calibration_sweep`.
Save the result with `save_calibration_table` and load it back with
`bias_correction.load_calibration_table(path)`; pass it as
`bias_calibration=...` to `bootstrap_losvd`/`compute_losvd_moments`, or
overwrite the file at `bias_correction._DEFAULT_TABLE_PATH` to replace the
package default entirely.

Example (mirrors `dev_notes/calibration/build_emiles_calibration.py`, the
script actually used to build this package's own shipped default table --
swap the template-loading lines for your own library and this is your
recalibration script)::

    from kinextract.calibrate_bias import run_calibration_sweep, save_calibration_table
    rows = run_calibration_sweep(
        templates=my_template_fluxes, template_wave=my_template_wave,
        weights=my_fixed_mixture_weights,
        sigma_grid=[20, 40, 80, 140, 220, 300, 400],
        snr_grid=[15, 30, 60, 100, 150],
        n_reps=15,
    )
    save_calibration_table(rows, "my_calibration.txt")
"""
from __future__ import annotations

import time
import warnings
from dataclasses import replace

import numpy as np

from .aic_regularization import estimate_m_eff_local
from .config import FitConfig
from .forward_model import build_template_stack
from .gh_moments import gauss_hermite_model, compute_losvd_moments


def _true_moments(vel: float, sig: float, h4_true: float = 0.0, h3_true: float = 0.0,
                   n: int = 200_001) -> tuple[float, float]:
    """Numerically-integrated true (V, sigma) of the Gauss-Hermite density
    used to build the mock -- for h3=h4=0 truth this equals (vel, sig)
    to high precision, computed this way anyway for consistency with the
    rest of the package's own convention (see `gh_moments`'s module
    docstring on why the raw parameters aren't always exactly the moments).
    `h4_true != 0` gives a genuine non-Gaussian (real-wing) truth -- see
    `run_calibration_sweep`'s own docstring for why this matters.

    `h3_true != 0` shifts the true V away from `vel` by the exact closed
    form `sqrt(3)*sig*h3_true` (`dev_notes/validation_suite/FINDINGS.md`) --
    computed numerically here (not via the closed form directly) so this
    function stays the single source of truth regardless of which moment
    is nonzero, and so a future h4-and-h3-both-nonzero cell (not currently
    exercised by `run_calibration_sweep`, which sweeps them one at a time)
    would still get a correct numerical answer.
    """
    vv = np.linspace(vel - 15 * sig, vel + 15 * sig, n)
    bb = np.maximum(gauss_hermite_model(vv, 1.0, vel, sig, h3_true, h4_true), 0.0)
    tot = np.trapezoid(bb, vv)
    m1 = np.trapezoid(bb * vv, vv) / tot
    m2 = np.sqrt(np.trapezoid(bb * (vv - m1) ** 2, vv) / tot)
    return float(m1), float(m2)


def run_calibration_sweep(templates, template_wave: np.ndarray, weights,
                           sigma_grid, snr_grid, v_true: float = 0.0, n_reps: int = 15,
                           wavefitmin: float = 8400.0, wavefitmax: float = 8800.0,
                           wave_obs: np.ndarray | None = None, h4_true: float = 0.0,
                           h3_true: float = 0.0,
                           nsig: int = 7, seed_base: int = 0, verbose: bool = True,
                           lam_grid: list[float] | None = None) -> list[dict]:
    """Synthetic sigma/h4 bias sweep against a *specific* template library
    and mixture, using the real `fit_losvd` multi-template engine (any
    `n_templates >= 1` -- pass a length-1 `templates` list to calibrate a
    single-template setup instead).

    Parameters
    ----------
    templates : list of ndarray, each on `template_wave`'s grid.
    template_wave : ndarray -- the templates' own wavelength grid.
    weights : ndarray, shape (n_templates,), OR a no-argument callable
        returning a fresh weight vector each call (to also average over
        weight-recovery noise across replicates). A fixed array is
        reused for every replicate.
    sigma_grid, snr_grid : true-sigma and S/N values to sweep (every
        combination is run).
    v_true : fixed true velocity for every cell (default 0.0 -- V bias was
        found consistent with pure noise in the original calibration and
        is not itself corrected by `bias_correction`; not re-swept here
        unless you have reason to check it for your own setup).
    n_reps : replicates per (sigma, S/N) cell. 15-25 is a reasonable
        floor for a usable mean/std; more tightens the calibration's own
        uncertainty at proportional cost.
    wavefitmin, wavefitmax : fit window (Angstrom, rest frame).
    wave_obs : the observed-frame wavelength grid to build spectra on;
        defaults to a MUSE-like grid (4750-9350 Angstrom, 1.25 Angstrom/pix)
        if not given.
    h4_true : default 0.0 (plain Gaussian truth, this function's original
        behavior). Set > 0 for a genuine non-Gaussian (real high-velocity
        wing) truth -- every calibration/ablation sweep in this project
        before this parameter existed used h4_true=0 exclusively, which
        cannot detect a method that suppresses or inflates real wing
        signal since there was never any real wing signal in the mock to
        begin with. Use a nonzero value whenever wing fidelity, not just
        aggregate sigma bias, is the thing being validated.
    h3_true : default 0.0 (same rationale as `h4_true`, but for the odd
        moment). Unlike `h4_true`, a nonzero `h3_true` also shifts the true
        V away from `v_true` by the exact closed form
        `sqrt(3)*sig*h3_true` (`_true_moments` computes this numerically,
        not via the closed form, so it stays correct even if a future
        caller sweeps h3 and h4 together) -- `v_bias_mean`/`v_bias_std` in
        the returned rows are always scored against that shifted true V,
        not `v_true` itself, so an h3-nonzero calibration table's `v_bias`
        column is a genuine residual bias after accounting for the known
        GH-parameterization shift, not the shift itself. Swept one moment
        at a time (h3 xor h4 nonzero) by every caller so far -- this
        function does not itself prevent setting both, but no calibration
        table built from it has exercised that combination yet.
    nsig : the mock's own true LOSVD is built on a `+-nsig*sigma` grid
        (7 matches this package's own general convention).
    seed_base : offset for the noise RNG seed (for reproducibility across
        separate sweep runs that shouldn't share noise draws).
    verbose : print progress per cell.
    lam_grid : regularization strengths to sweep at *every* (sigma, S/N)
        cell -- default `None` means `[2.0]`, reproducing this function's
        original fixed-lam behavior exactly (same noise draws, same
        fits, bit-for-bit). Pass more than one value (e.g. the same grid
        `aic_regularization.select_lambda_aic_local` scans) to build a
        real per-`lam` calibration set -- needed before a correction can
        be validly applied to a fit made with *adaptive* regularization
        rather than this package's fixed `lam=2.0` default. Feed the
        result to `save_calibration_table` and load it back with
        `bias_correction.load_per_lam_calibration_table` (one 2-D
        `(sigma, S/N)` table per `lam` value, interpolated between the
        two nearest `lam`s at query time -- see `PerLamCalibration`'s own
        docstring for why this, rather than treating `m_eff` as a third
        interpolation axis, is the validated design). `m_eff` (the fit's
        own effective degrees of freedom at that `lam`) is still computed
        and recorded per cell below as a useful diagnostic column, but is
        no longer used as a calibration query key. The same noise draw is
        reused across every `lam` in the grid for a given (sigma, S/N,
        rep) -- a paired design that reduces sweep-to-sweep noise in the
        bias-vs-lam trend without changing what each individual `lam`'s
        own mean/std estimate is.

    Returns
    -------
    List of dicts, one per (sigma, S/N, lam) cell, with keys matching
    `save_calibration_table`'s expected columns: `sigma` (true), `snr`,
    `n`, `v_bias_mean`, `v_bias_std`, `sig_bias_mean`, `sig_bias_std`,
    `h4_mean`, `h4_std`, `lam`, `m_eff_mean`, `m_eff_std`. Pass to
    `save_calibration_table` to write a file `load_calibration_table` can
    read.
    """
    if lam_grid is None:
        lam_grid = [2.0]
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
        vmin, vmax = v_true - nsig * sig, v_true + nsig * sig
        cfg = FitConfig(vmin=vmin, vmax=vmax)
        v_grid = cfg.build_velocity_grid()
        b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, v_true, sig, h3_true, h4_true), 0.0)
        b_truth /= b_truth.sum()
        A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
        true_v, true_s = _true_moments(v_true, sig, h4_true, h3_true)

        for snr in snr_grid:
            # per-lam accumulators -- rep is the outer loop (so every lam in
            # lam_grid is evaluated against the *same* noise draw for a given
            # rep, see this function's own docstring on why that's a
            # deliberate, safe choice) and lam the inner one.
            per_lam = {lam: dict(v=[], s=[], h3=[], h4=[], m_eff=[]) for lam in lam_grid}
            for rep in range(n_reps):
                w = weights if weights_is_fixed else np.asarray(weights(), float)
                galaxy_noiseless = sum(w[k] * (A_stack[k] @ b_truth) for k in range(n_templates))
                noise_level = np.median(galaxy_noiseless) / snr
                seed = seed_base + rep + int(sig * 10) + int(snr * 100)
                rng = np.random.default_rng(seed)
                galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
                errors = np.full(len(wave_fit), noise_level)
                A_stack_w = A_stack / errors[None, :, None]
                y_w = galaxy / errors

                for lam in lam_grid:
                    cfg_lam = replace(cfg, lam=lam)
                    m_eff, _chi2, result = estimate_m_eff_local(A_stack_w, y_w, cfg_lam, lam)
                    m = compute_losvd_moments(v_grid, result["b"])
                    if np.isfinite(m["vherm"]) and np.isfinite(m["sherm"]):
                        per_lam[lam]["v"].append(m["vherm"] - true_v)
                        per_lam[lam]["s"].append(m["sherm"] - true_s)
                        per_lam[lam]["h3"].append(m["h3"])
                        per_lam[lam]["h4"].append(m["h4"])
                        per_lam[lam]["m_eff"].append(m_eff)

            for lam in lam_grid:
                d = per_lam[lam]
                n = len(d["s"])
                if n < 3:
                    warnings.warn(f"run_calibration_sweep: sigma={sig}, S/N={snr}, lam={lam} had "
                                   f"only {n}/{n_reps} finite fits -- skipping this cell.",
                                   UserWarning, stacklevel=2)
                    continue
                row = dict(
                    sigma=float(sig), snr=float(snr), n=n,
                    v_bias_mean=float(np.mean(d["v"])), v_bias_std=float(np.std(d["v"], ddof=1)) if n > 1 else 0.0,
                    sig_bias_mean=float(np.mean(d["s"])), sig_bias_std=float(np.std(d["s"], ddof=1)) if n > 1 else 0.0,
                    h3_mean=float(np.mean(d["h3"])), h3_std=float(np.std(d["h3"], ddof=1)) if n > 1 else 0.0,
                    h4_mean=float(np.mean(d["h4"])), h4_std=float(np.std(d["h4"], ddof=1)) if n > 1 else 0.0,
                    lam=float(lam),
                    m_eff_mean=float(np.mean(d["m_eff"])), m_eff_std=float(np.std(d["m_eff"], ddof=1)) if n > 1 else 0.0,
                )
                rows.append(row)
                if verbose:
                    print(f"sigma={sig:6.0f}  S/N={snr:6.0f}  lam={lam:6.2f}  n={n:3d}  "
                          f"sig_bias={row['sig_bias_mean']:+7.2f}+/-{row['sig_bias_std']:.2f}  "
                          f"({100*row['sig_bias_mean']/true_s:+6.1f}%)  "
                          f"h4={row['h4_mean']:+.4f}  m_eff={row['m_eff_mean']:6.2f}  "
                          f"t={time.time()-t0:.0f}s", flush=True)
    return rows


def save_calibration_table(rows: list[dict], path: str, header_comment: str = "") -> None:
    """Write `run_calibration_sweep`'s output to a file
    `bias_correction.load_calibration_table` can read.

    Always writes the 3 trailing `lam`/`m_eff_mean`/`m_eff_std` columns
    now (present in every row `run_calibration_sweep` returns, even for
    the default `lam_grid=[2.0]` case) -- `load_calibration_table` treats
    a file with only the first 10 columns (written by a version of this
    function older than the `m_eff` axis) as equally valid, so this is
    not a breaking format change for anything already on disk.
    """
    with open(path, "w") as f:
        if header_comment:
            f.write(f"# {header_comment}\n")
        f.write(f"{'sigma':>6} {'S/N':>6} {'n':>4}   "
                f"{'V_bias_mean':>12} {'V_bias_std':>11}   "
                f"{'sig_bias_mean':>14} {'sig_bias_std':>13}   "
                f"{'sig_bias_pct':>13}   "
                f"{'h4_mean':>9} {'h4_std':>8}   "
                f"{'lam':>8} {'m_eff_mean':>10} {'m_eff_std':>9}\n")
        for r in rows:
            pct = 100 * r["sig_bias_mean"] / r["sigma"] if r["sigma"] else float("nan")
            f.write(f"{r['sigma']:6.0f} {r['snr']:6.0f} {r['n']:4d}   "
                    f"{r['v_bias_mean']:+12.2f} {r['v_bias_std']:11.2f}   "
                    f"{r['sig_bias_mean']:+14.2f} {r['sig_bias_std']:13.2f}   "
                    f"{pct:+12.1f}%   "
                    f"{r['h4_mean']:+9.4f} {r['h4_std']:8.4f}   "
                    f"{r['lam']:8.3f} {r['m_eff_mean']:10.3f} {r['m_eff_std']:9.3f}\n")
