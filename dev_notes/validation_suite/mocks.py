"""Reusable mock-generation for the randomized bias-validation suite.

Built in response to a real finding (2026-07-28): the (V, sigma, h3, h4)
parameters that DEFINE a truncated Gauss-Hermite series are not the same
thing as that curve's own mean/std/skewness/kurtosis. They coincide only in
the h3, h4 -> 0 limit. Away from that limit the divergence is large and has
nothing to do with clipping or fitting:

    V=100, sigma=200, h3=0.03, h4=0.00  ->  true curve mean = 109.7 (NO clip)
    V=100, sigma=200, h3=0.00, h4=0.05  ->  true curve sigma = 219 (NO clip)

Over the randomization ranges this suite uses by default (sigma 40-400,
h3/h4 in +/-0.1), 53% of draws have |V shift| > 10 km/s and 68% have
|sigma shift| > 5%, from the GH-parameterization mismatch ALONE, before any
noise or fitting enters. Clipping (the LOSVD going negative and being
clamped) is a real, separate, second-order effect on top of this -- it
reaches a few percent of mass at the extremes of the range, well below the
parameterization mismatch.

The consequence: nominal generator parameters are NEVER a valid "truth" to
score fitting bias against once h3/h4 leave a narrow neighborhood of zero.
Every case built here reports BOTH:
  - the nominal parameters (for reproducibility / provenance), and
  - `truth_moments`, from `compute_losvd_moments` on the actual (clipped,
    normalized) discretized curve -- the only correct basis for a bias
    comparison.
and a `generator_mismatch` block (truth_moments - nominal) so any downstream
analysis can tell, case by case, how much of an apparent trend is coming
from the generator rather than from the fit.

See dev_notes/validation_suite/FINDINGS.md for the full writeup.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from kinextract import (
    FitConfig, build_template_stack, gauss_hermite_model, compute_losvd_moments,
    load_emiles_templates, build_spline_truth,
)

EMILES_GRID_DEFAULT = "src/kinextract/template_data/emiles_muse_grid.npz"
WAVEMIN, STEP, N_PIX = 4750.0, 1.25, 3681
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0

# Two labeled presets. "safe" keeps h3/h4 in a range where the GH-parameterization
# mismatch stays modest (consistent with typically-observed early-type-galaxy
# values); "wide" is the stress-test range that produint mostly-mismatched
# "truth" curves -- useful for probing robustness, but never for a headline
# bias number unless generator_mismatch is reported alongside it.
RANGE_PRESETS = {
    "safe": dict(v=(0.0, 200.0), sigma=(40.0, 300.0), h3=(-0.05, 0.05), h4=(-0.05, 0.05),
                 snr=(20.0, 400.0)),
    "wide": dict(v=(0.0, 200.0), sigma=(40.0, 400.0), h3=(-0.1, 0.1), h4=(-0.1, 0.1),
                 snr=(20.0, 400.0)),
}


@dataclass
class MockCase:
    params: dict
    cfg: FitConfig
    v_grid: np.ndarray
    A_stack: np.ndarray
    template_wave: np.ndarray
    names: list
    ages: np.ndarray
    metals: np.ndarray
    b_truth_unit: np.ndarray
    galaxy_noiseless: np.ndarray
    truth_moments: dict
    generator_mismatch: dict
    clipped_mass_frac: float
    n_local_maxima: int
    wave_fit: np.ndarray = field(repr=False, default=None)


def observed_wavelength_grid():
    obs = WAVEMIN + np.arange(N_PIX) * STEP
    return obs[(obs >= WAVEFITMIN) & (obs <= WAVEFITMAX)]


WIGGLE_SCALE_RANGE = (0.05, 0.25)  # spline-truth analog of h3/h4: random log-space perturbation size


def draw_random_params(rng: np.random.Generator, fixed: dict | None = None,
                        preset: str = "safe", gaussian_only: bool = False,
                        n_templates: int = 4, emiles_ages=None, emiles_metals=None,
                        truth_kind: str = "gh") -> dict:
    """Draw one full parameter set, honoring any key present in `fixed`.

    `fixed` values are used as-is (not redrawn); everything else is randomized
    independently for this draw. This is the single point where "sweep this
    axis, randomize everything else" is implemented -- callers just build
    `fixed={axis: value}` per grid point.

    `truth_kind`: "gh" draws h3/h4 as the GH shape parameters (subject to the
    generator mismatch documented in this module's docstring). "spline" draws
    a `wiggle_scale` instead -- there is no GH shape being targeted, so h3/h4
    are not applicable and are set to NaN; `gaussian_only` then means
    `wiggle_scale=0` (a pure Gaussian core), the spline-truth analog of
    `h3=h4=0`.
    """
    fixed = fixed or {}
    ranges = RANGE_PRESETS[preset]

    def pick(key, lo_hi):
        if key in fixed:
            return fixed[key]
        return rng.uniform(*lo_hi)

    V = pick("v", ranges["v"])
    sigma = pick("sigma", ranges["sigma"])
    snr = pick("snr", ranges["snr"])

    if truth_kind == "gh":
        if gaussian_only:
            h3 = fixed.get("h3", 0.0)
            h4 = fixed.get("h4", 0.0)
        else:
            h3 = pick("h3", ranges["h3"])
            h4 = pick("h4", ranges["h4"])
        wiggle_scale = float("nan")
    elif truth_kind == "spline":
        if "h3" in fixed or "h4" in fixed:
            raise ValueError("truth_kind='spline' has no GH h3/h4 to fix -- sweep 'wiggle_scale' instead")
        h3 = h4 = float("nan")
        wiggle_scale = 0.0 if gaussian_only else pick("wiggle_scale", WIGGLE_SCALE_RANGE)
        # drawn from `rng` (not derived from wiggle_scale/fixed) so the actual wiggle
        # PATTERN varies across replicates even when wiggle_scale itself is fixed --
        # otherwise every replicate at a given magnitude would build an identical shape.
        spline_seed = int(rng.integers(0, 2**31 - 1))
    else:
        raise ValueError(f"unknown truth_kind={truth_kind!r}, expected 'gh' or 'spline'")
    spline_seed = spline_seed if truth_kind == "spline" else None

    if emiles_ages is None or emiles_metals is None:
        raise ValueError("emiles_ages/emiles_metals must be supplied (from the template grid file)")
    if "ages" in fixed:
        age_choice = np.asarray(fixed["ages"])
    else:
        age_choice = rng.choice(emiles_ages, size=n_templates, replace=True)
    if "metals" in fixed:
        metal_choice = np.asarray(fixed["metals"])
    else:
        metal_choice = rng.choice(emiles_metals, size=n_templates, replace=True)
    if "weights" in fixed:
        weights = np.asarray(fixed["weights"], dtype=float)
        weights = weights / weights.sum()
    else:
        weights = rng.dirichlet(np.ones(n_templates))

    return dict(v=float(V), sigma=float(sigma), h3=float(h3), h4=float(h4),
                wiggle_scale=float(wiggle_scale), spline_seed=spline_seed,
                snr=float(snr), ages=age_choice, metals=metal_choice, weights=weights,
                n_templates=n_templates, preset=preset, gaussian_only=gaussian_only,
                truth_kind=truth_kind)


def _count_local_maxima(y: np.ndarray) -> int:
    d = np.diff(y)
    return int(np.sum((d[:-1] > 0) & (d[1:] < 0)))


def build_mock(params: dict, emiles_grid_path: str = EMILES_GRID_DEFAULT,
               wave_fit: np.ndarray | None = None) -> MockCase:
    """Build one full mock case (templates, LOSVD, noiseless spectrum) from a
    drawn parameter dict. Pure numpy -- no noise is added here (see `runner.py`
    for that), so this function is fully deterministic given `params`.

    Dispatches on `params["truth_kind"]`:

    - "gh": the truncated Gauss-Hermite generator. `generator_mismatch` here
      means what this module's docstring describes -- the gap between the GH
      series' own defining coefficients and the resulting curve's actual
      moments, which is large and well-characterized; NOT a translation loss
      through the fitting basis (confirmed with zero clipping and zero
      fitting -- see FINDINGS.md).
    - "spline": `mocks.build_spline_truth` -- a curve defined directly as
      exp(spline), with NO Gauss-Hermite step at all, so there is no
      generator-vs-moments gap to speak of by construction. Here
      `generator_mismatch` instead reports how much incidental h3/h4 the
      random log-space wiggle happened to introduce relative to its (0, 0)
      nominal Gaussian core -- a different quantity, not directly comparable
      across truth_kind, which is why every row carries `truth_kind`
      explicitly rather than leaving it implicit.
    """
    if wave_fit is None:
        wave_fit = observed_wavelength_grid()

    select = list(zip([float(a) for a in params["ages"]], [float(m) for m in params["metals"]]))
    templates, template_wave, names, ages, metals = load_emiles_templates(emiles_grid_path, select=select)

    V, S = params["v"], params["sigma"]
    cfg = FitConfig(vmin=V - 7 * S, vmax=V + 7 * S)
    v_grid = cfg.build_velocity_grid()
    truth_kind = params.get("truth_kind", "gh")

    if truth_kind == "gh":
        h3, h4 = params["h3"], params["h4"]
        raw = gauss_hermite_model(v_grid, amp=1.0, vel=V, sig=S, h3=h3, h4=h4)
        neg_mass = -np.trapezoid(np.minimum(raw, 0.0), v_grid)
        pos_mass = np.trapezoid(np.maximum(raw, 0.0), v_grid)
        clipped_mass_frac = float(neg_mass / (pos_mass + neg_mass)) if (pos_mass + neg_mass) > 0 else 0.0
        n_local_maxima = _count_local_maxima(raw)
        b_truth = np.maximum(raw, 0.0)
        b_truth_unit = b_truth / np.trapezoid(b_truth, v_grid)
        truth_moments = compute_losvd_moments(v_grid, b_truth_unit, fit_h3h4=True)
        generator_mismatch = dict(
            dv=float(truth_moments["vherm"] - V),
            dsigma=float(truth_moments["sherm"] - S),
            dsigma_frac=float((truth_moments["sherm"] - S) / S) if S else float("nan"),
            dh3=float(truth_moments["h3"] - h3),
            dh4=float(truth_moments["h4"] - h4),
        )
    elif truth_kind == "spline":
        b_truth_unit, _coeffs = build_spline_truth(
            v_grid, vel=V, sigma=S, degree=cfg.degree, n_interior_knots=cfg.resolved_n_interior_knots(),
            wiggle_scale=params["wiggle_scale"], seed=params.get("spline_seed", 0))
        clipped_mass_frac = 0.0  # exp(.) is never negative by construction
        n_local_maxima = _count_local_maxima(b_truth_unit)
        truth_moments = compute_losvd_moments(v_grid, b_truth_unit, fit_h3h4=True)
        generator_mismatch = dict(
            dv=float(truth_moments["vherm"] - V),
            dsigma=float(truth_moments["sherm"] - S),
            dsigma_frac=float((truth_moments["sherm"] - S) / S) if S else float("nan"),
            dh3=float(truth_moments["h3"]),  # relative to a nominal (0, 0) Gaussian core
            dh4=float(truth_moments["h4"]),
        )
    else:
        raise ValueError(f"unknown truth_kind={truth_kind!r}, expected 'gh' or 'spline'")

    A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
    weights = np.asarray(params["weights"], dtype=float)
    galaxy_noiseless = sum(weights[k] * (A_stack[k] @ b_truth_unit) for k in range(len(templates)))

    return MockCase(
        params=params, cfg=cfg, v_grid=v_grid, A_stack=A_stack, template_wave=template_wave,
        names=names, ages=ages, metals=metals, b_truth_unit=b_truth_unit,
        galaxy_noiseless=galaxy_noiseless, truth_moments=truth_moments,
        generator_mismatch=generator_mismatch, clipped_mass_frac=clipped_mass_frac,
        n_local_maxima=n_local_maxima, wave_fit=wave_fit,
    )


# `build_spline_truth` (the no-Gauss-Hermite-step alternative truth generator
# used by `truth_kind="spline"` above) has been promoted to
# `kinextract.losvd.build_spline_truth` (imported at the top of this
# file) -- it is now the package-wide default mock-truth generator
# (`build_mock_cube`'s own default), not just this suite's own A/B
# comparison tool. See that function's docstring for the full rationale.
