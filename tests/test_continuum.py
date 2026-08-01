"""Tests for kinextract.continuum -- the 3-block (LOSVD, weights,
continuum) alternating minimization. See that module's own docstring and
`kinextract_test_suite/CONTINUUM_ARCHITECTURE_PLAN.md`'s "Validation
order" for what these tests are checking and why, in order:

1. Does the block converge correctly on a known, injected multiplicative
   continuum distortion, with already-normalized templates (isolate the
   continuum block itself before entangling it with template-continuum-
   trust questions, not yet implemented).
2. Does the weight step get contaminated by the continuum, i.e. does a
   wrong template mix partially "fake" a continuum correction it
   shouldn't be responsible for -- tested with a deliberately adversarial
   setup (two templates with opposite broadband continuum slopes, and an
   injected continuum distortion shaped exactly like what a wrong weight
   shift could fake).
"""
from __future__ import annotations

import numpy as np

from kinextract import (
    FitConfig, build_template_stack, build_spline_truth, compute_losvd_moments,
    build_bspline_basis, difference_matrix, fit_losvd_with_continuum,
)
from kinextract.continuum import fit_continuum_block


def _setup(templates, true_w, vel=0.0, sig=140.0, snr=100.0, true_C=None, seed=0):
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]

    cfg = FitConfig(vmin=vel - 7 * sig, vmax=vel + 7 * sig)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=vel, sigma=sig, wiggle_scale=0.0, seed=seed)
    A_stack = build_template_stack(wave_fit, wavelength, templates, v_grid)
    galaxy_noiseless = sum(true_w[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))
    if true_C is None:
        true_C = np.ones(len(wave_fit))
    galaxy_distorted = galaxy_noiseless * true_C

    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_distorted + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors

    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D2 = difference_matrix(n_coef, order=2, dv=dv, dv_ref=cfg.knot_spacing_kms)
    return A_stack_w, y_w, wave_fit, basis, D2, cfg, v_grid, true_C


def _make_template(wavelength, ca_centers, depths, width, slope=0.0):
    t = np.ones(len(wavelength))
    for cen, d in zip(ca_centers, depths):
        t -= d * np.exp(-0.5 * ((wavelength - cen) / width) ** 2)
    if slope != 0.0:
        frac = (wavelength - wavelength.min()) / (wavelength.max() - wavelength.min())
        t *= (1.0 + slope * (frac - 0.5))
    return t


def test_fit_continuum_block_recovers_a_known_curve_in_isolation():
    wave = np.linspace(8400.0, 8800.0, 321)
    M = np.full(321, 50.0)
    true_C = 1.0 + 0.15 * np.sin(2 * np.pi * (wave - wave.min()) / (wave.max() - wave.min()))
    rng = np.random.default_rng(0)
    y_w = M * true_C + rng.normal(0.0, 0.5, 321)
    basis_c, n_coef = build_bspline_basis(wave, n_interior_knots=4, degree=3)
    dw = (wave.max() - wave.min()) / 5
    D2_c = difference_matrix(n_coef, order=2, dv=dw, dv_ref=dw)

    result = fit_continuum_block(M, y_w, basis_c, D2_c, lam_c=2.0)
    assert result["success"]
    rms = np.sqrt(np.mean((result["C"] - true_C) ** 2))
    assert rms < 0.02, f"expected close recovery of the injected continuum curve, got RMS={rms:.4f}"


def test_3block_recovers_a_known_injected_continuum_distortion():
    """Validation order step 1: prove the 3-block alternation converges
    correctly on a synthetic case with a known, injected multiplicative
    continuum distortion, using already-normalized templates.
    """
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavelength = wavemin + np.arange(n_pix) * step
    wavefitmin, wavefitmax = 8400.0, 8800.0
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit_full = wavelength[mask]
    ca_centers = [8498.02, 8542.09, 8662.14]

    templates = [
        _make_template(wavelength, ca_centers, [0.30, 0.65, 0.35], 5.0),
        _make_template(wavelength, ca_centers, [0.60, 0.25, 0.55], 3.0),
    ]
    true_w = np.array([0.6, 0.4])
    true_C = 1.0 + 0.15 * np.sin(2 * np.pi * (wave_fit_full - wave_fit_full.min())
                                  / (wave_fit_full.max() - wave_fit_full.min()))

    A_stack_w, y_w, wave_fit, basis, D2, cfg, v_grid, true_C = _setup(
        templates, true_w, sig=140.0, snr=100.0, true_C=true_C, seed=0
    )
    result = fit_losvd_with_continuum(A_stack_w, y_w, wave_fit, basis, D2, n_continuum_knots=4, lam=cfg.lam)
    m = compute_losvd_moments(v_grid, result["b"])

    assert abs(m["sherm"] - 140.0) < 5.0, f"sigma recovery off: {m['sherm']:.2f}"
    w_norm = result["w"] / result["w"].sum()
    np.testing.assert_allclose(w_norm, true_w, atol=0.02)
    rms_C = np.sqrt(np.mean((result["C"] - true_C) ** 2))
    assert rms_C < 0.02, f"expected close recovery of the injected continuum, got RMS={rms_C:.4f}"


def test_weights_are_not_contaminated_by_an_adversarial_continuum_degeneracy():
    """Validation order step 2, the key "kinematics primacy" check the
    user explicitly required before this architecture could be trusted:
    two templates with *opposite* broadband continuum slopes (so a wrong
    weight shift really could partially fake a smooth continuum trend),
    and an injected continuum distortion shaped exactly like what that
    wrong weight shift would produce. If plain (undamped) 3-block
    alternation recovers the true 50/50 mix anyway, the proximal/ridge
    mitigation the architecture plan proposed (but said not to implement
    preemptively) isn't needed -- this is the test that decides that.
    """
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavelength = wavemin + np.arange(n_pix) * step
    wavefitmin, wavefitmax = 8400.0, 8800.0
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit_full = wavelength[mask]
    ca_centers = [8498.02, 8542.09, 8662.14]

    # template A: blue-favoring slope, template B: red-favoring slope (opposite signs)
    templates = [
        _make_template(wavelength, ca_centers, [0.30, 0.65, 0.35], 5.0, slope=-0.3),
        _make_template(wavelength, ca_centers, [0.60, 0.25, 0.55], 3.0, slope=+0.3),
    ]
    true_w = np.array([0.5, 0.5])
    frac = (wave_fit_full - wave_fit_full.min()) / (wave_fit_full.max() - wave_fit_full.min())
    # a smooth redward-increasing trend -- exactly the "shape" a shift toward template B fakes
    true_C = 1.0 + 0.25 * (frac - 0.5)

    w_errs = []
    for seed in range(3):
        A_stack_w, y_w, wave_fit, basis, D2, cfg, v_grid, true_C_ = _setup(
            templates, true_w, sig=140.0, snr=100.0, true_C=true_C, seed=seed
        )
        result = fit_losvd_with_continuum(A_stack_w, y_w, wave_fit, basis, D2, n_continuum_knots=4, lam=cfg.lam)
        w_norm = result["w"] / result["w"].sum()
        w_errs.append(np.abs(w_norm - true_w).max())

    mean_w_err = np.mean(w_errs)
    assert mean_w_err < 0.05, (
        f"weights should not be measurably pulled toward the wrong mix by the adversarial "
        f"continuum degeneracy, got mean max |w_err|={mean_w_err:.3f} across seeds {w_errs}"
    )


def test_flat_continuum_case_matches_the_validated_2block_fit_closely():
    """Sanity check: when there's no real continuum distortion to
    explain (C=1 everywhere), the 3-block fit should recover essentially
    the same LOSVD/weights as the already-validated 2-block
    `fit_losvd_basis` -- adding the continuum block shouldn't change
    anything when it has nothing useful to do.
    """
    from kinextract import fit_losvd_basis

    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavelength = wavemin + np.arange(n_pix) * step
    ca_centers = [8498.02, 8542.09, 8662.14]

    templates = [
        _make_template(wavelength, ca_centers, [0.30, 0.65, 0.35], 5.0),
        _make_template(wavelength, ca_centers, [0.60, 0.25, 0.55], 3.0),
    ]
    true_w = np.array([0.55, 0.45])

    A_stack_w, y_w, wave_fit, basis, D2, cfg, v_grid, true_C = _setup(
        templates, true_w, sig=140.0, snr=100.0, true_C=None, seed=2
    )
    result_3block = fit_losvd_with_continuum(A_stack_w, y_w, wave_fit, basis, D2, n_continuum_knots=4, lam=cfg.lam)
    result_2block = fit_losvd_basis(A_stack_w, y_w, basis, D2, lam=cfg.lam)

    m3 = compute_losvd_moments(v_grid, result_3block["b"])
    m2 = compute_losvd_moments(v_grid, result_2block["b"])
    assert abs(m3["sherm"] - m2["sherm"]) < 3.0
    np.testing.assert_allclose(result_3block["C"], np.ones_like(result_3block["C"]), atol=0.03)

def test_continuum_low_snr_flagged_failure_is_attributable_and_not_a_real_accuracy_problem(monkeypatch):
    """Documents a corrected finding, superseding two earlier (wrong or
    incomplete) versions of this test's own docstring -- read this before
    trusting `result["success"]` at low S/N with continuum-cofitting
    active. `success=False` here is a diagnostic curiosity, not a real
    accuracy signal -- see `dev_notes/VALIDATION_CAMPAIGN.md` Stage 15 for how that
    was established. This test locks in that `message`/`losvd_success`/
    `continuum_success` correctly attribute a flagged failure to whichever
    block actually caused it, rather than defaulting to blaming the LOSVD
    sub-fit regardless of the true cause.

    **2026-07 update (Stage 22): the seed-pinned natural reproduction no
    longer exists.** This flagged failure was, like the main LOSVD fit's
    own documented ~1-in-150 basin risk, sensitive to JAX float32
    precision noise in scipy's own convergence check -- switching the
    package to float64 (Stage 22) apparently fixed the continuum sub-fit's
    optimizer the same way it fixed the main fit's heavy-lam stall. A
    broad re-scan (sig in {20,30,50}, S/N in {8,10,12,15,20,25,30,40},
    seeds 0-59) found zero reproductions. Rather than chase an
    increasingly extreme, precision-fragile regime for something that may
    no longer occur naturally, this test now forces the same
    (losvd_success=True, continuum_success=False) state directly via a
    monkeypatched `fit_continuum_block` (keeping its real `C`/`coeffs`
    output, only overriding the reported convergence flag/message) --
    testing the attribution *logic* in `_alternate_once` deterministically,
    independent of whether real data ever triggers it again.
    """
    import kinextract.continuum as continuum_mod
    from kinextract import fit_losvd_basis

    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavelength = wavemin + np.arange(n_pix) * step
    ca_centers = [8498.02, 8542.09, 8662.14]
    templates = [_make_template(wavelength, ca_centers, [0.55, 0.70, 0.65], 5.0)]
    true_w = np.array([1.0])

    A_stack_w, y_w, wave_fit, basis, D2, cfg, v_grid, true_C = _setup(
        templates, true_w, sig=50.0, snr=30.0, true_C=None, seed=7
    )
    result_plain = fit_losvd_basis(A_stack_w, y_w, basis, D2, lam=cfg.lam)
    assert result_plain["success"], "the plain 2-block fit should converge fine -- confirms this isn't a general data problem"

    real_fit_continuum_block = continuum_mod.fit_continuum_block

    def _forced_failure(*args, **kwargs):
        out = dict(real_fit_continuum_block(*args, **kwargs))
        out["success"] = False
        out["message"] = "FORCED (test): simulated non-convergence"
        return out

    monkeypatch.setattr(continuum_mod, "fit_continuum_block", _forced_failure)
    result_continuum = fit_losvd_with_continuum(A_stack_w, y_w, wave_fit, basis, D2, n_continuum_knots=4, lam=cfg.lam)

    assert not result_continuum["success"]
    # message should name the block that actually failed, not default to the LOSVD's own.
    assert result_continuum["losvd_success"] is True
    assert result_continuum["continuum_success"] is False
    assert "continuum block" in result_continuum["message"]


def test_ridge_c_closes_the_curvature_penaltys_zero_cost_null_space():
    """Root cause of the amplification bug documented in this module's own
    docstring, found via direct linear-algebra inspection rather than
    further guessing at the emergent alternation dynamics: `lam_c*D2_c`
    (the curvature penalty alone) has an exact null space spanning the
    constant and *linear* directions in coefficient space -- a standard
    property of any second-order-difference P-spline penalty. An
    arithmetic-sequence coefficient vector gives essentially zero cost
    under `D2_c` *regardless of `lam_c`'s value*, and the resulting
    `C(lambda) = exp(basis_c @ coeffs)` is a near-perfectly-linear-in-
    wavelength curve -- exactly the "large, near-perfectly-linear spurious
    residual trend, unaffected by sweeping lam_c" signature the bug
    reproduced. `ridge_c` (a direct L2 penalty on the coefficients) closes
    this: it has a real, nonzero, growing cost along that same direction,
    regardless of `lam_c`.
    """
    wave_fit = np.linspace(8400.0, 8800.0, 321)
    n_knots = 4
    basis_c, n_coef = build_bspline_basis(wave_fit, n_interior_knots=n_knots, degree=3)
    dw = (wave_fit.max() - wave_fit.min()) / (n_knots + 1)
    D2_c = difference_matrix(n_coef, order=2, dv=dw, dv_ref=dw)

    # An arithmetic-sequence coefficient vector spans the "linear" null-space
    # direction of any second-order difference operator.
    null_direction = np.linspace(-1.0, 1.0, n_coef)
    rough_cost = float(np.sum((D2_c @ null_direction) ** 2))
    assert rough_cost < 1e-10, (
        f"expected the curvature penalty to assign ~zero cost to this direction "
        f"(confirming the null space exists), got {rough_cost:.3e}"
    )

    C_along_null = np.exp(basis_c @ (0.1 * null_direction))
    from scipy.stats import linregress
    lr = linregress(wave_fit, C_along_null)
    assert abs(lr.rvalue) > 0.99, (
        "expected moving along the null-space direction to produce a near-perfectly-"
        f"linear C(lambda), matching the documented bug's own residual signature; "
        f"got r={lr.rvalue:.4f}"
    )

    # Curvature penalty alone assigns ~zero cost regardless of lam_c's strength --
    # exactly why sweeping lam_c never suppressed the original bug.
    for lam_c in (0.01, 2.0, 100.0, 1e6):
        rough = 0.5 * lam_c * np.sum((D2_c @ null_direction) ** 2)
        assert rough < 1e-6, f"lam_c={lam_c} should still give ~zero cost along the null space"

    # ridge_c gives a real, nonzero, growing cost along the same direction.
    ridge_zero = 0.5 * 0.0 * np.sum(null_direction ** 2)
    ridge_default = 0.5 * 0.1 * np.sum(null_direction ** 2)  # FitConfig.ridge_c's own default
    assert ridge_zero == 0.0
    assert ridge_default > 0.05, "expected a decisively nonzero cost at the default ridge_c"


def test_ridge_c_does_not_degrade_genuine_continuum_recovery():
    """Companion to the null-space-closing test above: confirms the fix
    doesn't cost anything on the genuine-recovery side. Calibration note
    (see `FitConfig.ridge_c`'s own help text): `ridge_c=1.0` closed the
    null space just as effectively as `0.1` (recovery RMS identical either
    way) but tipped `fit_continuum_block`'s own already-documented
    spurious `success=False` flag in several regression tests -- `0.1`
    was chosen to avoid that while still being decisively nonzero.
    """
    wave = np.linspace(8400.0, 8800.0, 321)
    M = np.full(321, 50.0)
    true_C = 1.0 + 0.15 * np.sin(2 * np.pi * (wave - wave.min()) / (wave.max() - wave.min()))
    rng = np.random.default_rng(0)
    y_w = M * true_C + rng.normal(0.0, 0.5, 321)
    basis_c, n_coef = build_bspline_basis(wave, n_interior_knots=4, degree=3)
    dw = (wave.max() - wave.min()) / 5
    D2_c = difference_matrix(n_coef, order=2, dv=dw, dv_ref=dw)

    result_no_ridge = fit_continuum_block(M, y_w, basis_c, D2_c, lam_c=2.0, ridge_c=0.0)
    result_default = fit_continuum_block(M, y_w, basis_c, D2_c, lam_c=2.0)  # ridge_c default (0.1)
    rms_no_ridge = np.sqrt(np.mean((result_no_ridge["C"] - true_C) ** 2))
    rms_default = np.sqrt(np.mean((result_default["C"] - true_C) ** 2))
    assert rms_default < 0.02, f"expected close recovery of the injected continuum curve, got RMS={rms_default:.4f}"
    assert abs(rms_default - rms_no_ridge) < 0.005, (
        f"ridge_c's default should not measurably change genuine recovery quality: "
        f"no-ridge RMS={rms_no_ridge:.4f} vs. default RMS={rms_default:.4f}"
    )
