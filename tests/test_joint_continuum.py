"""Tests for the joint continuum-in-the-model fit (kinextract.joint).

See kinextract/joint.py's module docstring for the motivation: the shipped
ALS/polynomial continuum's hyperparameter search was found to fall back to
an oversmoothed continuum on real data
(examples/notebooks/03_real_data_muse.ipynb), a structural issue with
fitting the continuum as a separate sub-problem rather than a bug specific
to either continuum family. The joint method folds a continuum directly
into the joint LOSVD/template optimization instead, using a penalized
B-spline (P-spline) basis -- an earlier raw-polynomial prototype was found
(via a dedicated stress test) to lack the flexibility to match realistic
continuum shapes, and to suffer from Vandermonde ill-conditioning at higher
order, motivating the switch.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from kinextract import FitConfig, set_verbose
from kinextract.joint import (
    build_initial_guess,
    build_pspline_design,
    evaluate_model_gp_joint,
    fit_joint,
    normalize_template_matrix,
)
from kinextract.losvd import fit_losvd_gauss_hermite
from kinextract.mocks import true_losvd_on_grid
from kinextract.spectrum import make_fit_state

set_verbose(False)

MUSE_DIR = Path(__file__).parent.parent / "examples" / "data" / "muse"
SPEC_FILE = MUSE_DIR / "bin0105sp.spec"


@pytest.fixture(scope="module")
def real_muse_state():
    """A real FitState built from the bundled MUSE example spectrum.

    Skipped if the bundled example data isn't present, matching the
    convention in tests/conftest.py's real_muse_fit fixture.
    """
    if not SPEC_FILE.exists():
        pytest.skip("bundled MUSE example data not found")
    data = np.loadtxt(SPEC_FILE)
    flux = data[:, 1]
    ferr = flux / 50.0
    cfg = FitConfig(
        template_list_file=str(MUSE_DIR / "Tlist"),
        template_dir=str(MUSE_DIR),
        wavemin_full=4750.0, step=1.25,
        wavefitmin=8400.0, wavefitmax=8750.0,
        zgal=0.001556,
        losvd_vmin=-300.0, losvd_vmax=300.0,
        fit_als_continuum=False,  # not using the shipped continuum machinery
        use_spectrum_errors=False,
        xlam=10000.0, xlam_auto=False,
        sigl=100.0, clean=False,
    )
    st, _ = make_fit_state(cfg, gal_file=str(SPEC_FILE), gal_errors=ferr)
    return st


def _realistic_template_weights(st, n_interior_knots, rng):
    """Fit the real spectrum once to get a realistic template mixture.

    Using a naive *uniform* weight across all 35 templates simultaneously
    turns out to be an unrealistic, near-degenerate mock setup -- confirmed
    directly: even the shipped, already-validated pipeline recovers V
    wildly wrong (1.3 km/s vs an 80 km/s truth) on a mock built that way,
    for the same underlying reason pPXF-style fitting always finds a sparse
    optimal template combination rather than averaging the whole library --
    a flat average of 35 different stellar spectra partially degenerates
    with LOSVD broadening itself. Matches the precedent in
    `kinextract.mocks.build_matched_mock`, which always derives its
    injected mock's non-LOSVD parameters (template weights, continuum) from
    an actual completed fit to real data, never from a naive uniform guess.
    """
    result, design = fit_joint(st, n_interior_knots=n_interior_knots, use_jax=True)
    w_real = result.x[st.nl:st.nl + st.nt]
    return w_real / np.sum(w_real)


def _build_synthetic_mock(st, v_true, sigma_true, cont_true_coeffs, n_interior_knots, rng, w_true=None):
    """Build a synthetic spectrum with a known injected LOSVD and continuum.

    Uses the joint model itself (`evaluate_model_gp_joint`) to generate the
    noiseless "truth" spectrum from a known parameter vector, then adds
    per-pixel Gaussian noise at the real target's own error level -- the
    same self-consistency validation pattern as
    `kinextract.mocks.build_matched_mock`, adapted for this module's
    parameter layout. `w_true` should be a realistic template mixture (see
    `_realistic_template_weights`), not a naive uniform average.
    """
    design, _knots = build_pspline_design(st.x, n_interior_knots)
    t_norm, _ = normalize_template_matrix(st)

    b_true = true_losvd_on_grid(st.xl, v_true, sigma_true)
    if w_true is None:
        w_true = _realistic_template_weights(st, n_interior_knots, rng)
    a_true = np.concatenate([b_true, w_true, cont_true_coeffs])

    gp_true, *_ = evaluate_model_gp_joint(a_true, st, design, t_norm)
    good = (st.gerr > 0.0) & (st.gerr < 1.0e9)
    noise = np.where(good, rng.normal(0.0, 1.0, size=gp_true.shape) * st.gerr, 0.0)
    g_mock = gp_true + noise

    st_mock = dataclasses.replace(st, g=g_mock)
    return st_mock, a_true, design, t_norm


def _smooth_continuum_curve(x, level=39000.0):
    """A smooth, genuinely-curved continuum shape (not just a flat level),
    as an explicit function of wavelength -- used to derive "true" P-spline
    coefficients via projection (see callers) rather than hand-picking
    coefficients directly, which (unlike raw polynomial coefficients) don't
    have an immediately obvious interpretable shape on their own.
    """
    xn = (x - np.median(x)) / ((x.max() - x.min()) / 2.0)
    return level + 1200.0 * xn + 600.0 * xn**2 - 300.0 * xn**3 + 200.0 * xn**4


def test_joint_fit_recovers_known_losvd_and_continuum(real_muse_state):
    """Core recovery test: known LOSVD + known continuum on a real template.

    The real MUSE template carries genuine Ca II absorption features, so
    this also directly tests the concern that motivated dropping ALS's
    asymmetric-reweighting/masking machinery: if the joint fit's continuum
    were getting biased downward by absorption troughs the way a naive
    symmetric baseline fit would be, the recovered continuum would
    systematically undershoot the injected truth. It doesn't have to -- the
    template*LOSVD component is what's responsible for reproducing the
    troughs.
    """
    st = real_muse_state
    rng = np.random.default_rng(12345)
    n_interior_knots = 10
    v_true, sigma_true = 80.0, 120.0

    design_true, _knots = build_pspline_design(st.x, n_interior_knots)
    cont_curve_true = _smooth_continuum_curve(st.x)
    cont_true_coeffs, *_ = np.linalg.lstsq(design_true, cont_curve_true, rcond=None)

    st_mock, a_true, design, t_norm = _build_synthetic_mock(
        st, v_true, sigma_true, cont_true_coeffs, n_interior_knots, rng
    )

    result, design_fit = fit_joint(st_mock, n_interior_knots=n_interior_knots, use_jax=True)
    assert result.success, result.message

    gp, b, w, cont_coeffs, cont = evaluate_model_gp_joint(result.x, st_mock, design_fit, t_norm)

    # LOSVD recovery: fit Gauss-Hermite moments to the recovered histogram
    # and compare to the injected truth.
    gh = fit_losvd_gauss_hermite(st.xl, b, fit_h3h4=True)
    assert abs(gh["vherm"] - v_true) < 10.0, f"V recovery off: {gh['vherm']} vs {v_true}"
    assert abs(gh["sherm"] - sigma_true) < 10.0, f"sigma recovery off: {gh['sherm']} vs {sigma_true}"

    # Continuum recovery: the fitted continuum should track the injected
    # truth shape closely, not just its overall level -- this is the whole
    # point (the shipped ALS/polynomial continuum was found to visibly fail
    # this on real data).
    cont_true = design @ cont_true_coeffs
    good = (st.gerr > 0.0) & (st.gerr < 1.0e9)
    rel_err = np.abs((cont[good] - cont_true[good]) / cont_true[good])
    assert np.median(rel_err) < 0.02, f"median continuum relative error too large: {np.median(rel_err):.4f}"

    # LOSVD normalization is a hard constraint, not a soft penalty -- must
    # be exact, not just "close to 1".
    assert np.isclose(np.sum(b), 1.0, atol=1e-8)


def test_joint_fit_matches_without_jax(real_muse_state):
    """The finite-difference (no-JAX) fallback path should agree with the
    JAX path on a quick, small problem -- confirms the two objective
    implementations (model.objective_joint vs
    model.build_jax_objective_value_and_grad) are consistent with each
    other, not just each internally self-consistent.
    """
    st = real_muse_state
    rng = np.random.default_rng(999)
    n_interior_knots = 6

    design_true, _knots = build_pspline_design(st.x, n_interior_knots)
    cont_curve_true = _smooth_continuum_curve(st.x, level=39000.0)
    cont_true_coeffs, *_ = np.linalg.lstsq(design_true, cont_curve_true, rcond=None)

    st_mock, a_true, design, t_norm = _build_synthetic_mock(
        st, 0.0, 100.0, cont_true_coeffs, n_interior_knots, rng
    )

    result_jax, design_jax = fit_joint(st_mock, n_interior_knots=n_interior_knots, use_jax=True, maxiter=2000)
    result_fd, design_fd = fit_joint(st_mock, n_interior_knots=n_interior_knots, use_jax=False, maxiter=2000, maxfun=20000)

    # Both should land in a comparable, low-chi2 region -- not necessarily
    # bit-identical (different gradient sources take different paths), but
    # not wildly different objective values either.
    assert result_jax.fun == pytest.approx(result_fd.fun, rel=0.1)


def test_normalize_template_matrix_removes_scale_spread(real_muse_state):
    """The bundled MUSE templates have a large (~14x) spread in per-template
    median level (confirmed directly during diagnosis -- this is what
    created a genuine scale degeneracy with a free joint continuum before
    this normalization was added). After normalization every template
    column's median should be 1.
    """
    st = real_muse_state
    t_norm, template_scale = normalize_template_matrix(st)
    assert np.allclose(np.median(t_norm, axis=0), 1.0)
    assert np.all(template_scale > 0)


def test_losvd_hard_normalization_is_exact(real_muse_state):
    """evaluate_model_gp_joint must renormalize b to sum to exactly 1
    regardless of the raw parameter vector's own LOSVD-block sum -- this is
    the fix for the degeneracy where a free continuum let sum(b) drift
    (confirmed empirically to ~0.4 before this fix, with the continuum
    inflating by the reciprocal factor to compensate).
    """
    st = real_muse_state
    n_interior_knots = 10
    design, _knots = build_pspline_design(st.x, n_interior_knots)
    t_norm, _ = normalize_template_matrix(st)

    rng = np.random.default_rng(7)
    cont = np.linspace(10.0, 39000.0, design.shape[1])
    for raw_scale in [0.3, 1.0, 3.0]:
        b_raw = rng.uniform(0.1, 1.0, size=st.nl) * raw_scale
        w = np.full(st.nt, 1.0 / st.nt)
        a = np.concatenate([b_raw, w, cont])
        _, b_used, *_ = evaluate_model_gp_joint(a, st, design, t_norm)
        assert np.isclose(np.sum(b_used), 1.0, atol=1e-8), f"raw_scale={raw_scale}"


def test_build_initial_guess_produces_finite_sane_guess(real_muse_state):
    """Sanity check on the initial-guess/bounds builder used by fit_joint:
    finite values, bounds actually bracket the initial guess, and the
    initial continuum guess is within an order of magnitude of the data's
    own scale (not some wildly wrong number)."""
    st = real_muse_state
    x0, bounds, xscale, design = build_initial_guess(st, n_interior_knots=10)
    assert np.all(np.isfinite(x0))
    assert np.all(np.isfinite(xscale)) and np.all(xscale > 0)
    for val, (lo, hi) in zip(x0, bounds):
        assert lo <= val <= hi

    good = (st.gerr > 0.0) & (st.gerr < 1.0e9)
    cont0 = x0[st.nl + st.nt:]
    cont_curve0 = design @ cont0
    ratio = np.median(cont_curve0[good]) / np.median(st.g[good])
    assert 0.1 < ratio < 10.0, f"initial continuum guess wildly off data scale: ratio={ratio:.3f}"


def test_run_spectral_fit_dispatches_to_joint_by_default():
    """continuum_method defaults to "joint": run_spectral_fit(cfg) with
    fit_als_continuum=True should route through kinextract.joint.run_joint_fit
    rather than the shipped ALS/polynomial outer loop, and return an
    "outputs" dict with the same keys the shipped path produces (so
    downstream consumers that only look at "outputs" -- plotting, etc. --
    don't need to know which continuum method actually ran), with neutral
    coff/coff2/A placeholders since the joint model has no such parameters.
    """
    if not SPEC_FILE.exists():
        pytest.skip("bundled MUSE example data not found")
    from kinextract import FitConfig, run_spectral_fit
    from kinextract.errors import LOSVDErrorEstimator
    from kinextract.validation import assess_recovery_bias

    data = np.loadtxt(SPEC_FILE)
    flux = data[:, 1]
    ferr = flux / 50.0
    cfg = FitConfig(
        template_list_file=str(MUSE_DIR / "Tlist"),
        template_dir=str(MUSE_DIR),
        wavemin_full=4750.0, step=1.25,
        wavefitmin=8400.0, wavefitmax=8750.0,
        zgal=0.001556,
        losvd_vmin=-300.0, losvd_vmax=300.0,
        fit_als_continuum=True,  # continuum_method left at its "joint" default
        use_spectrum_errors=False,
        sigl=100.0, clean=False,
        joint_n_sigl0_iter=1,  # keep the test fast; convergence itself is covered elsewhere
        map_maxiter=2000,
    )
    assert cfg.continuum_method == "joint"

    fit = run_spectral_fit(cfg, gal_file=str(SPEC_FILE), gal_errors=ferr)

    for key in ("gp", "b", "w", "wfrac", "tt", "coff", "coff2", "A",
                "rms", "chi2_red", "chi2_total", "nrms", "continuum", "paths"):
        assert key in fit["outputs"], f"missing outputs key: {key}"
    assert fit["outputs"]["coff"] == 0.0
    assert fit["outputs"]["coff2"] == 0.0
    assert fit["outputs"]["A"] == 1.0
    assert np.all(np.isfinite(fit["outputs"]["gp"]))
    assert np.isclose(np.sum(fit["outputs"]["b"]), 1.0, atol=1e-8)

    # The old bootstrap/mock-based tools don't understand the joint layout
    # yet -- they must refuse to run rather than silently misinterpreting
    # a_map, per the explicit product decision to block them for now.
    with pytest.raises(NotImplementedError):
        LOSVDErrorEstimator(fit, cfg)
    with pytest.raises(NotImplementedError):
        assess_recovery_bias(fit, cfg, v_true_grid=[0.0], sigma_true_grid=[100.0], n_seeds=1)


def test_prenormalized_mode_defaults_to_shipped_path_not_joint():
    """fit_als_continuum=False (pre-normalized mode) must NOT go through
    kinextract.joint by default, even though continuum_method="joint" is
    itself the default -- joint's sigl0 fixed-point iteration costs up to
    n_sigl0_iter * len(xlam_auto_grid) full optimizations per fit, so
    silently applying it to every ordinary pre-normalized fit would be an
    unrequested ~15x slowdown. Confirmed here via the "outputs" dict shape:
    the shipped path's coff/coff2/A are real fitted values (not the joint
    path's neutral 0.0/0.0/1.0 placeholders) for this config's icoff=2 mode.
    """
    if not SPEC_FILE.exists():
        pytest.skip("bundled MUSE example data not found")
    from kinextract import FitConfig, run_spectral_fit

    data = np.loadtxt(SPEC_FILE)
    flux = data[:, 1]
    ferr = flux / 50.0
    cfg = FitConfig(
        template_list_file=str(MUSE_DIR / "Tlist"),
        template_dir=str(MUSE_DIR),
        wavemin_full=4750.0, step=1.25,
        wavefitmin=8400.0, wavefitmax=8750.0,
        zgal=0.001556,
        losvd_vmin=-300.0, losvd_vmax=300.0,
        use_spectrum_errors=False,
        sigl=100.0, clean=False,
        map_maxiter=2000,
    )
    assert cfg.fit_als_continuum is False
    assert cfg.continuum_method == "joint"
    assert cfg.joint_prenorm is False

    fit = run_spectral_fit(cfg, gal_file=str(SPEC_FILE), gal_errors=ferr)
    assert "sigl0_trace" not in fit["outputs"], (
        "pre-normalized fit unexpectedly went through kinextract.joint"
    )


def test_joint_prenorm_opt_in_fixes_continuum_at_one():
    """cfg.joint_prenorm=True routes a pre-normalized (fit_als_continuum=False)
    fit through kinextract.joint anyway, for its v_center/sigl0/xlam-selection
    improvements -- with the continuum fixed at 1.0 rather than co-fit, since
    pre-normalized input has no genuine continuum left to fit.
    """
    if not SPEC_FILE.exists():
        pytest.skip("bundled MUSE example data not found")
    from kinextract import FitConfig, run_spectral_fit

    data = np.loadtxt(SPEC_FILE)
    flux = data[:, 1]
    ferr = flux / 50.0
    cfg = FitConfig(
        template_list_file=str(MUSE_DIR / "Tlist"),
        template_dir=str(MUSE_DIR),
        wavemin_full=4750.0, step=1.25,
        wavefitmin=8400.0, wavefitmax=8750.0,
        zgal=0.001556,
        losvd_vmin=-300.0, losvd_vmax=300.0,
        use_spectrum_errors=False,
        sigl=100.0, clean=False,
        joint_prenorm=True,
        joint_n_sigl0_iter=1,  # keep the test fast
        map_maxiter=2000,
    )
    assert cfg.fit_als_continuum is False

    fit = run_spectral_fit(cfg, gal_file=str(SPEC_FILE), gal_errors=ferr)
    assert "sigl0_trace" in fit["outputs"], "joint_prenorm=True should route through kinextract.joint"
    assert np.allclose(fit["outputs"]["continuum"], 1.0), (
        "continuum should be fixed at 1.0 in joint_prenorm mode, not co-fit"
    )
    assert fit["outputs"]["coff"] == 0.0
    assert np.isclose(np.sum(fit["outputs"]["b"]), 1.0, atol=1e-8)
