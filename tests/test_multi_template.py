"""Tests for kinextract.multi_template.

See `multi_template.py`'s own module docstring for the full story: an
initial multi-seed test used near-collinear templates by accident and
showed genuinely bimodal weight recovery; re-tested with genuinely
distinct template shapes, weight recovery was excellent and stable. Both
regimes are locked in here so a future change can't silently reintroduce
confusion about which is which. Also see that docstring for why the
outer multi-start over initial weight vectors was tested (100 random
starts) and dropped as unnecessary -- `fit_losvd_basis` now
runs a single equal-weights start by default, so `objective_gap` is
`None` unless an explicit multi-point `w0_pool` is passed.

**These tests take a few seconds each**: each fit is one alternating
trajectory (single starting point), several inner iterations, each a
full multi-start LOSVD sub-fit.
"""
from __future__ import annotations

import numpy as np

from kinextract import (
    FitConfig, build_template_stack, build_spline_truth,
    fit_losvd, compute_losvd_moments,
)


def _make_template(wavelength, centers, depths, width):
    t = np.ones_like(wavelength)
    for cen, d in zip(centers, depths):
        t -= d * np.exp(-0.5 * ((wavelength - cen) / width) ** 2)
    return t


def _mock_multi_template_case(templates, true_w, vel=0.0, sig=140.0, nsig=7, snr=100.0, seed=0):
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]

    vmin, vmax = vel - nsig * sig, vel + nsig * sig
    cfg = FitConfig(vmin=vmin, vmax=vmax)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=vel, sigma=sig, wiggle_scale=0.0, seed=seed)

    A_stack = build_template_stack(wave_fit, wavelength, templates, v_grid)
    galaxy_noiseless = sum(true_w[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors
    return A_stack_w, y_w, cfg, v_grid, b_truth


def _distinct_templates(wavelength):
    ca_centers = [8498.02, 8542.09, 8662.14]
    return [
        _make_template(wavelength, ca_centers, [0.30, 0.65, 0.35], 5.0),
        _make_template(wavelength, ca_centers, [0.60, 0.25, 0.55], 3.0),
        _make_template(wavelength, ca_centers, [0.45, 0.45, 0.20], 7.0),
    ]


def _collinear_templates(wavelength):
    ca_centers = [8498.02, 8542.09, 8662.14]
    return [
        _make_template(wavelength, ca_centers, [0.35, 0.45, 0.40], 5.0),
        _make_template(wavelength, ca_centers, [0.65, 0.85, 0.75], 5.0),
        _make_template(wavelength, ca_centers, [0.20, 0.30, 0.25], 5.0),
    ]


def _full_wavelength():
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    return wavemin + np.arange(n_pix) * step


def test_fit_losvd_runs_and_recovers_known_case():
    wavelength = _full_wavelength()
    templates = _distinct_templates(wavelength)
    true_w = np.array([0.5, 0.3, 0.2])
    A_stack_w, y_w, cfg, v_grid, b_truth = _mock_multi_template_case(templates, true_w, seed=0)

    result = fit_losvd(A_stack_w, y_w, cfg)

    assert result["success"]
    assert np.all(np.isfinite(result["b"])) and np.all(result["b"] > 0.0)
    assert np.all(result["w"] > 0.0)
    assert result["objective_gap"] is None  # single-start default -- see module docstring

    w_norm = result["w"] / result["w"].sum()
    np.testing.assert_allclose(w_norm, true_w, atol=0.05)

    gh = compute_losvd_moments(v_grid, result["b"])
    assert abs(gh["vherm"] - 0.0) < 10.0
    assert abs(gh["sherm"] - 140.0) < 15.0


def test_weight_recovery_is_stable_for_distinct_templates_across_seeds():
    """Locks in the "well-behaved" regime from multi_template.py's own
    docstring: with genuinely distinct template shapes, weight recovery
    should be tight and consistent across independent noise draws, not
    just correct on one lucky seed.
    """
    wavelength = _full_wavelength()
    templates = _distinct_templates(wavelength)
    true_w = np.array([0.5, 0.3, 0.2])

    w_norms = []
    for seed in range(5):
        A_stack_w, y_w, cfg, v_grid, b_truth = _mock_multi_template_case(templates, true_w, seed=seed)
        result = fit_losvd(A_stack_w, y_w, cfg)
        w_norms.append(result["w"] / result["w"].sum())
    w_norms = np.array(w_norms)

    assert np.all(w_norms.std(axis=0) < 0.03), f"expected tight weight recovery, got std={w_norms.std(axis=0)}"


def test_weight_recovery_is_unstable_for_near_collinear_templates():
    """Characterization test, not a "should pass" sanity check: with
    near-collinear templates (differing mainly in an overall depth
    scale, not in shape), weight recovery is expected to be genuinely
    unstable across noise realizations -- a property of the template
    library's own conditioning, not a bug in the joint fit. If this test
    starts failing because the instability disappeared, that would be
    surprising and worth understanding, not just deleting the test.
    """
    wavelength = _full_wavelength()
    templates = _collinear_templates(wavelength)
    true_w = np.array([0.5, 0.3, 0.2])

    w_norms = []
    for seed in range(5):
        A_stack_w, y_w, cfg, v_grid, b_truth = _mock_multi_template_case(templates, true_w, seed=seed)
        result = fit_losvd(A_stack_w, y_w, cfg)
        w_norms.append(result["w"] / result["w"].sum())
    w_norms = np.array(w_norms)

    assert np.any(w_norms.std(axis=0) > 0.05), (
        f"expected at least one template's weight to be unstable across seeds for a "
        f"near-collinear library, got std={w_norms.std(axis=0)}"
    )


def test_max_iter_zero_raises_a_clear_error():
    """Found during stress testing: max_iter=0 used to crash with
    `TypeError: 'NoneType' object is not subscriptable` (the alternating
    loop never runs, so `best` stays None and the final `best["b"]`
    lookup fails) -- a confusing error pointing nowhere near the actual
    cause. Should fail clearly at the boundary instead. `max_iter` is
    only exposed on the lower-level `fit_losvd_basis`, not the
    `FitConfig`-based `_cfg` wrapper.
    """
    import pytest
    from kinextract.multi_template import fit_losvd_basis
    from kinextract.losvd import build_bspline_basis, difference_matrix

    wavelength = _full_wavelength()
    templates = _distinct_templates(wavelength)
    true_w = np.array([0.5, 0.3, 0.2])
    A_stack_w, y_w, cfg, v_grid, b_truth = _mock_multi_template_case(templates, true_w, seed=0)

    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D2 = difference_matrix(n_coef, order=2, dv=dv, dv_ref=cfg.knot_spacing_kms)
    with pytest.raises(ValueError, match="max_iter"):
        fit_losvd_basis(A_stack_w, y_w, basis, D2, max_iter=0)


def test_fit_losvd_continuum_knots_requires_wave_fit():
    import pytest

    wavelength = _full_wavelength()
    templates = _distinct_templates(wavelength)
    true_w = np.array([0.5, 0.3, 0.2])
    A_stack_w, y_w, cfg, v_grid, b_truth = _mock_multi_template_case(templates, true_w, seed=0)
    cfg_continuum = FitConfig(vmin=cfg.vmin, vmax=cfg.vmax, continuum_knots=4)

    with pytest.raises(ValueError, match="wave_fit"):
        fit_losvd(A_stack_w, y_w, cfg_continuum)


def test_fit_losvd_with_continuum_knots_set_runs_and_returns_C():
    """`FitConfig.continuum_knots` is the new (as of this refactor)
    standard-path way to turn on continuum-cofitting, replacing the
    previously separate `fit_losvd_with_continuum` as the only entry
    point -- both now share the same underlying alternation driver (see
    `multi_template._alternate_once`).
    """
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]

    templates = _distinct_templates(wavelength)
    true_w = np.array([0.5, 0.3, 0.2])
    sig = 140.0
    cfg = FitConfig(vmin=-7 * sig, vmax=7 * sig, continuum_knots=4)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=0.0, sigma=sig, wiggle_scale=0.0, seed=0)
    A_stack = build_template_stack(wave_fit, wavelength, templates, v_grid)
    galaxy_noiseless = sum(true_w[k] * (A_stack[k] @ b_truth) for k in range(len(templates)))
    noise_level = np.median(galaxy_noiseless) / 100.0
    rng = np.random.default_rng(0)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors

    result = fit_losvd(A_stack_w, y_w, cfg, wave_fit=wave_fit)
    assert result["success"]
    assert "C" in result and "continuum_coeffs" in result
    np.testing.assert_allclose(result["C"], np.ones_like(result["C"]), atol=0.05)


def test_degenerate_nnls_on_first_iteration_fails_gracefully_instead_of_crashing():
    """Found via code review, confirmed by reproduction: if NNLS drives every
    template weight to exactly zero on the very first alternating iteration
    (a pathological/adversarial noise draw, e.g. `y_w` anti-correlated with
    every nonnegative template combination), `_alternate_once`'s `best`
    tracker was still `None` when the loop broke out of the degenerate
    branch, and the final `result = dict(b=best["b"], ...)` lookup crashed
    with `TypeError: 'NoneType' object is not subscriptable`. This is
    directly reachable from `bootstrap_losvd`: a single unlucky noise draw
    among many replicates could crash an entire otherwise-healthy multi-
    minute bootstrap call. Now falls back to the first iteration's own
    (w0-based) LOSVD sub-fit, explicitly flagged `success=False`, instead
    of crashing -- `w` stays the (nonzero) starting weight vector rather
    than the degenerate all-zero NNLS output, so `w / w.sum()` downstream
    (in `bootstrap_losvd`'s `w_boot` normalization, for instance) doesn't
    divide by zero either.
    """
    from kinextract import FitConfig
    from kinextract.multi_template import fit_losvd

    rng = np.random.default_rng(0)
    cfg = FitConfig(vmin=-500.0, vmax=500.0)
    v_grid = cfg.build_velocity_grid()
    n_templates, n_pix = 2, 50
    A_stack = np.abs(rng.normal(size=(n_templates, n_pix, len(v_grid))))
    # adversarial: y_w entirely negative -> NNLS on a nonnegative A_stack @ b
    # (b >= 0 via the exp-link) can never reduce chi2 by using any positive
    # weight, so it drives every weight to exactly zero.
    y_w = -np.abs(rng.normal(size=n_pix)) * 5.0

    result = fit_losvd(A_stack, y_w, cfg)  # must not raise
    assert result["success"] is False
    assert "degenerate NNLS" in result["message"]
    assert np.all(np.isfinite(result["w"]))
    assert result["w"].sum() > 0


class TestFitLosvdParallel:
    """`fit_losvd_parallel`'s own core promise: fitting bins across worker
    processes must be bit-identical to fitting them one at a time in a
    serial loop -- pure wall-clock orchestration, zero accuracy cost. No
    prior test exercised this at all (found during the 2026-07-30 dead-
    code audit); this closes that gap.
    """

    @staticmethod
    def _bins(wavelength, templates):
        return [
            _mock_multi_template_case(templates, [1.0], vel=v, sig=s, seed=seed)
            for v, s, seed in [(0.0, 80.0, 0), (20.0, 140.0, 1), (-15.0, 60.0, 2)]
        ]

    def test_n_jobs_1_matches_a_serial_fit_losvd_loop(self):
        from kinextract import fit_losvd_parallel

        wavemin, step, n_pix = 4750.0, 1.25, 3681
        wavelength = wavemin + np.arange(n_pix) * step
        templates = [_make_template(wavelength, [8498.02, 8542.09, 8662.14], [0.5, 0.6, 0.4], 5.0)]

        bins = self._bins(wavelength, templates)
        A_stack_list = [b[0] for b in bins]
        y_w_list = [b[1] for b in bins]
        cfg_list = [b[2] for b in bins]

        serial = [fit_losvd(A, y, cfg) for A, y, cfg in zip(A_stack_list, y_w_list, cfg_list)]
        parallel = fit_losvd_parallel(A_stack_list, y_w_list, cfg_list, n_jobs=1)

        assert len(parallel) == len(serial) == 3
        for s, p in zip(serial, parallel):
            np.testing.assert_array_equal(s["b"], p["b"])
            np.testing.assert_array_equal(s["w"], p["w"])
            assert s["lam_used"] == p["lam_used"]

    def test_a_real_process_pool_is_bit_identical_to_serial(self):
        """The actual thing this function exists for: n_jobs > 1 spawns
        real worker processes (spawn context, thread-pinned) -- must
        still reproduce exactly what a serial loop gets, per bin, and in
        the same order the inputs were given.
        """
        from kinextract import fit_losvd_parallel

        wavemin, step, n_pix = 4750.0, 1.25, 3681
        wavelength = wavemin + np.arange(n_pix) * step
        templates = [_make_template(wavelength, [8498.02, 8542.09, 8662.14], [0.5, 0.6, 0.4], 5.0)]

        bins = self._bins(wavelength, templates)
        A_stack_list = [b[0] for b in bins]
        y_w_list = [b[1] for b in bins]
        cfg_list = [b[2] for b in bins]

        serial = [fit_losvd(A, y, cfg) for A, y, cfg in zip(A_stack_list, y_w_list, cfg_list)]
        parallel = fit_losvd_parallel(A_stack_list, y_w_list, cfg_list, n_jobs=2)

        for i, (s, p) in enumerate(zip(serial, parallel)):
            np.testing.assert_array_equal(s["b"], p["b"], err_msg=f"bin {i} mismatch")
            np.testing.assert_array_equal(s["w"], p["w"], err_msg=f"bin {i} mismatch")

    def test_mismatched_list_lengths_raise_a_clear_error(self):
        from kinextract import fit_losvd_parallel

        wavemin, step, n_pix = 4750.0, 1.25, 3681
        wavelength = wavemin + np.arange(n_pix) * step
        templates = [_make_template(wavelength, [8498.02, 8542.09, 8662.14], [0.5, 0.6, 0.4], 5.0)]
        bins = self._bins(wavelength, templates)
        A_stack_list = [b[0] for b in bins]
        y_w_list = [b[1] for b in bins]
        cfg_list = [b[2] for b in bins]

        import pytest
        with pytest.raises(ValueError, match="same length"):
            fit_losvd_parallel(A_stack_list, y_w_list[:-1], cfg_list, n_jobs=1)
        with pytest.raises(ValueError, match="one entry per bin"):
            fit_losvd_parallel(A_stack_list, y_w_list, cfg_list[:-1], n_jobs=1)
