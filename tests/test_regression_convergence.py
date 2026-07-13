"""Regression-convergence guardrails.

These three tests exist because of a real regression found in this
session: raising ``FitConfig.n_losvd_bins`` from 29 to 89 was shipped
without recalibrating the auto-xlam discrepancy-principle search for the
new bin count. Combined with a separate, unrelated tuning mistake
(``xlam_discrepancy_nsigma=1.0`` in notebook 02), the auto-xlam search ran
away to ``xlam ~ 5e7`` and the realistic mock fit failed to converge
(``success=False``), while the real MUSE data fit degraded and everything
got noticeably slower. None of this tripped any existing test, because no
test asserted convergence/timing/xlam-sanity on these specific, previously
known-good scenarios -- it was only caught by the user noticing the
regression by eye.

Each test below fits one of the three scenarios manually checked during
that investigation (basic synthetic mock, realistic multi-template E-MILES
mock with joint continuum, real MUSE data with joint continuum) at
whatever ``FitConfig`` defaults currently ship, and asserts:

- ``result.success is True``
- the auto-selected ``xlam`` stays within a generous but finite multiple of
  ``cfg.xlam_auto_grid[-1]`` (catches an "xlam ran away" case even if
  L-BFGS-B still happens to report ``success=True`` some other way)
- wall-clock time stays under a generous budget (directly targets the
  "everything is slower now" symptom)
- for the two synthetic mocks, recovered V/sigma land within a
  pre-characterized tolerance of the known truth

These are deliberately NOT exact reproductions of the notebooks (no file
I/O round-trips, no plotting) -- they reuse the same mock-building/
FitConfig-construction helpers the notebooks and benchmarks/scenarios.py
already use, just wired directly into pytest so a future config-default
change that reintroduces this class of bug fails the normal test suite
instead of waiting for a human to notice a wiggly plot.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinextract import FitConfig, run_spectral_fit
from kinextract.fitting import fit_losvd_gauss_hermite

from benchmarks.mock_spectrum import MUSE_CAII, build_mock_spectrum, write_mock_to_disk
from benchmarks.scenarios import Scenario, scenario_to_fitconfig


def _assert_healthy_fit(fit, cfg, label, *, max_seconds, xlam_grid_multiple=100.0):
    result = fit["result"]
    st = fit["state"]
    assert result.success, (
        f"{label}: fit did not converge ({result.message}) -- this is exactly the "
        f"symptom of the n_losvd_bins/xlam_discrepancy_nsigma regression this test "
        f"guards against."
    )
    xlam_ceiling = xlam_grid_multiple * cfg.xlam_auto_grid[-1]
    assert st.xlam <= xlam_ceiling, (
        f"{label}: selected xlam={st.xlam:.4g} exceeds {xlam_grid_multiple:.0f}x the "
        f"top of xlam_auto_grid ({cfg.xlam_auto_grid[-1]:.4g}) -- the auto-xlam search "
        f"likely ran away (see fitting._discrepancy_principle_search's bracket-expansion "
        f"warning); check n_losvd_bins/xlam_discrepancy_nsigma/xlam_auto_grid."
    )
    assert fit["elapsed_seconds"] <= max_seconds, (
        f"{label}: fit took {fit['elapsed_seconds']:.1f}s, budget was {max_seconds:.0f}s -- "
        f"this is the 'everything got slower' regression symptom."
    )


def _timed_run_spectral_fit(cfg, gal_file):
    t0 = time.perf_counter()
    fit = run_spectral_fit(cfg, gal_file=gal_file)
    fit["elapsed_seconds"] = time.perf_counter() - t0
    return fit


@pytest.mark.slow
def test_basic_synthetic_mock_converges_fast_and_recovers_truth(tmp_path):
    """Mirrors 01_basic_mock_fit.ipynb: single synthetic template, no continuum."""
    v_true, sigma_true = 80.0, 140.0
    mock = build_mock_spectrum(
        instrument=MUSE_CAII, v_true=v_true, sigma_true=sigma_true, snr_target=50.0,
        continuum_mode="none", template_role="matched", losvd_shape="gaussian",
        use_real_template=False, noise_seed=1000,
    )
    files = write_mock_to_disk(mock, tmp_path, prefix="basic_mock")

    cfg = FitConfig(
        template_list_file=str(files["tlist"]), template_dir=str(tmp_path),
        outdir=str(tmp_path),
        wavemin_full=MUSE_CAII.wavemin_full, step=MUSE_CAII.step,
        wavefitmin=MUSE_CAII.wavefitmin, wavefitmax=MUSE_CAII.wavefitmax,
        zgal=0.0, fit_continuum=False,
        xlam_auto=True, xlam_max_peaks=1,
        sigl=100.0, joint_n_sigl0_iter=1, use_spectrum_errors=True,
        clean=False, map_maxiter=10000, map_ftol=1e-8, print_every=999999,
    )
    fit = _timed_run_spectral_fit(cfg, gal_file=str(files["spec"]))
    _assert_healthy_fit(fit, cfg, "basic synthetic mock", max_seconds=30.0)

    gh = fit_losvd_gauss_hermite(fit["state"].xl, fit["outputs"]["b"], fit_h3h4=True)
    assert abs(gh["vherm"] - v_true) < 15.0, f"recovered V={gh['vherm']:.1f}, truth={v_true}"
    assert abs(gh["sherm"] - sigma_true) < 15.0, f"recovered sigma={gh['sherm']:.1f}, truth={sigma_true}"


@pytest.mark.slow
def test_realistic_emiles_mock_converges_fast_and_recovers_truth(tmp_path):
    """Mirrors 02_realistic_mock_fit.ipynb: real template + joint continuum co-fit.

    This is the exact scenario that caught the regression: with
    n_losvd_bins=89 + xlam_discrepancy_nsigma=1.0 this failed to converge
    (xlam ~ 5e7); at the current package defaults it should converge fast.
    """
    v_true, sigma_true = 80.0, 140.0
    scenario = Scenario(
        scenario_id="regression_guardrail_realistic_mock", axis="regression",
        instrument=MUSE_CAII, v_true=v_true, sigma_true=sigma_true, snr_target=50.0,
        template_role="matched", losvd_shape="gaussian",
    )
    mock = build_mock_spectrum(
        instrument=scenario.instrument, v_true=scenario.v_true, sigma_true=scenario.sigma_true,
        snr_target=scenario.snr_target, continuum_mode="joint",
        template_role=scenario.template_role, losvd_shape=scenario.losvd_shape,
        use_real_template=True, noise_seed=1000,
    )
    files = write_mock_to_disk(mock, tmp_path, prefix="realistic_mock")

    # Explicit xlam_criterion/xlam_discrepancy_nsigma so this test exercises
    # whatever FitConfig's own real defaults are (scenario_to_fitconfig's own
    # defaults are "chi2"/1.0, kept for its unrelated continuum-mode
    # comparison suite -- not what a real user gets).
    cfg = scenario_to_fitconfig(
        scenario, continuum_mode="joint", outdir=str(tmp_path),
        template_list_file=str(files["tlist"]), template_dir=str(tmp_path),
        xlam_criterion="discrepancy", xlam_discrepancy_nsigma=0.3,
    )
    fit = _timed_run_spectral_fit(cfg, gal_file=str(files["spec"]))
    _assert_healthy_fit(fit, cfg, "realistic E-MILES mock (joint continuum)", max_seconds=45.0)

    gh = fit_losvd_gauss_hermite(fit["state"].xl, fit["outputs"]["b"], fit_h3h4=True)
    assert abs(gh["vherm"] - v_true) < 20.0, f"recovered V={gh['vherm']:.1f}, truth={v_true}"
    assert abs(gh["sherm"] - sigma_true) < 20.0, f"recovered sigma={gh['sherm']:.1f}, truth={sigma_true}"


@pytest.mark.slow
def test_real_muse_data_converges_fast(tmp_path):
    """Mirrors 03_real_data_muse.ipynb: real MUSE spectrum, joint continuum,
    kinextract's own bundled E-MILES grid, narrow Ca II window.

    No ground truth to check here (real data) -- only convergence/xlam-
    sanity/timing, exactly what degraded in the real regression report
    ("we used to recover a very good LOSVD for NGC 5102's MUSE data that has
    now degraded significantly... the whole code now is slower").
    """
    muse_dir = Path(__file__).parent.parent / "examples" / "data" / "muse"
    spec_file = muse_dir / "bin0105sp.spec"
    emiles_npz = Path(__file__).parent.parent / "examples" / "data" / "emiles" / "kinextract_emiles_grid.npz"
    if not spec_file.exists() or not emiles_npz.exists():
        pytest.skip("bundled MUSE example data or E-MILES grid not found")

    fit_ages_gyr = [3.0, 4.5, 6.5, 9.5, 14.0]
    fit_metals = [-0.35, -0.25, 0.06, 0.15]
    template_select = [(age, metal) for age in fit_ages_gyr for metal in fit_metals]

    cfg = FitConfig(
        template_npz_file=str(emiles_npz), template_npz_select=template_select,
        outdir=str(tmp_path),
        wavemin_full=4750.0, step=1.25,
        wavefitmin=8400.0, wavefitmax=8750.0,
        zgal=0.001556, losvd_vmin=-300.0, losvd_vmax=300.0,
        fit_continuum=True, use_spectrum_errors=True,
        xlam_auto=True, sigl=60.0, clean=True,
        map_maxiter=20000, print_every=999999,
    )
    fit = _timed_run_spectral_fit(cfg, gal_file=str(spec_file))
    _assert_healthy_fit(fit, cfg, "real MUSE data (NGC 5102)", max_seconds=90.0)
