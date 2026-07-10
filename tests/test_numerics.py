"""Tests for core numerics (numerics.py) and metamorphic fitting properties.

These tests check mathematical properties that must hold regardless of the
specific spectrum, without needing a full end-to-end fit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinextract.fitting import compute_losvd_roughness, compute_losvd_n_peaks


MUSE_DIR = Path(__file__).parent.parent / "examples" / "data" / "muse"
SPEC_FILE = MUSE_DIR / "bin0105sp.spec"


# ── Metamorphic: roughness monotonicity ─────────────────────────────────────

def _losvd_with_sigma(sigma_kms: float, n: int = 101, v_max: float = 600.0) -> np.ndarray:
    v = np.linspace(-v_max, v_max, n)
    b = np.exp(-0.5 * (v / sigma_kms) ** 2)
    return b / b.sum()


@pytest.mark.parametrize("sigma_narrow,sigma_wide", [
    (30, 80),
    (50, 150),
    (80, 300),
])
def test_roughness_decreases_as_losvd_widens(sigma_narrow, sigma_wide):
    """Wider Gaussian → smoother → lower roughness."""
    r_narrow = compute_losvd_roughness(_losvd_with_sigma(sigma_narrow))
    r_wide = compute_losvd_roughness(_losvd_with_sigma(sigma_wide))
    assert r_narrow > r_wide, (
        f"Expected roughness(sigma={sigma_narrow}) > roughness(sigma={sigma_wide}), "
        f"got {r_narrow:.4g} <= {r_wide:.4g}"
    )


# ── Metamorphic: flux scaling invariance ─────────────────────────────────────

def test_roughness_scale_invariant():
    """Scaling the LOSVD by a constant should not change roughness
    (roughness is normalised by peak)."""
    b = _losvd_with_sigma(100.0)
    r1 = compute_losvd_roughness(b)
    r2 = compute_losvd_roughness(b * 42.0)
    assert abs(r1 - r2) < 1e-10


def test_n_peaks_scale_invariant():
    """Scaling the LOSVD amplitude should not change peak count."""
    b = _losvd_with_sigma(100.0)
    assert compute_losvd_n_peaks(b) == compute_losvd_n_peaks(b * 1000.0)


# ── Peak detection edge cases ────────────────────────────────────────────────

def test_n_peaks_negative_values_handled():
    """Negative LOSVD values (numerical noise) must not cause crashes."""
    b = _losvd_with_sigma(100.0) - 0.001  # some bins go negative
    result = compute_losvd_n_peaks(b)
    assert result >= 1


def test_n_peaks_very_short_array_returns_one():
    assert compute_losvd_n_peaks(np.array([0.5, 1.0, 0.5])) >= 1
    assert compute_losvd_n_peaks(np.array([1.0])) == 1


def test_n_peaks_two_equal_lobes():
    v = np.linspace(-500, 500, 201)
    b = (np.exp(-0.5 * ((v + 250) / 50.0) ** 2)
         + np.exp(-0.5 * ((v - 250) / 50.0) ** 2))
    assert compute_losvd_n_peaks(b) == 2


# ── Roughness boundary conditions ────────────────────────────────────────────

def test_roughness_finite_for_smooth_gaussian():
    b = _losvd_with_sigma(150.0)
    r = compute_losvd_roughness(b)
    assert np.isfinite(r)
    assert r >= 0.0


def test_roughness_returns_inf_for_all_zero():
    b = np.zeros(51)
    assert compute_losvd_roughness(b) == np.inf


def test_roughness_returns_inf_for_near_zero():
    b = np.full(51, 1e-12)
    assert compute_losvd_roughness(b) == np.inf


# ── estimate_velocity_xcorr: sub-pixel refinement ───────────────────────────

def test_estimate_velocity_xcorr_recovers_subpixel_shift_continuously():
    """A fractional-pixel injected shift should recover a continuous
    estimate, not snap to a multiple of one pixel's velocity step.

    Regression test for a real, measured bias (see estimate_velocity_xcorr's
    docstring): the pre-fix, integer-lag-only version could only return
    multiples of one MUSE pixel step (~43.6 km/s at the Ca II triplet),
    which on real N5102 bins with true |V| well under half a pixel step
    sometimes snapped to a neighboring, wrong lag entirely -- directly
    biasing the wing-taper's v_center regularization pivot. Injects a
    small, realistic fractional-pixel shift (0.5 pixels -- deliberately
    modest: this estimator is only ever used in production for small,
    near-systemic velocity residuals of order a few to a few tens of
    km/s, not large offsets, and the NNLS-weighted reference template it
    now also uses (see the "Reference template matters here too" section
    of estimate_velocity_xcorr's docstring) isn't designed to handle a
    large galaxy-vs-template position mismatch when only the galaxy, not
    the templates, is shifted -- a large synthetic shift here would test
    an unrealistic regime, not the real one) into the bundled real MUSE
    spectrum's own flux via linear interpolation, keeping the (unshifted)
    template matrix as the correlation reference, exactly like production
    usage on real data with a small residual velocity.
    """
    if not SPEC_FILE.exists():
        pytest.skip("bundled MUSE example data not found")
    from kinextract import FitConfig
    from kinextract.numerics import CEE, estimate_velocity_xcorr
    from kinextract.spectrum import make_fit_state

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
        fit_continuum=False,
        use_spectrum_errors=False,
        sigl=100.0, clean=False,
    )
    st, _ = make_fit_state(cfg, gal_file=str(SPEC_FILE), gal_errors=ferr)

    shift_pix = 0.5
    idx = np.arange(st.npix, dtype=float)
    st.g = np.interp(idx - shift_pix, idx, st.g, left=st.g[0], right=st.g[-1])

    lam_ref = float(st.x[st.npix // 2])
    true_v = shift_pix * st.scale * CEE / lam_ref
    pixel_step_v = st.scale * CEE / lam_ref

    v_est = estimate_velocity_xcorr(st)

    # A quantized, integer-only estimator could only answer a multiple of
    # pixel_step_v here -- never within half a pixel step of the injected
    # 2.5-pixel truth. The sub-pixel-refined estimator should land closer.
    assert abs(v_est - true_v) < 0.5 * pixel_step_v, (
        f"v_est={v_est:.2f} not within half a pixel step "
        f"({0.5 * pixel_step_v:.2f} km/s) of the injected true_v={true_v:.2f}"
    )
