"""Regression tests for the kinematic-fit clean/protect-window logic.

These guard against a real bug found in review: with the (then-default)
``protect_absorption_only=True``, `_update_clean_mask`'s protect-window
exemption was a silent no-op, because it only exempted resid<0 pixels,
which the one-sided sigma-clip already never rejects regardless of
`protect_mask`.  Real Ca II triplet pixels with a positive residual
(e.g. from continuum/template mismatch) were then clipped out of the
kinematic fit with zero protection, despite `clean_protect_ca_triplet=True`.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from kinextract.masking import _update_clean_mask, clean_ca_half_width_A
from kinextract.config import FitConfig


def _run_update_clean_mask(g, gp, protect_mask, protect_absorption_only, sigma_clip=3.0):
    """Call _update_clean_mask with a stubbed forward model.

    evaluate_model_gp is imported locally inside _update_clean_mask, so
    patching kinextract.numerics.evaluate_model_gp affects the next call.
    """
    st = SimpleNamespace(g=np.asarray(g, float))
    base_gerr = np.ones_like(st.g)
    good_mask = np.ones_like(st.g, dtype=bool)
    with patch("kinextract.numerics.evaluate_model_gp", return_value=(np.asarray(gp, float),)):
        new_mask, sigma = _update_clean_mask(
            st, a_best=None, base_gerr=base_gerr, good_mask=good_mask,
            sigma_clip=sigma_clip, protect_mask=protect_mask,
            protect_absorption_only=protect_absorption_only,
        )
    return new_mask, sigma


def test_protect_mask_shields_positive_residual_pixel_by_default():
    """A protected pixel with data above the model (resid > 0) must survive."""
    n = 50
    g = np.zeros(n)
    gp = np.zeros(n)
    # One large positive-residual outlier inside the protected window.
    outlier = 25
    g[outlier] = 20.0
    # Small residual noise everywhere else so robust_sigma is well-defined.
    rng = np.random.default_rng(0)
    g += rng.normal(0, 0.1, n)

    protect = np.zeros(n, dtype=bool)
    protect[outlier] = True

    new_mask, _ = _run_update_clean_mask(g, gp, protect, protect_absorption_only=False)
    assert new_mask[outlier], "protected pixel was clipped despite protect_mask=True"


def test_unprotected_positive_residual_pixel_is_still_clipped():
    """The same outlier, with no protection, must be rejected as usual."""
    n = 50
    g = np.zeros(n)
    gp = np.zeros(n)
    outlier = 25
    g[outlier] = 20.0
    rng = np.random.default_rng(0)
    g += rng.normal(0, 0.1, n)

    new_mask, _ = _run_update_clean_mask(g, gp, protect_mask=None, protect_absorption_only=False)
    assert not new_mask[outlier], "unprotected emission-like outlier should still be clipped"


def test_protect_absorption_only_true_is_a_documented_noop_for_positive_residuals():
    """protect_absorption_only=True must not rescue a positive-residual outlier.

    This is the exact behavior that made clean_protect_ca_triplet=True
    silently ineffective before the fix (which changed the *default* to
    protect_absorption_only=False); the flag is retained for backward
    compatibility and this test documents/locks in what it actually does.
    """
    n = 50
    g = np.zeros(n)
    gp = np.zeros(n)
    outlier = 25
    g[outlier] = 20.0
    rng = np.random.default_rng(0)
    g += rng.normal(0, 0.1, n)

    protect = np.zeros(n, dtype=bool)
    protect[outlier] = True

    new_mask, _ = _run_update_clean_mask(g, gp, protect, protect_absorption_only=True)
    assert not new_mask[outlier], (
        "protect_absorption_only=True unexpectedly protected a positive-residual "
        "pixel; if this now passes, the no-op behavior has changed and this "
        "test (and its docstring) should be revisited"
    )


def test_negative_residual_never_clipped_regardless_of_protection():
    """Absorption-like residuals (data below model) are never clipped, protected or not."""
    n = 50
    g = np.zeros(n)
    gp = np.zeros(n)
    deep = 25
    gp[deep] = 20.0  # model much higher than data => resid < 0
    rng = np.random.default_rng(0)
    g += rng.normal(0, 0.1, n)

    new_mask, _ = _run_update_clean_mask(g, gp, protect_mask=None, protect_absorption_only=False)
    assert new_mask[deep], "negative-residual (absorption-like) pixel should never be clipped"


def test_default_config_fully_protects_by_default():
    """FitConfig's default must be the protective (non-no-op) setting."""
    assert FitConfig().clean_protect_absorption_only is False


@pytest.mark.parametrize("sigl_kms,expected_min_hw", [(50.0, 6.0), (400.0, 11.5)])
def test_clean_ca_half_width_scales_with_velocity_pad(sigl_kms, expected_min_hw):
    """Effective Ca II protect half-width widens for broad LOSVDs via the velocity pad."""
    cfg = FitConfig(clean_ca_half_width=6.0, clean_protect_velocity_pad_kms=sigl_kms)
    hw = clean_ca_half_width_A(8662.14, cfg)
    assert hw >= expected_min_hw - 0.5
