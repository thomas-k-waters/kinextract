"""Tests for kinextract.config's once-per-process warning dedup.

The "knot spacing much finer than knot_spacing_kms" warning embeds
per-call values (vmin/vmax/dv/effective_lam) in its message, which
defeats Python's own warnings-module "once" action (keyed on exact
message text) -- a real notebook, multi-bin run showed this warning
printing far more than once despite that filter being registered. These
tests lock in the plain once-per-process flag that replaced it. See
test_losvd.py's test_narrow_grid_with_floored_knots_warns_about_effective_lam
for the underlying warning-condition tests themselves (the threshold,
the NSIG=7 convention staying clean, etc.) -- these
tests only cover the dedup behavior on top of that condition."""
from __future__ import annotations

import warnings

import pytest

from kinextract.config import FitConfig, _reset_knot_spacing_warning_dedup

# A grid narrow enough that min_interior_knots (not knot_spacing_kms) sets the
# knot count, resolving to a knot spacing far finer than knot_spacing_kms's
# default -- the same construction test_losvd.py's own test uses.
_NARROW = dict(vmin=80 - 2 * 40, vmax=80 + 2 * 40)


def test_knot_spacing_warning_fires_on_first_narrow_grid_config():
    _reset_knot_spacing_warning_dedup()
    with pytest.warns(UserWarning, match="much finer than"):
        FitConfig(lam=1e6, lam_rule="fixed", **_NARROW, continuum_knots=None)


def test_knot_spacing_warning_does_not_fire_again_for_a_different_narrow_config():
    """The real bug: two different narrow configs produce two different
    message texts (different lam/effective_lam), which used to defeat
    "once" entirely. A second, textually-distinct trigger must still be
    suppressed."""
    _reset_knot_spacing_warning_dedup()
    with pytest.warns(UserWarning, match="much finer than"):
        FitConfig(lam=1e6, lam_rule="fixed", **_NARROW, continuum_knots=None)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        FitConfig(lam=2e6, lam_rule="fixed", **_NARROW, continuum_knots=None)  # different values, same condition


def test_knot_spacing_warning_dedup_reset_helper_re_arms_it():
    _reset_knot_spacing_warning_dedup()
    with pytest.warns(UserWarning, match="much finer than"):
        FitConfig(lam=1e6, lam_rule="fixed", **_NARROW, continuum_knots=None)

    _reset_knot_spacing_warning_dedup()
    with pytest.warns(UserWarning, match="much finer than"):
        FitConfig(lam=1e6, lam_rule="fixed", **_NARROW, continuum_knots=None)


def test_sigma_scaled_lam_only_raises_lam_and_is_capped():
    import numpy as np
    from kinextract import sigma_scaled_lam
    assert sigma_scaled_lam(230.0) == 4096.0
    assert sigma_scaled_lam(350.0) == 4096.0
    assert np.isclose(sigma_scaled_lam(450.0), 4096.0 * (450 / 350) ** 6)
    assert sigma_scaled_lam(900.0) == 131072.0
    assert sigma_scaled_lam(float("nan")) == 4096.0
