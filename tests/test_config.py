"""Tests for kinextract.config's once-per-process warning dedup.

The "knot spacing much finer than knot_spacing_kms" warning embeds
per-call values (vmin/vmax/dv/effective_lam) in its message, which
defeats Python's own warnings-module "once" action (keyed on exact
message text) -- a real notebook, multi-bin run showed this warning
printing far more than once despite that filter being registered. These
tests lock in the plain once-per-process flag that replaced it. See
test_losvd.py's test_narrow_grid_with_floored_knots_warns_about_effective_lam
for the underlying warning-condition tests themselves (order-2 vs.
order-3 thresholds, the NSIG=7 convention staying clean, etc.) -- these
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
        FitConfig(penalty_order=2, lam=10.0, lam_selection="fixed", **_NARROW)


def test_knot_spacing_warning_does_not_fire_again_for_a_different_narrow_config():
    """The real bug: two different narrow configs produce two different
    message texts (different lam/effective_lam), which used to defeat
    "once" entirely. A second, textually-distinct trigger must still be
    suppressed."""
    _reset_knot_spacing_warning_dedup()
    with pytest.warns(UserWarning, match="much finer than"):
        FitConfig(penalty_order=2, lam=10.0, lam_selection="fixed", **_NARROW)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        FitConfig(penalty_order=3, lam=1e6, lam_selection="fixed", **_NARROW)  # different values, same condition


def test_knot_spacing_warning_dedup_reset_helper_re_arms_it():
    _reset_knot_spacing_warning_dedup()
    with pytest.warns(UserWarning, match="much finer than"):
        FitConfig(penalty_order=2, lam=10.0, lam_selection="fixed", **_NARROW)

    _reset_knot_spacing_warning_dedup()
    with pytest.warns(UserWarning, match="much finer than"):
        FitConfig(penalty_order=2, lam=10.0, lam_selection="fixed", **_NARROW)
