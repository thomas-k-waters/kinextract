"""Tests for the wing-taper machinery ported from the original
`kinextract` pipeline: `tapered_lam_vector`, `core_width_sigma`, and the
Greville-abscissae bookkeeping that lets a per-row penalty strength be
placed on the velocity axis.

Scope note, deliberately explicit: these test that the ported mechanism
is *mechanically correct*. They do NOT assert that turning it on fixes
the real-data sigma inflation it was ported for -- it does not, at the
regularization strengths selection currently picks, which is why
`FitConfig.lam_wing` defaults to 0.0. See `dev_notes/PORT_STATUS.md`.
"""
from __future__ import annotations

import numpy as np

from kinextract import (FitConfig, tapered_lam_vector, penalty_row_velocities,
                             gauss_hermite_model)
from kinextract.losvd import greville_abscissae, core_width_sigma


def test_greville_abscissae_span_the_grid_and_are_monotonic():
    v = FitConfig(vmin=-400.0, vmax=400.0).build_velocity_grid()
    g = greville_abscissae(v, 16)
    assert g[0] == -400.0 and g[-1] == 400.0
    assert np.all(np.diff(g) > 0)


def test_penalty_row_velocities_have_one_entry_per_penalty_row():
    from kinextract import difference_matrix
    v = FitConfig(vmin=-400.0, vmax=400.0).build_velocity_grid()
    n_coef = len(greville_abscissae(v, 16))
    for order in (2, 3):
        D = difference_matrix(n_coef, order=order)
        assert len(penalty_row_velocities(v, 16, order)) == D.shape[0]


def test_taper_is_flat_in_the_core_and_ramps_in_the_wings():
    row_v = np.array([0.0, 50.0, 100.0, 200.0, 400.0])
    lam, sigl0, sfac = 1.0, 50.0, 1.8
    out = tapered_lam_vector(row_v, lam, sigl0, 0.0, sfac=sfac, power=4.0, lam_wing=1.0)
    # inside sfac*sigl0 = 90 km/s the strength is exactly lam
    assert out[0] == lam and out[1] == lam
    # beyond, it ramps monotonically and matches the 4th-power law
    assert out[2] > lam and out[3] > out[2] and out[4] > out[3]
    expected_400 = lam * ((400.0 / sigl0) / sfac) ** 4
    assert np.isclose(out[4], expected_400)


def test_wing_strength_is_decoupled_from_the_core_strength():
    """The whole point of `lam_wing`: a small core `lam` must not drag the
    wing ramp down with it (the failure that made the coupled form inert).
    """
    row_v = np.array([0.0, 400.0])
    coupled = tapered_lam_vector(row_v, 1.0, 50.0, 0.0)                    # lam_wing=None -> lam
    decoupled = tapered_lam_vector(row_v, 1.0, 50.0, 0.0, lam_wing=512.0)
    assert coupled[0] == decoupled[0] == 1.0          # core untouched either way
    assert decoupled[1] > 100 * coupled[1]            # wings vastly stronger


def test_taper_disabled_reproduces_a_uniform_penalty():
    row_v = np.linspace(-400.0, 400.0, 9)
    out = tapered_lam_vector(row_v, 3.0, 50.0, 0.0, lam_wing=0.0)
    np.testing.assert_allclose(out, 3.0)


def test_core_width_sigma_ignores_a_wing_pedestal():
    """The reason `core_width_sigma` exists: a second-moment sigma is
    inflated by a pedestal, so seeding the taper with it makes the fixed
    point reinforce the defect. A half-maximum width must not move.
    """
    v = np.linspace(-400.0, 400.0, 801)
    core = np.exp(-0.5 * (v / 50.0) ** 2)
    with_pedestal = core + 0.05

    clean = core_width_sigma(v, core)
    contaminated = core_width_sigma(v, with_pedestal)
    assert abs(clean - 50.0) < 1.0
    assert abs(contaminated - clean) < 3.0

    # ...whereas the second moment is badly inflated by the same pedestal
    def second_moment(b):
        w = np.maximum(b, 0.0)
        mu = np.sum(v * w) / np.sum(w)
        return np.sqrt(np.sum(w * (v - mu) ** 2) / np.sum(w))
    assert second_moment(with_pedestal) > 2 * second_moment(core)
