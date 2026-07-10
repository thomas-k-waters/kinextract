"""Tests for kinextract.fitting's discrepancy-principle xlam selector.

These are unit tests of the search *algorithm* itself
(:func:`kinextract.fitting._discrepancy_principle_search` and
:func:`kinextract.fitting._v_recovery_is_sane`), using a synthetic,
analytically-known ``evaluate_xlam`` callable rather than a real spectral
fit -- fast and deterministic, and isolates the root-finding/fallback logic
from the (separately tested, much more expensive) end-to-end fitting
machinery.
"""
from __future__ import annotations

import numpy as np
import pytest

from kinextract.fitting import _discrepancy_principle_search, _v_recovery_is_sane

# ── _discrepancy_principle_search: root-finding convergence ─────────────────

def _make_monotone_evaluator(chi2_min: float = 100.0, ngood: int = 100,
                              slope: float = 5.0, xlam_ref: float = 1.0,
                              n_peaks: int = 1, v_rec: float = 0.0, sigma_rec: float = 100.0):
    """Build a synthetic evaluate_xlam whose chi2 rises linearly in
    log10(xlam) above `xlam_ref`: chi2(xlam) = chi2_min + slope *
    max(0, log10(xlam/xlam_ref)). Analytically invertible, so the expected
    converged xlam for a given target is known exactly.
    """
    def evaluate_xlam(xlam: float) -> dict:
        rise = slope * max(0.0, np.log10(xlam / xlam_ref))
        return dict(chi2_total=chi2_min + rise, ngood=ngood, n_peaks=n_peaks,
                    v_rec=v_rec, sigma_rec=sigma_rec)
    return evaluate_xlam


def test_discrepancy_search_converges_near_analytic_target():
    """For a known linear-in-log10(xlam) chi2 curve, the search should land
    within the bisection tolerance of the analytically-derived target xlam.
    """
    chi2_min, ngood, slope, xlam_ref = 100.0, 100, 5.0, 1.0
    nsigma = 1.0
    evaluate_xlam = _make_monotone_evaluator(chi2_min, ngood, slope, xlam_ref)

    target_rise = nsigma * float(np.sqrt(2.0 * ngood)) * (chi2_min / ngood)
    expected_xlam = xlam_ref * 10.0 ** (target_rise / slope)

    best_xlam, trials = _discrepancy_principle_search(
        evaluate_xlam, xlam_lo=1e-3, xlam_hi=1e2,
        max_peaks=1, v_center_est=0.0, nsigma=nsigma,
    )

    assert best_xlam == pytest.approx(expected_xlam, rel=0.2)
    assert len(trials) >= 2
    # The selected trial's chi2 must actually be >= the target (search always
    # errs toward "at least as much regularization as the target implies").
    chi2_min_trial = min(t["chi2_total"] for t in trials.values())
    target = chi2_min_trial * (1.0 + nsigma * float(np.sqrt(2.0 / ngood)))
    assert trials[best_xlam]["chi2_total"] >= target - 1e-6


def test_discrepancy_search_never_evaluates_same_xlam_twice():
    """The trials cache must dedupe identical xlam values across bracket
    expansion and bisection (each is an expensive full MAP refit in the
    real caller)."""
    calls = []

    def counting_evaluator(xlam):
        calls.append(xlam)
        return _make_monotone_evaluator()(xlam)

    _best_xlam, trials = _discrepancy_principle_search(
        counting_evaluator, xlam_lo=1e-3, xlam_hi=1e2,
        max_peaks=1, v_center_est=0.0,
    )
    assert len(calls) == len(set(calls)), "same xlam evaluated more than once"
    assert len(trials) == len(calls)


def test_discrepancy_search_expands_bracket_when_target_not_bracketed():
    """If the initial xlam_hi doesn't reach the target, the search must
    expand geometrically (not just give up) until it does."""
    # slope small enough that xlam_hi=10 alone doesn't reach a large target.
    evaluate_xlam = _make_monotone_evaluator(chi2_min=1000.0, ngood=1000, slope=1.0, xlam_ref=1.0)
    best_xlam, trials = _discrepancy_principle_search(
        evaluate_xlam, xlam_lo=1.0, xlam_hi=10.0,
        max_peaks=1, v_center_est=0.0, nsigma=1.0,
        max_bracket_expansions=6,
    )
    # sqrt(2*1000) ~ 44.7 chi2-unit rise needed; slope=1 per decade means
    # ~45 decades -- way beyond xlam_hi=10 without expansion.
    assert max(trials.keys()) > 10.0, "bracket was never expanded"
    assert best_xlam > 10.0


def test_discrepancy_search_gives_up_gracefully_if_target_unreachable():
    """A chi2(xlam) curve that's essentially flat (e.g. numerically clipped)
    should not loop forever -- max_bracket_expansions caps the search and
    the largest xlam tried is returned."""
    def flat_evaluator(xlam):
        return dict(chi2_total=100.0, ngood=100, n_peaks=1, v_rec=0.0, sigma_rec=100.0)

    best_xlam, trials = _discrepancy_principle_search(
        flat_evaluator, xlam_lo=1.0, xlam_hi=10.0,
        max_peaks=1, v_center_est=0.0, nsigma=1.0,
        max_bracket_expansions=3,
    )
    # 1 (lo) + 1 (initial hi) + 3 expansions = 5 evaluations, no bisection
    # possible since the target is never reached.
    assert len(trials) == 5
    assert best_xlam == max(trials.keys())


# ── Unimodality fallback ─────────────────────────────────────────────────────

def test_discrepancy_search_falls_back_when_target_region_is_multipeaked():
    """If the xlam that meets the discrepancy target is multi-peaked, the
    search must fall back to the smallest *unimodal* candidate that still
    meets the target, not silently return the multi-peaked one."""
    # Candidates are only ever evaluated at grid/bisection points; force a
    # small enough bracket that we know exactly which xlam values get tried.
    peaks_by_xlam = {}

    def evaluator(xlam):
        rise = 5.0 * max(0.0, np.log10(xlam))
        # Mark everything below 10 as multi-peaked (simulating an
        # under-regularized, noisy LOSVD), 10 and above as unimodal.
        n_peaks = 2 if xlam < 10.0 else 1
        peaks_by_xlam[xlam] = n_peaks
        return dict(chi2_total=100.0 + rise, ngood=100, n_peaks=n_peaks,
                    v_rec=0.0, sigma_rec=100.0)

    best_xlam, trials = _discrepancy_principle_search(
        evaluator, xlam_lo=1.0, xlam_hi=1e4,
        max_peaks=1, v_center_est=0.0, nsigma=0.5,
    )
    assert trials[best_xlam]["n_peaks"] <= 1, "returned a multi-peaked candidate"


# ── _v_recovery_is_sane ──────────────────────────────────────────────────────

def test_v_recovery_sane_accepts_consistent_candidate():
    trials = {
        1.0: dict(v_rec=10.0, sigma_rec=50.0, n_peaks=1),
        2.0: dict(v_rec=12.0, sigma_rec=50.0, n_peaks=1),
        3.0: dict(v_rec=11.0, sigma_rec=50.0, n_peaks=1),
    }
    assert _v_recovery_is_sane(2.0, trials, v_center_est=11.0, max_peaks=1)


def test_v_recovery_sane_rejects_outlier_candidate():
    """A candidate wildly discrepant from both the xcorr estimate and every
    other trial (the real failure mode this guards against -- a spurious
    local optimum at one xlam) must be rejected."""
    trials = {
        1.0: dict(v_rec=10.0, sigma_rec=50.0, n_peaks=1),
        2.0: dict(v_rec=11.0, sigma_rec=50.0, n_peaks=1),
        3.0: dict(v_rec=200.0, sigma_rec=50.0, n_peaks=1),  # spurious outlier
    }
    assert not _v_recovery_is_sane(3.0, trials, v_center_est=10.5, max_peaks=1)


def test_v_recovery_sane_ignores_multipeaked_trials_in_mad_comparison():
    """Other trials that are themselves multi-peaked shouldn't count toward
    the 'other candidates' consensus used for the MAD check."""
    trials = {
        1.0: dict(v_rec=10.0, sigma_rec=50.0, n_peaks=1),
        2.0: dict(v_rec=11.0, sigma_rec=50.0, n_peaks=1),
        3.0: dict(v_rec=500.0, sigma_rec=50.0, n_peaks=3),  # multi-peaked, excluded
        4.0: dict(v_rec=10.5, sigma_rec=50.0, n_peaks=1),
    }
    assert _v_recovery_is_sane(4.0, trials, v_center_est=10.5, max_peaks=1)
