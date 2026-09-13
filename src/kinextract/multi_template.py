"""Joint LOSVD + template-mixture-weight fitting via alternating
minimization: fix the mixture weights `w` and fit the LOSVD (an exact
call into `kinextract.losvd.fit_log_spline_map` against the
effective design matrix `A_eff(w) = sum_k w_k * A_stack[k]`), then fix the
LOSVD and solve for `w` via NNLS (convex, exact). Repeat until the joint
objective stops improving.

Works for any `n_templates >= 1` -- a single template is simply the
`n_templates=1` case of this same problem (NNLS trivially returns
`w=[c]` for some `c>0`).

**Weight/amplitude scale is not independently identifiable.** The model
is invariant under `(w, b) -> (c*w, b/c)` for any `c > 0`, so the
*absolute* scale `fit_losvd` lands on is arbitrary. `compute_losvd_moments`
only depends on `b`'s shape, not its scale, and template mixture
fractions should always be read off as `w / w.sum()`, not `w` itself.

**No multi-start over the initial weight vector by default.** Alternating
minimization on this biconvex problem could in principle land in a
coordinate-wise local optimum; `w0_pool` lets you pass multiple starting
weight vectors if a specific template library needs it, but the default
single equal-weights start is sufficient for well- and poorly-conditioned
libraries alike. A poorly-conditioned (near-collinear) template library
does produce genuinely unstable weight recovery across different noise
realizations of the *same* underlying data -- an identifiability property
of the data and template library, visible directly in the per-step NNLS
design matrix's condition number, not an optimizer failure.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import nnls

from .config import FitConfig
from .losvd import (
    build_bspline_basis,
    default_initial_coeffs,
    difference_matrix,
    fit_log_spline_map,
)


def _alternate_once(A_stack: np.ndarray, y_w: np.ndarray, basis: np.ndarray, D2: np.ndarray,
                     lam: float, ridge_lam: float, w0: np.ndarray, max_iter: int, tol: float,
                     patience: int = 2, losvd_single_start: bool = False,
                     continuum_basis: np.ndarray | None = None, D2_c: np.ndarray | None = None,
                     lam_c: float = 2.0, ridge_c: float = 0.1,
                     losvd_warm_start: bool = True,
                     initial_coeffs: np.ndarray | None = None) -> dict:
    """One alternating-minimization trajectory from a single starting
    weight vector `w0`, to convergence or `max_iter` iterations. Two- or
    three-block, controlled by `continuum_basis`.

    Tracks the best-seen solution across all iterations (not just the
    last), stopping after `patience` iterations with no improvement --
    each half-step's own L-BFGS-B sub-solve has its own floating-point
    noise, so the joint objective typically oscillates in a tight band
    rather than settling to a bit-identical fixed point.

    `losvd_single_start` (default False): if True, only the FIRST
    iteration's LOSVD sub-fit uses a single, deterministic starting point
    (`default_initial_coeffs`, the data-driven ridge estimate) instead of
    `fit_log_spline_map`'s own default multi-start pool -- later iterations
    still warm-start from the previous iteration's own converged
    coefficients exactly as they do when this is False (see
    `FitConfig.losvd_single_start`'s own docstring for a 2026-07-30 bug
    this behavior fixes: forcing the deterministic start on every
    iteration, not just the first, disabled warm-starting entirely and
    made this ablation flag slower than the thing it was meant to be a
    faster alternative to). Exists for isolating the multi-start search's
    own marginal effect on recovered bias -- not recommended for
    real fits, since `fit_log_spline_map`'s own docstring documents a real,
    otherwise-invisible failure mode (~1-in-150 fits landing in a
    meaningfully-worse local minimum) that multi-start exists to catch.

    `continuum_basis`/`D2_c`/`lam_c`/`ridge_c` (default `continuum_basis=None`,
    off -- reproduces the plain two-block `(b, w)` alternation exactly):
    when `continuum_basis` is given, this becomes the three-block
    `(b, w, C)` alternation from `kinextract.continuum` -- see that
    module's own docstring for the algorithm, and `fit_continuum_block`'s
    own docstring specifically for why `ridge_c` exists. `fit_losvd`
    reaches this same path automatically whenever `cfg.continuum_knots`
    is set -- there is no separate continuum-cofitting entry point.

    `initial_coeffs` (default `None`): seeds the LOSVD warm-start slot
    before iteration 0, letting a caller who already has a nearby-lam or
    nearby-fit solution skip even the *first* iteration's multi-start
    search -- see `select_lambda_aic_local`'s `warm_start_lam_grid` option.
    Ignored if `losvd_warm_start=False`.
    """
    n_templates = A_stack.shape[0]
    n_pix = A_stack.shape[1]
    w = np.asarray(w0, dtype=float).copy()
    has_continuum = continuum_basis is not None
    C = np.ones(n_pix) if has_continuum else None
    continuum_coeffs = np.zeros(continuum_basis.shape[1]) if has_continuum else None

    history = []
    best = None
    best_objective = np.inf
    stale = 0
    # Warm-start state for the LOSVD sub-fit across alternating iterations.
    # The multi-start basin search only has to happen ONCE, on the first
    # iteration: it exists to pick the right basin (see fit_log_spline_map's
    # docstring), and `w` moves only slightly per iteration afterwards, so
    # re-running a cold 3-start search every iteration re-solves an already-
    # solved problem. Profiled directly on the standard 3-template mock: a
    # default fit ran 114 L-BFGS-B optimizations, of which ~2/3 were these
    # redundant re-searches. The first iteration's own multi-start gap is
    # kept as the reported diagnostic, since that is the iteration where the
    # basin was actually chosen.
    #
    # `initial_coeffs`, if given, seeds this same warm-start slot before the
    # loop even starts -- letting a caller skip iteration 0's own multi-start
    # too, e.g. `select_lambda_aic_local` warm-starting each lam grid point
    # from the previous (neighbouring) lam's converged solution. `None`
    # (the default) reproduces the original cold-start-on-iteration-0
    # behavior exactly.
    warm_coeffs = initial_coeffs
    first_objective_gap = None

    for iteration in range(max_iter):
        # Block 1: fix (w, [C]), solve b
        A_mix = np.tensordot(w, A_stack, axes=([0], [0]))
        A_eff = C[:, None] * A_mix if has_continuum else A_mix
        if losvd_single_start and iteration == 0:
            # Ablates ONLY the iteration-0 multi-start *search* (this flag's whole
            # documented purpose -- "isolating the multi-start search's own marginal
            # effect"), not the separate iteration-to-iteration warm-start mechanism
            # below. Previously this branch fired on every iteration, forcing a fresh
            # deterministic ridge start (and therefore a from-scratch L-BFGS-B
            # optimization) each time instead of ever warm-starting from the
            # previous, already-converged iteration -- found directly (2026-07-30):
            # this made `losvd_single_start=True` fits ~3x *slower* than full
            # multi-start on a real-templates/order-3 case (14-15s vs 4.5-6s),
            # the opposite of its own docstring's "~2.5x faster" claim, which was
            # evidently measured before this interaction mattered as much as it
            # does under the current order-3/adaptive-lam-grid default. Iterations
            # after the first now warm-start exactly like the non-single-start
            # path (see the `elif` below), matching what `losvd_single_start`'s
            # docstring already claimed it did.
            coeffs0 = default_initial_coeffs(A_eff, y_w, basis, ridge_lam=ridge_lam)
        elif losvd_warm_start:
            coeffs0 = warm_coeffs  # None (full multi-start) unless `initial_coeffs` seeded it
        else:
            coeffs0 = None  # legacy: a cold multi-start search on every iteration
        result_b = fit_log_spline_map(A_eff, y_w, basis, D2, lam=lam, ridge_lam=ridge_lam,
                                       coeffs0=coeffs0)
        b = result_b["b"]
        coeffs = result_b["coeffs"]
        warm_coeffs = coeffs
        if first_objective_gap is None:
            first_objective_gap = result_b.get("objective_gap")

        # Block 2: fix (b, [C]), solve w
        if has_continuum:
            design_w = np.stack([C * (A_stack[k] @ b) for k in range(n_templates)], axis=1)
        else:
            design_w = np.stack([A_stack[k] @ b for k in range(n_templates)], axis=1)
        w, _ = nnls(design_w, y_w)
        if not np.any(w > 0):
            # degenerate: NNLS drove every weight to exactly zero (a bad
            # start or pathological data) -- stop rather than divide by a
            # zero effective design matrix on the next iteration. If this
            # happens on the very first iteration, `best` is still None,
            # so fall back to this iteration's own (w0-based) LOSVD
            # sub-fit, explicitly flagged as failed -- a single
            # pathological noise draw among many `bootstrap_losvd`
            # replicates then produces one flagged-bad replicate instead
            # of aborting the whole bootstrap call.
            if best is None:
                best = dict(b=b, w=w0.copy(), coeffs=coeffs, objective=np.inf,
                            success=False,
                            message="degenerate NNLS: all template weights driven to zero",
                            losvd_success=bool(result_b["success"]),
                            continuum_success=None,
                            losvd_objective_gap=first_objective_gap,
                            C=C.copy() if has_continuum else None,
                            continuum_coeffs=continuum_coeffs.copy() if has_continuum else None)
            break

        # Block 3 (continuum only): fix (b, w), solve C
        continuum_success = True
        if has_continuum:
            # Local import: continuum.py imports fit_losvd_basis (below) at
            # module level, so a top-level import here would be circular --
            # deferred until actually needed, same pattern continuum.py
            # itself already uses for its own gh_moments import.
            from .continuum import fit_continuum_block
            M = np.tensordot(w, A_stack, axes=([0], [0])) @ b
            result_c = fit_continuum_block(M, y_w, continuum_basis, D2_c, lam_c=lam_c, ridge_c=ridge_c,
                                            coeffs0=continuum_coeffs)
            C = result_c["C"]
            continuum_coeffs = result_c["coeffs"]
            continuum_success = bool(result_c["success"])

        A_mix = np.tensordot(w, A_stack, axes=([0], [0]))
        A_final = C[:, None] * A_mix if has_continuum else A_mix
        resid = A_final @ b - y_w
        chi2 = 0.5 * np.sum(resid ** 2)
        # `lam` may be a per-row vector (the wing taper) -- weight row-wise.
        rough = 0.5 * float(np.sum(np.asarray(lam) * (D2 @ coeffs) ** 2))
        if has_continuum:
            rough += 0.5 * lam_c * np.sum((D2_c @ continuum_coeffs) ** 2)
        objective = chi2 + rough
        history.append(objective)

        if best is None or objective < best_objective - tol * max(abs(best_objective), 1.0):
            best_objective = objective
            losvd_success = bool(result_b["success"])
            # Attribute the message to whichever block(s) actually failed --
            # see `continuum.fit_continuum_block`'s docstring for why
            # continuum_success=False is common and not itself a sign of a
            # bad fit.
            if has_continuum and not continuum_success and not losvd_success:
                message = f"losvd block: {result_b['message']}; continuum block: {result_c['message']}"
            elif has_continuum and not continuum_success:
                message = f"continuum block: {result_c['message']}"
            elif not losvd_success:
                message = f"losvd block: {result_b['message']}"
            else:
                message = str(result_b["message"])
            best = dict(b=b, w=w, coeffs=coeffs, objective=objective,
                        success=losvd_success and continuum_success,
                        message=message, losvd_success=losvd_success,
                        continuum_success=continuum_success if has_continuum else None,
                        losvd_objective_gap=first_objective_gap,
                        C=C.copy() if has_continuum else None,
                        continuum_coeffs=continuum_coeffs.copy() if has_continuum else None)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    result = dict(
        b=best["b"], w=best["w"], coeffs=best["coeffs"], n_iter=iteration + 1,
        objective=best["objective"], objective_history=np.array(history),
        success=best["success"], message=best["message"],
        losvd_success=best["losvd_success"], continuum_success=best["continuum_success"],
        losvd_objective_gap=best["losvd_objective_gap"],
    )
    if has_continuum:
        result["C"] = best["C"]
        result["continuum_coeffs"] = best["continuum_coeffs"]
    return result


def fit_losvd_basis(A_stack: np.ndarray, y_w: np.ndarray, basis: np.ndarray, D2: np.ndarray,
                     lam: float = 2.0, ridge_lam: float = 1.0,
                     w0_pool: list[np.ndarray] | None = None,
                     max_iter: int = 20, tol: float = 1e-4, patience: int = 2,
                     losvd_single_start: bool = False,
                     continuum_basis: np.ndarray | None = None, D2_c: np.ndarray | None = None,
                     lam_c: float = 2.0, ridge_c: float = 0.1,
                     losvd_warm_start: bool = True,
                     initial_coeffs: np.ndarray | None = None) -> dict:
    """Jointly recover the LOSVD and template mixture weights, given an
    already-built spline basis and roughness-penalty operator. Prefer
    `fit_losvd` (builds these from a `FitConfig`) unless you need direct
    control over the basis/penalty construction.

    Parameters
    ----------
    A_stack : ndarray, shape (n_templates, n_pix, n_bins)
        Noise-whitened per-template design matrices, from
        `kinextract.forward_model.build_template_stack`.
    y_w : ndarray, shape (n_pix,)
        Noise-whitened observed flux.
    basis, D2, lam, ridge_lam : as in `kinextract.losvd.fit_log_spline_map`.
    w0_pool : optional explicit pool of starting weight vectors (each
        shape `(n_templates,)`) for a multi-start search over the whole
        alternating procedure. Defaults to a single equal-weights start.
    max_iter : maximum alternating iterations per starting point.
    tol : relative convergence tolerance on the joint objective between
        successive iterations.
    losvd_single_start : passed through to `_alternate_once` -- see its
        own docstring. Default False (multi-start, the safe/recommended
        setting).
    continuum_basis, D2_c, lam_c, ridge_c : optional continuum block (default
        `continuum_basis=None`, off -- reproduces the plain two-block
        fit exactly). When given, adds a third alternating block that
        jointly recovers a shared multiplicative continuum -- see
        `kinextract.continuum` for the algorithm; prefer calling
        `fit_losvd` with `cfg.continuum_knots` set, which builds
        `continuum_basis`/`D2_c` from `wave_fit` automatically, rather
        than building them by hand for this lower-level function.
        `ridge_c` (default 0.1) is a direct L2 penalty on the
        continuum coefficients -- see `continuum.fit_continuum_block`'s
        own docstring for why it exists.
    initial_coeffs : optional single starting guess in LOSVD coefficient
        space, applied to every trajectory in `w0_pool` -- see
        `_alternate_once`'s own docstring. Lets a caller who already has a
        nearby solution (e.g. a neighbouring `lam` value's converged fit)
        skip iteration 0's multi-start search too, not just later
        iterations'.

    Returns
    -------
    dict with `b` (recovered LOSVD -- see this module's docstring on the
    `(w, b)` scale non-identifiability), `w` (recovered template weights;
    normalize via `w / w.sum()` for mixture fractions), `coeffs`,
    `success` (`losvd_success and continuum_success` -- see
    `losvd_success`/`continuum_success` below for which block it actually
    reflects), `message` (attributed to whichever block(s) actually
    failed -- see `continuum.fit_continuum_block`'s own docstring for why
    `continuum_success=False` there is common and does not by itself
    indicate a bad fit), `losvd_success`, `continuum_success` (`None` if
    continuum-cofitting is off), `n_iter` (from the winning trajectory),
    `n_coef`, `n_templates`, `lam`, `objective`, `objective_gap` (gap
    between the best and second-best starting *weight vector's* final
    objective -- `None` unless `w0_pool` has more than one entry),
    `losvd_objective_gap` (the winning trajectory's own LOSVD sub-fit
    multi-start gap, from `fit_log_spline_map`'s own `objective_gap` --
    a different, independent diagnostic that applies regardless of
    `w0_pool` size, since `fit_log_spline_map` always multi-starts
    internally), `objective_history` (the winning trajectory's objective
    per iteration).
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1 (got {max_iter}) -- at least one alternating "
                          "iteration is required to produce a result.")

    n_templates, n_pix, n_bins = A_stack.shape
    n_coef = basis.shape[1]

    if w0_pool is None:
        w0_pool = [np.ones(n_templates)]

    trajectories = [
        _alternate_once(A_stack, y_w, basis, D2, lam, ridge_lam, w0, max_iter, tol,
                         patience=patience,
                         losvd_single_start=losvd_single_start,
                         continuum_basis=continuum_basis, D2_c=D2_c, lam_c=lam_c, ridge_c=ridge_c,
                         losvd_warm_start=losvd_warm_start, initial_coeffs=initial_coeffs)
        for w0 in w0_pool
    ]
    objectives = [t["objective"] for t in trajectories]
    best_idx = int(np.argmin(objectives))
    best = trajectories[best_idx]
    objective_gap = None
    if len(trajectories) > 1:
        sorted_obj = sorted(objectives)
        objective_gap = sorted_obj[1] - sorted_obj[0]

    result = dict(
        b=best["b"], w=best["w"], coeffs=best["coeffs"],
        success=best["success"], message=best["message"], n_iter=best["n_iter"],
        losvd_success=best["losvd_success"], continuum_success=best["continuum_success"],
        n_coef=n_coef, n_templates=n_templates, lam=lam,
        objective=best["objective"], objective_gap=objective_gap,
        losvd_objective_gap=best["losvd_objective_gap"],
        objective_history=best["objective_history"],
    )
    if continuum_basis is not None:
        result["C"] = best["C"]
        result["continuum_coeffs"] = best["continuum_coeffs"]
        result["n_coef_continuum"] = continuum_basis.shape[1]
        result["lam_c"] = lam_c
        result["ridge_c"] = ridge_c
    return result


def fit_losvd(A_stack: np.ndarray, y_w: np.ndarray, cfg: FitConfig,
              wave_fit: np.ndarray | None = None,
              initial_coeffs: np.ndarray | None = None) -> dict:
    """Run a LOSVD + template-mixture-weight fit end-to-end from a
    `FitConfig`: builds the spline basis and velocity-spacing-scaled
    roughness-penalty operator from `cfg`, then calls `fit_losvd_basis`.

    `A_stack`/`y_w` must already correspond to `cfg.build_velocity_grid()`
    -- build `A_stack` on that same grid via
    `kinextract.forward_model.build_template_stack` before calling
    this. Works for any `n_templates >= 1`, including a single template
    (pass a length-1 list to `build_template_stack`).

    `wave_fit` : the observed wavelength grid `A_stack`/`y_w` correspond
        to. Only required if `cfg.continuum_knots is not None` (raises
        `ValueError` if missing in that case) -- turns on continuum-
        cofitting (see `kinextract.continuum`), building the
        continuum's own wavelength-based B-spline basis/penalty from
        `cfg.continuum_knots`/`cfg.lam_c`. `None` (the default) reproduces
        this package's original two-block-only behavior exactly.

    `cfg.lam_selection` (default `'aic_local'`): if set, `lam` is chosen
    per fit via `aic_regularization.select_lambda_aic_local` before doing
    anything else below, then treated exactly like an explicitly-passed
    fixed `lam` for the rest of this call -- see `FitConfig.lam_selection`'s
    own help text for why this is the default and what it costs. Supports
    continuum-cofitting (`cfg.continuum_knots` set) -- `estimate_m_eff_local`'s
    own docstring has the joint LOSVD+continuum linearization this relies
    on. The `lam` actually used is always reported back as `result['lam_used']`.

    `initial_coeffs` : optional single starting guess in LOSVD coefficient
    space, passed straight through to the fixed-`lam` fit at the end of
    this call -- see `fit_losvd_basis`'s own docstring. Ignored when
    `cfg.lam_selection == 'aic_local'` and no wing taper is active (that
    path's own lam-grid scan, `select_lambda_aic_local`, has its own
    independent warm-start chaining across the grid -- see its
    `warm_start_lam_grid` option -- so a single caller-supplied seed
    would only ever help the first, cheapest grid point).
    """
    if cfg.continuum_knots is not None and wave_fit is None:
        raise ValueError(
            "fit_losvd: cfg.continuum_knots is set but wave_fit was not provided -- "
            "continuum-cofitting needs the observed wavelength grid to build its own "
            "wavelength-based B-spline basis (continuum acts on wavelength, not velocity)."
        )

    lam_selection_result = None
    if cfg.lam_selection == "aic_local":
        # Local import: aic_regularization.py imports fit_losvd from this module at
        # module level, so a top-level import here would be circular -- same pattern
        # continuum.py/_alternate_once already use for fit_continuum_block.
        from .aic_regularization import select_lambda_aic_local
        # parsimony_delta=2.0 and the sub-knot-sigma candidate guard for
        # the order-3 penalty only -- see select_lambda_aic_local's own
        # docstring for the evidence and for why both stay OFF for order 2.
        is_d3 = cfg.penalty_order == 3
        lam_selection_result = select_lambda_aic_local(
            A_stack, y_w, cfg, np.asarray(cfg.resolved_lam_grid()),
            parsimony_delta=2.0 if is_d3 else 0.0,
            min_resolvable_sigma_frac=0.5 if is_d3 else 0.0,
            max_edge_amplitude_frac=cfg.max_edge_amplitude_frac,
            warm_start_lam_grid=True,
            self_normalize_lam_grid=cfg.self_normalize_lam_grid,
            wave_fit=wave_fit)
        cfg = dataclasses.replace(cfg, lam=lam_selection_result.best_lam, lam_selection="fixed")

    # The selection scan already fit at every candidate lam, so the fit at the
    # winning lam is reusable rather than repeated -- including under continuum-
    # cofitting now that estimate_m_eff_local's own joint linearization supports it.
    reusable_selection_fit = None
    if lam_selection_result is not None and lam_selection_result.best_fit is not None:
        reusable_selection_fit = lam_selection_result.best_fit

    v_grid = cfg.build_velocity_grid()
    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D = difference_matrix(n_coef, order=cfg.penalty_order, dv=dv, dv_ref=cfg.knot_spacing_kms)

    continuum_basis = D2_c = None
    if cfg.continuum_knots is not None:
        continuum_basis, n_coef_c = build_bspline_basis(wave_fit, n_interior_knots=cfg.continuum_knots, degree=3)
        dw = (wave_fit.max() - wave_fit.min()) / (cfg.continuum_knots + 1)
        D2_c = difference_matrix(n_coef_c, order=2, dv=dw, dv_ref=dw)

    if reusable_selection_fit is not None:
        result = reusable_selection_fit
    else:
        result = fit_losvd_basis(A_stack, y_w, basis, D, lam=cfg.lam, ridge_lam=cfg.ridge_lam,
                                  losvd_single_start=cfg.losvd_single_start,
                                  continuum_basis=continuum_basis, D2_c=D2_c, lam_c=cfg.lam_c,
                                  ridge_c=cfg.ridge_c, initial_coeffs=initial_coeffs)

    result["lam_used"] = cfg.lam
    result["lam_selection_result"] = lam_selection_result
    return result


def _fit_losvd_worker(args):
    """Top-level (picklable) per-bin worker for `fit_losvd_parallel` --
    process pools need a module-level function, not a closure, since
    workers unpickle it by import path."""
    A_stack, y_w, cfg, wave_fit = args
    return fit_losvd(A_stack, y_w, cfg, wave_fit=wave_fit)


def _available_cpu_count() -> int:
    """The number of CPUs this *process* can actually use -- not just the
    number physically present on the machine it happens to be running on.

    `os.cpu_count()` reports the whole node's core count regardless of
    how many this job was actually allocated. On a shared SLURM node
    (e.g. Great Lakes), that is a real, well-known multiprocessing
    pitfall: a job given `--cpus-per-task=8` on a 128-core node sees
    `os.cpu_count() == 128`, so `resolve_n_jobs`'s old `n_jobs=None`
    default (before this existed) would spawn up to 128 worker
    processes -- 16x this job's actual allocation, degrading this job's
    own throughput (thread/process thrashing across only 8 real cores)
    and starving every other job sharing that node. This is very likely
    what "the parallelization needs to be robust... especially when I
    start running on Greatlakes" was pointing at.

    Cross-checks every signal available and takes the tightest (safest,
    smallest) one, so a wrong or missing signal on any one platform
    never causes overcommit -- only ever causes *less* parallelism than
    the true allocation, which is merely slower, not disruptive to other
    jobs on a shared node:

    - `os.sched_getaffinity(0)` (Linux only): the actual cpuset/cgroup-
      restricted core count for this process -- ground truth when
      available, since it reflects whatever confinement mechanism was
      used (SLURM's cgroup plugin or otherwise), not SLURM specifically.
    - `SLURM_CPUS_PER_TASK` / `SLURM_JOB_CPUS_PER_NODE`: SLURM's own
      account of this job's allocation -- checked even when
      `sched_getaffinity` is available (a cross-check, since some SLURM
      configurations don't actually confine via cgroups), and the
      primary signal on platforms without `sched_getaffinity` (macOS,
      this package's own dev machine). `SLURM_JOB_CPUS_PER_NODE` can be
      like `"8"` or `"8(x2)"` for a multi-node job requesting the same
      per-node count twice -- only the leading integer is used, since
      that is this *task's* own per-node share regardless of node count.
    - `os.cpu_count()`: the whole-machine fallback, used only when
      neither of the above is available (a plain laptop/workstation,
      where it is already the right answer).
    """
    import os
    candidates = []

    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except AttributeError:
        pass  # not available on macOS/Windows

    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        val = os.environ.get(var)
        if val:
            try:
                candidates.append(int(val.split("(")[0]))
            except ValueError:
                pass

    candidates.append(os.cpu_count() or 1)
    return max(1, min(candidates))


def resolve_n_jobs(n_jobs: int | None, per_job_bytes: float, cap_fraction: float = 0.5) -> int:
    """Pick a worker-process count for the `*_parallel` functions that
    won't overcommit *this* machine's memory, while still using
    everything available on a bigger one -- e.g. this package's own
    16GB-RAM dev laptop vs. a large-memory HPC node (Great Lakes).

    `n_jobs` given explicitly (not `None`) always wins, unchanged -- the
    escape hatch for a user who knows their own machine's headroom and
    wants to force more (or less) parallelism than the automatic
    estimate below would pick. This is the only case that matters on a
    large-memory cluster: pass `n_jobs=<your core count>` and this
    function is a no-op.

    `n_jobs=None` (every `*_parallel` function's own default) starts from
    `_available_cpu_count()` -- this job's own actual CPU allocation, not
    necessarily the whole machine's (see that function's own docstring
    for why this matters on a shared SLURM node) -- then caps it so
    `n_jobs * per_job_bytes` doesn't exceed `cap_fraction` (default half)
    of *available* system memory -- leaving headroom for whatever the
    caller already holds (e.g. a per-bin design-matrix list built before
    the parallel call, or a real MUSE cube still in memory). Available
    memory is estimated via `psutil` if installed (accurate -- accounts
    for other running processes/apps), falling back to *total* physical
    memory via `os.sysconf` on POSIX systems (conservative but portable,
    no new required dependency) if `psutil` isn't present, or, failing
    both (e.g. Windows without `psutil`), just `_available_cpu_count()`
    unchanged -- today's exact behavior, never worse than before this
    existed.

    This scales in both directions with no code change needed: a laptop
    with only a few GB free gets a small, safe worker count automatically;
    an HPC node with hundreds of GB gets this job's own full CPU
    allocation (the memory cap essentially never binds there) -- see this
    package's own `dev_notes/` for the real crash this was built to
    prevent (a 16GB machine, a multi-GB real cube, and
    `os.cpu_count()`-many workers each needing their own per-bin working
    memory, all at once).
    """
    import os
    cpu_default = _available_cpu_count()
    if n_jobs is not None:
        return n_jobs
    if per_job_bytes <= 0:
        return cpu_default

    available_bytes = None
    try:
        import psutil
        available_bytes = psutil.virtual_memory().available
    except ImportError:
        try:
            available_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            available_bytes = None

    if available_bytes is None:
        return cpu_default

    memory_capped = max(1, int((available_bytes * cap_fraction) // per_job_bytes))
    return max(1, min(cpu_default, memory_capped))


def _limit_worker_threads():
    """`ProcessPoolExecutor` initializer: pin BLAS/OpenMP thread pools to 1
    per worker, and default JAX to the CPU backend, before `kinextract`
    (and therefore numpy/JAX) gets imported in this fresh process.

    Without the BLAS/OpenMP pinning, each of `n_jobs` worker *processes*
    independently lets numpy's BLAS backend (and JAX's own CPU/Eigen
    thread pool) spawn threads across every physical core -- `n_jobs`
    processes each trying to use all cores oversubscribes them by a
    factor of `n_jobs`, which measurably erodes the parallel speedup
    (confirmed directly: forcing single-threaded BLAS via environment
    variables improved an 8-worker, 40-bin case from 1.53x to 1.70x --
    still not linear, since this is a heterogeneous-core Apple Silicon
    machine, but a real, free improvement from eliminating
    oversubscription). OG `kinextract`'s own parallel bootstrap solves
    the identical problem via `threadpoolctl`; this uses plain
    environment variables instead to avoid a new dependency for a
    kinextract package that doesn't otherwise need one.

    `JAX_PLATFORMS` defaults to `"cpu"` (via `setdefault`, so an already-set
    value -- a user deliberately wanting GPU -- is never overridden) for the
    same reason: `kinextract.__init__` and `laplace_uncertainty` both import
    jax unconditionally, so every worker process in *every* `*_parallel`
    function imports it, not just NUTS-based ones. On a GPU-visible SLURM
    node, `n_jobs` worker processes each independently trying to initialize
    a CUDA/ROCm backend for what is otherwise plain per-bin CPU linear
    algebra is wasted device-init time at best and device contention/errors
    at worst -- this was previously only worked around externally (the
    `losvderrors` SLURM wrapper script sets this env var by hand before
    launching), which silently stopped protecting anyone who ran a
    `*_parallel` function outside that one script. Setting the same default
    here makes every `*_parallel` function safe on a GPU node "for every
    user who downloads kinextract", not just this framework's own scripts.
    """
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    # `select_lambda_aic_local` constructs a FitConfig-equivalent check for every
    # candidate lam in its own scan grid, not just the one finally selected -- so
    # FitConfig's own "knot spacing much finer than knot_spacing_kms is calibrated
    # against" warning can fire once per candidate per bin per replicate, easily
    # hundreds of times across a real multi-bin run (confirmed directly:
    # `examples/notebooks/07_full_galaxy_analysis.ipynb`, see
    # `dev_notes/real_data_sigma/FINDINGS.md`'s warning-frequency entry). Dedup for
    # that warning now lives in `config.py` itself (a plain once-per-process flag,
    # not a `warnings.filterwarnings("once", ...)` registration here) -- the message
    # embeds per-call values (vmin/vmax/dv/effective_lam), so Python's own "once"
    # action (keyed on exact message text) barely deduplicated anything in
    # practice; see `config.py`'s own `_KNOT_SPACING_WARNING_SHOWN` for why, and why
    # that fix also needs no registration here to cover serial (non-`_parallel`)
    # callers too.


def fit_losvd_parallel(A_stack_list, y_w_list, cfg_list, wave_fit=None, n_jobs=None):
    """Fit many independent LOSVD problems -- e.g. every spatial bin of a
    dartboard-binned galaxy cube -- in parallel via a process pool.

    Every bin's fit is completely independent (no shared state between
    bins), so this is pure wall-clock speedup with zero accuracy cost --
    identical per-bin numerics to calling `fit_losvd` serially in a loop,
    just distributed across worker processes. This targets a different
    cost than `select_lambda_aic_local`'s `warm_start_lam_grid` (which
    speeds up *one* fit): a full-galaxy analysis with hundreds of bins was
    measured at "over an hour" mainly because those hundreds of
    already-optimized single-bin fits were run one at a time.

    Parameters
    ----------
    A_stack_list, y_w_list : sequences of the same length, one entry per
        bin, each matching `fit_losvd`'s own `A_stack`/`y_w`.
    cfg_list : a single `FitConfig` (reused for every bin) or a sequence
        of one `FitConfig` per bin -- pass a sequence if, e.g., each bin
        gets its own `vmin`/`vmax` from a two-stage per-bin refit (the
        common real-data pattern: a wide first-pass grid, then a
        `V +/- n*sigma`-centered second pass per bin).
    wave_fit : passed through to every `fit_losvd` call, only needed if
        `cfg.continuum_knots` is set. A single array (shared across every
        bin) or a sequence of one per bin. `None` (default) matches
        `fit_losvd`'s own default (continuum-cofitting off).
    n_jobs : worker process count. `None` (default) uses `os.cpu_count()`,
        memory-permitting -- see `resolve_n_jobs`'s own docstring for how
        the automatic cap scales from a small laptop to a large-memory
        HPC node with no code change needed. Pass an explicit value to
        override the automatic estimate either way. `1` disables the
        process pool and runs serially in-process -- useful for
        debugging (real tracebacks, no pickling) or when the bin count
        is small enough that pool startup would dominate.

        **On heterogeneous performance/efficiency-core hardware (e.g.
        Apple Silicon), `os.cpu_count()` is not necessarily the fastest
        choice -- measure your own machine.** On an 8-logical-core M1 (4
        performance + 4 efficiency), `n_jobs=4` measurably beat
        `n_jobs=8` (21.3s vs 24.8s for 80 bins) -- adding the efficiency
        cores made things *worse*, not better, presumably from cache/
        memory-bandwidth contention with the performance cores. There is
        no portable way to detect the performance-core count from Python
        without a new dependency, so this is not auto-corrected; pass
        `n_jobs` explicitly if `os.cpu_count()` is not giving you the
        expected scaling. On homogeneous (cloud/cluster/server) hardware,
        `os.cpu_count()` is the right default.

    Returns
    -------
    list of `fit_losvd`'s own result dicts, one per bin, in the same
    order as the inputs -- not a generator, so a failure partway through
    surfaces before any results are used, rather than silently on first
    access.

    Uses `concurrent.futures.ProcessPoolExecutor` with the `spawn` start
    method (not `fork`): each worker imports `kinextract` fresh and
    initializes its own independent JAX/CPU context, which is the
    documented-safe way to combine JAX with multiprocessing (a forked
    process can inherit JAX's internal thread pool in a broken state).
    This costs one-time worker startup/JIT-warmup latency, paid once per
    worker (workers are reused across all bins assigned to them), not
    once per bin. Each worker also pins its own BLAS/OpenMP thread pools
    to 1 (`_limit_worker_threads`) before importing `kinextract`,
    since `n_jobs` processes each defaulting to all-core BLAS threading
    would oversubscribe the machine by a factor of `n_jobs` -- the same
    problem OG `kinextract`'s own parallel bootstrap solves via
    `threadpoolctl`.

    **Measured** (`dev_notes/speed/`, 3-E-MILES-template mock, order-3
    default, adaptive lam selection, on the M1 above): ~2.1x wall-clock
    speedup per bin at `n_jobs=4` on 80 bins vs. a serial loop -- real,
    but well short of the naive 4x from 4 workers, since worker-startup
    and per-process JIT-warmup costs don't fully amortize at this bin
    count/machine. Expect a bigger win, closer to linear in the
    performance-core count, at larger bin counts (a real multi-hundred-
    bin galaxy analysis) and on homogeneous multi-core hardware.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    n_bins = len(A_stack_list)
    if len(y_w_list) != n_bins:
        raise ValueError(f"A_stack_list and y_w_list must have the same length "
                          f"(got {n_bins} and {len(y_w_list)}).")
    cfgs = cfg_list if isinstance(cfg_list, (list, tuple)) else [cfg_list] * n_bins
    if len(cfgs) != n_bins:
        raise ValueError(f"cfg_list, given as a sequence, must have one entry per bin "
                          f"(got {len(cfgs)} for {n_bins} bins).")
    waves = wave_fit if isinstance(wave_fit, (list, tuple)) else [wave_fit] * n_bins
    if len(waves) != n_bins:
        raise ValueError(f"wave_fit, given as a sequence, must have one entry per bin "
                          f"(got {len(waves)} for {n_bins} bins).")

    jobs = list(zip(A_stack_list, y_w_list, cfgs, waves))

    if n_jobs == 1:
        return [_fit_losvd_worker(job) for job in jobs]

    # 4x the largest single design matrix -- a rough but real budget for the whitened
    # copy, tensordot products, and other working arrays one job needs at once (see
    # resolve_n_jobs's own docstring for why this is memory-aware, not just cpu_count()).
    per_job_bytes = 4 * max((a.nbytes for a in A_stack_list), default=0)
    n_jobs = resolve_n_jobs(n_jobs, per_job_bytes)
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=get_context("spawn"),
                              initializer=_limit_worker_threads) as pool:
        return list(pool.map(_fit_losvd_worker, jobs))
