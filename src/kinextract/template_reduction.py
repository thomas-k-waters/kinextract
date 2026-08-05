"""Iterative template-set reduction: fit every spatial bin against a
large candidate template grid, keep only the templates any bin actually
uses (NNLS weight above a floor), refit the whole bin set against that
smaller candidate grid, and repeat until the *union* of surviving
templates across bins stops shrinking -- a data-driven way to go from
"the full E-MILES grid" to "the handful of stellar populations this
target's spectra actually need," reusable across a whole cube instead of
hand-picking a template recipe per target.

**Why the union, not each bin's own set, has to stabilize.** A single
bin's own NNLS solution can flip between two near-degenerate templates
from one iteration to the next (the same weight/template-mixture
degeneracy `multi_template.py`'s own module docstring describes) without
the *set of templates useful to the cube as a whole* changing at all.
Requiring every individual bin's own selection to stop changing would
make the loop sensitive to that noise; requiring only the union to
stabilize is the more robust criterion and is what actually answers the
question this exists to answer -- "what template subset does this
target need," not "what does this one bin prefer this iteration."

**Monotonic by construction.** Each iteration's candidate grid is a
subset of the indices selected from the *previous* iteration's fits, so
the union can only shrink or stay the same from one iteration to the
next -- it can never grow back. That means this loop is guaranteed to
either stabilize or shrink to nothing within `max_iterations`, not
oscillate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FitConfig
from .forward_model import build_template_stack
from .multi_template import fit_losvd


@dataclass
class TemplateReductionResult:
    """`union_selected` : sorted indices into the *original* full
    candidate template list -- the recommended, reusable subset.
    `per_bin_history` : `per_bin_history[bin_index]` is a list of dicts,
    one per iteration that bin actually participated in
    (`{iteration, n_templates, selected_indices, weights, b, chi2_red}`,
    `selected_indices`/`weights` already mapped back to original-grid
    indices) -- this is what a `chi^2` (or shape/bias)-vs-`n_templates`
    plateau plot is built from directly, with no extra fitting.
    `converged` : whether the union stabilized before `max_iterations`.
    """
    union_selected: np.ndarray
    per_bin_history: list[list[dict]]
    converged: bool
    n_iterations: int
    threshold_used: float
    max_iterations: int
    bins_subset_used: np.ndarray | None


def reduce_templates_iteratively(bin_specs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                                  templates: list[np.ndarray], template_wave: np.ndarray,
                                  cfg: FitConfig, threshold: float = 1e-3, max_iterations: int = 10,
                                  bins_subset: list[int] | None = None) -> TemplateReductionResult:
    """Run the iterative reduction described in this module's own
    docstring.

    Parameters
    ----------
    bin_specs : one `(wave_fit, flux_fit, errors)` tuple per spatial bin
        (rest-frame wavelength, flux, per-pixel noise -- the same shape
        `multi_template.fit_losvd`'s own inputs are built from).
    templates, template_wave : the full candidate template grid (e.g.
        every E-MILES SSP from `emiles.load_emiles_templates(path)` with
        `select=None`).
    cfg : `FitConfig` -- only its velocity-grid/fitting settings are
        used; every bin is fit at the same `cfg`.
    threshold : normalized-weight floor for "selected" (`w / w.sum() >
        threshold`) -- matches this package's own existing convention
        (`04_template_calibration.ipynb`'s `NNLS_THRESHOLD`).
    max_iterations : hard cap on refit rounds, independent of whether the
        union has actually stabilized -- this loop can be expensive
        (every iteration refits every participating bin against however
        many templates survived so far), so this is a real cost control,
        not just a safety net.
    bins_subset : optional list of indices into `bin_specs` to restrict
        the (expensive) loop to -- e.g. one angular sector's worth of
        radial bins for a target where azimuthal symmetry is a reasonable
        assumption. `None` (default) uses every bin.

    Returns
    -------
    `TemplateReductionResult`. Also see `save_template_subset`/
    `load_template_subset` in this module for persisting it.
    """
    n_templates_full = len(templates)
    participating = np.arange(len(bin_specs)) if bins_subset is None else np.asarray(bins_subset)

    v_grid = cfg.build_velocity_grid()
    active_indices = np.arange(n_templates_full)
    per_bin_history: list[list[dict]] = [[] for _ in range(len(bin_specs))]
    converged = False
    n_iterations = 0

    for iteration in range(max_iterations):
        n_iterations = iteration + 1
        active_templates = [templates[i] for i in active_indices]
        bin_selected_sets = []

        for bidx in participating:
            wave_fit, flux_fit, errors = bin_specs[bidx]
            A_stack = build_template_stack(wave_fit, template_wave, active_templates, v_grid)
            A_stack_w = A_stack / errors[None, :, None]
            y_w = flux_fit / errors

            result = fit_losvd(A_stack_w, y_w, cfg)
            w_norm = result["w"] / result["w"].sum()
            local_selected = np.where(w_norm > threshold)[0]
            global_selected = active_indices[local_selected]

            A_eff_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0]))
            chi2_red = float(np.sum((A_eff_w @ result["b"] - y_w) ** 2)) / len(wave_fit)

            per_bin_history[bidx].append(dict(
                iteration=iteration, n_templates=len(active_indices),
                selected_indices=global_selected, weights=w_norm[local_selected],
                b=result["b"], chi2_red=chi2_red,
            ))
            bin_selected_sets.append(set(global_selected.tolist()))

        union_now = np.array(sorted(set().union(*bin_selected_sets)) if bin_selected_sets else [], dtype=int)

        if set(union_now.tolist()) == set(active_indices.tolist()):
            converged = True
            break
        if len(union_now) == 0:
            # every participating bin's weights collapsed to nothing above threshold --
            # stop rather than fit with an empty candidate set next iteration.
            break
        active_indices = union_now

    return TemplateReductionResult(
        union_selected=active_indices, per_bin_history=per_bin_history, converged=converged,
        n_iterations=n_iterations, threshold_used=threshold, max_iterations=max_iterations,
        bins_subset_used=(np.asarray(bins_subset) if bins_subset is not None else None),
    )


def save_template_subset(result: TemplateReductionResult, bin_table: list[dict],
                          template_names: list[str], template_ages: np.ndarray,
                          template_metals: np.ndarray, filepath: str) -> None:
    """Save a `TemplateReductionResult` to a single `.npz`, following
    `uncertainty.save_bootstrap`'s flattening convention. Each bin's
    *final* iteration's weights are stored densely (`(n_bins,
    n_templates_full)`, zero where a template wasn't selected) rather
    than as ragged per-bin arrays -- avoids the `allow_pickle=True` route
    `emiles.load_emiles_templates`'s own `allow_pickle=False` convention
    deliberately avoids, at the trivial cost of some unused zeros.
    """
    n_bins = len(result.per_bin_history)
    n_templates_full = len(template_names)
    weight_matrix = np.zeros((n_bins, n_templates_full))
    converged_per_bin = np.zeros(n_bins, dtype=bool)
    n_iterations_per_bin = np.zeros(n_bins, dtype=int)

    for bidx, history in enumerate(result.per_bin_history):
        if not history:
            continue
        last = history[-1]
        weight_matrix[bidx, last["selected_indices"]] = last["weights"]
        n_iterations_per_bin[bidx] = len(history)
        if len(history) >= 2:
            converged_per_bin[bidx] = (set(history[-1]["selected_indices"].tolist())
                                        == set(history[-2]["selected_indices"].tolist()))
        else:
            converged_per_bin[bidx] = result.converged

    bin_id = np.array([row.get("bin_id", i) for i, row in enumerate(bin_table)]) if bin_table else np.arange(n_bins)
    r_mid = np.array([row.get("r_mid", np.nan) for row in bin_table]) if bin_table else np.full(n_bins, np.nan)
    theta_mid = np.array(
        [row.get("theta_mid") if row.get("theta_mid") is not None else np.nan for row in bin_table]
    ) if bin_table else np.full(n_bins, np.nan)

    np.savez(
        filepath,
        weight_matrix=weight_matrix, bin_id=bin_id, r_mid=r_mid, theta_mid=theta_mid,
        converged_per_bin=converged_per_bin, n_iterations_per_bin=n_iterations_per_bin,
        union_selected=result.union_selected,
        threshold_used=np.asarray(result.threshold_used), max_iterations=np.asarray(result.max_iterations),
        template_names=np.array(template_names), template_ages=np.asarray(template_ages),
        template_metals=np.asarray(template_metals),
    )


def load_template_subset(filepath: str) -> dict:
    """Load a template subset saved by `save_template_subset`. Returns a
    flat dict of arrays, keyed the same way as the arguments passed to
    `save_template_subset` (plus `union_selected`/`threshold_used`/
    `max_iterations`).
    """
    data = np.load(filepath, allow_pickle=False)
    return {k: data[k] for k in data.files}
