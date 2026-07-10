"""Full-posterior (NUTS/HMC) binned-LOSVD fitting.

The reported fit is a full Bayesian posterior over the flat parameter
vector -- the non-parametric LOSVD histogram ``b``, template weights
``w``, and continuum/amplitude nuisance parameters -- sampled via
NumPyro's NUTS sampler, with the **posterior mean** reported as the point
estimate. Given a skewed posterior (the common case near the instrumental
resolution limit, where the LOSVD is only weakly identified), the mean is
the minimum-MSE point estimate under squared loss; the mode is not, and
can be systematically biased toward the prior in exactly this regime (see
:class:`~kinextract.config.FitConfig`'s "Known limitations" section for
validated recovery behavior near the resolution limit).

The LOSVD's wing-taper smoothness prior (extra regularization strength for
bins far from the line core) is recentered on a data-driven velocity
estimate (:func:`estimate_velocity_xcorr`) rather than the fixed grid's
zero point, so it doesn't asymmetrically suppress whichever tail of the
LOSVD happens to sit farther from grid-zero for a nonzero true velocity.

Continuum-cofitting (``cfg.fit_continuum=True``) is not supported here --
:func:`fit_state_bayesian` raises ``NotImplementedError`` up front for
that case. Only pre-normalized-mode fits (``cfg.fit_continuum=False``)
are supported: the continuum multiplier passed into the model is fixed at
all-ones, so it's a nuisance-free constant rather than something this
module needs to converge via an outer loop. Regularization-strength
(``xlam``) selection uses a cheap MAP-based grid search
(:func:`kinextract.fitting._auto_select_xlam`) as an internal heuristic --
picking a regularization strength is a hyperparameter-search problem, not
the final reported measurement -- so only *one* full NUTS posterior is
run per fit, at the very end, once ``xlam`` has converged.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from ._utils import log
from .config import FitConfig
from .numerics import _wing_taper_lam_vec, estimate_velocity_xcorr
from .state import FitState

_WING_TAPER_SFAC = 1.8  # matches the legacy Fortran convention (see numerics.py)


def build_numpyro_model(st: FitState, v_center: float):
    """Build the NumPyro model function for one LOSVD/template/continuum-
    offset posterior, at fixed regularization strength ``st.xlam``.

    ``b`` (the LOSVD histogram) and ``w`` (template weights) are both,
    physically, compositional/simplex-like quantities -- a normalized
    histogram and a set of mixture weights, respectively (the forward model
    only ever uses ``w`` through the scale-invariant ratio
    ``(t @ w) / sum(w)``, so its absolute magnitude is unidentified by the
    likelihood, and ``b`` must satisfy ``sum(b) == 1``). Each is sampled as
    an unconstrained latent Gaussian pushed through a softmax:
    ``z_b/z_w ~ Normal(0, 1)``, ``b = softmax(z_b)``, ``w = softmax(z_w)``.
    This exactly enforces non-negativity and ``sum() == 1`` by construction,
    giving NUTS a naturally-scaled space to explore rather than a curved,
    box-constrained one -- a bounded-``Uniform`` parameterization with an ad
    hoc normalization penalty produces pathological NUTS trajectories
    (hundreds to thousands of leapfrog steps per transition, saturating the
    tree-depth ceiling regardless of ``target_accept_prob`` tuning), the
    classic signature of poorly-scaled, artificially box-constrained
    geometry. ``b``/``w`` are registered as
    ``numpyro.deterministic`` sites (not just local variables), so
    ``mcmc.get_samples()`` returns them directly under the ``"b"``/``"w"``
    keys as their physical (non-negative, sum-to-1) values -- every
    downstream consumer (:func:`_posterior_mean_vector`,
    :func:`kinextract.plotting.plot_losvd_posterior`, etc.) reads them
    exactly as it would any other named site.

    The forward-model math (template mixture, LOSVD interpolation via the
    precomputed ``losvd_j0/j1/w`` tables, pixel-scatter convolution via
    ``ip_map``/``ip_mask``, continuum multiplication, and the
    ``continuum_poly_mode`` additive/multiplicative correction) is reframed
    as a log-joint-density: the negative log-posterior (up to an additive
    constant) is ``0.5*chi2 + 0.5*smooth`` -- the ``0.5`` on both terms
    is what makes ``chi2`` a genuine Gaussian log-likelihood and ``smooth``
    (the already-quadratic wing-tapered second-difference roughness
    penalty, recentered per :func:`_wing_taper_lam_vec`) a genuine
    log-prior at the *same* relative strength that a given ``xlam`` means
    for the MAP-based ``xlam`` selection (:func:`kinextract.fitting._auto_select_xlam`)
    that picks it -- omitting the ``0.5`` on ``smooth`` alone would leave
    the prior twice as strong, relative to the likelihood, than that
    selection intends. There is no separate normalization-penalty term:
    ``sum(b) == 1`` already holds exactly by construction via the softmax
    parameterization above.

    The continuum multiplier is a runtime *argument* of the returned model
    (``model(cont)``), not a value baked into the closure -- passing it as
    an argument (with a fixed array shape) lets JAX/NumPyro reuse one
    compiled trace across calls instead of recompiling from scratch each
    time. In practice ``cont`` is always all-ones here (only
    pre-normalized-mode fits, ``cfg.fit_continuum=False``, are supported
    by :func:`fit_state_bayesian`), but the model still accepts it as an
    explicit argument for the (unsupported) continuum-cofit case.

    Parameters
    ----------
    st : FitState
        Fit state providing the data, template matrix, and precomputed
        LOSVD/pixel-shift interpolation tables.
    v_center : float
        Velocity to recenter the wing-taper regularization on (see
        :func:`estimate_velocity_xcorr`). Fixed for the whole fit.

    Returns
    -------
    callable
        A NumPyro model function of one argument, ``model(cont)``,
        suitable for ``numpyro.infer.NUTS``.
    """
    nl, nt, npix = st.nl, st.nt, st.npix
    icoff = int(st.icoff)
    fit_global_amp = bool(st.fit_global_amp)
    fortran_template_mixture = bool(st.fortran_template_mixture)
    fit_continuum = bool(st.fit_continuum)
    poly_mode = str(getattr(st, "continuum_poly_mode", "none"))
    poly_bound = float(getattr(st, "continuum_poly_bound", 0.1))
    poly_x = (
        jnp.asarray(st.continuum_poly_x, dtype=jnp.float64)
        if poly_mode != "none" and st.continuum_poly_x is not None
        else None
    )

    t = jnp.asarray(st.t, dtype=jnp.float64)
    outside_tpl = jnp.asarray(st.outside_tpl, dtype=bool)
    losvd_j0 = jnp.asarray(st.losvd_j0, dtype=jnp.int32)
    losvd_j1 = jnp.asarray(st.losvd_j1, dtype=jnp.int32)
    losvd_w = jnp.asarray(st.losvd_w, dtype=jnp.float64)
    ip_safe = jnp.clip(jnp.asarray(st.ip_map, dtype=jnp.int32), 0, npix - 1)
    ip_mask_2d = jnp.asarray(st.ip_mask, dtype=jnp.float64)
    coffi, coff2i = float(st.coffi), float(st.coff2i)
    resd = float(st.resd)
    iskip = int(st.iskip)
    xlam = float(st.xlam)
    sigl0 = float(st.sigl0)

    fit_mask_np = np.zeros(npix, dtype=bool)
    fit_mask_np[iskip:npix - iskip] = True
    fit_mask = jnp.asarray(fit_mask_np)

    lam_vec = jnp.asarray(_wing_taper_lam_vec(st.xl, xlam, sigl0, v_center), dtype=jnp.float64)

    log_amp_center = 0.0
    if fit_global_amp:
        tpl0 = np.asarray(st.t)[:, 0]
        good_amp = (np.asarray(st.gerr) < 1e9) & np.isfinite(tpl0) & (tpl0 != 0)
        amp_ref = (
            float(np.median(np.asarray(st.g)[good_amp]) / np.median(tpl0[good_amp]))
            if good_amp.any() else 1.0
        )
        if not np.isfinite(amp_ref) or amp_ref <= 0:
            amp_ref = 1.0
        log_amp_center = float(np.log(amp_ref))

    def model(cont, g, gerr):
        cont_arr = cont
        z_b = numpyro.sample("z_b", dist.Normal(0.0, 1.0).expand([nl]).to_event(1))
        b = numpyro.deterministic("b", jax.nn.softmax(z_b))
        z_w = numpyro.sample("z_w", dist.Normal(0.0, 1.0).expand([nt]).to_event(1))
        w = numpyro.deterministic("w", jax.nn.softmax(z_w))

        if icoff == 1:
            coff = numpyro.sample("coff", dist.Uniform(coffi - 0.8, coffi + 4.0))
            coff2 = numpyro.sample("coff2", dist.Uniform(-0.7, 0.7))
        elif icoff == 2:
            coff = coffi
            coff2 = numpyro.sample("coff2", dist.Uniform(-0.7, 0.7))
        else:
            coff = coffi
            coff2 = coff2i

        if fit_global_amp:
            logA = numpyro.sample("logA", dist.Normal(log_amp_center, 3.0))
            A = numpyro.deterministic("A", jnp.exp(logA))
        else:
            A = 1.0
        poly = numpyro.sample("poly", dist.Uniform(-poly_bound, poly_bound)) if poly_x is not None else 0.0

        # w sums to exactly 1 by construction (softmax), so no zero-guard is
        # needed for the overall normalization anymore -- only the per-pixel
        # "all templates invalid here" edge case (s2o) below still needs one.
        if fortran_template_mixture:
            tval = t @ w
        else:
            valid_t = jnp.where(outside_tpl, 0.0, t)
            s2o = 1.0 - jnp.sum(outside_tpl * w[None, :], axis=1)
            s2o = jnp.where(s2o == 0.0, 1.0, s2o)
            tval = jnp.sum(valid_t * w[None, :], axis=1) / s2o

        temp = (tval + coff) / (coff + 1.0) + coff2

        y2 = b[losvd_j0] + (b[losvd_j1] - b[losvd_j0]) * losvd_w
        s = jnp.sum(y2)
        sum_b = jnp.sum(b)
        scale = jnp.where(s != 0.0, sum_b / s, 1.0)
        ynew = y2 * scale

        contrib = temp[:, None] * ynew[None, :] * ip_mask_2d
        xs_contrib = ynew[None, :] * ip_mask_2d
        gp = jnp.zeros(npix, dtype=jnp.float64).at[ip_safe].add(contrib)
        xs = jnp.zeros(npix, dtype=jnp.float64).at[ip_safe].add(xs_contrib)
        suml = jnp.sum(ynew)
        gp = jnp.where(xs != 0.0, gp * suml / xs, gp)
        gp = gp * A

        if poly_x is not None:
            if poly_mode == "additive":
                gp = gp + poly * poly_x
            elif poly_mode == "multiplicative":
                gp = gp * (1.0 + poly * poly_x)

        if fit_continuum:
            gp = gp * cont_arr

        valid = (
            fit_mask & jnp.isfinite(g) & jnp.isfinite(gp) & jnp.isfinite(gerr)
            & (gerr > 0.0) & (gerr < 1.0e9)
        )
        resid = jnp.where(valid, (g - gp) / gerr, 0.0)
        numpyro.factor("loglik", -0.5 * jnp.sum(resid ** 2))

        left = (b[1] - 2.0 * b[0]) ** 2
        right = (b[nl - 2] - 2.0 * b[nl - 1]) ** 2
        mid = (b[2:] - 2.0 * b[1:-1] + b[:-2]) ** 2
        terms = jnp.concatenate([jnp.array([left]), mid, jnp.array([right])])
        smooth = jnp.sum(lam_vec * terms) / resd
        # -0.5 makes `smooth` a genuine log-prior at the same relative
        # strength, for a given xlam, that xlam_auto's MAP-based selection
        # intends (see the module docstring).
        numpyro.factor("logprior_smooth", -0.5 * smooth)
        # sum(b) == 1 exactly by construction (b = softmax(z_b)), so no
        # separate normalization-penalty term is needed (sum_b above is
        # used only for the LOSVD interpolation scale).

    return model


def _assemble_flat_vector(st: FitState, b, w, coff: float = 0.0, coff2: float = 0.0,
                           A: float = 1.0, poly: float = 0.0) -> np.ndarray:
    """Assemble a flat parameter vector in the same layout as
    :func:`kinextract.spectrum.build_initial_guess_nonparam`'s ``a0``
    (``b``, ``w``, then the icoff-dependent continuum-offset block, then
    an optional global amplitude, then an optional polynomial-continuum
    coefficient), for feeding a posterior-mean vector into
    :func:`kinextract.numerics.evaluate_model_gp`.
    """
    parts = [np.asarray(b, float), np.asarray(w, float)]
    if st.icoff == 1:
        parts.append(np.array([coff, coff2]))
    elif st.icoff == 2:
        parts.append(np.array([coff2]))
    if st.fit_global_amp:
        parts.append(np.array([A]))
    if getattr(st, "continuum_poly_mode", "none") != "none":
        parts.append(np.array([poly]))
    return np.concatenate(parts)


def _nudge_inside(x: np.ndarray, lo: float, hi: float, frac: float = 1e-3) -> np.ndarray:
    """Clip ``x`` to lie strictly inside ``(lo, hi)``, margined by ``frac``
    of the interval width.

    L-BFGS-B (used for ``cfg.clean``/xlam auto-selection's cheap MAP
    refits) very commonly converges with one or
    more box constraints *active* -- a parameter sitting exactly at its
    bound is a normal, expected outcome for a bound-constrained optimizer,
    not a bug. NumPyro's constrained-to-unconstrained initialization
    transform, however, requires a strictly interior point (an exact
    boundary value raises "Cannot find valid initial parameters" outright).
    This nudges any such point back inside before it's used to initialize
    NUTS.
    """
    margin = frac * (hi - lo)
    return np.clip(x, lo + margin, hi - margin)


def _init_params_from_flat(st: FitState, a: np.ndarray) -> dict:
    """Inverse of :func:`_assemble_flat_vector`: split a flat parameter
    vector into the named sites NumPyro's ``init_to_value`` expects.

    ``b``/``w`` are ``numpyro.deterministic`` sites (see
    :func:`build_numpyro_model`), so the actual *sample* sites needing
    initialization are their unconstrained pre-softmax latents, ``z_b``/
    ``z_w`` -- not ``b``/``w`` directly (``init_to_value`` only applies to
    real sample sites). Given a physical initial vector ``b_val``/``w_val``
    (from the MAP-derived ``a_start``), ``z = log(clip(value, eps, None))``
    is a valid initial unconstrained value: ``softmax(log(x))_i = x_i /
    sum(x_j)``, which exactly reproduces ``b_val``/``w_val`` renormalized to
    sum to 1 (which the MAP-derived values already approximately satisfy).
    ``coff``/``coff2``/``A``/``poly`` remain bounded-Uniform scalars, so
    :func:`_nudge_inside` is still used for those below.
    """
    a = np.asarray(a, float)
    nl, nt = st.nl, st.nt
    i = 0
    eps = 1e-12
    b_val = np.clip(a[i:i + nl], eps, None)
    w_val = np.clip(a[i + nl:i + nl + nt], eps, None)
    params = {"z_b": jnp.asarray(np.log(b_val)), "z_w": jnp.asarray(np.log(w_val))}
    i += nl + nt
    if st.icoff == 1:
        coff_lo, coff_hi = st.coffi - 0.8, st.coffi + 4.0
        params["coff"] = jnp.asarray(_nudge_inside(np.array([a[i]]), coff_lo, coff_hi))[0]
        params["coff2"] = jnp.asarray(_nudge_inside(np.array([a[i + 1]]), -0.7, 0.7))[0]
        i += 2
    elif st.icoff == 2:
        params["coff2"] = jnp.asarray(_nudge_inside(np.array([a[i]]), -0.7, 0.7))[0]
        i += 1
    if st.fit_global_amp:
        params["logA"] = jnp.asarray(np.log(np.clip(a[i], 1e-300, None)))
        i += 1
    if getattr(st, "continuum_poly_mode", "none") != "none":
        bound = float(getattr(st, "continuum_poly_bound", 0.1))
        params["poly"] = jnp.asarray(_nudge_inside(np.array([a[i]]), -bound, bound))[0]
        i += 1
    return params


def build_mcmc(model, init_params: dict, *, num_warmup: int = 50, num_samples: int = 75,
               num_chains: int = 4, dense_mass: bool = True, target_accept_prob: float = 0.8,
               max_tree_depth: int = 10, progress_bar: bool = False) -> MCMC:
    """Build a NUTS/MCMC object for ``model``, without running it.

    ``init_params`` is handled via the kernel's ``init_strategy``
    (:func:`numpyro.infer.init_to_value`). ``z_b``/``z_w`` (the unconstrained
    latents underlying the softmax-parameterized ``b``/``w``, see
    :func:`build_numpyro_model`) and the remaining bounded-Uniform scalar
    sites (``coff``/``coff2``/``A``/``poly``) both need constrained-space
    values transformed to whatever unconstrained space NUTS actually samples
    in -- passing constrained values directly to ``MCMC.run(...,
    init_params=...)`` would silently be wrong, since that argument is
    documented to expect already-unconstrained values.

    ``chain_method="vectorized"`` runs all chains as one batched/``vmap``-ed
    computation instead of one after another, which is dramatically faster
    on this problem and, unlike a sequential run, does not show divergent
    transitions.

    ``dense_mass=True`` is necessary for reliable convergence: the LOSVD
    bins are strongly correlated through the smoothness prior, which a
    dense mass matrix captures and a diagonal one cannot -- a diagonal
    mass matrix shows a large fraction of divergent transitions and poor
    R-hat.

    ``target_accept_prob=0.8`` (the NumPyro default) gives clean
    convergence (max R-hat close to 1, well-behaved divergence fraction)
    with short, natural NUTS trajectories (tens to low hundreds of
    leapfrog steps/transition, not saturating ``max_tree_depth``).
    """
    nuts = NUTS(model, init_strategy=numpyro.infer.init_to_value(values=init_params),
                dense_mass=dense_mass, target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth)
    return MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                progress_bar=progress_bar, chain_method="vectorized")


def run_nuts_fit(mcmc: MCMC, cont, g, gerr, *, seed: int = 0):
    """Run ``mcmc`` (already configured with the desired ``init_params``
    via :func:`build_mcmc`) at continuum ``cont`` and data ``(g, gerr)``,
    and return the posterior samples plus convergence diagnostics.

    ``g``/``gerr`` are passed here as explicit runtime arguments rather than
    closure constants in :func:`build_numpyro_model`, so that fitting many
    different spectra of the same shape -- the realistic HPC/notebook-batch
    scenario this package is built for -- can reuse one compiled JAX trace
    across all of them: JAX's JIT treats closed-over arrays as compile-time
    constants baked into the traced computation, not inputs it can reuse
    the trace for, so closing over per-spectrum data would force a full
    retrace on every new noise realization even with the same problem
    shape.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Built by :func:`build_mcmc` for this iteration's init_params.
    cont : ndarray
        Current continuum multiplier, passed through to the model (always
        all-ones in the supported pre-normalized-mode case).
    g, gerr : ndarray
        Observed spectrum and per-pixel error, passed through to the
        model (``gerr`` already encodes any masking via the ``BIG``
        sentinel, exactly as elsewhere in this package).
    seed : int
        PRNG seed for this run.

    Returns
    -------
    samples : dict
        Posterior samples for every named site, each shape
        ``(num_samples * num_chains, ...)``.
    diagnostics : dict
        ``n_divergent`` (int), ``n_total`` (int), ``r_hat`` (dict of
        per-site max R-hat across chains, only when ``num_chains > 1``),
        ``mean_num_steps``/``max_num_steps`` (float/int -- NUTS trajectory
        length, i.e. leapfrog steps per transition, averaged/maxed over all
        post-warmup samples across chains; a direct measure of posterior
        geometry difficulty, independent of raw per-step cost -- see
        :mod:`kinextract.bayesian`'s performance notes), and ``mcmc`` (the
        underlying ``numpyro.infer.MCMC`` object, for callers wanting
        ``mcmc.print_summary()`` or further inspection).
    """
    num_chains = mcmc.num_chains
    mcmc.run(jax.random.PRNGKey(seed), cont, g, gerr, extra_fields=("num_steps",))

    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    n_divergent = int(np.sum(np.asarray(extra.get("diverging", np.zeros(1)))))
    n_total = int(next(iter(samples.values())).shape[0])

    num_steps = np.asarray(extra.get("num_steps", np.zeros(1)))
    mean_num_steps = float(np.mean(num_steps))
    max_num_steps = int(np.max(num_steps))

    r_hat = {}
    if num_chains > 1:
        from numpyro.diagnostics import summary as numpyro_summary
        samples_by_chain = mcmc.get_samples(group_by_chain=True)
        stats = numpyro_summary(samples_by_chain)
        for site, s in stats.items():
            if "r_hat" in s:
                r_hat[site] = float(np.max(np.asarray(s["r_hat"])))

    if n_divergent > 0:
        log(f"WARNING: NUTS reported {n_divergent}/{n_total} divergent transitions "
            f"-- posterior summaries may not be reliable.")

    return samples, dict(n_divergent=n_divergent, n_total=n_total, r_hat=r_hat,
                          mean_num_steps=mean_num_steps, max_num_steps=max_num_steps, mcmc=mcmc)


def _posterior_mean_vector(st: FitState, samples: dict) -> tuple[np.ndarray, dict]:
    """Reduce posterior samples to a flat point-estimate vector (posterior
    mean of every site) plus the per-site mean arrays, for use as the
    final reported best-fit parameters.
    """
    b_mean = np.asarray(samples["b"]).mean(axis=0)
    w_mean = np.asarray(samples["w"]).mean(axis=0)
    coff_mean = float(np.asarray(samples["coff"]).mean()) if "coff" in samples else st.coffi
    coff2_mean = float(np.asarray(samples["coff2"]).mean()) if "coff2" in samples else st.coff2i
    A_mean = float(np.asarray(samples["A"]).mean()) if "A" in samples else 1.0
    poly_mean = float(np.asarray(samples["poly"]).mean()) if "poly" in samples else 0.0
    a_vec = _assemble_flat_vector(st, b_mean, w_mean, coff_mean, coff2_mean, A_mean, poly_mean)
    return a_vec, dict(b_mean=b_mean, w_mean=w_mean, coff_mean=coff_mean,
                        coff2_mean=coff2_mean, A_mean=A_mean, poly_mean=poly_mean)


def fit_state_bayesian(st: FitState, cfg: FitConfig, a0: np.ndarray, bounds: list):
    """Run the full Bayesian LOSVD/template/continuum fit, called from
    :func:`kinextract.fitting.run_spectral_fit`.

    Only pre-normalized-mode fits are supported (``cfg.fit_continuum=False``):
    this raises ``NotImplementedError`` immediately if ``cfg.fit_continuum``
    is True, since continuum-cofitting has no NUTS-compatible implementation
    here (see the module docstring). Pre-normalize the spectrum first (see
    :func:`kinextract.continuum.asymmetric_least_squares_continuum` and
    ``examples/notebooks/06_prenormalized_workflow.ipynb``), or use the MAP
    joint path (:mod:`kinextract.joint`) if a cofit continuum with
    bootstrap uncertainty estimates is needed instead.

    ``cfg.clean=True`` (iterative sigma-clipping of outlier pixels) is
    handled as a one-time MAP-based preprocessing pass
    (:func:`kinextract.fitting.run_iterative_clean_map`) before any
    posterior sampling happens: identifying cosmic rays/bad columns is a
    data-quality step, not the scientific measurement, so it doesn't need
    full posterior sampling.

    Regularization-strength (``xlam``) auto-selection uses a cheap
    MAP-based grid search (see the module docstring). One full NUTS
    posterior is run at the very end, at ``cfg``'s configured (fast)
    sampling budget -- if that posterior fails the reliability gate (poor
    R-hat or too many divergences, more likely for real data with more
    structure than a smooth synthetic mock), it is retried once at a
    substantially larger, separately-validated budget rather than silently
    reporting an unconverged result. Whichever posterior passes (or the
    last one tried, if neither does) is what gets reported.

    Parameters
    ----------
    st : FitState
        Fit state to update in place (``st.xlam``, ``st.continuum_mult``,
        etc.).
    cfg : FitConfig
        Run configuration. Must have ``cfg.fit_continuum=False``.
        NUTS-specific fields (``nuts_num_warmup``, ``nuts_num_samples``,
        ``nuts_num_chains``, ``nuts_seed``) fall back to sensible defaults
        if unset.
    a0 : ndarray
        Initial flat parameter vector (from
        :func:`kinextract.spectrum.build_initial_guess_nonparam`), used
        to seed the first NUTS run.
    bounds : list of tuple
        Unused here (NUTS samples in unconstrained space with automatic
        transforms rather than box-constrained optimization); accepted
        only so the call site in ``run_spectral_fit`` does not need to
        branch on fit method.

    Returns
    -------
    result : SimpleNamespace
        ``x`` (posterior-mean flat vector), ``success`` (bool, True unless
        divergence fraction or R-hat fails the reliability thresholds),
        ``message`` (str), ``mcmc`` (the final ``numpyro.infer.MCMC``
        object -- ``mcmc.get_samples()`` gives the raw posterior draws for
        every named site, used by e.g.
        :func:`kinextract.plotting.plot_losvd_posterior` to draw
        credible-interval error bars), ``diagnostics`` (dict with
        ``n_divergent``, ``n_total``, ``r_hat`` (per-site dict),
        ``max_r_hat`` (scalar, the same value used in the ``success`` gate
        above and quoted in ``message``), ``mean_num_steps``/
        ``max_num_steps`` (NUTS trajectory length -- leapfrog steps per
        transition -- a direct measure of posterior geometry difficulty),
        and ``setup_wall_time_s``/``nuts_wall_time_s`` (wall-clock split
        between everything before the final NUTS call -- cleaning, xlam
        selection -- and the final NUTS call(s) themselves, which
        typically dominate total runtime; the latter includes the retry
        time if a budget escalation happened).
    """
    from types import SimpleNamespace

    from .fitting import _auto_select_xlam, _auto_select_xlam_discrepancy, run_iterative_clean_map
    from .masking import build_clean_protect_mask

    if getattr(cfg, "fit_continuum", False):
        raise NotImplementedError(
            "kinextract.bayesian does not support continuum-cofitting "
            "(cfg.fit_continuum=True): only pre-normalized-mode fits "
            "(cfg.fit_continuum=False) are supported for the Bayesian/NUTS "
            "path. Pre-normalize the spectrum first (see "
            "kinextract.continuum.asymmetric_least_squares_continuum and "
            "examples/notebooks/06_prenormalized_workflow.ipynb), or use "
            "the MAP joint path (kinextract.joint) if you need a cofit "
            "continuum with bootstrap uncertainty estimates."
        )

    t_start = time.perf_counter()

    if getattr(cfg, "jax_enable_x64", True):
        # Critical for NUTS: without this, JAX silently runs every array in
        # this module in float32, and float32 gradients through the
        # leapfrog integrator cause severe convergence failures (R-hat
        # far from 1 even with a generous warmup/samples/chains budget).
        # This needs its own explicit call here since xlam auto-selection
        # (which enables it as a side effect via _fit_map_once) can be
        # skipped entirely (cfg.xlam_auto=False).
        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass

    map_kwargs = dict(
        map_gtol=cfg.map_gtol, map_maxls=cfg.map_maxls,
        use_scaled_optimizer=cfg.use_scaled_optimizer,
        use_jax_objective=cfg.use_jax_objective, jax_enable_x64=cfg.jax_enable_x64,
    )

    if getattr(cfg, "clean", False):
        # Bad-pixel identification (cosmic rays, bad columns, reduction
        # artifacts) doesn't need posterior sampling -- it's a data-quality
        # preprocessing step, not the scientific measurement. The MAP-based
        # sigma-clipping machinery masks outliers once, up front (mutates
        # st.gerr in place); the full NUTS fit below then runs on the
        # cleaned data, so no MAP point estimate feeds into the reported
        # kinematics themselves.
        log("Pre-cleaning (MAP-based sigma-clip) before Bayesian fit")
        pm = build_clean_protect_mask(st, cfg)
        _, good_mask = run_iterative_clean_map(
            st, np.asarray(a0, float), bounds,
            cfg.map_maxiter, cfg.map_ftol, cfg.map_maxfun, cfg.print_every,
            cfg.clean_sigma, cfg.clean_maxiter, cfg.clean_minpix,
            pm, cfg.clean_protect_absorption_only,
            getattr(cfg, "clean_bloom_pixels", 0),
            **map_kwargs,
        )
        st.clean_good_mask = good_mask

    if getattr(cfg, "xlam_auto", False):
        xlam_grid = getattr(cfg, "xlam_auto_grid", (1e2, 1e3, 1e4, 1e5, 1e6, 1e7))
        if str(getattr(cfg, "xlam_criterion", "discrepancy")).lower() == "discrepancy":
            _auto_select_xlam_discrepancy(
                st, cfg, np.asarray(a0, float), bounds,
                xlam_grid=xlam_grid, map_kwargs=map_kwargs,
                nsigma=float(getattr(cfg, "xlam_discrepancy_nsigma", 0.3)),
            )
        else:
            _auto_select_xlam(
                st, cfg, np.asarray(a0, float), bounds,
                xlam_grid=xlam_grid,
                smooth_threshold=float(getattr(cfg, "xlam_smooth_threshold", 0.25)),
                map_kwargs=map_kwargs,
            )

    # Recomputed here (post-cleaning/xlam-selection) rather than reused from
    # st.v_center (set once, pre-cleaning, in make_fit_state): cleaning can
    # remove cosmic-ray/artifact pixels that would otherwise bias the
    # cross-correlation estimate.
    v_center = estimate_velocity_xcorr(st)
    st.v_center = v_center
    seed = int(getattr(cfg, "nuts_seed", 0))
    final_cfg = dict(
        num_warmup=int(getattr(cfg, "nuts_num_warmup", 50)),
        num_samples=int(getattr(cfg, "nuts_num_samples", 75)),
        num_chains=int(getattr(cfg, "nuts_num_chains", 4)),
        dense_mass=True, target_accept_prob=0.8,
    )

    a_start = np.asarray(a0, float).copy()

    model = build_numpyro_model(st, v_center)
    init_params = _init_params_from_flat(st, a_start)
    cont = jnp.asarray(getattr(st, "continuum_mult", np.ones(st.npix)), dtype=jnp.float64)
    g_arr = jnp.asarray(st.g, dtype=jnp.float64)
    gerr_arr = jnp.asarray(st.gerr, dtype=jnp.float64)
    setup_wall_time_s = time.perf_counter() - t_start

    # The default (num_warmup, num_samples) is tuned for fast convergence on
    # typical problems, but some spectra -- particularly real data with more
    # structure than a smooth synthetic mock -- genuinely need more sampling
    # to converge reliably. Rather than silently reporting an unconverged
    # posterior, retry once at a substantially larger, separately-validated
    # budget if the first attempt's reliability gate fails.
    budgets = [(final_cfg["num_warmup"], final_cfg["num_samples"])]
    if budgets[0] != (750, 1000):
        budgets.append((750, 1000))

    t_nuts = time.perf_counter()
    for attempt, (num_warmup, num_samples) in enumerate(budgets):
        attempt_cfg = dict(final_cfg, num_warmup=num_warmup, num_samples=num_samples)
        if attempt > 0:
            log(f"NUTS budget ({budgets[attempt - 1]}) did not converge reliably; "
                f"retrying with a larger budget ({num_warmup}+{num_samples}x"
                f"{attempt_cfg['num_chains']})")
        else:
            log(f"Final NUTS posterior at converged continuum "
                f"({num_warmup}+{num_samples}x{attempt_cfg['num_chains']}, vectorized)")
        mcmc = build_mcmc(model, init_params, **attempt_cfg)
        samples, diagnostics = run_nuts_fit(mcmc, cont, g_arr, gerr_arr, seed=seed)

        n_div = diagnostics.get("n_divergent", 0)
        n_tot = diagnostics.get("n_total", 1)
        r_hat_vals = diagnostics.get("r_hat", {})
        max_r_hat = max(r_hat_vals.values()) if r_hat_vals else float("nan")
        # >=10% divergent transitions, or R-hat too far from 1 (chains not mixing --
        # e.g. too few warmup/samples for this many parameters), makes the posterior
        # untrustworthy regardless of how the point estimate looks.
        success = (n_div < 0.1 * n_tot) and (np.isnan(max_r_hat) or max_r_hat < 1.1)
        if success or attempt == len(budgets) - 1:
            break

    nuts_wall_time_s = time.perf_counter() - t_nuts
    diagnostics["setup_wall_time_s"] = setup_wall_time_s
    diagnostics["nuts_wall_time_s"] = nuts_wall_time_s
    log(f"Phase timing: setup={setup_wall_time_s:.1f}s NUTS={nuts_wall_time_s:.1f}s "
        f"mean_num_steps={diagnostics.get('mean_num_steps', float('nan')):.1f} "
        f"max_num_steps={diagnostics.get('max_num_steps', 'n/a')}")
    a_best, _ = _posterior_mean_vector(st, samples)

    message = f"NUTS complete: {n_div}/{n_tot} divergent transitions, max R-hat={max_r_hat:.4f}"
    if not success:
        log(f"WARNING: {message} -- treating this fit as unreliable (success=False)")

    # Expose the scalar max R-hat directly (the per-site r_hat dict alone
    # forces every caller to re-derive this max themselves).
    diagnostics["max_r_hat"] = max_r_hat

    from .numerics import objective_map
    f_best = float(objective_map(a_best, st))

    return SimpleNamespace(x=a_best, fun=f_best, success=success, message=message,
                            mcmc=diagnostics.get("mcmc"), diagnostics=diagnostics)
