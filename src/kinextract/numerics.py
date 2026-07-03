"""Core numerics for the non-parametric LOSVD fit.

This module implements the forward model and objective function that sit at
the heart of ``kinextract``: given a trial parameter vector (an LOSVD
histogram ``b`` over ``nl`` velocity bins, template weights ``w``, and a
handful of continuum/amplitude nuisance parameters), :func:`evaluate_model_gp`
convolves the stellar template mixture with the LOSVD (via a discrete
pixel-scatter scheme driven by the precomputed ``ip_map`` table) to produce a
model spectrum, and :func:`objective_map` compares that model to the data to
form

    objective = chi2 + smoothness_penalty + 0.1 * |sum(b) - 1|

where ``chi2`` is the usual (data - model)^2 / sigma^2 sum over good pixels
and the smoothness penalty is a second-derivative roughness penalty on ``b``
whose strength ``xlam`` is tapered up in the LOSVD wings (see
:func:`_compute_smoothness`). Two parallel implementations of the objective
and its gradient are provided: fast Numba-JIT loops
(:func:`_compute_chi2`, :func:`_compute_smoothness`,
:func:`_convolve_losvd_numba`) used with scipy finite-difference gradients,
and an optional JAX-jitted value-and-gradient builder
(:func:`_build_jax_objective_value_and_grad`) that supplies exact analytic
gradients to the L-BFGS-B optimizer in :mod:`kinextract.fitting` for a large
speedup. :func:`objective_components` and
:func:`compute_weighted_template_spectrum` are diagnostic helpers used to
inspect a fit after the fact.

Compiled JAX kernels are cached process-wide in ``_JAX_VG_CACHE`` (see
:func:`_get_or_build_jax_vg`) so repeated fits sharing the same problem
shape/data don't each pay a fresh ~10-120s XLA compilation. Anyone touching
that cache should read the docstrings on :func:`_array_fingerprint` and
:func:`_get_or_build_jax_vg` first: an earlier version keyed the cache on
shape/scalar parameters alone, which let two *different* fits (different
template, different LOSVD grid) silently share one compiled kernel whenever
their shapes and scalars happened to coincide, evaluating one fit's data
against another's template/grid with no error -- and a later version fixed
that but serialized all compilation through one process-wide lock, which
was safe but defeated `ThreadPoolExecutor`-based parallelism across
distinct problem shapes. Both are fixed (content-fingerprinted keys,
per-key locks), but the failure mode is subtle enough to call out here.
Separately, running many fits concurrently -- multiple threads in one
process, multiple separate processes, or under unrelated external CPU
load -- was verified (``benchmarks/``) to *not* perturb numerical results;
the CPU-contention scare that motivated checking this turned out to be
tied to the pre-fix locking bug's pathological process state, not a
general reproducibility risk.
"""
from __future__ import annotations
import warnings
import time
import threading as _threading
import numpy as np
try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None
    warnings.warn(
        "JAX is not installed or failed to import. The MAP optimizer and "
        "bootstrap will fall back to scipy finite-difference gradients, which "
        "are 20-100x slower per L-BFGS-B step. Install with: pip install jax[cpu]",
        RuntimeWarning, stacklevel=1,
    )
try:
    from numba import njit
except Exception:
    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else lambda f: f
from .state import FitState, getnlosvd_fast_from_b


# =============================================================================
# JAX cache (from preamble)
# =============================================================================

# Process-wide cache for JAX JIT-compiled value+grad functions.
# Keyed by a shape-signature tuple so that the XLA kernel is compiled once per
# unique (nl, nt, npix, …) combination and then shared by ALL callers in the
# same process — including every bootstrap thread.  Without this, every call to
# _build_jax_objective_value_and_grad creates a new Python closure → new JAX
# trace → new XLA compilation (~30-120 s each).  With 1000 bootstrap replicates
# that cost dominates the runtime.
_JAX_VG_CACHE: dict = {}
# _JAX_VG_CACHE_LOCK only guards creating/fetching a *per-key* lock below; the
# actual (slow, ~10-120s) XLA compilation happens under that per-key lock, not
# this one. Sharing a single lock across all keys serialized every distinct
# compilation process-wide -- fine when almost all callers shared a handful of
# keys, but became a severe bottleneck once _array_fingerprint (see below)
# made most keys effectively unique per scenario: six ThreadPoolExecutor
# workers hitting six different new keys at once still ended up compiling one
# at a time, turning a 6-way-parallel benchmark run into an effectively
# 1-way-serial one.
_JAX_VG_CACHE_LOCK = _threading.Lock()
_JAX_VG_KEY_LOCKS: dict = {}


def _array_fingerprint(*arrays) -> int:
    """Fast content fingerprint for arrays baked into a JAX-compiled closure.

    ``_build_jax_objective_value_and_grad`` closes over ``st.t``, ``st.xl``,
    ``st.outside_tpl``, and the LOSVD/pixel interpolation tables
    (``losvd_j0/j1/w``, ``ip_map``, ``ip_mask``) as compiled-in constants --
    two FitState instances can share every *shape* (nl, nt, npix, ...) and
    scalar (xlam, sigl0, ...) in :func:`_get_or_build_jax_vg`'s cache key
    while having completely different underlying data (e.g. a different
    template file, or a different LOSVD velocity grid from a different
    ``sigl``/``losvd_vmin``/``losvd_vmax``). Without this fingerprint in the
    key, the second such fit would silently reuse the first's compiled
    kernel -- evaluating the *first* fit's template/grid against the
    *second* fit's data, with no error or warning, just a wrong answer.
    """
    h = 0
    for arr in arrays:
        h = hash((h, np.asarray(arr).tobytes()))
    return h


def _get_or_build_jax_vg(st):
    """Return the cached JAX value+grad function for this FitState shape.

    Thread-safe: double-checked locking with a *per-key* lock, so two threads
    racing to build the same new key wait for one compile and share its
    result, while threads building two *different* new keys compile fully in
    parallel rather than serializing through a single process-wide lock.
    """
    # When icoff==0 or icoff==2, coffi/coff2i are baked into the JAX closure as
    # Python float constants.  Include them in the key so two spectra with the
    # same shape but different fixed coff values don't share a compiled kernel.
    content_fp = _array_fingerprint(
        st.t, st.outside_tpl, st.xl, st.losvd_j0, st.losvd_j1, st.losvd_w,
        st.ip_map, st.ip_mask,
    )
    key = (
        int(st.nl), int(st.nt), int(st.npix), int(st.nlosvd),
        int(st.icoff), bool(st.fit_global_amp),
        bool(st.fortran_template_mixture), bool(st.fit_als_continuum),
        float(st.xlam), float(st.sigl0),
        float(st.coffi)  if int(st.icoff) in (0, 2) else 0.0,
        float(st.coff2i) if int(st.icoff) == 0       else 0.0,
        content_fp,
    )
    if key in _JAX_VG_CACHE:
        return _JAX_VG_CACHE[key]
    with _JAX_VG_CACHE_LOCK:
        key_lock = _JAX_VG_KEY_LOCKS.setdefault(key, _threading.Lock())
    with key_lock:
        if key not in _JAX_VG_CACHE:
            _JAX_VG_CACHE[key] = _build_jax_objective_value_and_grad(st)
    return _JAX_VG_CACHE[key]


# =============================================================================
# Section 8 - Core numerics
# =============================================================================

@njit(cache=True)
def _compute_chi2(
    g: np.ndarray, gp: np.ndarray, gerr: np.ndarray,
    iskip: int, npix: int,
) -> float:
    """Sum of squared, error-normalised residuals over the good fit region.

    Computes ``sum_i ((g[i] - gp[i]) / gerr[i])**2`` over the interior pixel
    range ``[iskip, npix - iskip)``, skipping any pixel whose error is
    non-positive, effectively infinite (``>= 1e9``, the sentinel used
    elsewhere in the package to mark clipped/masked pixels), or where any of
    ``g``, ``gp``, ``gerr`` is non-finite. This is the ``chi2`` term of the
    objective described in the module docstring. Numba-JIT compiled for
    speed since it runs on every objective evaluation during optimization.

    Parameters
    ----------
    g : ndarray
        Observed (data) spectrum, length ``npix``.
    gp : ndarray
        Model spectrum evaluated at the current parameters, length ``npix``.
    gerr : ndarray
        Per-pixel 1-sigma error estimate, length ``npix``. Pixels with
        ``gerr >= 1e9`` are treated as masked/excluded.
    iskip : int
        Number of pixels to skip at each end of the spectrum (edge effects
        from the LOSVD convolution).
    npix : int
        Total number of pixels in the spectrum.

    Returns
    -------
    float
        The chi-squared statistic, summed (not reduced) over good pixels.
    """
    chi2 = 0.0
    for i in range(iskip, npix - iskip):
        if gerr[i] > 0.0 and gerr[i] < 1.0e9:
            if np.isfinite(g[i]) and np.isfinite(gp[i]) and np.isfinite(gerr[i]):
                r = (g[i] - gp[i]) / gerr[i]
                chi2 += r * r
    return chi2


@njit(cache=True)
def _compute_smoothness(
    b: np.ndarray, xl: np.ndarray,
    xlam: float, sigl0: float, resd: float, nl: int,
) -> float:
    """Second-derivative roughness penalty on the LOSVD, with wing tapering.

    Computes ``sum_j lam_j * (second difference of b at j)**2 / resd``, where
    the second difference uses a reflecting boundary condition at the first
    and last bins (``b[1] - 2*b[0]`` and ``b[nl-2] - 2*b[nl-1]``) and
    ``lam_j`` is the local regularization strength (see Notes). This is the
    ``smoothness_penalty`` term of the objective described in the module
    docstring: it discourages bin-to-bin wiggles in the recovered LOSVD
    while still allowing it to take an arbitrary (non-parametric) shape.
    Numba-JIT compiled since it runs on every objective evaluation.

    Parameters
    ----------
    b : ndarray
        Trial LOSVD histogram, length ``nl``.
    xl : ndarray
        Velocity grid (km/s) on which `b` is defined, length ``nl``.
    xlam : float
        Base (central) regularization strength.
    sigl0 : float
        Characteristic LOSVD velocity dispersion (km/s) used to define the
        "core" vs. "wing" regions for the taper.
    resd : float
        Normalization divisor (typically related to the number of degrees
        of freedom / residual scale), applied to the summed penalty.
    nl : int
        Number of LOSVD bins.

    Returns
    -------
    float
        The smoothness penalty value.

    Notes
    -----
    The regularization weight is not spatially uniform: for bins with
    ``|xl[j]| > 1.8 * sigl0`` (i.e. beyond 1.8 sigma into the LOSVD wings),
    ``lam`` is scaled up by a factor ``(|xl[j] / sigl0| / 1.8)**4``. This
    intentionally suppresses noise-driven wiggles far from the line core,
    where the data carry little information about the LOSVD shape, while
    leaving the regularization near the peak comparatively loose so genuine
    kinematic structure can be recovered. The factor of 1.8 and the quartic
    form exactly match a legacy Fortran convention and are kept as-is for
    compatibility with historical fits; this is a deliberate design choice,
    not a bug.
    """
    sfac = 1.8
    smooth = 0.0
    for j in range(nl):
        lam = xlam
        if abs(xl[j]) > sfac * sigl0:
            lam = xlam * (abs(xl[j] / sigl0) / sfac) ** 4
        if j == 0:
            smooth += lam * (b[1] - 2 * b[0]) ** 2
        elif j == nl - 1:
            smooth += lam * (b[nl - 2] - 2 * b[nl - 1]) ** 2
        else:
            smooth += lam * (b[j + 1] - 2 * b[j] + b[j - 1]) ** 2
    return smooth / resd


@njit(cache=True)
def _convolve_losvd_numba(
    temp: np.ndarray,
    ynew: np.ndarray,
    ip_map: np.ndarray,
    ip_mask: np.ndarray,
    npix: int,
    nlosvd: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convolve a template spectrum with the LOSVD via discrete pixel scatter.

    For every template pixel ``i`` and every LOSVD velocity bin ``j``, this
    scatters the contribution ``temp[i] * ynew[j]`` into output pixel
    ``ip_map[i, j]`` (when ``ip_mask[i, j]`` is True), and separately
    accumulates the LOSVD weight ``ynew[j]`` into a normalization array
    ``xs`` at the same output pixel. This implements the convolution of the
    stellar template with the (non-parametric) LOSVD as a scatter operation
    rather than a dense matrix-vector product, mirroring the legacy Fortran
    algorithm's pixel bookkeeping exactly. Numba-JIT compiled: this is the
    single hottest loop in the fit, executed on every objective/gradient
    evaluation.

    Parameters
    ----------
    temp : ndarray
        Template spectrum (already weighted/mixed and continuum-corrected),
        length ``npix``.
    ynew : ndarray
        LOSVD values resampled onto the per-pixel velocity grid, length
        ``nlosvd``.
    ip_map : ndarray of int
        Precomputed (npix, nlosvd) table mapping each (pixel, LOSVD-bin)
        pair to an output pixel index. Built with Fortran-style NINT
        rounding (round-half-away-from-zero); see
        :func:`kinextract.io.nint_fortran` and
        :func:`kinextract.state.precompute_ip_map`.
    ip_mask : ndarray of bool
        (npix, nlosvd) mask; True where the corresponding ``ip_map`` entry
        is a valid in-range output pixel index.
    npix : int
        Number of output (data) pixels.
    nlosvd : int
        Number of LOSVD velocity bins used in the resampled grid.

    Returns
    -------
    gp : ndarray
        Unnormalized scattered model spectrum, length ``npix``.
    xs : ndarray
        Per-pixel sum of scattered LOSVD weights, length ``npix``, used by
        the caller to renormalize ``gp`` (dividing out uneven pixel
        coverage from the discrete scatter).

    Notes
    -----
    The pixel indices in ``ip_map`` are produced with Fortran ``NINT``
    (round-half-away-from-zero) rather than NumPy's default round-half-to-
    even, purely for bit-for-bit compatibility with legacy output formats
    from the original Fortran pipeline.
    """
    gp = np.zeros(npix, dtype=np.float64)
    xs = np.zeros(npix, dtype=np.float64)

    for i in range(npix):
        ti = temp[i]
        for j in range(nlosvd):
            if ip_mask[i, j]:
                ip = ip_map[i, j]
                yj = ynew[j]
                gp[ip] += ti * yj
                xs[ip] += yj

    return gp, xs


def evaluate_model_gp(
    a: np.ndarray, st: FitState, apply_continuum: bool = True,
) -> tuple:
    """Evaluate the forward model spectrum for a trial parameter vector.

    Unpacks the flat parameter vector `a` into its physical components (the
    LOSVD histogram `b`, template weights `w`, continuum offsets, and
    optional global amplitude / polynomial term), forms the weighted
    template mixture, convolves it with the LOSVD via the discrete
    pixel-scatter scheme (:func:`_convolve_losvd_numba`, driven by the
    precomputed ``ip_map``/``ip_mask`` tables on `st`), and applies any
    continuum corrections. This is the forward model whose mismatch with
    the observed spectrum forms the chi2 term of :func:`objective_map`.

    Parameters
    ----------
    a : ndarray
        Flat parameter vector, laid out as: ``nl`` LOSVD bin values,
        ``nt`` template weights, then (depending on ``st.icoff``) 0, 1, or
        2 continuum offset parameters, then an optional global amplitude
        (if ``st.fit_global_amp``), then an optional polynomial continuum
        coefficient (if ``st.continuum_poly_mode != "none"``).
    st : FitState
        Precomputed fit state holding the data, template matrix, velocity
        grids, and configuration flags needed to evaluate the model.
    apply_continuum : bool, optional
        If True (default), multiply the model by the current ALS continuum
        (``st.continuum_mult``) when ``st.fit_als_continuum`` is set. Set to
        False to obtain the continuum-free model, e.g. for diagnostics.

    Returns
    -------
    gp : ndarray
        Model spectrum, length ``npix``.
    b : ndarray
        LOSVD histogram used to build the model, length ``nl`` (clipped to
        a small positive floor).
    w : ndarray
        Template weights used to build the model, length ``nt`` (clipped to
        a small positive floor).
    coff : float
        Additive continuum offset parameter.
    coff2 : float
        Second continuum offset parameter.
    A : float
        Global amplitude scaling applied to the model (1.0 if
        ``st.fit_global_amp`` is False).

    Notes
    -----
    When ``st.fortran_template_mixture`` is False, pixels outside any
    template's wavelength coverage are excluded via ``st.outside_tpl`` (a
    boolean mask), rather than via the fragile ``T == 1.0`` sentinel value
    used by the legacy Fortran/early-Python code.
    """
    nl, nt, npix = st.nl, st.nt, st.npix
    i = 0
    b = np.maximum(a[i:i + nl], 1e-6);  i += nl
    w = np.maximum(a[i:i + nt], 1e-12); i += nt

    if st.icoff == 0:
        coff, coff2 = st.coffi, st.coff2i
    elif st.icoff == 1:
        coff, coff2 = a[i], a[i + 1]; i += 2
    elif st.icoff == 2:
        coff, coff2 = st.coffi, a[i]; i += 1
    else:
        raise ValueError(f"icoff must be 0, 1, or 2; got {st.icoff}")

    if st.fit_global_amp:
        A = a[i]; i += 1
    else:
        A = 1.0
    if st.continuum_poly_mode != "none":
        poly = a[i]; i += 1
    else:
        poly = 0.0

    sum2 = float(np.sum(w)) or 1.0
    if st.fortran_template_mixture:
        tval = st.t @ w / sum2
    else:
        # Exclude pixels outside any template's coverage.
        # Uses outside_tpl (bool) instead of the old T==1.0 sentinel.
        valid_t = np.where(st.outside_tpl[:, None], 0.0, st.t)
        s2o = sum2 - (st.outside_tpl[:, None] * w[None, :]).sum(axis=1)
        s2o = np.where(s2o == 0, 1.0, s2o)
        tval = (valid_t * w[None, :]).sum(axis=1) / s2o

    temp = (tval + coff) / (coff + 1.0) + coff2

    _, ynew = getnlosvd_fast_from_b(st, b)

    gp, xs = _convolve_losvd_numba(
        temp,
        ynew,
        st.ip_map,
        st.ip_mask,
        npix,
        st.nlosvd,
    )

    suml = float(np.sum(ynew))
    gp = np.where(xs != 0, gp * suml / xs, gp)

    gp = gp * A

    if st.continuum_poly_mode == "additive" and st.continuum_poly_x is not None:
        gp = gp + poly * st.continuum_poly_x
    elif st.continuum_poly_mode == "multiplicative" and st.continuum_poly_x is not None:
        gp = gp * (1.0 + poly * st.continuum_poly_x)

    if apply_continuum and st.fit_als_continuum:
        cont = getattr(st, "continuum_mult", None)
        if cont is not None:
            gp = gp * cont

    return gp, b, w, coff, coff2, A


def objective_map(a: np.ndarray, st: FitState) -> float:
    """Scalar objective function minimised by the MAP LOSVD/template fit.

    Evaluates the forward model via :func:`evaluate_model_gp` and combines
    the chi2 goodness-of-fit with the LOSVD smoothness penalty and a soft
    normalization constraint:

        objective = chi2 + smoothness_penalty + 0.1 * |sum(b) - 1|

    where ``chi2`` is computed by :func:`_compute_chi2` over the good
    (unmasked, finite, non-clipped) pixels, and ``smoothness_penalty`` is
    the wing-tapered second-derivative roughness penalty on the LOSVD `b`
    computed by :func:`_compute_smoothness` with regularization strength
    ``st.xlam``. The final term softly enforces that the LOSVD histogram
    integrates to 1 (i.e. represents a normalized probability distribution)
    without a hard constraint, so the optimizer remains a simple bound-
    constrained problem solvable by L-BFGS-B. This is the function (or,
    when ``use_jax_objective=True``, its JAX-autodiff counterpart built by
    :func:`_build_jax_objective_value_and_grad`) minimised in
    :mod:`kinextract.fitting`.

    Parameters
    ----------
    a : ndarray
        Flat trial parameter vector; see :func:`evaluate_model_gp` for its
        layout.
    st : FitState
        Fit state providing the data, errors, velocity grid, and
        regularization settings. Its ``ntot`` counter (total objective
        evaluations) and ``time_eval_model`` (cumulative model-evaluation
        wall time) are incremented as a side effect, for diagnostics.

    Returns
    -------
    float
        The scalar objective value to be minimised.
    """
    st.ntot += 1
    t0 = time.perf_counter()
    gp, b, *_ = evaluate_model_gp(a, st)
    if not hasattr(st, "time_eval_model"):
        st.time_eval_model = 0.0
    st.time_eval_model += time.perf_counter() - t0
    chi2 = _compute_chi2(st.g, gp, st.gerr, st.iskip, st.npix)
    smooth = _compute_smoothness(b, st.xl, st.xlam, st.sigl0, st.resd, st.nl)
    return float(chi2 + smooth + 1e-1 * abs(float(np.sum(b)) - 1.0))


def _build_jax_objective_value_and_grad(st: FitState):
    """Build a JAX-jitted value-and-gradient function for the MAP objective.

    Re-expresses the same objective computed by :func:`objective_map`
    (chi2 + wing-tapered smoothness penalty + LOSVD normalization penalty)
    as a pure JAX function of the trial parameter vector `a`, closing over
    the shape-dependent, effectively-static quantities from `st` (grid
    sizes, template matrix, LOSVD interpolation tables, pixel-scatter maps,
    fixed continuum offsets, ``xlam``/``sigl0``). ``jax.value_and_grad`` is
    then applied and the result wrapped in ``jax.jit``, so calling the
    returned function yields both the objective value and its exact
    (autodiff, not finite-difference) gradient with respect to `a` in a
    single traced/compiled kernel. This lets the L-BFGS-B optimizer in
    :mod:`kinextract.fitting` (``use_jax_objective=True``) use analytic
    gradients instead of scipy's default finite-difference approximation,
    giving a large speedup over both the finite-difference NumPy/Numba path
    and the legacy Fortran quasi-Newton solver (which also used
    finite-difference gradients).

    Parameters
    ----------
    st : FitState
        Fit state describing the problem shape and fixed inputs. Only its
        static structure (array shapes, dtype-relevant flags, and
        currently-fixed scalars such as ``coffi``/``coff2i``, ``xlam``,
        ``sigl0``) is baked into the compiled kernel; genuinely dynamic
        per-call data (the observed spectrum `g`, its errors, and the ALS
        continuum) are passed as explicit runtime arguments instead (see
        Notes).

    Returns
    -------
    callable or None
        A function ``f(a, g, gerr, cont) -> (value, grad)`` returning the
        objective value (float) and its gradient with respect to `a`
        (ndarray) as NumPy types, ready to hand to ``scipy.optimize.minimize``.
        Returns None if JAX is not installed.

    Notes
    -----
    Both `g` (observed spectrum) and `cont` (ALS continuum) are explicit
    arguments, not closure variables. This is critical for performance:

    * `cont` changes between ALS outer iterations, so making it explicit
      lets the same compiled kernel serve all outer iterations without
      recompilation.
    * `g` changes between bootstrap replicates, so making it explicit means
      the module-level ``_JAX_VG_CACHE`` (see :func:`_get_or_build_jax_vg`)
      can share the single compiled kernel across all 1000+ bootstrap
      draws. Without this, each replicate would trigger a fresh XLA
      compilation (30-120 s each).

    ``jax.value_and_grad`` differentiates with respect to the first
    positional argument (`a`) only; `g`, `gerr`, and `cont` are treated as
    non-differentiated dynamic inputs.
    """
    if jax is None or jnp is None:
        return None

    nl = int(st.nl)
    nt = int(st.nt)
    npix = int(st.npix)
    icoff = int(st.icoff)
    fit_global_amp = bool(st.fit_global_amp)
    fortran_template_mixture = bool(st.fortran_template_mixture)
    fit_als_continuum = bool(st.fit_als_continuum)

    t = jnp.asarray(st.t, dtype=jnp.float64)
    outside_tpl = jnp.asarray(st.outside_tpl, dtype=bool)
    losvd_j0 = jnp.asarray(st.losvd_j0, dtype=jnp.int32)
    losvd_j1 = jnp.asarray(st.losvd_j1, dtype=jnp.int32)
    losvd_w = jnp.asarray(st.losvd_w, dtype=jnp.float64)
    ip_map = jnp.asarray(st.ip_map, dtype=jnp.int32)
    ip_mask = jnp.asarray(st.ip_mask, dtype=bool)
    # gerr is NOT captured here; it is passed explicitly at call time so that
    # sigma-clip updates (st.gerr changes) and bootstrap replicates all reuse
    # the single compiled kernel without recompilation.

    coffi = float(st.coffi)
    coff2i = float(st.coff2i)
    xlam = float(st.xlam)
    sigl0 = float(st.sigl0)
    resd = float(st.resd)
    iskip = int(st.iskip)

    fit_mask = np.zeros(npix, dtype=bool)
    fit_mask[iskip:npix - iskip] = True
    fit_mask = jnp.asarray(fit_mask, dtype=bool)

    sfac = 1.8
    lam_vec = np.array([
        xlam * (max(abs(float(st.xl[j])) / sigl0, sfac) / sfac) ** 4
        if abs(float(st.xl[j])) > sfac * sigl0
        else xlam
        for j in range(nl)
    ], dtype=float)
    lam_vec = jnp.asarray(lam_vec, dtype=jnp.float64)

    # Keep 2D shape (npix, nlosvd) to avoid jnp.tile/repeat inside the jitted function.
    ip_safe = jnp.clip(jnp.asarray(ip_map, dtype=jnp.int32), 0, npix - 1)
    ip_mask_2d = jnp.asarray(ip_mask, dtype=jnp.float64)

    def _smooth_penalty(b):
        left = (b[1] - 2.0 * b[0]) ** 2
        right = (b[nl - 2] - 2.0 * b[nl - 1]) ** 2
        mid = (b[2:] - 2.0 * b[1:-1] + b[:-2]) ** 2
        terms = jnp.concatenate([jnp.array([left]), mid, jnp.array([right])])
        return jnp.sum(lam_vec * terms) / resd

    def _ynew_from_b(b):
        y2 = b[losvd_j0] + (b[losvd_j1] - b[losvd_j0]) * losvd_w
        s = jnp.sum(y2)
        sum_b = jnp.sum(b)
        scale = jnp.where(s != 0.0, sum_b / s, 1.0)
        return y2 * scale

    # g, gerr_dyn, and cont are explicit arguments so JAX traces them as dynamic
    # arrays reused across outer iterations and bootstrap draws without recompile.
    # jax.value_and_grad differentiates w.r.t. the first argument (a) only.
    def _objective(a, g, gerr_dyn, cont):
        i = 0
        b = jnp.maximum(a[i:i + nl], 1e-6)
        i += nl
        w = jnp.maximum(a[i:i + nt], 1e-12)
        i += nt

        if icoff == 0:
            coff = coffi
            coff2 = coff2i
        elif icoff == 1:
            coff = a[i]
            coff2 = a[i + 1]
            i += 2
        else:
            coff = coffi
            coff2 = a[i]
            i += 1

        A = a[i] if fit_global_amp else 1.0

        sum2 = jnp.sum(w)
        sum2 = jnp.where(sum2 != 0.0, sum2, 1.0)
        if fortran_template_mixture:
            tval = (t @ w) / sum2
        else:
            valid_t = jnp.where(outside_tpl[:, None], 0.0, t)
            s2o = sum2 - jnp.sum(outside_tpl[:, None] * w[None, :], axis=1)
            s2o = jnp.where(s2o == 0.0, 1.0, s2o)
            tval = jnp.sum(valid_t * w[None, :], axis=1) / s2o

        temp = (tval + coff) / (coff + 1.0) + coff2
        ynew = _ynew_from_b(b)

        # Gather-style scatter: (npix, nlosvd) outer product, then scatter to output.
        # Broadcasting avoids the large jnp.tile/repeat intermediates.
        contrib = temp[:, None] * ynew[None, :] * ip_mask_2d   # (npix, nlosvd)
        xs_contrib = ynew[None, :] * ip_mask_2d                 # (npix, nlosvd)
        gp = jnp.zeros(npix, dtype=jnp.float64).at[ip_safe].add(contrib)
        xs = jnp.zeros(npix, dtype=jnp.float64).at[ip_safe].add(xs_contrib)

        suml = jnp.sum(ynew)
        gp = jnp.where(xs != 0.0, gp * suml / xs, gp)
        gp = gp * A

        if fit_als_continuum:
            gp = gp * cont

        valid = (
            fit_mask
            & jnp.isfinite(g)
            & jnp.isfinite(gp)
            & jnp.isfinite(gerr_dyn)
            & (gerr_dyn > 0.0)
            & (gerr_dyn < 1.0e9)
        )

        resid = (g - gp) / gerr_dyn
        chi2 = jnp.sum(jnp.where(valid, resid * resid, 0.0))
        smooth = _smooth_penalty(b)
        norm_pen = 1e-1 * jnp.abs(jnp.sum(b) - 1.0)
        return chi2 + smooth + norm_pen

    obj_vg = jax.jit(jax.value_and_grad(_objective))

    def _value_and_grad_np(
        a_np: np.ndarray,
        g_np: np.ndarray,
        gerr_np: np.ndarray,
        cont_np: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        val, grad = obj_vg(
            jnp.asarray(a_np, dtype=jnp.float64),
            jnp.asarray(g_np, dtype=jnp.float64),
            jnp.asarray(gerr_np, dtype=jnp.float64),
            jnp.asarray(cont_np, dtype=jnp.float64),
        )
        return float(val), np.asarray(grad, dtype=float)

    return _value_and_grad_np


def objective_components(a: np.ndarray, st: FitState) -> dict:
    """Break down the MAP objective into its individual terms.

    Re-evaluates the forward model at `a` and reports the chi2, smoothness
    penalty, LOSVD normalization penalty, and their sum separately, rather
    than only the combined scalar returned by :func:`objective_map`. Useful
    for diagnosing whether a fit is dominated by data mismatch or by
    regularization, e.g. when tuning ``xlam`` or investigating a poor fit.

    Parameters
    ----------
    a : ndarray
        Trial parameter vector; see :func:`evaluate_model_gp` for its layout.
    st : FitState
        Fit state providing the data, errors, velocity grid, and
        regularization settings.

    Returns
    -------
    dict
        Dictionary with keys:

        ``"chi2"``
            The chi-squared term (float).
        ``"smooth"``
            The LOSVD smoothness penalty term (float).
        ``"losvd_norm_penalty"``
            The ``0.1 * |sum(b) - 1|`` normalization penalty term (float).
        ``"total"``
            Sum of the three terms above; equal to the value
            :func:`objective_map` would return for the same `a` and `st`.
        ``"smooth_over_chi2"``
            Ratio of the smoothness penalty to chi2, a quick diagnostic of
            how strongly regularization is influencing the fit relative to
            the data (``nan`` if chi2 is zero).
    """
    gp, b, *_ = evaluate_model_gp(a, st)
    chi2   = _compute_chi2(st.g, gp, st.gerr, st.iskip, st.npix)
    smooth = _compute_smoothness(b, st.xl, st.xlam, st.sigl0, st.resd, st.nl)
    fadd   = 1e-1 * abs(float(np.sum(b)) - 1.0)
    return {
        "chi2": float(chi2), "smooth": float(smooth),
        "losvd_norm_penalty": float(fadd),
        "total": float(chi2 + smooth + fadd),
        "smooth_over_chi2": float(smooth / chi2) if chi2 else np.nan,
    }


def compute_weighted_template_spectrum(st: FitState, w: np.ndarray) -> np.ndarray:
    """Compute the weighted mixture of stellar templates for given weights.

    Forms the (unconvolved, pre-LOSVD) weighted combination of the template
    library ``st.t`` using weights `w`, i.e. the same template-mixing step
    performed inside :func:`evaluate_model_gp` before LOSVD convolution.
    Used to reconstruct the best-fit "input" template spectrum for plotting
    or output, without having to re-run the full forward model.

    Parameters
    ----------
    st : FitState
        Fit state holding the template matrix ``st.t`` (shape
        ``(npix, nt)``) and the ``outside_tpl`` coverage mask.
    w : ndarray
        Template weights, length ``nt`` (as fit by the optimizer).

    Returns
    -------
    ndarray
        Weighted template spectrum, length ``npix``. If the weights sum to
        a non-positive or non-finite value, returns an array of ones (a
        flat fallback spectrum) rather than dividing by zero.

    Notes
    -----
    When ``st.fortran_template_mixture`` is False, pixels outside any
    template's coverage are excluded from the weighted sum via
    ``st.outside_tpl``, matching the corresponding logic in
    :func:`evaluate_model_gp`.
    """
    w = np.asarray(w, float)
    sum2 = float(np.sum(w))
    if sum2 <= 0 or not np.isfinite(sum2):
        return np.ones(st.npix)
    if st.fortran_template_mixture:
        return st.t @ w / sum2
    valid_t = np.where(st.outside_tpl[:, None], 0.0, st.t)
    s2o = np.where(
        (sum2 - (st.outside_tpl[:, None] * w[None, :]).sum(axis=1)) == 0,
        1.0,
        sum2 - (st.outside_tpl[:, None] * w[None, :]).sum(axis=1),
    )
    return (valid_t * w[None, :]).sum(axis=1) / s2o
