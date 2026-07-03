"""Fit-state container and LOSVD/convolution precomputation for kinextract.

This module defines :class:`FitState`, the dataclass that carries the
entire numerical state of a single spectral fit (observed spectrum,
template matrix, LOSVD velocity grid, and precomputed index/weight tables
used to evaluate the template-LOSVD convolution). It also provides the
functions that build those precomputed tables once per fit
(`precompute_losvd_interp`, `precompute_ip_map`) and a fast helper
(`getnlosvd_fast_from_b`) that maps LOSVD histogram amplitudes onto the
fine velocity grid used internally for the convolution, so that the
optimizer's inner loop (in :mod:`kinextract.numerics`) can avoid recomputing
this geometry on every objective-function evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# =============================================================================
# Section 7 - FitState dataclass and precomputation
# =============================================================================

@dataclass
class FitState:
    """Mutable container holding the full numerical state of one spectral fit.

    A single `FitState` instance is built once per spatial bin (spaxel or
    aperture) being fit and is threaded through every stage of the
    pipeline: the chi-squared objective function, the roughness-penalty
    regularization, the ALS continuum co-fit, and the final output
    writers. It bundles the observed spectrum, the stellar/SSP template
    matrix, the non-parametric LOSVD velocity grid, and precomputed index
    maps that make the template-LOSVD convolution fast to evaluate
    repeatedly inside the optimizer.

    The LOSVD itself is represented non-parametrically: as a histogram of
    `nl` amplitudes on the velocity grid `xl` (km/s), recovered by
    chi-squared minimization with a roughness penalty controlled by `xlam`,
    rather than fit as a Gauss-Hermite parametric form.

    Parameters and Attributes are documented together below since this is a
    plain dataclass (all constructor parameters become attributes).

    Attributes
    ----------
    x : ndarray, shape (npix,)
        Galaxy wavelength grid, in Angstroms (rest frame after any
        redshift correction), for the pixels included in the fit.
    g : ndarray, shape (npix,)
        Observed galaxy flux at each pixel of `x`.
    gerr : ndarray, shape (npix,)
        Per-pixel flux uncertainty. Pixels to be excluded from the fit
        (bad regions, emission lines, template-coverage gaps) are marked
        by setting the corresponding entry to the sentinel `BIG`
        (see :mod:`kinextract._utils`) rather than removed, so array
        shapes stay aligned across the pipeline.
    t : ndarray, shape (npix, nt)
        Stellar/SSP template matrix on the galaxy grid, one column per
        template, each normalized to a flux level of order 1 (see
        :func:`kinextract.templates.build_template_matrix_fortran`).
    outside_tpl : ndarray of bool, shape (npix,)
        True where no template in `t` covers the pixel's wavelength.
        Replaces the fragile ``t == 1.0`` sentinel used in the original
        Fortran-derived logic, which could misidentify genuine in-range
        pixels whose value happened to equal 1.0 exactly.
    xlam : float
        Roughness-penalty regularization strength applied to the
        recovered LOSVD. Larger values force a smoother (lower curvature)
        LOSVD at the cost of chi-squared fit quality.
    xl : ndarray, shape (nl,)
        LOSVD velocity-bin centers, in km/s. Defines the non-parametric
        histogram grid on which the LOSVD amplitudes `b` are recovered.
    sigl0 : float
        Fiducial/reference velocity dispersion (km/s) used to scale the
        roughness penalty and to set the default extent of `xl` when an
        explicit velocity range is not supplied.
    npix : int
        Number of wavelength pixels in the fit region (length of `x`, `g`,
        `gerr`, and the first axis of `t`).
    nl : int
        Number of LOSVD velocity bins (length of `xl`); the LOSVD is
        represented as an `nl`-bin histogram, not a parametric form.
    nt : int
        Number of stellar/SSP templates (columns of `t`).
    iskip : int
        Number of pixels to skip at each end of the spectrum when
        computing chi-squared/writing outputs (edge pixels affected by
        convolution boundary effects).
    icoff : int
        Continuum-offset handling mode used by the objective function:
        which of `coffi`/`coff2i` are held fixed vs. fit as free
        parameters (see :mod:`kinextract.numerics` for the exact mapping).
    nlosvd : int
        Number of points on the fine internal velocity grid (`vgrid0`)
        used to evaluate the LOSVD during convolution; typically much
        finer than `nl` for numerical accuracy, and computed from the
        pixel scale and a reference wavelength so the convolution kernel
        is well resolved.
    scale : float
        Wavelength step of the galaxy grid `x`, in Angstroms per pixel.
    c1 : float
        Fortran-convention pixel-index offset (``1 - x[0] / scale``) used
        together with `facnew0` to compute integer pixel shifts for the
        LOSVD convolution via `nint_fortran`, preserving exact
        correspondence with the legacy Fortran pixel-indexing scheme.
    resd : float
        Cube of the wavelength pixel scale (``scale ** 3``); enters the
        roughness-penalty normalization inherited from the original
        Fortran formulation.
    coffi, coff2i : float
        Initial/fixed continuum-offset and continuum-slope parameters,
        interpreted according to `icoff`.
    prenormalized : bool
        True if the input spectrum was already continuum-normalized on
        disk (a ``.norm`` file), in which case the ALS continuum co-fit is
        skipped.
    fortran_template_mixture : bool
        True to reproduce the original Fortran convention for how
        template weights are combined into the model spectrum.
    fit_global_amp : bool
        True to fit a single overall flux-normalization amplitude shared
        across all templates, in addition to per-template weights.
    vgrid0 : ndarray or None, shape (nlosvd,)
        Fine internal velocity grid used for LOSVD evaluation during
        convolution (see `nlosvd`).
    facnew0 : ndarray or None, shape (nlosvd,)
        Per-point conversion factor from velocity (on `vgrid0`) to
        fractional pixel shift, used to build `ip_map`.
    losvd_j0, losvd_j1 : ndarray or None, shape (nlosvd,)
        Lower/upper bin indices into `xl` bracketing each point of
        `vgrid0`, precomputed by `precompute_losvd_interp` for fast linear
        interpolation of the LOSVD histogram onto the fine grid.
    losvd_w : ndarray or None, shape (nlosvd,)
        Linear interpolation weight (0-1) for `losvd_j1` at each point of
        `vgrid0` (weight for `losvd_j0` is ``1 - losvd_w``).
    ip_map : ndarray or None, shape (npix, nlosvd)
        Precomputed pixel-shift index map for the template-LOSVD
        convolution: ``ip_map[i, j]`` is the (zero-based, Fortran-NINT
        rounded) pixel index that velocity point `j` maps flux from onto
        output pixel `i`. Built once per fit by `precompute_ip_map` since
        it depends only on the (fixed) wavelength grid and velocity grid,
        not on the trial LOSVD amplitudes.
    ip_mask : ndarray or None, shape (npix, nlosvd)
        Boolean mask marking which entries of `ip_map` fall inside the
        valid pixel range ``[0, npix)``; out-of-range contributions are
        excluded from the convolution sum.
    t_err : ndarray or None, shape (npix, nt)
        Per-pixel template flux uncertainty matrix, normalized like `t`.
    f_template : float
        Pooled median fractional template flux error across all templates
        and pixels, used as a floor/inflation term for effective flux
        errors in the fit.
    continuum_mult : ndarray or None, shape (npix,)
        Multiplicative continuum model (from the ALS baseline co-fit)
        applied to the template-only model spectrum to reproduce the
        observed continuum shape.
    continuum_mask, continuum_mask_init : ndarray or None
        Boolean masks marking pixels treated as continuum (vs.
        absorption-dominated) during the ALS baseline fit;
        `continuum_mask_init` freezes the mask determined at
        initialization when `als_abs_clean_init_only` is enabled.
    fit_als_continuum : bool
        True to co-fit an asymmetric-least-squares (ALS) continuum
        baseline alongside the LOSVD and template weights, rather than
        relying on a pre-normalized input spectrum.
    ntot : int
        Running count of objective-function evaluations for the current
        optimization, used for progress logging.
    """

    x: np.ndarray
    g: np.ndarray
    gerr: np.ndarray
    t: np.ndarray
    # outside_tpl: True where at least one template does not cover the pixel.
    # Replaces the fragile t_missing = (T == 1.0) sentinel from the original.
    outside_tpl: np.ndarray
    c1: float
    resd: float
    xlam: float
    xl: np.ndarray
    sigl0: float
    npix: int
    nl: int
    iskip: int
    nt: int
    icoff: int
    nlosvd: int
    scale: float
    coffi: float = 0.0
    coff2i: float = 0.0
    prenormalized: bool = False
    norm_extra: dict = field(default_factory=dict)
    fortran_template_mixture: bool = True
    fit_global_amp: bool = False
    vgrid0: Optional[np.ndarray] = None
    facnew0: Optional[np.ndarray] = None
    losvd_j0: Optional[np.ndarray] = None
    losvd_j1: Optional[np.ndarray] = None
    losvd_w: Optional[np.ndarray] = None
    ip_map: Optional[np.ndarray] = None
    ip_mask: Optional[np.ndarray] = None
    t_err: Optional[np.ndarray] = None          # template error matrix (npix × nt)
    f_template: float = 0.0                    # pooled median fractional template error
    als_caii_refine_shift_A: float = 0.0       # Ca II measured wavelength shift (masking only)
    emission_pre_mask: Optional[np.ndarray] = None   # pixels pre-masked for emission lines
    continuum_mult: Optional[np.ndarray] = None
    continuum_mask: Optional[np.ndarray] = None
    continuum_mask_init: Optional[np.ndarray] = None   # absorption mask from init (for als_abs_clean_init_only)
    fit_als_continuum: bool = False
    als_lam_current: Optional[float] = None
    als_p_current: Optional[float] = None
    als_records: list = field(default_factory=list)
    continuum_poly_mode: str = "none"
    continuum_poly_x: Optional[np.ndarray] = None
    continuum_poly_bound: float = 0.1
    ntot: int = 0

    def __getstate__(self):
        """Serialize FitState for multiprocessing by converting numpy arrays to tuples."""
        state = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                # Store as (bytes, shape, dtype) tuple for compact serialization
                state[k] = ("__ndarray__", v.tobytes(), v.shape, str(v.dtype))
            elif isinstance(v, dict):
                state[k] = v
            else:
                state[k] = v
        return state

    def __setstate__(self, state):
        """Deserialize FitState after unpickling, reconstructing numpy arrays."""
        for k, v in state.items():
            if isinstance(v, tuple) and len(v) == 4 and v[0] == "__ndarray__":
                # Reconstruct ndarray from (marker, bytes, shape, dtype)
                _, array_bytes, shape, dtype_str = v
                arr = np.frombuffer(array_bytes, dtype=np.dtype(dtype_str))
                self.__dict__[k] = arr.reshape(shape)
            else:
                self.__dict__[k] = v


def precompute_losvd_interp(st: FitState) -> None:
    """Precompute linear-interpolation indices/weights from `xl` onto `vgrid0`.

    The LOSVD is recovered as an `nl`-bin histogram on the coarse velocity
    grid `st.xl`, but the template-LOSVD convolution is evaluated on the
    finer internal grid `st.vgrid0`. This function precomputes, once per
    fit, the bracketing bin indices (`losvd_j0`, `losvd_j1`) and linear
    interpolation weight (`losvd_w`) needed to resample any trial LOSVD
    histogram `b` from `xl` onto `vgrid0` cheaply inside the optimizer's
    inner loop (see `getnlosvd_fast_from_b`), avoiding a fresh
    `searchsorted` call on every objective-function evaluation.

    Parameters
    ----------
    st : FitState
        Fit state to update in place. Requires `st.xl` and `st.vgrid0` to
        already be set. Populates `st.losvd_j0`, `st.losvd_j1`, and
        `st.losvd_w`.

    Returns
    -------
    None
        Results are stored directly on `st`.
    """
    j1 = np.searchsorted(st.xl, st.vgrid0, side="right")
    j0 = j1 - 1
    j0 = np.clip(j0, 0, len(st.xl) - 1)
    j1 = np.clip(j1, 0, len(st.xl) - 1)
    denom = st.xl[j1] - st.xl[j0]
    w = np.zeros_like(st.vgrid0, float)
    m = (denom != 0) & (j0 != j1)
    w[m] = (st.vgrid0[m] - st.xl[j0][m]) / denom[m]
    st.losvd_j0, st.losvd_j1, st.losvd_w = j0, j1, w


def precompute_ip_map(st: FitState) -> None:
    """Precompute the (npix, nlosvd) pixel-shift index map for convolution.

    For every combination of output pixel and fine-grid velocity point,
    computes which input pixel's template flux contributes to it once
    shifted by that velocity — i.e. the discretized template-LOSVD
    convolution kernel geometry. Because the wavelength grid `st.x` and
    velocity grid `st.vgrid0`/`st.facnew0` are fixed for the duration of a
    fit, this mapping is independent of the trial LOSVD amplitudes and is
    computed exactly once per fit rather than on every objective-function
    evaluation.

    Pixel indices are rounded using `kinextract.io.nint_fortran`
    (round-half-away-from-zero) rather than NumPy's default rounding, to
    exactly reproduce the legacy Fortran pipeline's pixel-indexing
    convention.

    Parameters
    ----------
    st : FitState
        Fit state to update in place. Requires `st.c1`, `st.x`, and
        `st.facnew0` to already be set. Populates `st.ip_map` and
        `st.ip_mask`.

    Returns
    -------
    None
        Results are stored directly on `st`: `st.ip_map` holds the
        (zero-based) pixel index for each (pixel, velocity-point) pair,
        and `st.ip_mask` marks which of those indices fall within the
        valid range ``[0, npix)``.
    """
    from .io import nint_fortran
    ip = nint_fortran(st.c1 + st.x[:, None] * st.facnew0[None, :]) - 1
    st.ip_map = ip
    st.ip_mask = (ip >= 0) & (ip < st.npix)


def getnlosvd_fast_from_b(st: FitState, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample a trial LOSVD histogram onto the fine internal velocity grid.

    Linearly interpolates the `nl`-bin LOSVD amplitude vector `b` (defined
    on `st.xl`) onto the finer grid `st.vgrid0` using the indices/weights
    precomputed by `precompute_losvd_interp`, then rescales the result so
    its total sum matches the sum of the original histogram `b`
    (preserving total LOSVD normalization/flux under resampling). This is
    the fast per-evaluation step used inside the optimizer to turn trial
    LOSVD amplitudes into the fine-grid profile needed for convolution with
    the template matrix.

    Parameters
    ----------
    st : FitState
        Fit state providing the precomputed interpolation tables
        (`losvd_j0`, `losvd_j1`, `losvd_w`) and the fine velocity-to-pixel
        conversion factors `facnew0`.
    b : ndarray, shape (nl,)
        Trial LOSVD histogram amplitudes on the coarse grid `st.xl`.

    Returns
    -------
    facnew0 : ndarray, shape (nlosvd,)
        Velocity-to-pixel-shift conversion factors on the fine grid
        (passed through unchanged from `st.facnew0`, returned alongside the
        resampled LOSVD for convenience at the call site).
    y2 : ndarray, shape (nlosvd,)
        LOSVD amplitudes `b` resampled onto the fine grid `st.vgrid0` and
        renormalized to preserve the total sum of `b`.
    """
    b = np.asarray(b, float)
    y2 = b[st.losvd_j0] + (b[st.losvd_j1] - b[st.losvd_j0]) * st.losvd_w
    s = float(np.sum(y2))
    return st.facnew0, y2 * (float(np.sum(b)) / s if s else 1.0)
