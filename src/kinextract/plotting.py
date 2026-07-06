"""Diagnostic plots for kinextract spectral fits.

This module provides Matplotlib-based diagnostic plots for inspecting the
output of the main spectral-fitting pipeline: :func:`plot_fit` compares the
observed spectrum to the best-fit model and its residuals; :func:`plot_losvd`
shows the recovered non-parametric line-of-sight velocity distribution
(LOSVD) together with its post-hoc Gauss-Hermite (GH) characterization; and
:func:`plot_als_continuum` gives a detailed view of the Asymmetric
Least-Squares (ALS) continuum fit, including masked pixel regions and
labeled stellar/nebular absorption and emission features. The module-level
constant :data:`PROMINENT_STELLAR_LINES` is a curated reference table of
rest-frame wavelengths used to annotate these plots. These functions are
purely for visual inspection and do not modify any fit results.
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

from ._utils import BIG
from .continuum import _STELLAR_ABSORPTION_LINE_TABLES, _center_to_fit_frame
from .losvd import fit_losvd_gauss_hermite

# =============================================================================
# Shared plot style
# =============================================================================
#
# Color convention used across all plots in this module (not enforced by the
# rcParams color cycle below, since plots pass explicit `color=` per series --
# listed here so new panels stay consistent):
#   steelblue  - observed/input data
#   tomato     - MAP model / recovered fit
#   grey       - truth / reference / masked-out
#   darkcyan   - secondary metric (e.g. dispersion on a twin axis)
#
# Applied automatically on import so every caller -- the plot_* functions
# below, or a notebook doing its own Matplotlib calls -- gets consistent
# styling with no separate style-file lookup.
if plt is not None:
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(
            color=["steelblue", "tomato", "darkcyan", "grey", "goldenrod", "mediumpurple"]
        ),
    })


# =============================================================================
# Section 16 - Diagnostic plots
# =============================================================================

def _mask_to_spans(boolean_mask: np.ndarray, x: np.ndarray) -> list[tuple[float, float]]:
    """Convert a boolean pixel mask to a list of (x_lo, x_hi) contiguous spans."""
    spans = []
    i = 0
    n = len(boolean_mask)
    while i < n:
        if not boolean_mask[i]:
            i += 1
            continue
        i0 = i
        while i + 1 < n and boolean_mask[i + 1]:
            i += 1
        spans.append((float(x[i0]), float(x[i])))
        i += 1
    return spans


# Curated short list of the most prominent stellar features for reference overlays.
# Drawn as thin gray dashed lines when mark_prominent_lines=True.
# (center_A, label)
"""Reference table of prominent stellar and nebular line rest wavelengths.

A tuple of ``(center_wavelength_angstrom, label)`` pairs spanning the
optical/near-IR range typically covered by MUSE and STIS spectra
(H Balmer, Fe I/Lick indices, Mg b triplet, Na I D, Ca I/II, TiO molecular
bands, K I, O I, and common nebular emission lines such as [O III], H-alpha,
[N II], [S II]). Used by :func:`plot_als_continuum` (and available for use
by other diagnostic plots) to draw thin reference tick marks and labels at
these wavelengths, independent of which pixels are actually masked in a
given fit. Wavelengths are rest-frame air/vacuum values as conventionally
tabulated in the stellar-population and Lick-index literature; this is a
plotting aid, not an authoritative atomic line list.
"""
PROMINENT_STELLAR_LINES: tuple[tuple[float, str], ...] = (
    # ── H Balmer ─────────────────────────────────────────────────────────────
    (4861.33, r"H$\beta$"),
    # ── Fe I / blends (Lick indices) ─────────────────────────────────────────
    (5015.20, "Fe I"),          # Fe5015 Lick index
    (5270.40, "Fe I"),          # Fe5270 Lick index
    (5328.04, "Fe I"),          # Fe5335 Lick index
    (5406.00, "Fe I"),          # Fe5406 Lick index
    (5709.38, "Fe I"),          # Fe5709 Lick index
    (5780.38, "Fe I"),          # Fe5782 Lick index
    # ── Mg b triplet + MgH ───────────────────────────────────────────────────
    (5138.70, "MgH"),           # A²Π-X²Σ⁺ bandhead; gravity-sensitive
    (5167.32, "Mg b"),
    (5172.68, "Mg b"),
    (5183.60, "Mg b"),
    (5528.41, "Mg I"),
    # ── Na I D doublet — very strong in K giants; IMF-sensitive ─────────────
    (5889.95, "Na I D"),
    (5895.92, "Na I D"),
    # ── Ca I ─────────────────────────────────────────────────────────────────
    (6122.22, "Ca I"),
    (6162.17, "Ca I"),
    (6347.23, "Ca I"),
    (6373.48, "Ca I"),
    # ── Ba II (s-process; luminous K giants) ─────────────────────────────────
    (6496.90, "Ba II"),
    # ── TiO molecular bands ───────────────────────────────────────────────────
    # γ' system: ~5847, ~6159, ~6651 Å
    (5847.00, r"TiO $\gamma'$"),
    (6159.00, r"TiO $\gamma'$"),
    (6651.00, r"TiO $\gamma'$"),
    # γ system: 7054–7676 Å (strongest optical TiO complex)
    (7055.00, r"TiO $\gamma$"),
    (7126.00, r"TiO $\gamma$"),
    (7589.00, r"TiO $\gamma$"),
    (7676.00, r"TiO $\gamma$"),
    # δ system
    (8432.00, r"TiO $\delta$"),
    (8859.00, r"TiO $\delta$"),
    # ── K I resonance doublet — gravity-sensitive ─────────────────────────────
    (7664.91, "K I"),
    (7698.96, "K I"),
    # ── O I triplet (photospheric) ────────────────────────────────────────────
    (7771.94, "O I"),
    (7774.17, "O I"),
    (7775.39, "O I"),
    # ── Na I NIR doublet — IMF-sensitive ─────────────────────────────────────
    (8183.27, "Na I"),
    (8194.82, "Na I"),
    # ── Ca II infrared triplet ────────────────────────────────────────────────
    (8498.02, "Ca II"),
    (8542.09, "Ca II"),
    (8662.14, "Ca II"),
    # ── Fe I in CaT region ───────────────────────────────────────────────────
    (8514.08, "Fe I"),
    (8621.61, "Fe I"),
    (8688.63, "Fe I"),
    # ── Mg I ─────────────────────────────────────────────────────────────────
    (8806.76, "Mg I"),
    # ── Near-IR: Pa7, Ca I, CN ───────────────────────────────────────────────
    (9015.00, "Pa8"),
    (9229.00, r"Pa$\delta$"),
    (9257.00, "Ca I"),
    (9350.00, "CN"),
    # ── Common nebular emission ───────────────────────────────────────────────
    (5006.84, "[O III]"),
    (5875.62, "He I"),
    (6300.30, "[O I]"),
    (6562.80, r"H$\alpha$"),
    (6583.45, "[N II]"),
    (6716.44, "[S II]"),
    (6730.82, "[S II]"),
    (7135.80, "[Ar III]"),
    (9068.60, "[S III]"),
)


def plot_fit(fit: dict) -> None:
    """Plot the observed spectrum against the best-fit model and residuals.

    Draws a two-panel figure: the top panel overlays the observed spectrum
    (``fit["state"].g``) and the best-fit model spectrum
    (``fit["outputs"]["gp"]``), marking any masked/rejected pixels
    (``gerr >= 1e9``) in orange; the bottom panel shows the fractional
    residual ``(data - model) / model`` with the RMS (computed over good
    pixels only) annotated and marked with dashed reference lines. This is
    the primary quick-look diagnostic for judging overall fit quality.

    Parameters
    ----------
    fit : dict
        Output of ``run_spectral_fit()``. Must contain ``fit["state"]`` (a
        :class:`~kinextract.state.FitState` with the observed spectrum ``g``,
        wavelength grid ``x``, and per-pixel error ``gerr``) and
        ``fit["outputs"]["gp"]`` (the best-fit model spectrum).

    Returns
    -------
    None
        Displays the figure via ``plt.show()``; does not return a value.
    """
    st = fit["state"]
    gp = fit["outputs"]["gp"]
    good = st.gerr < 1e9
    resid = (st.g - gp) / gp

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax = axes[0]
    ax.plot(st.x, st.g, "k", lw=1, label="data")
    ax.plot(st.x, gp, color="tab:red", lw=1.5, label="model")
    if (~good).any():
        ax.scatter(st.x[~good], st.g[~good], s=12, color="tab:orange",
                   label="masked", zorder=5)
    ax.set_ylabel("Normalised flux" if st.prenormalized else "Flux")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_ylim(st.g.min() * 0.9, st.g.max() * 1.1)

    ax = axes[1]
    ax.axhline(0, color="0.5", lw=1)
    ax.plot(st.x, resid, color="tab:blue", lw=1)
    if good.any():
        rms = np.sqrt(np.mean(resid[good] ** 2))
        ax.axhline( rms, color="0.7", ls="--", lw=1)
        ax.axhline(-rms, color="0.7", ls="--", lw=1)
        if (~good).any():
            ax.scatter(st.x[~good], resid[~good], s=12, color="tab:orange",
                       label="masked", zorder=5)
        ax.text(0.01, 0.95, f"RMS={rms:.4g}", transform=ax.transAxes, va="top")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("(Data - model) / model")
    ax.set_ylim(-0.05, 0.05)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_losvd(fit: dict) -> None:
    """Plot the recovered non-parametric LOSVD with its Gauss-Hermite overlay.

    Left panel: the non-parametric LOSVD ``b`` (histogram over velocity bins
    ``xl``, km/s) recovered by the main fit, overlaid with a post-hoc
    Gauss-Hermite (GH) fit computed via
    :func:`kinextract.losvd.fit_losvd_gauss_hermite` (legacy fortran H3/H4
    convention). The fitted mean velocity V is marked with a vertical dashed
    line, and the panel is annotated with the GH moments (V, sigma, h3, h4)
    alongside the simpler flux-weighted moments (v1, v2) for comparison.
    Right panel: the fractional weight assigned to each stellar template in
    the fit (``fit["outputs"]["wfrac"]``), useful for spotting templates
    that dominate or are unused. A one-line text summary of the GH fit is
    also printed to stdout.

    Parameters
    ----------
    fit : dict
        Output of ``run_spectral_fit()``. Must contain ``fit["state"].xl``
        (the LOSVD velocity grid, km/s), ``fit["outputs"]["b"]`` (the
        recovered LOSVD amplitudes), and ``fit["outputs"]["wfrac"]``
        (per-template weight fractions).

    Returns
    -------
    None
        Displays the figure via ``plt.show()`` and prints a GH moment
        summary to stdout; does not return a value.
    """
    st = fit["state"]
    b = fit["outputs"]["b"]
    wfrac = fit["outputs"]["wfrac"]
    gh = fit_losvd_gauss_hermite(st.xl, b, fit_h3h4=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(st.xl, b, "o-", color="tab:blue", label="LOSVD")
    if np.all(np.isfinite(gh["model"])):
        ax.plot(st.xl, gh["model"], color="tab:red", lw=1.8, label="GH fit")
    ax.axvline(gh["vherm"], color="tab:red", ls="--", lw=1.3)

    annotation = (
        rf"$V = {gh['vherm']:.2f}$, $\sigma = {gh['sherm']:.2f}$" "\n"
        rf"$h_3 = {gh['h3']:.3f}$, $h_4 = {gh['h4']:.3f}$" "\n"
        rf"(moments: $V = {gh['v1']:.2f}$, $\sigma = {gh['v2']:.2f}$)"
    )
    ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.4", alpha=0.85))
    ax.set_xlabel("Velocity [km/s]")
    ax.set_ylabel("LOSVD")
    ax.legend()
    ax.grid(alpha=0.25)

    axes[1].bar(np.arange(1, len(wfrac) + 1), wfrac, color="tab:gray")
    axes[1].set_xlabel("Template index")
    axes[1].set_ylabel("Weight fraction")
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_losvd_posterior(fit: dict, confidence: float = 0.68, max_gh_draws: int = 300) -> dict:
    """Plot the recovered LOSVD with posterior credible-interval error bars.

    The optional Bayesian fit path (:func:`kinextract.bayesian.fit_state_bayesian`,
    not used by default -- see that function's module docstring for when
    to reach for it instead of the default MAP+bootstrap pipeline) reports
    a full posterior over the LOSVD histogram, not just a point estimate --
    this is the direct, native way to see uncertainty on it, as an
    alternative to the bootstrap/Laplace-based
    ``kinextract.errors.plot_losvd_with_errors`` for fits run through that
    (NUTS-based) path. Left panel: the posterior mean LOSVD
    (matching ``fit["outputs"]["b"]``) with a shaded
    ``confidence``-fraction credible band (from the raw posterior draws,
    not a Gaussian approximation) and a scatter of per-draw
    Gauss-Hermite (V, sigma) points to show how correlated the kinematic
    uncertainty is. Right panel: posterior mean/credible-interval error
    bars on the Gauss-Hermite moments (V, sigma, h3, h4) themselves,
    computed by re-fitting :func:`kinextract.losvd.fit_losvd_gauss_hermite`
    to each posterior draw individually (not just the mean LOSVD) --
    this correctly propagates LOSVD-shape uncertainty into the moments
    rather than assuming they're independent.

    Parameters
    ----------
    fit : dict
        Output of :func:`kinextract.bayesian.fit_state_bayesian` (called
        directly; ``run_spectral_fit()``'s default MAP output has no
        posterior). Must contain ``fit["state"].xl`` and
        ``fit["result"].mcmc`` (the NUTS ``numpyro.infer.MCMC`` object).
    confidence : float, optional
        Credible-interval width, e.g. 0.68 (~1-sigma) or 0.95.
    max_gh_draws : int, optional
        Cap on how many posterior draws get an individual Gauss-Hermite
        fit (each is itself a small optimization; fitting every draw of a
        large posterior is unnecessary for a stable uncertainty estimate
        and would be slow). Draws are evenly subsampled if there are more
        than this.

    Returns
    -------
    dict
        ``{"V": (mean, lo, hi), "sigma": (...), "h3": (...), "h4": (...)}``
        where ``lo``/``hi`` are the ``confidence``-fraction credible
        interval bounds, for programmatic use alongside the plot.

    Raises
    ------
    ValueError
        If ``fit["result"]`` has no ``mcmc`` attribute (e.g. it came from
        the MAP path, which has no posterior to draw error bars from --
        use :func:`plot_losvd` for that case instead).
    """
    st = fit["state"]
    mcmc = getattr(fit["result"], "mcmc", None)
    if mcmc is None:
        raise ValueError(
            "fit['result'] has no posterior (mcmc is None) -- plot_losvd_posterior "
            "only works for fits from kinextract.bayesian.fit_state_bayesian. Use "
            "plot_losvd() for a point-estimate-only fit, e.g. from the default "
            "MAP+bootstrap pipeline (run_spectral_fit())."
        )

    samples = mcmc.get_samples()
    b_draws = np.asarray(samples["b"])  # (n_draws, nl)
    n_draws = b_draws.shape[0]

    alpha = (1.0 - confidence) / 2.0
    b_mean = b_draws.mean(axis=0)
    b_lo = np.quantile(b_draws, alpha, axis=0)
    b_hi = np.quantile(b_draws, 1.0 - alpha, axis=0)

    idx = np.linspace(0, n_draws - 1, min(n_draws, max_gh_draws)).astype(int)
    moments = {"vherm": [], "sherm": [], "h3": [], "h4": []}
    for i in idx:
        gh_i = fit_losvd_gauss_hermite(st.xl, b_draws[i], fit_h3h4=True)
        if gh_i["fit_success"]:
            moments["vherm"].append(gh_i["vherm"])
            moments["sherm"].append(gh_i["sherm"])
            moments["h3"].append(gh_i["h3"])
            moments["h4"].append(gh_i["h4"])

    def _summarize(key):
        arr = np.asarray(moments[key])
        return float(np.mean(arr)), float(np.quantile(arr, alpha)), float(np.quantile(arr, 1.0 - alpha))

    summary = {
        "V": _summarize("vherm"), "sigma": _summarize("sherm"),
        "h3": _summarize("h3"), "h4": _summarize("h4"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.fill_between(st.xl, b_lo, b_hi, color="tab:blue", alpha=0.25,
                     label=f"{confidence:.0%} credible interval")
    ax.plot(st.xl, b_mean, "o-", color="tab:blue", label="Posterior mean LOSVD")
    ax.axvline(summary["V"][0], color="tab:red", ls="--", lw=1.3)
    annotation = (
        rf"$V = {summary['V'][0]:.2f}^{{+{summary['V'][2] - summary['V'][0]:.2f}}}"
        rf"_{{-{summary['V'][0] - summary['V'][1]:.2f}}}$" "\n"
        rf"$\sigma = {summary['sigma'][0]:.2f}^{{+{summary['sigma'][2] - summary['sigma'][0]:.2f}}}"
        rf"_{{-{summary['sigma'][0] - summary['sigma'][1]:.2f}}}$" "\n"
        rf"$h_3 = {summary['h3'][0]:.3f}$, $h_4 = {summary['h4'][0]:.3f}$"
        f" ({len(moments['vherm'])}/{len(idx)} draws)"
    )
    ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.4", alpha=0.85))
    ax.set_xlabel("Velocity [km/s]")
    ax.set_ylabel("LOSVD")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    names = ["V", "sigma", "h3", "h4"]
    means = [summary[n][0] for n in names]
    lo_err = [summary[n][0] - summary[n][1] for n in names]
    hi_err = [summary[n][2] - summary[n][0] for n in names]
    ax.errorbar(range(len(names)), means, yerr=[lo_err, hi_err], fmt="o",
                color="tab:blue", capsize=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"${n}$" if n in ("h3", "h4") else n for n in names])
    ax.set_ylabel("Value")
    ax.set_title(f"Gauss-Hermite moments ({confidence:.0%} credible interval)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    print(
        f"V={summary['V'][0]:.4f} [{summary['V'][1]:.4f}, {summary['V'][2]:.4f}]  "
        f"sigma={summary['sigma'][0]:.4f} [{summary['sigma'][1]:.4f}, {summary['sigma'][2]:.4f}]  "
        f"h3={summary['h3'][0]:.4f} [{summary['h3'][1]:.4f}, {summary['h3'][2]:.4f}]  "
        f"h4={summary['h4'][0]:.4f} [{summary['h4'][1]:.4f}, {summary['h4'][2]:.4f}]"
    )
    return summary


def plot_als_continuum(
    fit: dict,
    cfg=None,
    mark_prominent_lines: bool = True,
) -> None:
    """Diagnostic plot for Asymmetric Least-Squares (ALS) continuum fits.

    Produces a three-panel figure for fits run with ``cfg.fit_als_continuum
    = True``: the top panel shows the observed spectrum, the full best-fit
    model, and the fitted ALS continuum baseline, with shaded spans marking
    detected-emission and detected-absorption (ALS) pixel regions and
    labeled reference lines; the middle panel shows the same data and model
    divided by the ALS continuum (continuum-normalized view, useful for
    judging line depths independent of the baseline shape); the bottom
    panel shows fractional residuals ``(data - model) / model``. Individual
    rejected pixels (masked for reasons other than detected emission, e.g.
    bad regions, template coverage gaps, or sigma-clipping) are overplotted
    as translucent red points in all three panels. If ``fit["state"]`` was
    not run in ALS continuum mode, this function prints a notice and
    returns without plotting.

    Parameters
    ----------
    fit : dict
        Output of ``run_spectral_fit()``. Must contain ``fit["state"]``,
        ``fit["outputs"]["gp"]`` (model spectrum), and
        ``fit["outputs"]["continuum"]`` (fitted ALS continuum baseline).
    cfg : FitConfig, optional
        When supplied, the plot additionally overlays masked regions and
        line labels for detected emission, ALS-detected absorption, and any
        user-specified extra mask windows/lines. Labels are drawn ONLY for
        lines whose pixels are actually present in the corresponding mask,
        so labeling is always consistent with the shaded regions shown.
        If omitted, only the curated prominent-line reference ticks (see
        ``mark_prominent_lines``) are drawn.
    mark_prominent_lines : bool, optional
        When True (default), draw thin gray reference ticks and labels for
        the curated list of strongest stellar/nebular lines
        (:data:`PROMINENT_STELLAR_LINES`) that fall within the fit
        wavelength window, regardless of masking status.

    Returns
    -------
    None
        Displays the figure via ``plt.show()``; does not return a value.
    """
    st = fit["state"]
    if not st.fit_als_continuum:
        print("Not in ALS continuum mode — nothing to plot.")
        return

    gp = fit["outputs"]["gp"]
    cont = fit["outputs"]["continuum"]
    good = st.gerr < 1e9

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1]},
    )

    # ── Build mask layers ────────────────────────────────────────────────────
    # all pixels with gerr >= BIG (after full fit: bad regions, emission pre-mask,
    # template gaps, sigma-clipped outliers).
    pre_masked = st.gerr >= BIG

    # Emission pre-mask (stored by make_fit_state before any fitting).
    em_mask_bool = getattr(st, "emission_pre_mask", None)
    if em_mask_bool is None:
        em_mask_bool = np.zeros(st.npix, dtype=bool)
    else:
        em_mask_bool = np.asarray(em_mask_bool, bool)

    # ALS continuum absorption mask: pixels excluded from ALS but still in spectral fit.
    cont_mask = getattr(st, "continuum_mask", None)
    if cont_mask is not None:
        als_abs_excl = ~np.asarray(cont_mask, bool) & ~pre_masked
    else:
        als_abs_excl = np.zeros(st.npix, dtype=bool)

    # "Other rejected" pixels: pre-masked for reasons other than detected emission
    # (bad regions, template coverage gaps, sigma-clipped outliers from fitting).
    other_rejected = pre_masked & ~em_mask_bool

    # Panel 0 + 1: emission pre-mask (red)
    for x0, x1 in _mask_to_spans(em_mask_bool, st.x):
        axes[0].axvspan(x0, x1, color="tab:red", alpha=0.22, zorder=0,
                        label="_nolegend_")
        axes[1].axvspan(x0, x1, color="tab:red", alpha=0.22, zorder=0,
                        label="_nolegend_")
    # Panel 0 only: ALS absorption mask (purple)
    for x0, x1 in _mask_to_spans(als_abs_excl, st.x):
        axes[0].axvspan(x0, x1, color="tab:purple", alpha=0.22, zorder=0,
                        label="_nolegend_")

    # ── Helper: is this line center actually represented in a mask? ──────────
    def _line_is_masked(cen: float, hw: float, mask: np.ndarray) -> bool:
        pix = (st.x >= cen - hw) & (st.x <= cen + hw)
        return bool(pix.any() and mask[pix].any())

    # ── Collect line markers — ONLY for lines whose pixels are in the mask ───
    # Labels are tied to the actual mask arrays so they only appear when
    # there is corresponding shading.
    abs_lines:   list[tuple[str, float]] = []   # (label, center_rest)
    em_lines:    list[tuple[str, float]] = []
    extra_lines: list[tuple[str, float]] = []

    if cfg is not None:
        wmin = float(getattr(cfg, "wavefitmin", st.x.min()))
        wmax = float(getattr(cfg, "wavefitmax", st.x.max()))
        em_hw  = float(getattr(cfg, "mask_emission_line_half_width_A", 5.0))
        abs_hw = float(getattr(cfg, "als_auto_mask_half_width_A", 5.0))

        # Frame-conversion helper: apply the same als_mask_center_shift_A that
        # the masking functions use, so all plotted lines land in the same place.
        def _to_plot_frame(cen_rest: float) -> float:
            return _center_to_fit_frame(cen_rest, "rest", cfg)

        # Emission lines: label only if pixel actually in pre_masked.
        if getattr(cfg, "mask_emission_lines_in_fit", True):
            seen_em: set[float] = set()
            mask_paschen = bool(getattr(cfg, "mask_paschen_lines_in_fit", False))

            for table in _STELLAR_ABSORPTION_LINE_TABLES:
                for tag, name, cen_rest in table:
                    is_emission_candidate = tag == "em"
                    is_paschen_candidate = mask_paschen and tag == "paschen"

                    if not (is_emission_candidate or is_paschen_candidate):
                        continue

                    if cen_rest in seen_em:
                        continue

                    if cen_rest < wmin - em_hw or cen_rest > wmax + em_hw:
                        continue

                    cen_plot = _to_plot_frame(cen_rest)

                    if _line_is_masked(cen_plot, em_hw, pre_masked):
                        em_lines.append((name, cen_plot))
                        seen_em.add(cen_rest)

        # ALS absorption lines: label only if pixel actually in als_abs_excl.
        if getattr(cfg, "als_auto_mask_abs_lines", True) and als_abs_excl.any():
            mask_paschen = bool(getattr(cfg, "als_auto_mask_paschen", False))
            seen_abs: set[float] = set()
            for table in _STELLAR_ABSORPTION_LINE_TABLES:
                for tag, name, cen_rest in table:
                    if tag == "em" or cen_rest in seen_abs:
                        continue
                    if tag == "paschen" and not mask_paschen:
                        continue
                    if cen_rest < wmin or cen_rest > wmax:
                        continue
                    cen_plot = _to_plot_frame(cen_rest)
                    if _line_is_masked(cen_plot, abs_hw, als_abs_excl):
                        abs_lines.append((name, cen_plot))
                        seen_abs.add(cen_rest)

        # Extra user lines: label only if pixel in protect_extra or als_abs_excl.
        # protect_extra covers the user-supplied als_extra_mask_windows.
        protect_extra = np.zeros(st.npix, dtype=bool)
        extra_frame = getattr(cfg, "als_extra_mask_frame", "rest")
        for lo_raw, hi_raw in getattr(cfg, "als_extra_mask_windows", ()):
            lo = float(lo_raw) / (1.0 + cfg.zgal) if extra_frame == "observed" else float(lo_raw)
            hi = float(hi_raw) / (1.0 + cfg.zgal) if extra_frame == "observed" else float(hi_raw)
            protect_extra |= (st.x >= lo) & (st.x <= hi)
        seen_extra: set[float] = set()
        extra_combined = protect_extra | als_abs_excl
        extra_line_frame = getattr(cfg, "als_extra_mask_line_frame", "rest")
        for item in getattr(cfg, "als_extra_mask_lines", ()):
            c_raw = float(item[0])
            name = item[2] if len(item) >= 3 else f"{c_raw:.1f} Å"
            c_rest = c_raw / (1.0 + cfg.zgal) if extra_line_frame == "observed" else c_raw
            hw = float(item[1]) if len(item) >= 2 else abs_hw
            c_plot = _to_plot_frame(c_rest)
            if c_rest not in seen_extra and _line_is_masked(c_plot, hw, extra_combined):
                extra_lines.append((name, c_plot))
                seen_extra.add(c_rest)
        for item in getattr(cfg, "extra_absorption_lines", ()):
            c_raw = float(item[0])
            hw = float(item[1]) if len(item) >= 2 else abs_hw
            name = item[2] if len(item) >= 3 else f"{c_raw:.1f} Å"
            c_plot = _to_plot_frame(c_raw)
            if c_raw not in seen_extra and _line_is_masked(c_plot, hw, extra_combined):
                extra_lines.append((name, c_plot))
                seen_extra.add(c_raw)

    else:
        def _to_plot_frame(cen_rest: float) -> float:  # type: ignore[misc]
            return cen_rest

    # ── Prominent reference lines (independent of masking) ──────────────────
    ref_lines: list[tuple[str, float]] = []
    if mark_prominent_lines:
        xlo, xhi = float(st.x.min()), float(st.x.max())
        # Avoid duplicating lines already shown as masked labels
        already_labeled = {c for _, c in abs_lines + em_lines + extra_lines}
        for cen_rest, name in PROMINENT_STELLAR_LINES:
            cen_plot = _to_plot_frame(cen_rest)
            if xlo <= cen_plot <= xhi and cen_plot not in already_labeled:
                ref_lines.append((name, cen_plot))

    # ── Data, model, continuum (top panel) ──────────────────────────────────
    ax = axes[0]
    _show_errs = getattr(cfg, "use_spectrum_errors", True)
    ax.plot(st.x, st.g, "k", lw=0.8, label="data")
    if _show_errs:
        gerr_plot = np.where(st.gerr < BIG, st.gerr, np.nan)
        ax.fill_between(st.x, st.g - gerr_plot, st.g + gerr_plot,
                        color="0.8", step="mid", alpha=0.5, label="error")
    ax.plot(st.x, gp, color="C0", lw=1.5, label="model")
    ax.plot(st.x, cont, color="tab:orange", lw=1.2, ls="--", label="ALS continuum")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_patches = []
    if em_mask_bool.any():
        legend_patches.append(Patch(color="tab:red", alpha=0.4, label="Detected Emission"))
    if als_abs_excl.any():
        legend_patches.append(Patch(color="tab:purple", alpha=0.4, label="Detected Absorption (ALS)"))
    if other_rejected.any():
        legend_patches.append(Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="tab:red",
            markersize=4, alpha=0.5, label="rejected pixels",
        ))
    if ref_lines:
        legend_patches.append(Line2D([0], [0], color="0.65", lw=0.7, ls="--",
                                     label="prominent lines (ref)"))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_patches,
              labels=labels + [p.get_label() for p in legend_patches],
              fontsize=8, loc="lower right", frameon=True, fancybox=True, framealpha=0.85)
    ax.set_ylabel("Flux")
    ax.grid(alpha=0.25)
    good_finite = good & np.isfinite(st.g)
    ylo = float(np.nanmin(st.g[good_finite])) if good_finite.any() else float(st.g.min())
    yhi = float(np.nanmax(st.g[good_finite])) if good_finite.any() else float(st.g.max())
    ax.set_ylim(ylo * 0.9, yhi * 1.1)

    for name, cen in ref_lines:
        ax.axvline(cen, color="0.60", lw=0.7, ls="--", alpha=0.7, zorder=1)
        ax.text(cen, yhi * 1.05, name, fontsize=5, rotation=90,
                ha="center", va="bottom", color="0.45", clip_on=True)
    for name, cen in em_lines:
        ax.axvline(cen, color="tab:red", lw=0.8, ls=":", alpha=0.85, zorder=3)
        ax.text(cen, yhi * 1.03, name, fontsize=5, rotation=90,
                ha="center", va="bottom", color="tab:red", clip_on=True)
    for name, cen in abs_lines:
        ax.axvline(cen, color="tab:purple", lw=0.8, ls=":", alpha=0.8, zorder=3)
        ax.text(cen, yhi * 1.03, name, fontsize=5, rotation=90,
                ha="center", va="bottom", color="tab:purple", clip_on=True)
    for name, cen in extra_lines:
        ax.axvline(cen, color="tab:green", lw=0.8, ls="--", alpha=0.85, zorder=3)
        ax.text(cen, yhi * 1.05, name, fontsize=5, rotation=90,
                ha="center", va="bottom", color="tab:green", clip_on=True)

    # ── Continuum-normalised (middle panel) ──────────────────────────────────
    ax = axes[1]
    cont_safe = np.where(np.abs(cont) > 0, cont, np.nan)
    g_norm = st.g / cont_safe
    ax.plot(st.x, g_norm, "k", lw=0.8, label="data / ALS continuum")
    ax.plot(st.x, gp / cont_safe, color="C0", lw=1.5, label="model / ALS continuum")
    if _show_errs:
        ax.fill_between(st.x, (st.g - gerr_plot) / cont_safe,
                        (st.g + gerr_plot) / cont_safe,
                        color="0.8", step="mid", alpha=0.5)
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    good_norm = good & np.isfinite(g_norm)
    ylo_n = float(np.nanmin(g_norm[good_norm])) if good_norm.any() else 0.8
    yhi_n = float(np.nanmax(g_norm[good_norm])) if good_norm.any() else 1.2
    ax.set_ylim(ylo_n * 0.95, yhi_n * 1.05)
    ax.set_ylabel("Continuum-normalised")

    em1_patches: list = []
    if em_mask_bool.any():
        em1_patches.append(Patch(color="tab:red", alpha=0.4,
                                 label="Detected Emission (spectral fitting)"))
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(handles=h1 + em1_patches,
              labels=l1 + [p.get_label() for p in em1_patches],
              fontsize=8)
    ax.grid(alpha=0.25)
    for _, cen in ref_lines:
        ax.axvline(cen, color="0.65", lw=0.7, ls="--", alpha=0.6, zorder=1)
    for name, cen in em_lines:
        ax.axvline(cen, color="tab:red", lw=0.8, ls=":", alpha=0.85, zorder=3)
        ax.text(cen, yhi_n * 1.03, name, fontsize=5, rotation=90,
                ha="center", va="bottom", color="tab:red", clip_on=True)
    for _, cen in abs_lines:
        ax.axvline(cen, color="tab:purple", lw=0.8, ls=":", alpha=0.8, zorder=3)
    for _, cen in extra_lines:
        ax.axvline(cen, color="tab:green",  lw=0.8, ls="--", alpha=0.8, zorder=3)

    # ── Residuals (bottom panel) ─────────────────────────────────────────────
    ax = axes[2]
    resid = (st.g - gp) / np.where(np.abs(gp) > 0, gp, np.nan)
    ax.plot(st.x, resid, color="C0", lw=0.8)
    ax.axhline(0, color="0.5", ls="--", lw=1)
    if good.any():
        rms = np.sqrt(np.nanmean(resid[good] ** 2))
        ax.text(0.01, 0.95, f"RMS={rms:.4g}", transform=ax.transAxes, va="top")
    ax.set_xlabel(r"Wavelength ($\AA$, rest frame)")
    ax.set_ylabel(r"(Data $-$ model) / model")
    ax.grid(alpha=0.25)
    ax.set_ylim(-0.1, 0.1)
    for _, cen in ref_lines:
        ax.axvline(cen, color="0.65", lw=0.7, ls="--", alpha=0.6, zorder=1)
    for _, cen in em_lines:
        ax.axvline(cen, color="tab:red",    lw=0.8, ls=":", alpha=0.8, zorder=3)
    for _, cen in abs_lines:
        ax.axvline(cen, color="tab:purple", lw=0.8, ls=":", alpha=0.8, zorder=3)
    for _, cen in extra_lines:
        ax.axvline(cen, color="tab:green",  lw=0.8, ls="--", alpha=0.8, zorder=3)

    # ── Rejected-pixel scatter (all 3 panels) ────────────────────────────────
    # Show every pixel that is flagged as bad (not part of a labeled ALS
    # absorption or detected emission region) as a red dot so it's clear
    # which individual pixels were removed from the fit.
    if other_rejected.any():
        rej_x = st.x[other_rejected]
        rej_g = st.g[other_rejected]
        rej_gn = rej_g / cont_safe[other_rejected]
        rej_r  = resid[other_rejected]
        scatter_kw = dict(s=10, color="tab:red", alpha=0.35, zorder=4,
                          linewidths=0, label="_nolegend_")
        axes[0].scatter(rej_x, np.clip(rej_g,  ylo, yhi),    **scatter_kw)
        axes[1].scatter(rej_x, np.clip(rej_gn, ylo_n, yhi_n), **scatter_kw)
        axes[2].scatter(rej_x, np.clip(rej_r, -0.1,  0.1),   **scatter_kw)

    plt.tight_layout()
    plt.show()
