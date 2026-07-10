"""
Command-line entry point: ``python -m kinextract <spectrum> [config.toml]``.

Runs the full non-parametric LOSVD fit on a single spectrum using
:func:`~kinextract.fitting.run_spectral_fit`, then displays the standard
diagnostic plots (spectral fit or co-fit continuum, plus the recovered
LOSVD). Intended for quick interactive checks of a single spectrum; for
batch processing or notebook use, call :func:`~kinextract.run_spectral_fit`
directly with a :class:`~kinextract.FitConfig` instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ._utils import log
from .config import load_config_from_toml
from .fitting import run_spectral_fit

# =============================================================================
# Section 17 - Entry point
# =============================================================================

def main() -> None:
    """Parse CLI arguments, run the fit, and show the diagnostic plots.

    Reads the spectrum path from ``sys.argv[1]`` and an optional TOML
    config path from ``sys.argv[2]`` (defaulting to ``kinextract.config``
    next to the spectrum file). Prints usage and exits with status 1 if no
    spectrum path is given.
    """
    if len(sys.argv) < 2:
        print(
            "Usage: python kinextract.py </your/path/spectrum_file> [</your/path/config_file>]\n"
            "\n"
            "  The TOML config file sets all fit parameters.\n"
            "  The spectrum file is passed separately on the command line.\n"
            "  If no config file is supplied, kinextract.config is searched for\n"
            "  in the same directory as the spectrum file.\n"
        )
        sys.exit(1)

    spectrum_path = sys.argv[1]

    config_path = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else str(Path(spectrum_path).with_name("kinextract.config"))
    )

    log(f"Reading spectrum from {spectrum_path}")
    log(f"Reading config file from {config_path}")

    cfg = load_config_from_toml(config_path)

    fit = run_spectral_fit(cfg, gal_file=spectrum_path)

    log(f"chi2={fit['chi2']:.4g}  ngood={fit['ngood']}  chi2_red={fit['outputs']['chi2_red']:.4g}")

    from .plotting import plot_continuum, plot_fit, plot_losvd

    if cfg.fit_continuum:
        plot_continuum(fit)
    else:
        plot_fit(fit)

    plot_losvd(fit)


if __name__ == "__main__":
    main()
