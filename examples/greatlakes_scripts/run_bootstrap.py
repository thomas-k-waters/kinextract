#!/usr/bin/env python3
"""Standalone bootstrap error estimation for a single spectrum.

This script runs a full MAP fit followed by bootstrap and Laplace error
estimation for one spectrum file. It's a standalone tool for testing or
re-running error estimation on an existing MAP fit outside of the normal
cluster workflow -- kinematicextraction itself submits its own generated
helper script (.run_kinextract_with_errors.py) rather than this one.

Requires the kinextract package to be importable (pip installed, or
lib/kinextract/src on PYTHONPATH -- see kinematicextraction for the same
fallback pattern).

Usage
-----
    python run_bootstrap.py spectrum.spec kinextract.config [options]

See ``python run_bootstrap.py --help`` for the full option list.
"""

import argparse
from pathlib import Path

from kinextract import load_config_from_toml, run_spectral_fit, estimate_losvd_errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run bootstrap LOSVD error estimation for a single spectrum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("spec_file", help="Path to input .spec file")
    parser.add_argument("config_file", help="Path to kinextract.config (TOML)")
    parser.add_argument("--n_bootstrap", type=int, default=200,
                        help="Number of bootstrap replicates")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Worker processes for bootstrap (-1 = all CPUs)")
    parser.add_argument("--block_size", type=int, default=1,
                        help="Block length for block bootstrap (1 = IID residuals)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--prefix", default=None,
                        help="Output file prefix (defaults to spectrum stem)")
    parser.add_argument("--outdir", default=".",
                        help="Directory for output files")
    parser.add_argument("--no_refit_als", dest="refit_als", action="store_false",
                        default=True,
                        help="Skip ALS continuum re-estimation on each bootstrap draw")
    args = parser.parse_args()

    cfg = load_config_from_toml(args.config_file)
    cfg.outdir = args.outdir

    fit = run_spectral_fit(cfg, gal_file=args.spec_file)

    prefix = args.prefix or Path(args.spec_file).stem
    estimate_losvd_errors(
        fit,
        cfg,
        run_laplace=True,
        run_bootstrap=True,
        run_bias=True,
        n_bootstrap=args.n_bootstrap,
        block_size=args.block_size,
        n_jobs=args.n_jobs,
        bootstrap_seed=args.seed,
        refit_als=args.refit_als,
        plot=False,
        write_to_files=True,
        prefix=prefix,
        outdir=args.outdir,
    )
    print(f"Output files written for {prefix} in {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
