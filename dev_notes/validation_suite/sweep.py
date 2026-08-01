"""CLI runner: sweep ONE named axis over a grid of fixed values, randomizing
every other axis (sigma, h3, h4, V, S/N, template ages/metals/weights) fresh
on every replicate. This is the single reusable engine behind "test sigma
bias while randomizing everything else", "test S/N bias while randomizing
everything else", etc. -- rather than a new one-off script per question.

Usage:
    python sweep.py --axis snr   --values 20,40,60,100,150,200,300,400 --n-reps 15 --out results/sweep_snr.csv
    python sweep.py --axis sigma --values 40,80,140,220,300,400        --n-reps 15 --out results/sweep_sigma.csv
    python sweep.py --axis h3    --values -0.08,-0.04,0,0.04,0.08      --n-reps 15 --out results/sweep_h3.csv
    python sweep.py --axis sigma --values 40,80,140,220,300,400 --gaussian-only --out results/sweep_sigma_gaussian.csv
    python sweep.py --axis sigma --values 40,80,140,220,300,400 --truth-kind spline --out results/sweep_sigma_spline.csv

IMPORTANT for h3/h4 (or any axis whose grid starts negative): use the
`--values=` form (equals sign, not a space) -- `--values -0.05,-0.025,0,...`
makes argparse treat "-0.05,..." as an unrecognized option rather than the
value of `--values`, and fails with "argument --values: expected one
argument". `--values=-0.05,-0.025,0,...` sidesteps this entirely. This is a
standard argparse quirk with negative-leading option values, not specific to
this script -- but it silently truncated a full run of this suite once
(`set -e` in the calling shell script aborted at the first failure), so it is
called out here explicitly.

`--truth-kind spline` builds truths directly as exp(spline) -- no
Gauss-Hermite step at all (see `mocks.build_spline_truth`) -- for isolating
pure fitting/regularization bias from the GH-parameterization-mismatch
mechanism documented in `mocks.py` and `FINDINGS.md`. `--axis h3`/`--axis h4`
are only valid with the default `--truth-kind gh`; sweep `--axis wiggle_scale`
for the spline generator's own shape-randomization knob instead.

Every replicate uses an independent (fixed_value_index, rep_index) sub-seed
derived from a single `--seed` for the whole run, so "randomized, not the
same data every time" (the actual requirement) coexists with full run-to-run
reproducibility given the same `--seed` -- rerun with the same seed and you
get the exact same 15 random template/sigma/h3/h4/weight draws back.

Results are appended to `--out` one row at a time (flushed after every fit),
so a run that is interrupted or time-boxed still leaves a fully usable
partial CSV -- nothing is lost by stopping early.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mocks import EMILES_GRID_DEFAULT, RANGE_PRESETS, observed_wavelength_grid
from runner import run_one

AXES = ("v", "sigma", "h3", "h4", "snr", "wiggle_scale")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axis", required=True, choices=AXES, help="which quantity to sweep (fix per grid point)")
    p.add_argument("--values", required=True, help="comma-separated grid values for --axis")
    p.add_argument("--n-reps", type=int, default=15, help="randomized replicates per grid value (default 15)")
    p.add_argument("--preset", default="safe", choices=list(RANGE_PRESETS), help="randomization range preset")
    p.add_argument("--truth-kind", default="gh", choices=("gh", "spline"),
                   help="'gh' = truncated Gauss-Hermite truth (default); "
                        "'spline' = GH-free exp(spline) truth, see mocks.build_spline_truth")
    p.add_argument("--gaussian-only", action="store_true",
                   help="gh: force h3=h4=0.  spline: force wiggle_scale=0 (pure Gaussian core)")
    p.add_argument("--n-templates", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="master seed; determines every draw in this run")
    p.add_argument("--out", required=True, help="output CSV path (appended, not overwritten if it already has rows)")
    p.add_argument("--emiles-grid", default=EMILES_GRID_DEFAULT)
    args = p.parse_args()
    if args.truth_kind == "spline" and args.axis in ("h3", "h4"):
        p.error(f"--axis {args.axis} has no meaning with --truth-kind spline -- use --axis wiggle_scale")
    if args.truth_kind == "gh" and args.axis == "wiggle_scale":
        p.error("--axis wiggle_scale only applies to --truth-kind spline")
    return args


def main():
    args = parse_args()
    values = [float(v) for v in args.values.split(",")]
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    emiles_path = args.emiles_grid if os.path.isabs(args.emiles_grid) else os.path.join(root, args.emiles_grid)

    import numpy as _np
    grid_file = _np.load(emiles_path, allow_pickle=True)
    emiles_ages, emiles_metals = grid_file["ages"], grid_file["metals"]
    wave_fit = observed_wavelength_grid()

    out_path = args.out if os.path.isabs(args.out) else os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_header = not (os.path.exists(out_path) and os.path.getsize(out_path) > 0)

    total = len(values) * args.n_reps
    done = 0
    t_start = time.time()
    fieldnames = None
    with open(out_path, "a", newline="") as fh:
        writer = None
        for vi, val in enumerate(values):
            for rep in range(args.n_reps):
                sub_seed = args.seed * 1_000_003 + vi * 10_007 + rep
                rng = np.random.default_rng(sub_seed)
                row = run_one(
                    rng, noise_seed=sub_seed, fixed={args.axis: val},
                    preset=args.preset, gaussian_only=args.gaussian_only,
                    n_templates=args.n_templates, emiles_ages=emiles_ages, emiles_metals=emiles_metals,
                    emiles_grid_path=emiles_path, wave_fit=wave_fit, truth_kind=args.truth_kind,
                )
                row["sweep_axis"] = args.axis
                row["sweep_value"] = val
                row["rep"] = rep
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                writer.writerow(row)
                fh.flush()
                done += 1
                elapsed = time.time() - t_start
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] axis={args.axis} value={val:g} rep={rep} "
                      f"bias_sigma={row['bias_sigma']:+.2f} bias_h4={row['bias_h4']:+.4f} "
                      f"lam={row['lam_used']:g}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)
    print(f"\nwrote {done} rows to {out_path}")


if __name__ == "__main__":
    main()
