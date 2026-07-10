#!/usr/bin/env python3
"""Empirical calibration of the discrepancy-principle xlam selector's
``xlam_discrepancy_nsigma`` (kinextract.fitting._discrepancy_principle_search).

Cappellari (2017, MNRAS 466, 798, Sec. 3.5)'s pPXF ``REGUL`` convention
targets a chi2 rise of ``nsigma * sqrt(2*ngood)`` above the unregularized
fit, with nsigma=1.0.
A quick single-mock test (2026-07 validation review) found nsigma=1.0
selects a far-too-large xlam for kinextract's own problem (a 29-bin
non-parametric LOSVD histogram smoothed directly, vs. pPXF's typical use
smoothing many template weights for star-formation-history recovery) --
sigma bias +10 km/s vs. the existing "chi2" criterion's -4.8 km/s on one
mock. This module empirically sweeps candidate nsigma values across a
*representative* subset of benchmarks/scenarios.py's scenario grid (the
"sigma" and "snr" axes specifically -- the two most directly relevant to
how much genuine curvature chi2(xlam) has) with multiple noise replicates
each, to find whether a single nsigma value generalizes reasonably well
across conditions (a legitimate, defensible package-specific calibration)
or whether the "right" nsigma varies too much case-to-case to trust (a red
flag for the whole approach, per the module docstring's linked plan).

continuum_mode is held fixed at "none" (mock pre-divided by its own true
continuum, matching run_suite.py's existing oracle/ceiling convention) so
this isolates xlam-selection behavior from continuum-cofitting -- a
separate, independent research question already covered by run_suite.py's
own continuum_mode comparison.

Usage
-----
    python -m benchmarks.calibrate_xlam --n-workers 6 --replicates 5

Mirrors benchmarks/run_suite.py's design (checkpointed/resumable,
ThreadPoolExecutor + threadpoolctl BLAS-thread-limit-to-1, same
_ThreadLocalStdout log-redirection idiom) since that harness was already
built, fixed, and validated this session -- reusing it here rather than
writing new harness plumbing from scratch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmarks.mock_spectrum import build_mock_spectrum, write_mock_to_disk
from benchmarks.run_suite import _ThreadLocalStdout
from benchmarks.scenarios import Scenario, build_scenario_grid, scenario_to_fitconfig
from kinextract import run_spectral_fit
from kinextract.fitting import compute_losvd_n_peaks, fit_losvd_gauss_hermite

# name -> (xlam_criterion, xlam_discrepancy_nsigma). "chi2" is the existing
# package default, included as the baseline every candidate is judged
# against. nsigma candidates span roughly a decade below Cappellari's own
# nsigma=1.0 (already shown too large) down toward where a single mock's
# "good" xlam sat empirically (~1/8 of the nsigma=1.0 target).
XLAM_METHODS: dict[str, tuple[str, float]] = {
    "chi2_baseline": ("chi2", 1.0),  # nsigma unused for "chi2"
    "disc_n1.0": ("discrepancy", 1.0),
    "disc_n0.5": ("discrepancy", 0.5),
    "disc_n0.3": ("discrepancy", 0.3),
    "disc_n0.2": ("discrepancy", 0.2),
    "disc_n0.1": ("discrepancy", 0.1),
    "disc_n0.05": ("discrepancy", 0.05),
}

RESULT_FIELDS = [
    "scenario_id", "axis", "xlam_method", "replicate_seed",
    "instrument", "v_true", "sigma_true", "h3_true", "h4_true", "losvd_shape",
    "snr_target", "template_role", "emission_level", "xlam_max_peaks_used",
    "v_rec", "sigma_rec", "h3_rec", "h4_rec", "gh_fit_success", "n_peaks_recovered",
    "bias_v", "bias_sigma", "bias_h3", "bias_h4",
    "chi2_red", "xlam_selected", "n_good_pixels",
    "wall_time_s", "crashed", "error_message",
]


def run_one(scenario: Scenario, xlam_method: str, replicate_seed: int,
            results_dir: Path) -> dict:
    """Build the mock, run one MAP-only fit (continuum_mode="none" always),
    score against ground truth. Never raises -- see run_suite.run_one,
    same convention.
    """
    t0 = time.perf_counter()
    log_path = results_dir / "logs" / f"{scenario.scenario_id}__{xlam_method}__{replicate_seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: None for k in RESULT_FIELDS}
    row.update(
        scenario_id=scenario.scenario_id, axis=scenario.axis, xlam_method=xlam_method,
        replicate_seed=replicate_seed, instrument=scenario.instrument.name,
        v_true=scenario.v_true, sigma_true=scenario.sigma_true,
        h3_true=scenario.h3_true, h4_true=scenario.h4_true, losvd_shape=scenario.losvd_shape,
        snr_target=scenario.snr_target, template_role=scenario.template_role,
        emission_level=scenario.emission_level, crashed=False, error_message="",
    )
    xlam_criterion, nsigma = XLAM_METHODS[xlam_method]
    try:
        with open(log_path, "w") as logf:
            if isinstance(sys.stdout, _ThreadLocalStdout):
                sys.stdout.set_target(logf)
            work_dir = results_dir / "work" / f"{scenario.scenario_id}__{xlam_method}__{replicate_seed}"
            mock = build_mock_spectrum(
                instrument=scenario.instrument, v_true=scenario.v_true, sigma_true=scenario.sigma_true,
                snr_target=scenario.snr_target, continuum_mode="none",
                template_role=scenario.template_role, losvd_shape=scenario.losvd_shape,
                h3=scenario.h3_true, h4=scenario.h4_true, emission_level=scenario.emission_level,
                noise_seed=replicate_seed,
            )
            mock.flux_noisy = mock.flux_noisy / mock.true_continuum
            mock.errors = mock.errors / mock.true_continuum
            paths = write_mock_to_disk(mock, work_dir, prefix="mock")
            cfg = scenario_to_fitconfig(
                scenario, "none", outdir=str(work_dir),
                template_list_file=str(paths["tlist"]), template_dir=str(work_dir),
                xlam_criterion=xlam_criterion, xlam_discrepancy_nsigma=nsigma,
            )
            fit = run_spectral_fit(cfg, gal_file=str(paths["spec"]), write_outputs=False)
            st = fit["state"]
            b = fit["outputs"]["b"]
            gh = fit_losvd_gauss_hermite(st.xl, b, fit_h3h4=True)
            chi2_red = fit["chi2"] / max(fit["ngood"] - 1, 1)
            row.update(
                xlam_max_peaks_used=cfg.xlam_max_peaks,
                v_rec=gh["vherm"], sigma_rec=gh["sherm"], h3_rec=gh["h3"], h4_rec=gh["h4"],
                gh_fit_success=gh.get("fit_success", None), n_peaks_recovered=compute_losvd_n_peaks(b),
                bias_v=gh["vherm"] - scenario.v_true, bias_sigma=gh["sherm"] - scenario.sigma_true,
                bias_h3=gh["h3"] - scenario.h3_true, bias_h4=gh["h4"] - scenario.h4_true,
                chi2_red=chi2_red, xlam_selected=float(st.xlam), n_good_pixels=int(fit["ngood"]),
            )
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception as e:
        row.update(crashed=True, error_message=f"{type(e).__name__}: {e}")
        with open(log_path, "a") as logf:
            logf.write("\n--- EXCEPTION ---\n")
            logf.write(traceback.format_exc())
    finally:
        if isinstance(sys.stdout, _ThreadLocalStdout):
            sys.stdout.clear_target()
    row["wall_time_s"] = time.perf_counter() - t0
    return row


def load_checkpoint(checkpoint_path: Path) -> set[tuple]:
    if not checkpoint_path.exists():
        return set()
    done = set()
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row.get("scenario_id"), row.get("xlam_method"), row.get("replicate_seed")))
    return done


def append_checkpoint(checkpoint_path: Path, row: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown (not a git repo)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-workers", type=int, default=6)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results_xlam_calib")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=list(XLAM_METHODS.keys()),
                         choices=list(XLAM_METHODS.keys()))
    parser.add_argument("--only-axis", type=str, default=None,
                         help="restrict to one scenario axis, for quick reruns/debugging")
    args = parser.parse_args()

    if not isinstance(sys.stdout, _ThreadLocalStdout):
        sys.stdout = _ThreadLocalStdout(sys.stdout)

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / "checkpoint.jsonl"

    if not checkpoint_path.exists():
        append_checkpoint(checkpoint_path, {
            "_header": True, "git_commit": _git_commit(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenario_id": None, "xlam_method": None, "replicate_seed": None,
        })

    scenarios = build_scenario_grid()
    # Default scope: "base" + the "sigma"/"snr" axes -- the two axes most
    # directly relevant to how much genuine curvature chi2(xlam) has (see
    # module docstring). Use --only-axis to narrow further or widen back to
    # the full grid if the reduced scope looks promising.
    if args.only_axis:
        scenarios = [s for s in scenarios if s.axis == args.only_axis]
    else:
        scenarios = [s for s in scenarios if s.axis in ("base", "sigma", "snr")]

    done = load_checkpoint(checkpoint_path)
    jobs = [
        (s, method, 1000 + k)
        for s in scenarios
        for method in args.methods
        for k in range(args.replicates)
        if (s.scenario_id, method, 1000 + k) not in done
    ]
    print(f"{len(jobs)} jobs remaining ({len(done)} already checkpointed)", flush=True)
    if not jobs:
        print("Nothing to do -- all jobs already checkpointed.", flush=True)
        return 0

    from threadpoolctl import threadpool_limits
    t_start = time.perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(run_one, s, method, seed, results_dir): (s, method, seed)
                for (s, method, seed) in jobs
            }
            n_done, n_crashed = 0, 0
            for future in as_completed(futures):
                s, method, seed = futures[future]
                row = future.result()  # run_one never raises
                append_checkpoint(checkpoint_path, row)
                n_done += 1
                n_crashed += int(row.get("crashed", False))
                elapsed = time.perf_counter() - t_start
                eta = elapsed / n_done * (len(jobs) - n_done)
                status = "CRASHED" if row.get("crashed") else "ok"
                print(f"[{n_done}/{len(jobs)}] {s.scenario_id}/{method}/{seed}: {status} "
                      f"({row.get('wall_time_s', 0):.0f}s) "
                      f"-- {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, {n_crashed} crashed so far",
                      flush=True)

    print(f"xlam calibration complete: {n_done} jobs, {n_crashed} crashed. "
          f"Results: {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
