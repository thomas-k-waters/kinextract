#!/usr/bin/env python3
"""Overnight-runnable fast-tier driver for the kinextract validation suite.

Builds mock spectra across the scenario grid (benchmarks/scenarios.py),
fits each with kinextract (MAP fit only -- no bootstrap/Laplace, see
run_tier2_slow.py for that), and appends one result row per
(scenario, continuum_mode, replicate) to a checkpoint file, so an
interrupted run can simply be restarted with the same command.

Usage
-----
    python -m benchmarks.run_suite --n-workers 6 --replicates 8

Parallelism reuses the exact idiom kinextract's own bootstrap error
estimation uses internally (src/kinextract/errors.py): a
ThreadPoolExecutor (not multiprocessing -- JAX's XLA runtime is already
initialized in the parent process and threads share it) wrapped in a
threadpoolctl BLAS-thread-limit-to-1 context, so n_workers threads x 1 BLAS
thread each = full CPU utilization without oversubscription.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from kinextract import run_spectral_fit
from kinextract.fitting import fit_losvd_gauss_hermite, compute_losvd_n_peaks

from benchmarks.mock_spectrum import build_mock_spectrum, write_mock_to_disk
from benchmarks.scenarios import build_scenario_grid, scenario_to_fitconfig, Scenario


class _ThreadLocalStdout:
    """A ``sys.stdout`` replacement that dispatches each thread's writes to a
    thread-local target file.

    ``contextlib.redirect_stdout`` mutates the single process-global
    ``sys.stdout``, which is unsafe when multiple ``ThreadPoolExecutor``
    workers each try to redirect it concurrently (one thread's ``__exit__``
    can restore ``sys.stdout`` to a file another thread already closed,
    causing "I/O operation on closed file" elsewhere). Install this once at
    startup instead; each worker thread calls :meth:`set_target` /
    :meth:`clear_target` around its own fit rather than using
    ``redirect_stdout``.
    """

    def __init__(self, real_stdout):
        self._real = real_stdout
        self._local = threading.local()

    def set_target(self, f) -> None:
        self._local.target = f

    def clear_target(self) -> None:
        self._local.target = None

    def write(self, s: str) -> int:
        target = getattr(self._local, "target", None)
        return (target or self._real).write(s)

    def flush(self) -> None:
        target = getattr(self._local, "target", None)
        (target or self._real).flush()

RESULT_FIELDS = [
    "scenario_id", "axis", "continuum_mode", "replicate_seed",
    "instrument", "v_true", "sigma_true", "h3_true", "h4_true", "losvd_shape",
    "snr_target", "template_role", "emission_level", "xlam_max_peaks_used",
    "v_rec", "sigma_rec", "h3_rec", "h4_rec", "gh_fit_success", "n_peaks_recovered",
    "bias_v", "bias_sigma", "bias_h3", "bias_h4",
    "chi2_red", "xlam_selected", "n_good_pixels",
    "wall_time_s", "crashed", "error_message",
]


def run_one(scenario: Scenario, continuum_mode: str, replicate_seed: int,
            results_dir: Path) -> dict:
    """Build the mock, run one MAP-only fit, score against ground truth.

    Never raises: any exception is caught and recorded as a failure row
    (crashed=True) so one bad scenario cannot abort the overnight run.
    """
    t0 = time.perf_counter()
    log_path = results_dir / "logs" / f"{scenario.scenario_id}__{continuum_mode}__{replicate_seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: None for k in RESULT_FIELDS}
    row.update(
        scenario_id=scenario.scenario_id, axis=scenario.axis, continuum_mode=continuum_mode,
        replicate_seed=replicate_seed, instrument=scenario.instrument.name,
        v_true=scenario.v_true, sigma_true=scenario.sigma_true,
        h3_true=scenario.h3_true, h4_true=scenario.h4_true, losvd_shape=scenario.losvd_shape,
        snr_target=scenario.snr_target, template_role=scenario.template_role,
        emission_level=scenario.emission_level, crashed=False, error_message="",
    )
    try:
        with open(log_path, "w") as logf:
            if isinstance(sys.stdout, _ThreadLocalStdout):
                sys.stdout.set_target(logf)
            work_dir = results_dir / "work" / f"{scenario.scenario_id}__{continuum_mode}__{replicate_seed}"
            mock = build_mock_spectrum(
                instrument=scenario.instrument, v_true=scenario.v_true, sigma_true=scenario.sigma_true,
                snr_target=scenario.snr_target, continuum_mode=continuum_mode,
                template_role=scenario.template_role, losvd_shape=scenario.losvd_shape,
                h3=scenario.h3_true, h4=scenario.h4_true, emission_level=scenario.emission_level,
                noise_seed=replicate_seed,
            )
            if continuum_mode == "none":
                mock.flux_noisy = mock.flux_noisy / mock.true_continuum
                mock.errors = mock.errors / mock.true_continuum
            paths = write_mock_to_disk(mock, work_dir, prefix="mock")
            cfg = scenario_to_fitconfig(
                scenario, continuum_mode, outdir=str(work_dir),
                template_list_file=str(paths["tlist"]), template_dir=str(work_dir),
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
        # clean up per-replicate scratch files -- only the log + result row are kept
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
            done.add((row.get("scenario_id"), row.get("continuum_mode"), row.get("replicate_seed")))
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
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--continuum-modes", nargs="+", default=["als", "poly_amp"])
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
            "scenario_id": None, "continuum_mode": None, "replicate_seed": None,
        })

    scenarios = build_scenario_grid()
    if args.only_axis:
        scenarios = [s for s in scenarios if s.axis == args.only_axis]

    # BASE also gets the "none" oracle continuum mode, in addition to whatever
    # --continuum-modes was given, as a ceiling reference.
    modes_by_scenario = {
        s.scenario_id: (list(args.continuum_modes) + (["none"] if s.scenario_id == "base" else []))
        for s in scenarios
    }

    done = load_checkpoint(checkpoint_path)
    jobs = [
        (s, mode, 1000 + k)
        for s in scenarios
        for mode in modes_by_scenario[s.scenario_id]
        for k in range(args.replicates)
        if (s.scenario_id, mode, 1000 + k) not in done
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
                executor.submit(run_one, s, mode, seed, results_dir): (s.scenario_id, mode, seed)
                for (s, mode, seed) in jobs
            }
            n_done, n_crashed = 0, 0
            for future in as_completed(futures):
                s, mode, seed = futures[future]
                row = future.result()  # run_one never raises
                append_checkpoint(checkpoint_path, row)
                n_done += 1
                n_crashed += int(row.get("crashed", False))
                elapsed = time.perf_counter() - t_start
                eta = elapsed / n_done * (len(jobs) - n_done)
                status = "CRASHED" if row.get("crashed") else "ok"
                print(f"[{n_done}/{len(jobs)}] {s.scenario_id}/{mode}/{seed}: {status} "
                      f"({row.get('wall_time_s', 0):.0f}s) "
                      f"-- {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, {n_crashed} crashed so far",
                      flush=True)

    print(f"Fast tier complete: {n_done} jobs, {n_crashed} crashed. "
          f"Results: {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
