"""Shared low-level utilities for kinextract.

This module has no dependencies on the rest of the package and is imported
by nearly every other module. It defines the physical constant used to
convert between wavelength shift and line-of-sight velocity (``CEE``), the
sentinel value used to mark masked/bad pixels in flux-error arrays (``BIG``),
and small logging/timing helpers used throughout the LOSVD-fitting pipeline
to report progress on long-running fits.
"""
from __future__ import annotations

import time

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CEE = 299792.458   # speed of light, km/s (IAU 2012)
BIG = 1e10     # sentinel value for masked pixels

_T0 = time.perf_counter()
_VERBOSE = True


def set_verbose(verbose: bool) -> None:
    """Enable or disable :func:`log`'s progress messages package-wide.

    ``kinextract`` logs progress (stage start/end timing, xlam/ALS search
    steps, etc.) unconditionally via :func:`log` throughout the fitting
    pipeline. Call ``set_verbose(False)`` once (e.g. at the top of a
    notebook or script) to silence all of it -- useful when running many
    fits back-to-back and only the final results matter, not the internal
    progress trace. Independent of `FitConfig.print_every`, which only
    controls the separate per-iteration objective-evaluation counter inside
    a single optimization.

    Parameters
    ----------
    verbose : bool
        If False, :func:`log` becomes a no-op until re-enabled.
    """
    global _VERBOSE
    _VERBOSE = bool(verbose)


def log(msg: str) -> None:
    """Print a timestamped progress message.

    Prefixes `msg` with the elapsed wall-clock time (in seconds) since the
    module was first imported, and flushes immediately so messages interleave
    correctly with output from long-running fits (including those run in
    subprocesses/workers during bootstrap error estimation). No-op if
    :func:`set_verbose` has been called with ``False``.

    Parameters
    ----------
    msg : str
        Message to print.
    """
    if _VERBOSE:
        print(f"[{time.perf_counter() - _T0:9.2f}s] {msg}", flush=True)


class Timer:
    """Context manager that logs the start, end, and duration of a code block.

    On entry it emits a ``START <label>`` message via :func:`log`; on exit it
    emits an ``END <label> (<seconds>s)`` message. Used throughout the fitting
    pipeline to profile expensive stages (I/O, template interpolation, LOSVD
    precomputation, optimization) without adding a full profiling dependency.

    Parameters
    ----------
    label : str
        Human-readable name of the block being timed, included in both the
        start and end log messages.
    """

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        log(f"START {self.label}")
        return self

    def __exit__(self, *_):
        log(f"END   {self.label} ({time.perf_counter() - self.t0:.2f}s)")
