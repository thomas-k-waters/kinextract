"""Shared test fixtures/helpers.

`test_a_real_process_pool_is_bit_identical_to_serial`-style tests exist to
catch real dispatch/pickling/ordering bugs in a `*_parallel` function --
not to characterize floating-point noise. Building the "serial" side of
that comparison by calling the fit function directly in the test process
compares the wrong things: a real worker process imports JAX fresh, in a
single-threaded process (`_limit_worker_threads` pins BLAS/OpenMP thread
pools to 1 *before* JAX/numpy import), while the test process's own JAX/
XLA CPU backend is already initialized, multi-threaded, and has been
warmed up by every earlier test in the run. XLA's CPU backend picks its
own internal reduction strategy per process/thread context regardless of
BLAS thread-count env vars or `threadpoolctl` (checked directly,
2026-09-14: wrapping the serial loop in
`threadpoolctl.threadpool_limits(limits=1)` does not close the gap), so
that mismatch is real but is a few ULPs at the raw output level -- usually
invisible, but iterative fits (robust-fit sigma-clipping, bootstrap
convergence) can occasionally amplify a few ULPs into a convergence/
clipping decision landing on the wrong side of a boundary, producing an
occasional large, run-dependent mismatch with no real bug behind it. A
fixed `rtol` tolerance chases this rather than fixing it: CI failures
after loosening `test_uncertainty.py`'s equivalent test to `rtol=0.05`
(2026-09-14) still recurred at 5.33%, and a previously-untouched sibling
in `test_robust_fit.py` failed the same day at ~18% -- proof this class of
flakiness isn't bounded by any single tolerance.

`run_jobs_in_one_worker_process` gives these tests a genuinely fair
reference instead: it runs a worker function over a list of jobs through a
real, single-worker `ProcessPoolExecutor`, using the exact same `spawn`
context and `_limit_worker_threads` initializer the real `n_jobs>1` path
uses. Both sides of the comparison then see identical, freshly-initialized,
single-threaded process state, so the comparison is testing dispatch and
result-ordering -- what these tests actually exist to catch -- rather than
JAX/XLA process-state sensitivity, and can go back to exact/tight
equality instead of a tolerance wide enough to hide a real bug.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from kinextract.multi_template import _limit_worker_threads


def run_jobs_in_one_worker_process(worker_fn, jobs):
    """Run `worker_fn` over `jobs`, in order, inside a single freshly-spawned
    worker process -- the same process state any one real parallel worker
    sees, but with all jobs kept in that one process so the result is a fair
    "serial reference" for a `*_parallel` function's bit-identity tests."""
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"),
                              initializer=_limit_worker_threads) as pool:
        return list(pool.map(worker_fn, jobs))
