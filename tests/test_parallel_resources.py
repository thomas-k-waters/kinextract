"""Tests for kinextract.multi_template's worker-pool resource-sizing
helpers (_available_cpu_count, resolve_n_jobs, _limit_worker_threads) --
the pieces responsible for the *_parallel functions not overcommitting
CPU or memory on a shared HPC node (e.g. Great Lakes via SLURM)."""
from __future__ import annotations

import os

import pytest

from kinextract.multi_template import _available_cpu_count, _limit_worker_threads, resolve_n_jobs


def test_available_cpu_count_prefers_sched_affinity_over_cpu_count(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(4)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
    assert _available_cpu_count() == 4


def test_available_cpu_count_uses_slurm_cpus_per_task_when_no_affinity_call(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
    assert _available_cpu_count() == 8


def test_available_cpu_count_parses_slurm_job_cpus_per_node_multinode_form(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "8(x2)")
    assert _available_cpu_count() == 8


def test_available_cpu_count_takes_tightest_signal_when_several_disagree(monkeypatch):
    """A wrong/stale signal should never cause overcommit -- the minimum
    of every available signal wins."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
    assert _available_cpu_count() == 4


def test_available_cpu_count_falls_back_to_cpu_count_with_no_other_signals(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
    assert _available_cpu_count() == 8


def test_available_cpu_count_ignores_unparseable_slurm_env_value(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
    assert _available_cpu_count() == 8


def test_resolve_n_jobs_explicit_value_always_wins(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert resolve_n_jobs(32, per_job_bytes=1e9) == 32


def test_resolve_n_jobs_none_is_capped_by_slurm_allocation_not_whole_node(monkeypatch):
    """The real Great Lakes scenario this whole mechanism exists for: a job
    allocated a handful of cores on a big shared node must never spawn a
    worker per *node* core."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
    # per_job_bytes small enough that memory never binds -- CPU allocation should.
    assert resolve_n_jobs(None, per_job_bytes=1.0) == 8


def test_limit_worker_threads_sets_single_threaded_blas_env_vars(monkeypatch):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "JAX_PLATFORMS"):
        monkeypatch.delenv(var, raising=False)
    _limit_worker_threads()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert os.environ[var] == "1"
    assert os.environ["JAX_PLATFORMS"] == "cpu"


def test_limit_worker_threads_does_not_override_explicit_jax_platforms(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "gpu")
    _limit_worker_threads()
    assert os.environ["JAX_PLATFORMS"] == "gpu"
