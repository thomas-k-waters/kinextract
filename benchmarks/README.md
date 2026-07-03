# kinextract validation suite

A diagnostic, overnight-runnable test suite that generates mock spectra with
known ground-truth kinematics across many astrophysical regimes (star
clusters, low/high-mass black holes, different S/N, different instruments,
emission contamination, non-Gaussian/double-peaked LOSVDs) and fits them with
kinextract, comparing recovered V/sigma/h3/h4 against truth. Its central
purpose is a head-to-head comparison between the two continuum-fitting modes
(`fit_als_continuum=True` vs. the newer raw-flux + multiplicative-polynomial
+ global-amplitude mode), since real cross-validation against the legacy
Fortran pipeline found the ALS approach can create a genuine identifiability
degeneracy with the simultaneously-fit LOSVD.

This is a diagnostic/reporting tool, distinct from and independent of the
pytest suite in `tests/` (which only tests `benchmarks/mock_spectrum.py`'s
own correctness, in `tests/test_mock_spectrum.py`).

## Running the fast tier (MAP fits only, ~1.5-3 hours)

```bash
cd /Users/waterstk/Documents/Gultekin_Astrophysics/kinextract
python -m benchmarks.run_suite --n-workers 6 --replicates 8
```

This is resumable: if interrupted (killed, machine sleep, etc.), just run
the same command again -- `benchmarks/results/checkpoint.jsonl` tracks
completed `(scenario, continuum_mode, replicate_seed)` triples and only
remaining jobs are (re-)run.

Useful flags:
- `--only-axis sigma` -- restrict to one scenario axis (see `scenarios.py`
  for the full axis list: `sigma`, `snr`, `instrument`, `template`,
  `emission`, `losvd_shape`, `bug`), for quick reruns/debugging.
- `--continuum-modes als` -- restrict to one continuum mode.
- `--replicates 2` -- fewer noise realizations per scenario, for a fast smoke test.

## Generating the report

```bash
python -m benchmarks.report
```

Writes `benchmarks/results/report.md` (headline continuum-mode comparison
table, bias-vs-axis plots, flagged-scenario table, runtime table) plus
`raw_results.csv` and `plots/*.png`. Can be re-run at any time, including
while `run_suite.py` is still running (against whatever's checkpointed so far).

## Adjusting what counts as "falls short"

`benchmarks/score.py`'s `THRESHOLDS` dict defines the numeric bias/chi2_red
cutoffs used to flag a scenario in the report. These are proposals (see the
docstring for the reasoning behind each), not fixed truth -- edit them
directly to match your own science precision requirements, then re-run
`python -m benchmarks.report` (no need to re-run the fits).

## Slow tier (Laplace + bootstrap error bars on a curated subset)

Not yet implemented -- see the plan this suite was built from
(`~/.claude/plans/sorted-chasing-rainbow-agent-adb527942b716d2eb.md`)
Section 5.3 for the design (select the BASE scenario, the two named `bug_*`
scenarios, and the worst-scoring fast-tier point per axis; run
`estimate_losvd_errors(..., n_bootstrap=30)` on each to check whether the
ALS-vs-poly bias difference exceeds the bootstrap-estimated uncertainty).

## Dependencies

`pandas` is required for `score.py`/`report.py` (not for `run_suite.py`
itself) -- install via `pip install -e ".[benchmarks]"` from the repo root.
