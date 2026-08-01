# benchmarks

This directory previously held a diagnostic suite (`run_suite.py`,
`mock_spectrum.py`, `calibrate_xlam.py`, `scenarios.py`, `score.py`,
`report.py`) built against the pre-1.0 architecture: `kinextract.joint`'s
P-spline-in-the-model continuum-cofitting, `fit_continuum=True`,
`continuum_poly_mode`. None of that API exists in this version -- the
non-parametric LOSVD engine and continuum-cofitting mechanism were both
rebuilt from scratch (see `../VALIDATION_CAMPAIGN.md`), so those scripts
would not run and have been removed rather than left as silently-broken
dead code.

**The current equivalent lives in `../dev_notes/validation_suite/`**:
`sweep.py` (randomized bias/coverage sweeps across sigma/S-N/h3/h4, GH and
spline truth generators, the "safe" preset used for the package's own
validation campaign), `coverage_sweep.py` (does a reported bootstrap
interval actually cover truth at the rate it claims?), `mocks.py` (the
randomized mock generator both scripts share), and `build_notebook.py`
(regenerates a results notebook from the sweep CSVs). These are actively
maintained and were rerun end-to-end as part of the 1.0 merge -- see
`../dev_notes/HISTORY.md` for the full index of what's been validated and
where to find it.
