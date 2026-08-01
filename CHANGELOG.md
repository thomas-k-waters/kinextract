# Changelog

## 1.0.0 -- from-scratch rebuild

This release replaces the pre-1.0 `kinextract` (0.4.0) entirely with what
was developed independently as `kinextract_lite`: a from-scratch, more
thoroughly validated reimplementation of the same non-parametric LOSVD
fitting idea, not an incremental update. `src/`, `tests/`, and
`examples/` are wholesale replacements; `dev_notes/` is additive -- the
old project's investigation history (`stress_test/` harness, EMILES-vs-
real-star template comparisons, `n_losvd_bins`/auto-`xlam` regression
testing, MUSE/STIS real-data validation, joint-mode V-bias remediation)
is preserved alongside the new project's own (indexed in
`dev_notes/HISTORY.md`), not discarded.

**Not ported, by deliberate scope decision:**
- `kinextract.joint` (the P-spline-in-the-model continuum-cofitting
  method) -- superseded by a different architecture: a shared
  multiplicative continuum, alternated against the LOSVD/template fit in
  the same optimization (`kinextract.continuum`,
  `kinextract.multi_template`'s 3-block alternation).
- `kinextract.bayesian` (optional NumPyro full-posterior/NUTS path) -- no
  equivalent yet; noted as a possible future addition, not attempted here.
- The TOML config loader (`load_config_from_toml`) and the `kinextract`
  CLI entry point -- `FitConfig` is now a plain dataclass, instantiated
  directly.
- `benchmarks/`'s old diagnostic suite (built against the removed
  `joint`/TOML API) -- removed; its role is now filled by
  `dev_notes/validation_suite/` (`sweep.py`, `coverage_sweep.py`), which
  is actively maintained and was rerun end-to-end as part of this merge.

**What's new relative to either prior codebase**, in brief (full detail in
`VALIDATION_CAMPAIGN.md`'s 27-stage record and `dev_notes/HISTORY.md`'s
topic index):
- A single fitting engine for any number of templates
  (`kinextract.fit_losvd`), replacing separate single/multi-template code
  paths.
- Data-driven regularization-strength selection via a bootstrap-estimated
  generalized AIC (`aic_regularization.py`), as an alternative to a fixed
  `lam`.
- A third-difference, shrink-toward-Gaussian penalty (`penalty_order=3`,
  the default) that measurably fixed a ~5%-everywhere LOSVD shape
  residual found in the prior architecture.
- Parametric-bootstrap uncertainty quantification with a documented,
  tested coverage story, plus an explicitly opt-in (not yet trusted for
  unsupervised use) synthetic-calibrated correction for regularization/
  shrinkage bias.
- AR(1) correlated-noise whitening (`noise_model.py`), closing a
  previously-unaddressed real-MUSE noise-correlation gap.
- E-MILES SSP template packing/LSF-matching as a first-class template
  source, IFU-cube spatial binning ("dartboard" binning, PA-aware,
  reflected `sin(theta)` sectors matching the legacy `sco_framework`
  pipeline's own convention) and iterative template-subset reduction, for
  full-galaxy analyses (`examples/notebooks/08_full_galaxy_analysis.ipynb`).
- A randomized, multi-seed validation suite
  (`dev_notes/validation_suite/`) scoring recovered moments against a
  mock's own truth-curve moments (not nominal Gauss-Hermite parameters),
  covering GH and spline-shaped truths, run as this project's standing
  regression check rather than one-off validation scripts.

See `README.md` for current package scope/status and
`dev_notes/HISTORY.md` for the full index of closed/open investigations.
