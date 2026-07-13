# Joint-mode velocity bias: current status and remediation plan

This document consolidates the investigation into the residual velocity
(V) recovery bias found while validating kinextract's joint continuum-cofit
mode (`cfg.fit_continuum=True`) against pPXF, and lays out concrete next
steps. The quantified summary of this finding is also recorded in
`FitConfig`'s class docstring ("Known limitations: joint-mode velocity
bias") — this document is the working investigation record behind that
summary, kept here rather than in the docstring since it's aimed at future
development work, not package users.

## The two distinct bias components

Everything below refers to a fixed synthetic-mock validation setup: E-MILES
templates, `n_losvd_bins=89`, a 1000A fit window, sigma swept from 30-350
km/s, true V=80 km/s, 3-5 noise seeds per condition.

**1. A small, low-sigma offset (sigma <~ 160 km/s).** Roughly -2 to -2.5
km/s at true V=+80, shrinking toward zero as true V does, and flipping sign
at true V=-80 (biased toward +1 to +3 km/s there). This looks like a
shrinkage-toward-the-grid-center effect (the LOSVD's velocity grid,
`[losvd_vmin, losvd_vmax]`, is symmetric around v=0 by convention), but is
**not understood** — see "Ruled out" below. It does not appear in an
equivalent pPXF (Gauss-Hermite parametric) fit of the same mock, which
shows near-zero V bias in this regime.

**2. A larger bias growing with sigma (sigma >~ 200 km/s).** Grows to
roughly -10 to -14 km/s by sigma=350. This component also appears, at
comparable (somewhat smaller) magnitude, in pPXF's own fit of the same
mock — it is likely the well-documented V/sigma/h3 covariance at low
per-resolution-element S/N (van der Marel & Franx 1993; the moment-
covariance discussion in Cappellari 2017), i.e. an expected property of
*any* reasonable estimator in this regime, not a kinextract-specific
defect. The right response to this component is probably not to chase it
to zero, but to confirm `residual_bootstrap`'s error bars have honest
coverage there (see "Not yet done" below) rather than trusting the point
estimate.

Component 1 is the one worth continuing to root-cause, since pPXF doesn't
show it and it persists even at sigma where everything else is
well-behaved (chi2_red ~ 1, no convergence issues).

## Ruled out (with the specific test and result)

All tests below used the same E-MILES mock generator, 89-bin LOSVD grid,
1000A window, sigma=70 unless noted.

| Hypothesis | Test | Result |
|---|---|---|
| `xlam_criterion` selection rule ("discrepancy" vs "chi2") | Full sigma sweep, both criteria | Essentially identical V/sigma bias under both |
| Wing-taper `v_center` recentering on/off | `joint_recenter_v=True` vs `False` | `False` much worse at low sigma, no difference at high sigma |
| Recentering *quality* | `v_center` forced to the *exact* true V (bypassing the real cross-correlation estimate) vs normal | Nearly identical results in every case (<0.5 km/s difference) |
| Fitting-template count/degeneracy | 20 vs 16 vs 4 fitting templates (4 being the true pair + 2 neighbors) | No improvement; slightly *worse* at high sigma with fewer templates |
| Generating-population complexity | Simple 2-SSP mock vs a broad, smooth ~20-SSP mixture spanning the whole fitting grid | Nearly identical bias pattern, slightly worse at high sigma for the broad mock |
| LOSVD grid centered on v=0 vs on the true V | `losvd_vmin/vmax` fixed at `+-vrange` vs `true_v +- vrange` | Markedly *worse*, and inconsistent in direction (not a clean sign flip) — grid-recentering interacts badly with something else in the fit, not a fix |

Also confirmed via a "cheat-start" diagnostic: initializing the optimizer
at the *exact* true (LOSVD, template weights) and re-optimizing from there
converges to the same biased answer the normal (uniform-start) fit finds,
with an equal-or-lower objective value than the true parameters achieve.
This rules out "bad initialization / stuck in a local optimum" — the
biased answer is a genuine, better optimum of the current objective for a
given noise realization, so any fix has to change the objective or model
structure, not the optimizer's starting point.

## Not yet tried (recommended next steps, in priority order)

1. **Isolate joint-mode continuum-cofitting from the LOSVD fit itself.**
   Every test above used `cfg.fit_continuum=True` (the joint continuum
   P-spline cofit). The shipped pre-normalized path (`cfg.fit_continuum=
   False`, `cfg.joint_prenorm=True`, no continuum nuisance parameters at
   all) has not been tested on this same mock/sigma grid. Running the
   identical sigma sweep through the pre-normalized path would show
   whether the low-sigma shrinkage is specific to the continuum cofit's
   added parameters/interactions, or a property of the non-parametric
   LOSVD fit in general (present either way). This is the highest-value
   untested lead and should be done first.
2. **Increase seed count for the low-sigma offset specifically.** All
   "ruled out" tests above used only 3 seeds per condition — enough to see
   a consistent-looking pattern, not enough to fully rule out per-seed
   noise shaping the apparent trend. Re-run the true-V-dependence sweep
   (V=0, 40, 80, -80 at a fixed low sigma) with 10-20 seeds to get a
   statistically solid characterization before designing further fixes.
3. **Inspect the recovered LOSVD histogram directly, not just its
   GH-fit summary.** Every diagnostic so far reduced the fit to V/sigma
   (or, in one case, GH-fit vs raw-moment). Plotting the actual `b(v)`
   array for a few biased cases against the true injected Gaussian would
   show *where* the discrepancy actually lives (a shifted peak, a
   secondary bump, an asymmetric tail) rather than inferring it indirectly
   from summary statistics.
4. **Re-examine the GH-moment extraction step with more seeds.** A first
   pass (`fit_losvd_gauss_hermite` vs a raw first-moment) found mixed
   results — the GH fit is *better* than the raw moment at sigma 160-250
   but *worse* at sigma=350 — on only 2 seeds. Worth redoing with more
   seeds and, if the crossover is real, checking whether
   `fit_losvd_gauss_hermite_higher` (h5/h6 terms) changes the picture at
   the highest sigma.
5. **REML/marginal-likelihood regularization-strength selection.** A
   separate, earlier-considered plan (see the discrepancy-principle
   vs. REML design in an earlier planning note) is a more principled
   xlam-selection alternative to both `"chi2"` and `"discrepancy"`.
   `xlam_criterion` choice was already ruled out as the *direct* driver of
   the V bias, so this is lower priority than items 1-3, but worth
   revisiting if those don't resolve it.

## Practical guidance until this is resolved

- Treat recovered V at sigma <~ 160 km/s as accurate to a few km/s, not
  exactly zero-bias.
- At sigma >~ 200 km/s, prefer the bootstrap error bars
  (`LOSVDErrorEstimator.residual_bootstrap`) over the point estimate for
  judging consistency with an expected/literature value — this regime's
  bias is likely a shared, expected property of any comparable method, not
  a kinextract-specific bug, so the correct response is honest
  uncertainty, not a point-estimate correction.
- `assess_recovery_bias`/`correct_recovered_losvd` do not currently
  support joint-mode fits at all (they raise on `cfg.fit_continuum=True`).
  There is no automated correction path for this mode yet.
