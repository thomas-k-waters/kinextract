# pPXF vs kinextract_lite on synthetic cubes with known truth

**Date: 2026-07-29/30 (overnight)**

## Why synthetic cubes, not just the real NGC 5102 comparison

`dev_notes/scratch_notebooks/ppxf_vs_kinextract_comparison.ipynb` already runs
both engines on the same dartboard-binned cube, but only against real data,
where there is no ground truth to score against -- only "the two methods
disagree," not "which one is right." `build_mock_cube` uses a single,
spatially-constant true `(V, sigma, h3, h4)` LOSVD at every spaxel, modulated
by a declining Sersic surface brightness against a constant noise floor, so
one cube's dartboard bins span a whole range of S/N *for the same known
truth* -- turning one cube into a free S/N sweep with an actual answer to
check against.

Script: `dev_notes/ppxf_comparison/synthetic_cube_comparison.py`. Both engines
fit the exact same per-bin spectra with the exact same E-MILES templates the
mock was built from (not the full grid) -- no template-selection freedom on
either side, isolating the LOSVD-recovery step itself. 41 dartboard bins/cube
(radial x angular, PA irrelevant here since there's no true rotation),
`RMAX=5"`, one draw per bin (no repeated noise realizations per S/N level --
see limitations).

## Four cubes, chosen to probe specific open questions

| cube | true sigma | shape | central S/N | probes |
|---|---|---|---|---|
| `baseline_sig50` | 50 | mild spline wiggle | ~1100 | general baseline |
| `high_sigma150` | 150 | mild spline wiggle | ~1100 | the "sloppy manifold" regime flagged open in `AUDIT_BRIEFING.md` |
| `genuine_wings_h3h4` | 80 (nominal) | GH, true h3=0.08, h4=0.12 | ~1100 | genuinely non-Gaussian truth -- `AUDIT_BRIEFING.md`'s "h3 severely suppressed" issue |
| `faint_sig80` | 80 | mild spline wiggle | ~300 (mostly S/N<15) | low-S/N robustness |

(Truth-curve moments differ slightly from the nominal config values quoted
above -- e.g. `genuine_wings_h3h4`'s actual truth is sherm=97.62, h3=0.0999,
h4=0.1003 -- this project's own established practice, see
`dev_notes/validation_suite/FINDINGS.md`'s GH-truth-mismatch entry, is to
always score against the curve's own moments, which is what's done
throughout.)

## Headline result: kinextract_lite wins on sigma in every cube, at every S/N above the noise floor

| cube | sigma MAE (kin / pPXF) | sigma RMSE (kin / pPXF) | bins where kin wins |
|---|---|---|---|
| baseline_sig50 | 8.85 / 12.18 | 11.70 / 18.04 | 66% |
| high_sigma150 | 23.02 / 29.68 | 38.15 / 42.45 | 63% |
| genuine_wings_h3h4 | 17.18 / 22.44 | 22.74 / 36.43 | 56% |
| faint_sig80 (all bins) | 48.23 / 41.43 | 85.52 / 56.79 | 73% |
| **all 164 bins pooled** | **24.32 / 26.43** | -- | **65%** |

`faint_sig80`'s aggregate MAE/RMSE is the one row where pPXF looks better --
misleading on its own, see below.

## The one place kinextract_lite is worse: truly pathological S/N (<~3/pixel)

Splitting `faint_sig80` by S/N threshold shows the aggregate number above is
driven entirely by a handful of extreme-low-S/N bins, not a real regime where
pPXF wins:

| S/N cut | n bins | kin sigma MAE | pPXF sigma MAE |
|---|---|---|---|
| all | 41 | 48.23 | 41.43 |
| >= 3 | 29 | 29.74 | 37.89 |
| >= 5 | 21 | 17.89 | 28.22 |
| >= 8 | 11 | 8.88 | 19.58 |
| < 3 | 12 | 92.90 | 49.97 |

Below S/N~3/pixel, kinextract_lite's flexible non-parametric LOSVD
occasionally produces genuinely wild point estimates (e.g. one bin recovered
sigma=375 km/s against a true 80; another 322 km/s) -- more extreme than
pPXF's worst failures in the same bins (its fixed 4-parameter Gauss-Hermite
form is more constrained and degrades more gracefully when there's
essentially no signal to fit). Once S/N clears roughly 5, kinextract_lite is
unambiguously better and the gap widens with S/N (2.2x better MAE by S/N>=8).
**Practical takeaway**: this package's own flexibility is a genuine
liability only in a regime (S/N<~3/pixel) where arguably neither method
should be trusted; nothing here suggests kinextract_lite is worse in any
regime a real observation would actually be used in.

## Where pPXF actually wins: h3/h4 when the truth is genuinely Gauss-Hermite-shaped

| cube | h3 MAE (kin / pPXF) | h4 MAE (kin / pPXF) |
|---|---|---|
| baseline_sig50 (truth h3=-0.03, h4~0) | 0.0323 / 0.0387 | **0.0012 / 0.0200** |
| high_sigma150 (truth h3=0.02, h4=-0.02) | 0.0177 / 0.0284 | 0.0228 / 0.0304 |
| genuine_wings_h3h4 (truth h3=0.10, h4=0.10) | 0.0929 / **0.0801** | 0.0973 / **0.0904** |
| faint_sig80 (truth h3=-0.03, h4~0, mostly low S/N) | 0.0626 / 0.0324 | 0.0627 / 0.0145 |

kinextract_lite's h4 recovery is dramatically better when the true LOSVD is
*near*-Gaussian (16x lower MAE in `baseline_sig50`) -- expected and correct:
the order-3 penalty's null space is any Gaussian log-density
(`VALIDATION_CAMPAIGN.md` Stage 22), so it shrinks toward exactly the right
answer there. That same shrinkage becomes a mild liability in
`genuine_wings_h3h4`, where the truth has real h3=0.10/h4=0.10 -- pPXF's
fixed Gauss-Hermite parametrization has no shrinkage to fight against a
truth that happens to already be exactly that shape, so it edges out
kinextract_lite on h3/h4 specifically there (both still recover sigma
better via kinextract_lite, though). This is the same tradeoff already
named in `PARAMETER_GUIDE.md` ("order 3 shrinks real h4 by ~0.03-0.05... a
regime where h4 is near the measurement noise floor regardless") and
`AUDIT_BRIEFING.md`'s open "h3 recovery is severely suppressed" issue --
this comparison gives that known limitation a concrete pPXF-relative
number for the first time, rather than just a synthetic-only bias measurement.

`faint_sig80`'s h3/h4 numbers mostly reflect the same low-S/N breakdown as
its sigma numbers above, not an independent effect.

## An important asymmetry in the comparison, not fully controlled for

pPXF here uses `degree=6` (additive) + `mdegree=2` (multiplicative)
polynomial continuum freedom, matching the real-data notebook's own
convention; kinextract_lite's fits used no continuum cofitting at all
(`cfg.continuum_knots=None`, matching this project's own default -- adaptive
lam-selection doesn't support continuum-cofitting yet, see `uncertainty.py`'s
docstring). Both methods see spectra built from the exact same templates
used to fit them, so there is no true continuum-shape mismatch to absorb in
this synthetic setup -- but pPXF's extra continuum freedom is still an
asymmetry in the two engines' effective degrees of freedom that this
comparison does not isolate. Worth a follow-up with continuum cofitting
matched on both sides (or removed from pPXF's call) before treating the
sigma-bias gap as entirely attributable to the LOSVD-parametrization
difference this comparison was designed to probe.

## Other honest limitations

- **One noise draw per bin, not repeated trials.** Each S/N level in the
  table above is one random realization, not an average over many --
  unlike `coverage_sweep.py`'s randomized-truth design elsewhere in this
  project. The per-cube MAE/RMSE numbers (pooling 41 bins spanning a wide
  S/N range) are more trustworthy than any single bin's own number.
- **41 bins/cube, one seed (0).** Same seed reused across all 4 cube
  configs (only the injected LOSVD/amplitude differ) -- a different seed
  would shuffle which specific bins land at which S/N, but the qualitative
  S/N-binned pattern above (kin wins above S/N~3-5, loses below) should be
  a real effect, not a seed artifact, given how cleanly it splits.
- **Only one template mixture, one galaxy profile, no rotation.** This
  isolates the LOSVD-recovery question cleanly but doesn't test the
  multi-template-identifiability or kinematic-PA-dependent effects the
  real-data dartboard notebook (and `AUDIT_BRIEFING.md`'s own "more
  templates costs identifiability" note) already documents separately.

## What this adds to the existing validation picture

The synthetic-only calibration sweeps elsewhere in this project (coverage
sweep, `build_emiles_calibration.py`) measure kinextract_lite's *own* bias
in isolation. This is the first head-to-head, same-data, same-templates
comparison against the field-standard alternative with ground truth
available, and it's a genuinely good result for the core recovery method:
kinextract_lite wins on sigma bias in every regime that isn't pathologically
low S/N, wins dramatically on h4 when the truth is near-Gaussian (its
designed-for regime), and loses only the specific, already-documented,
already-understood h3/h4-under-genuine-wings tradeoff -- not a surprise
finding, but a concrete number for a previously qualitative claim.
