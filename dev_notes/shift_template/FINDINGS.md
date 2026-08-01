# shift_template linear-interpolation bias: quantified and fixed (opt-in)

**Date: 2026-07-28**

## Background

Flagged in earlier validation work as an open item: `shift_template`
(`forward_model.py`) evaluates a Doppler-shifted template on the observed
wavelength grid via `np.interp` -- piecewise-linear interpolation. This
adds a fractional-shift-dependent effective broadening to real-data fits
(quoted earlier: 0 to ~17 km/s, period ~31 km/s across E-MILES design-
matrix columns), invisible to synthetic-mock validation because mocks are
conventionally built with the exact same design matrix the fitter uses --
whatever interpolation "error" exists cancels against itself by
construction. Real data has no such cancellation: the true redshift/
velocity offset between the galaxy and the template grid is arbitrary and
uncorrelated with the template's own native sampling.

## Quantified (no fitting involved -- isolates the mechanism itself)

`dev_notes/shift_template/diag01_interp_bias.py`. Ground truth: the
E-MILES native grid (confirmed uniform, 0.9 Angstrom/pix) cubic-spline-
resampled at 20x density, then linearly interpolated at the query points --
at 20x oversampling this is an accurate proxy for the continuous signal.

At ~8500 Angstrom, one native template pixel corresponds to
`c * 0.9 / 8500 ~ 31.7 km/s` -- matching the previously-quoted ~31 km/s
period exactly. Scanning velocity across one full period:

| fractional phase | linear max-rel error |
|---|---|
| 0.0 (on-grid) | 0.74-0.90% |
| ~0.5-0.6 (mid-pixel) | 1.13-1.17% (peak) |

The error is genuinely periodic in phase, not noise -- confirmed by
scanning two full periods and seeing the pattern repeat.

## The fix: Lanczos-windowed sinc resampling (opt-in)

`shift_template`/`build_template_stack` gained `method: "linear" |
"lanczos"` (default `"linear"`, unchanged) and `lanczos_a` (kernel
half-width, default 3). Requires a uniformly-sampled `template_wave`
(checked; raises `ValueError` otherwise) -- confirmed true for the shipped
E-MILES grid.

**Verified correct**: at exact grid alignment (query points landing
precisely on native template samples), Lanczos reproduces those samples to
numerical precision (`sinc(0)=1`, `sinc(nonzero integer)=0`) -- confirms
the implementation itself, not just its statistical behavior.

**Verified to fix the actual defect**: comparing linear and Lanczos against
the same oversampled ground truth, Lanczos's disagreement with that ground
truth does NOT show the same phase-periodic swing linear interpolation
does (max-rel error stays close to flat, ~0.56-0.59%, across the same
phase scan where linear swings 0.7-1.2%). What remains for Lanczos is a
smaller, phase-*independent* residual -- verified by a direct sanity check
that this floor is not itself a bug (exact-alignment case above) and is
better explained by real spectra not being perfectly band-limited (no
resampling kernel, sinc-based or otherwise, reproduces a non-band-limited
signal exactly) than by any periodic artifact. **This is the property that
actually matters for kinematics**: a phase-independent residual does not
masquerade as velocity-shift-dependent broadening the way linear
interpolation's periodic error can.

Unit test version of this comparison (`tests/test_forward_model.py::
test_shift_template_lanczos_removes_the_periodic_component_of_linear_
error`) uses the coefficient of variation of the max-error envelope across
one phase period as a cheap, robust proxy for "does the error swing
systematically with shift" -- asserts Lanczos's CV is less than half
linear's. Passes.

## Cost

`build_template_stack` (a 4-template, 150-velocity-bin case): 0.007s
(linear) vs. 0.468s (Lanczos a=3) -- ~67x slower in isolation, but this
function runs ONCE per fit setup (the design matrix is precomputed and
reused across every optimizer/NNLS iteration within `fit_losvd`), so this
adds roughly 6-9% to a typical ~5-8s total fit -- real, but affordable for
an opt-in correctness improvement.

## Status: opt-in, not the default

Consistent with this project's standing convention for every prior new
mechanism (`lam_wing`, `shape_aware_correction`, adaptive lam selection
before it was promoted, etc.): ships available and tested, but does not
silently change the numerical output of any existing fit. Promoting it to
default would need the same scale of validation this package's other
defaults have had (a stress-test grid showing it doesn't regress anything,
ideally direct confirmation on real data that the specific 0-17 km/s
effective-broadening signature shrinks). Not done tonight -- flagged as a
natural extension of the randomized validation-suite work
(`dev_notes/validation_suite/`) if real-data validation becomes a
priority.

## Verified

6 new tests in `tests/test_forward_model.py`, all passing: default-method
backward-compatibility (bit-identical to pre-existing `np.interp` calls),
exact-alignment correctness, the periodic-error-removal property itself,
non-uniform-grid rejection, unknown-method rejection, and
`build_template_stack`'s own method/lanczos_a plumbing reaching
`shift_template` correctly for both methods.
