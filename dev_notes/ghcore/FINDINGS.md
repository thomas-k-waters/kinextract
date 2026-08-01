# Option #1 (GH core x penalized spline): tested, and it fails for a
# structural reason. 2026-07-28

## What was built

    log b(v) = -w^2/2 + h3*H3(w) + h4*H4(w) + spline(v; c),   w = (v-V)/sigma

with only `c` penalized, so the penalty's null space is the 5-parameter
family {A, V, sigma, h3, h4}. Core parameters fit JOINTLY with the spline
(not anchored to a pilot fit -- that variant already failed earlier for a
different reason: pilot h3/h4 scatter injected as bias).

Prototype: `dev_notes/ghcore/proto.py`.

## Result: worse than both existing baselines

NB02 mock (V=30, sigma=90, h3=0.03, h4=0.05, S/N=60), 4 seeds, vs. the
truth curve's own moments (V=+34.6, sigma=98.4, h3=+0.036, h4=+0.050):

| method | max resid | recovered h4 |
|---|---|---|
| order 3 (current default) | 7.73% | +0.000 |
| order 2, best lam | **6.19%** | +0.033 |
| GH core x spline, best lam | 14.9% | **-0.014** |

and at large lam (where it should collapse to a clean parametric GH fit)
it degenerates completely: sigma runs to its lower bound (5 km/s) and max
residual exceeds 1000%.

## Root cause -- and it is not fixable by tuning

**In the exponent, the quartic term beats the quadratic.** `H4(w)` grows
as `w^4`, so `-w^2/2 + h4*H4(w)` turns positive and diverges for *any*
h4 > 0:

| w | 0 | 2 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| exponent at h4=+0.05 | +0.03 | -1.8 | +0.5 | +10 | +31 | +68 |

So the model cannot represent a positive-h4 LOSVD at all on a +-7 sigma
grid: b(v) blows up in the wings. The fits confirm the mechanism rather
than merely being consistent with it -- **recovered core h4 ran to
NEGATIVE values (-0.35, its bound) on a truth with h4 = +0.05**, because
the optimizer is fleeing the divergence, not fitting the data.

Two false escapes, both checked:

* **Work in linear space instead** (`(1 + h3H3 + h4H4) * exp(-w^2/2)`,
  the textbook GH form): the polynomial goes negative in the wings. On a
  +-7 sigma grid it stays positive only for |h4| < 0.0005 -- i.e. zero.
  This is the same truncated-GH pathology already documented in this
  package at h4 = -0.1.
* **Bound h4 tighter**: any bound loose enough to represent the truth
  (+0.05) is loose enough to diverge.

## The real lesson

**The Gauss-Hermite family is not a valid density family over a wide
velocity grid**, in either space. That is *why* "shrink toward a
Gaussian" (order 3) is well-posed and "shrink toward Gauss-Hermite" is
not. The target family has to be one that is positive and integrable by
construction.

## Corrected direction (NOT yet tested)

Replace the GH core with an always-positive 4-parameter family that
contains the Gaussian as a special case, and keep everything else:

    log b(v) = log f(v; V, sigma, skew, tail) + spline(v; c)

Candidates, in order of preference:

1. **Sinh-arcsinh (Jones & Pewsey 2009)** -- exactly 4 parameters
   (location, scale, skewness, tail-weight), strictly positive for all
   parameter values, reduces to the Gaussian at (skew=0, tail=1), and
   both tails decay. No divergence is possible by construction, which is
   precisely the failure above.
2. **Skew-t / skew-normal** -- positive, well-studied, but skew-normal
   cannot do kurtosis (so it cannot represent h4 at all), and skew-t's
   tail parameter is awkward near the Gaussian limit.
3. **Gaussian mixture (2 components)** -- always positive and very
   flexible, but its parameters are badly non-identifiable and it can
   represent the bimodal pathologies we are specifically trying to
   exclude.

Sinh-arcsinh is the one worth trying. Its (skew, tail) map onto h3/h4
monotonically near the Gaussian, so the null space is "all
near-Gaussian LOSVDs with skew and kurtosis" -- the family actually
wanted -- without ever leaving the space of valid densities.

---

## Sinh-arcsinh (SHASH) core — tested, and it settles the whole question

**Date: 2026-07-28**

Replaced the divergent GH core with a sinh-arcsinh core (Jones & Pewsey 2009):

    log b(v) = -S^2/2 + log1p(S^2)/2 - log1p(x^2)/2,
    x = (v-V)/sigma,  S = sinh(delta*asinh(x) - eps)

### It fixes the divergence

The GH core failed structurally: `H4(w)` grows like `w^4` and beats `-w^2/2`, so
`b` diverges for any h4>0. SHASH cannot diverge — verified across the parameter
box, edge log-density always strongly negative (Gaussian limit -32.9/-27.7;
heavy tails -4.4/-4.0; light tails -941/-716). Skew and kurtosis are carried by
`eps`/`delta`, which reshape the *argument* of a Gaussian rather than adding a
polynomial to the exponent. So SHASH is the right shape family.

### But it does not beat the spline, and here is why

SHASH core x spline, NB02 mock (V=30, sigma=90, h3=0.03, h4=0.05, 4-template
near-collinear E-MILES, S/N=60, 4 seeds), truth curve V=+34.6/98.4/+0.036/+0.050:

| lam | max resid | h4 |
|---|---|---|
| 1 (best) | 12.77% | +0.002 |
| 10 | 12.97% | -0.002 |
| 100 | 39.59% | -0.093 |
| >=1000 | collapses (sigma -> 2.8, resid >2000%) | |

vs. baselines: **order-3 spline 7.73% / h4=0.000, order-2 best-lam 6.19% /
h4=0.033.** SHASH loses to both. The high-lam collapse is a numerical dead zone:
small sigma sends `delta*asinh(x)` large, `S^2` overflows, `exp(-S^2/2)`
underflows to 0, and the gradient vanishes, so L-BFGS-B cannot climb back out.

### The decisive test: it is noise, not the null space

Strip the spline entirely and fit the **4-parameter SHASH core alone** — no
penalty, no null space, nothing for a bad penalty to hide in:

| case | max resid | sigma | h3 | h4 |
|---|---|---|---|---|
| S/N=60 | 11.51% | 96.5 | +0.043 | **-0.023** |
| S/N=200 | 3.45% | 97.6 | +0.036 | **+0.028** |
| S/N=1000 | 1.57% | 98.2 | +0.034 | **+0.045** |
| noiseless | 1.77% | 98.3 | +0.033 | **+0.050** |

truth: sigma=98.4, h3=+0.036, h4=+0.050.

Recovery is **exact when noiseless and degrades monotonically with S/N**. h4 walks
+0.050 -> +0.045 -> +0.028 -> -0.023 as S/N falls 1e4 -> 1000 -> 200 -> 60. A
4-parameter model with zero regularization has no null space to leak into and no
penalty bias, and it *still* lands at 11.5% at S/N=60.

**Conclusion: the NB02 residual floor at S/N=60 is set by photon noise and
template degeneracy, not by the roughness penalty's null space.** No penalty
redesign — D2, D3, SHASH, GH, unimodality, wing tapering — can push below it,
which retroactively explains why every one of those attempts failed. It also
means the existing spline is not the problem: at 6.19-7.73% it is already
*better* than a maximally-constrained 4-parameter parametric fit at 11.5%.

Also worth stating plainly: the truth curve is **not** inside the error bars at
S/N=60 because at S/N=60 the data do not determine h4. That is a property of the
mock, not a bug to fix.

### Control failure and correction (logged per standing rule)

The first run of the S/N ladder included a "true template weights held fixed"
control that returned 100% residual and railed to its bounds from every start.
That was **my bug, not a result**: `shash_b` normalizes to unit *integral* while
the mock's `bt` normalizes to unit *sum*, so the fixed-weight residual was off by
the grid spacing `dv`. NNLS silently absorbed the factor in the free-weight rows,
which is why only the control exploded. With `dv` restored the control passes —
cost 2.94e2, 1.77% resid, sigma=98.2, h3=+0.033, h4=+0.049 — confirming the
free-weight rows above are sound.

### Verdict

**Rejected as a replacement for the spline** (loses on residual, collapses at high
lam). Kept in `dev_notes/ghcore/shash.py` as the diagnostic that established the
noise floor. Prototype only — not promoted into `src/`.
