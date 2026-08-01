# The "skewed LOSVD" in 02_realistic_mock_fit -- diagnosed (2026-07-27)

## 1. The method is not broken. At S/N -> infinity it is exact.

Same mock, same free template weights, noise reduced to S/N=1e4:

| | V | sigma | h3 | h4 | max resid |
|---|---|---|---|---|---|
| truth curve | +34.6 | 98.4 | +0.036 | +0.050 | -- |
| recovered (noiseless) | +34.6 | 98.4 | +0.036 | +0.050 | **0.2%** |

Template weights recovered *exactly* ([0.15, 0.35, 0.35, 0.15]). So the
forward model, penalty, basis and selection are all sound. **Everything
seen at S/N=60 is noise-driven bias, not a structural defect.**

## 2. Part of the reported "bias" is a reporting bug

The notebook prints `truth` as the **nominal Gauss-Hermite parameters**
(V=30, sigma=90, h3=0.03). Those are not the moments of the curve it
plots. The truth *curve's* own moments -- measured through the same
estimator the fit is scored with -- are **V=+34.6, sigma=98.4,
h3=+0.036**. Comparing a moment estimator against GH parameters
manufactures ~5 km/s of V "bias" and ~8 km/s of sigma "bias" that is not
there. Fix the comparison before tuning anything against it.

## 3. The real shape defect: order-3 destroys h4

Truth has h4=+0.050 (a peakier-than-Gaussian core). Recovered h4, 6-seed
means, notebook-02 mock:

| penalty | h4 at its best lam | best max resid |
|---|---|---|
| order 3 (current default) | **+0.000 at every usable lam** | 7.73% |
| order 2 | +0.033 (lam=4) | **6.19%** |

Lower h4 = lower, broader peak -- exactly the peak deficit visible in the
plots. The order-3 null space is *any Gaussian*, so shrinking toward it
erases precisely the structure this truth has. This is the documented
Stage-22 tradeoff, now measured at curve level rather than moment level.

## 4. The 01-vs-02 comparison is not like-for-like

01's 3.4% max residual is **one lucky noise draw (seed 42)**. Averaged
over 6 seeds at the same sigma/S-N, notebook 01's own case gives:

| penalty | best max resid (6-seed mean), NB-01 Gaussian sigma=50 |
|---|---|
| order 3 | 7.44% (lam=4096) |
| order 2 | 7.51% (lam=1) |

So **typical** performance on the "easy" benchmark is ~7.5%, statistically
indistinguishable from notebook 02's 7.7%. Notebook 02 is not performing
worse than notebook 01; a lucky seed is being compared against a typical
one. Any target set from a single seed will not be reproducible.

## 5. lam selection is suboptimal for both orders

| case | order | selection picks | best available |
|---|---|---|---|
| NB-02 | 3 | lam=32768 -> 7.91% | lam=8 -> 7.73% |
| NB-02 | 2 | lam=1 -> 8.11% | lam=4 -> **6.19%** |

Order 2's optimum also *moves with sigma* (lam=1 at sigma=50, lam=4 at
sigma=90) and degrades sharply either side of it (lam=4 at sigma=50 gives
10.3%), which is why order 3 -- flat and robust in lam -- was preferred
despite killing h4. That is the genuine tension: **order 3 is robust but
erases h4; order 2 recovers h4 but needs a lam that selection does not
find.**

## 6. What the user's actual requirement implies

The requirement is *coverage*: "the true LOSVD within the error bars over
the whole profile." In the plots the truth's peak sits ~3x the bootstrap
band width above the recovered peak. So the failure is **bias larger than
the quoted uncertainty**, i.e. the bootstrap band does not include
regularization bias -- an already-documented property of this bootstrap
(it resamples noise, not the penalty's shrinkage). Two independent levers:
reduce the bias (penalty/lam), and make the band honest about it.

## Recommended next actions, in order of expected value

1. **Fix the truth-comparison bug in notebooks 01/02** (compare against
   the truth curve's own moments). Cheap, and removes phantom bias.
2. **Score everything over >=6 seeds, never one.** The current benchmark
   is a single draw.
3. **Decide the penalty order on evidence per regime**, or find a penalty
   whose null space is the *Gauss-Hermite* family rather than the Gaussian
   family -- that would keep order-3's robustness without erasing h4. This
   is the real open research step and is not a small change.
4. Make the uncertainty band include regularization bias, so coverage
   means what it claims.
