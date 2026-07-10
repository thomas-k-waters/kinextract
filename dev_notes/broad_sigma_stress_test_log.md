# Autonomous stress-test session: reliable LOSVD recovery across sigma regimes

Branch: `claude/broad-sigma-losvd-stress-test` (local only, not pushed; `main` is
untouched). Every meaningful step is a separate commit on this branch — use
`git log` to see the full sequence, `git diff <commit>^..<commit>` to inspect any
one step, `git checkout <commit> -- <path>` to revert a single file, or just
`git checkout main` to walk away from all of it.

Goal (from the user, verbatim intent): the realistic mock fit notebook should
reliably recover the true LOSVD within the error bars of the recovered LOSVD,
for as wide a sigma range as possible, with the high-sigma regime specifically
strong for the next project (NGC 4751).

## IMPORTANT recalibration finding (read this first)

Before diving into more sigma=140/350 testing, I checked NGC 4751's actual
data at `/Users/waterstk/Documents/Gultekin_Astrophysics/EHT_Project/data/N4751/`.
The legacy Fortran pipeline's own real kinematic extraction for this galaxy
(`misc/pallmc.out`, 40 bins spanning r=0.01-1.57 arcsec) shows:

  sigma: min=42.7, max=69.5, median=50.2 km/s

**Not anywhere near 349 km/s.** The 349 km/s figure is almost certainly a
large-aperture/effective-radius integrated velocity dispersion from a
different kind of measurement (e.g. a single big-aperture spectrum), not what
kinextract needs to extract per-bin from this actual MUSE data. N4751's real,
per-bin kinematics sit in essentially the *same* regime as N5102's own real
data (35-47 km/s) — call it sigma ~35-90 km/s with margin.

This matters a lot for how I'm prioritizing the rest of this session:
1. **Primary goal, directly serves both real galaxies**: bulletproof
   recovery across sigma ~30-100 km/s. This is the regime that actually
   matters for the data in hand.
2. **Secondary/stretch goal**: keep pushing the sigma=140-350 regime as far
   as I can, since the user did ask for "as wide a range as possible" and it's
   valuable general robustness — but I'm not letting it block or dominate the
   session now that I know it's not the regime NGC 4751's real data needs.

I have NOT verified whether the 349 km/s number applies to a different radius,
a different aperture, or a data-reduction detail I'm missing — flagging this
explicitly rather than assuming. Worth double-checking with the user when they
return, but proceeding on the safer assumption (their own real, already-reduced
data is the ground truth for what this notebook needs to validate against).

## Prior state (checkpoint 0, commit 1096f1d)

Summary of everything already tried before this autonomous session (see the
conversation this branch continues from, and the commit message itself):
- Confirmed MAP-anchored bootstrap error bounds match the legacy `.mcfit2`
  convention (errors.py).
- Confirmed real-template mismatch (fitting a mock built from templates NOT
  in the fitting library) causes severe bias -- resolved by generating mocks
  from real library templates.
- Confirmed the full 35-star MUSE library has severe, genuine multi-modal
  degeneracy at broad sigma (140 km/s: V recovered 65-164 across 5 noise
  seeds; sigma=350: even worse, V from -137 to +33).
- Confirmed restricting to G/K giants (10 stars, the user's own real
  practice) mostly fixes sigma=140 (V: 65-80 across seeds) but does NOT fix
  sigma=350 (V still -51 to +33).
- Built SVD-based template-basis reduction (`kinextract.templates.
  reduce_templates_svd`/`write_svd_reduced_templates`, `cfg.template_w_bounds`
  for the resulting mixed-sign weights) -- code is correct and tested (98/98
  tests pass), but empirically it's *not yet* an improvement: sigma=140
  becomes stable-but-biased (V~60 instead of 80), sigma=350 gets *worse*
  (more chaotic than the unreduced G/K library).

## Session log

(Entries added as I go, newest at the bottom.)
