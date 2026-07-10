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

### Checkpoint 2: comprehensive baseline sweep + E-MILES win (this is a big one)

Ran a proper multi-seed (5 seeds), multi-sigma (30-350 km/s) sweep, generating
from real templates and fitting back with the same library (no mismatch by
construction), reporting bias AND seed-to-seed scatter separately (bias >>
scatter means the error bars from that scatter would NOT cover truth, a
stricter and more honest test than point-estimate accuracy alone).

**G/K-restricted MUSE library (10 real stars)**: revealed something new --
even in the "good" sigma~30-90 regime, there's a *persistent, systematic*
V bias of -6 to -9.5 km/s (small scatter, ~1-2 km/s, but a real offset that
scatter alone wouldn't explain/cover). Traced this to one specific star,
`HD099648_av.dat` (G8Iab, a *supergiant*, not part of the true generating
population) absorbing 28-71% of the fitted weight in every seed tested, even
though it's not physically present -- luminosity-class-I stars apparently
have different enough line profiles from the class-III giants they're being
used to approximate that this creates a real, systematic pull. At high sigma
(200+) the same full G/K library becomes outright chaotic again (std=33 km/s
at sigma=350), matching earlier findings.

**E-MILES (20 SSP templates, old/moderate-metallicity subgrid, bundled with
the installed pPXF package -- no download needed)**: dramatically better at
*every* sigma tested, not just the low end:

| sigma | G/K MUSE: V bias (std) | E-MILES: V bias (std) |
|---|---|---|
| 30  | -9.5 (1.2)  | -2.0 (1.2) |
| 50  | -7.9 (1.3)  | -1.4 (1.1) |
| 70  | -7.0 (1.8)  | -1.6 (1.4) |
| 90  | -6.5 (2.2)  | -2.4 (1.8) |
| 120 | -8.1 (4.1)  | -2.7 (1.2) |
| 160 | -4.1 (3.8)  | -0.9 (1.7) |
| 200 | -7.9 (11.1) | -3.3 (3.5) |
| 250 | -20.5 (18.3)| -11.5 (3.6) |
| 300 | -51.9 (26.2)| -25.2 (7.4) |
| 350 | -73.5 (33.3)| -43.1 (14.6)|

E-MILES is a dense, physically smooth grid (25 ages x 6 metallicities in the
full set), fundamentally more well-behaved than a handful of individual,
idiosyncratic real stars -- exactly the pPXF-community rationale for using it,
now directly confirmed on this problem. sigma <= 160 looks close to solved
with E-MILES (bias consistent with the ~1-2 km/s scatter, i.e. error bars
built from that scatter should genuinely cover truth). sigma >= 200 still has
a real, if much smaller, residual bias to chase.

Next: (1) check whether restricting the MUSE library to III-only giants (no
supergiants) closes the gap without needing E-MILES, for completeness: (2)
root-cause the remaining E-MILES high-sigma bias; (3) move to finalizing
E-MILES as the primary template recommendation and rebuilding notebook 02
around it.
