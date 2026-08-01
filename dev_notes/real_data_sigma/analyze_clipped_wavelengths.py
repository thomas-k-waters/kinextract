"""Where does notebook 08's real-data cleaning (robust_scale=True,
clip_sigma=3.0) actually clip pixels? Loads the just-finished full-galaxy
cache and checks whether clipped rest-frame wavelengths cluster near the
Ca II triplet (`cleaning.CAII_TRIPLET_CENTERS_A`) -- the user's own
suspected failure mode -- or are scattered (more consistent with
continuum-shaped sky-residual contamination that per-pixel clipping can't
distinguish from signal). `wave_fit` is already rest-frame per bin (the
8400-8800 A fit window), so a real CaII effect should cluster tightly
across bins despite each bin's different systemic+rotation velocity,
while sky-residual contamination (fixed in observed frame) should smear
out across different rest-frame positions bin to bin.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

sys.path.insert(0, "src")
from kinextract.cleaning import CAII_TRIPLET_CENTERS_A, CAII_TRIPLET_HALF_WIDTH_A

CACHE = "examples/notebooks/cache/full_galaxy_rmax29.7_pa135.8_nt5_nb219_rho0.88_nboot15_seed0_clip3.0_iter3_robustTrue.pkl"


def main():
    with open(CACHE, "rb") as f:
        data = pickle.load(f)

    all_clipped_wave = []
    all_wave_fit = []
    per_bin_clip_frac = []
    for bin_id, d in data.items():
        wave_fit = d["wave_fit"]
        clipped_mask = d["clipped_mask"]
        all_clipped_wave.append(wave_fit[clipped_mask])
        all_wave_fit.append(wave_fit)
        per_bin_clip_frac.append(clipped_mask.mean())

    all_clipped_wave = np.concatenate(all_clipped_wave)
    n_total_clipped = len(all_clipped_wave)
    n_total_pix = sum(len(w) for w in all_wave_fit)

    print(f"Bins: {len(data)}")
    print(f"Total pixels: {n_total_pix}, total clipped: {n_total_clipped} "
          f"({100*n_total_clipped/n_total_pix:.1f}%)")
    print(f"Per-bin clip fraction: median {100*np.median(per_bin_clip_frac):.1f}%, "
          f"mean {100*np.mean(per_bin_clip_frac):.1f}%, "
          f"max {100*np.max(per_bin_clip_frac):.1f}%, "
          f"n_bins>10% {sum(f > 0.10 for f in per_bin_clip_frac)}/{len(data)}")

    dist_to_nearest_caii = np.min(
        np.abs(all_clipped_wave[:, None] - np.array(CAII_TRIPLET_CENTERS_A)[None, :]), axis=1)
    within_half_width = dist_to_nearest_caii <= CAII_TRIPLET_HALF_WIDTH_A
    print(f"\nClipped pixels within {CAII_TRIPLET_HALF_WIDTH_A} A of a CaII center: "
          f"{within_half_width.sum()}/{n_total_clipped} ({100*within_half_width.mean():.1f}%)")

    # Null baseline: what fraction of the fit window itself falls within
    # CAII_TRIPLET_HALF_WIDTH_A of a center, i.e. what we'd expect from
    # uniformly-scattered clipping with no CaII preference at all.
    wavemin, wavemax = 8400.0, 8800.0
    grid = np.linspace(wavemin, wavemax, 100000)
    dist_grid = np.min(np.abs(grid[:, None] - np.array(CAII_TRIPLET_CENTERS_A)[None, :]), axis=1)
    null_frac = (dist_grid <= CAII_TRIPLET_HALF_WIDTH_A).mean()
    print(f"Null (uniform-scatter) expectation: {100*null_frac:.1f}% of the fit window "
          f"falls within {CAII_TRIPLET_HALF_WIDTH_A} A of a CaII center")

    # Histogram of clipped wavelengths across the fit window, coarse bins,
    # to see if there's a dominant peak anywhere (CaII or otherwise).
    hist, edges = np.histogram(all_clipped_wave, bins=40, range=(wavemin, wavemax))
    print("\nClipped-wavelength histogram (40 bins across 8400-8800 A):")
    peak_idx = np.argsort(hist)[::-1][:8]
    for i in sorted(peak_idx):
        center = 0.5 * (edges[i] + edges[i + 1])
        nearest_caii = min(CAII_TRIPLET_CENTERS_A, key=lambda c: abs(c - center))
        print(f"  {edges[i]:.0f}-{edges[i+1]:.0f} A: {hist[i]:4d} clipped "
              f"(nearest CaII center {nearest_caii:.1f}, {abs(nearest_caii-center):.1f} A away)")


if __name__ == "__main__":
    main()
