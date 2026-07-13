"""Extract E-MILES SSP templates (bundled with pPXF) in the CaII-triplet
region, write them out in kinextract's template file format.

E-MILES is a dense, physically smooth grid (25 ages x 6 metallicities, here
subsampled) as opposed to the ~35 individual, idiosyncratic real stars in the
bundled MUSE library -- the standard pPXF-community remedy for the template
correlation/degeneracy problem confirmed in this stress-test session.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

EMILES_NPZ = "/opt/anaconda3/envs/astro/lib/python3.13/site-packages/ppxf/sps_models/spectra_emiles_9.0.npz"

WAVEMIN, STEP, N_PIX = 4750.0, 1.25, 3681
WAVELENGTH = WAVEMIN + np.arange(N_PIX) * STEP


def build_emiles_template_set(
    out_dir: str,
    age_idx: list[int] | None = None,
    metal_idx: list[int] | None = None,
    data_fwhm_A: float = 2.51,
) -> str:
    """Write a subgrid of E-MILES SSPs as kinextract template files + Tlist.

    Parameters
    ----------
    age_idx, metal_idx : list of int, optional
        Indices into the E-MILES age (25 values, 0.06-15.8 Gyr) / metallicity
        (6 values, -1.71 to +0.22 dex) grids to include. Default: a modest
        subgrid spanning old, moderate-to-high metallicity populations
        (typical of an early-type/bulge-dominated galaxy), not the full
        150-template grid, since young/metal-poor SSPs aren't physically
        relevant to this kind of target and needlessly reintroduce the
        degenerate-library problem this is meant to avoid.
    data_fwhm_A : float, optional
        E-MILES's own native resolution at ~8580 A (2.51 A FWHM per the
        bundled file's own `fwhm` array) -- returned for the caller to pass
        as `cfg.template_fwhm_A` alongside `cfg.data_fwhm_A` (MUSE's own
        LSF) for proper resolution matching.

    Returns
    -------
    str : path to the written Tlist.
    """
    d = np.load(EMILES_NPZ)
    lam = d["lam"]
    templates = d["templates"]  # (nlam, nages, nmetals)
    ages = d["ages"]
    metals = d["metals"]

    if age_idx is None:
        # Old populations only (>= ~2 Gyr), every other grid point to keep
        # the count modest -- indices 16, 18, 20, 22, 24 -> ages ~2.5-15.8 Gyr.
        age_idx = [16, 18, 20, 22, 24]
    if metal_idx is None:
        # Solar and moderately sub/super-solar -- indices 2,3,4,5 -> -0.71 to +0.22.
        metal_idx = [2, 3, 4, 5]

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    names = []
    for ai in age_idx:
        for mi in metal_idx:
            flux = templates[:, ai, mi]
            flux_on_grid = np.interp(WAVELENGTH, lam, flux)
            pos = flux_on_grid > 0
            med = np.nanmedian(flux_on_grid[pos]) if pos.any() else 1.0
            flux_on_grid = flux_on_grid / med if med > 0 else flux_on_grid
            name = f"emiles_age{ages[ai]:.3f}_z{metals[mi]:+.2f}.dat".replace("+", "p").replace("-", "m")
            np.savetxt(out_dir_p / name,
                       np.column_stack([WAVELENGTH, flux_on_grid, np.full(N_PIX, 0.001)]),
                       fmt="%10.4f  %14.8f  %12.8f")
            names.append(name)
    tlist_path = out_dir_p / "Tlist"
    tlist_path.write_text("\n".join(names) + "\n")
    print(f"Wrote {len(names)} E-MILES templates to {out_dir_p}")
    return str(tlist_path)


if __name__ == "__main__":
    build_emiles_template_set("dev_notes/stress_test/emiles_templates")
