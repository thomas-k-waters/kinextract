"""Broad-population variant of harness_emiles: generates mocks from a
SMOOTH, realistic mixture spanning most/all of the fitting SSP grid (not
just 2 discrete SSPs), to test whether the V bias found with the simple
2-SSP mock is an artifact of that mock's artificial simplicity relative to
the fitting library's flexibility, or a genuine kinextract weakness that
would also show up on a more realistically complex population."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndimage_shift

sys.path.insert(0, str(Path(__file__).parent))
from harness import WAVELENGTH, WAVEMIN, STEP, N_PIX, CEE, LAM_CENTER, write_spec_file  # noqa: E402
from prepare_emiles import EMILES_NPZ  # noqa: E402

FIT_AGE_IDX = [16, 18, 20, 22, 24]
FIT_METAL_IDX = [2, 3, 4, 5]

# Smooth, unimodal-in-both-dimensions weight profile spanning ALL 20 fitting
# templates -- old-age-dominated (typical quiescent early-type), roughly
# solar-peaked metallicity, no single template above ~10% of the light.
AGE_WEIGHTS = np.array([0.05, 0.10, 0.25, 0.30, 0.30])   # over FIT_AGE_IDX
METAL_WEIGHTS = np.array([0.15, 0.25, 0.35, 0.25])        # over FIT_METAL_IDX
assert abs(AGE_WEIGHTS.sum() - 1.0) < 1e-9
assert abs(METAL_WEIGHTS.sum() - 1.0) < 1e-9

BROAD_WEIGHTS_2D = np.outer(AGE_WEIGHTS, METAL_WEIGHTS)  # (5, 4), sums to 1.0
BROAD_WEIGHTS_FLAT = BROAD_WEIGHTS_2D.flatten()  # matches Tlist's ai-then-mi ordering


def load_emiles_flux(age_idx: int, metal_idx: int) -> np.ndarray:
    d = np.load(EMILES_NPZ)
    flux = d["templates"][:, age_idx, metal_idx]
    lam = d["lam"]
    on_grid = np.interp(WAVELENGTH, lam, flux)
    pos = on_grid > 0
    med = np.nanmedian(on_grid[pos]) if pos.any() else 1.0
    return on_grid / med if med > 0 else on_grid


def build_broad_mock(true_v: float, true_sigma: float, seed: int,
                      noise: float = 250.0, cont_level: float = 12_000.0):
    templates = []
    for ai in FIT_AGE_IDX:
        for mi in FIT_METAL_IDX:
            templates.append(load_emiles_flux(ai, mi))
    templates = np.array(templates)  # (20, npix)
    pop_template = np.tensordot(BROAD_WEIGHTS_FLAT, templates, axes=1)

    sigma_pix = true_sigma * LAM_CENTER / (CEE * STEP)
    shift_pix = true_v * LAM_CENTER / (CEE * STEP)
    gal_norm = ndimage_shift(gaussian_filter(pop_template, sigma_pix), +shift_pix)

    x_norm = (WAVELENGTH - WAVELENGTH.mean()) / (0.5 * (WAVELENGTH[-1] - WAVELENGTH[0]))
    slope = 1.0 + 0.80 * x_norm + 0.5 * x_norm**2 - 0.10 * x_norm**3
    hump = 0.50 * np.exp(-0.5 * ((x_norm - 0.20) / 0.45) ** 2)
    fringes = 0.01 * np.sin(2 * np.pi * (WAVELENGTH - WAVELENGTH[0]) / 50.0)
    cont = cont_level * (slope + hump + fringes)

    RNG = np.random.default_rng(seed)
    gal = gal_norm * cont + RNG.normal(0.0, noise, N_PIX)
    errs = np.full(N_PIX, noise)
    return gal, errs
