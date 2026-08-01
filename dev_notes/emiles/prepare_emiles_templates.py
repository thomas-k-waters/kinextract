"""Builds a curated E-MILES SSP template set, LSF-matched to MUSE, in
kinextract's plain-text template format (wavelength, flux) -- an
alternative to the 33-star MUSE Library of Stellar Spectra used
everywhere else in this package, to test whether E-MILES's much broader
spectral-type/age/metallicity coverage reduces bias from gaps in that
library, and because a general library is not optional for instruments
without their own dedicated one (e.g. HST STIS).

**Source**: the user's own downloaded E-MILES base grid
(`EMILES_BASTI_BASE_BI_FITS.tar.gz`, BaSTI isochrones, bimodal IMF,
scaled-solar/"baseFe") -- read directly via `astropy.io.fits`, not via
any third-party package's bundled/precomputed grid. kinextract has
no dependency on pPXF anywhere, including in prep scripts like this one;
an earlier draft of this script used pPXF's bundled E-MILES convenience
file as a shortcut and was corrected -- that file is someone else's
already-processed intermediate product with its own undocumented
resampling/selection choices, whereas the user's own tar is the primary
source data.

**Selected subgrid**: IMF slope bi1.30 (the standard reference slope),
5 ages (2, 5, 8, 11, 14 Gyr) x 5 metallicities (-0.66, -0.25, +0.06,
+0.26, +0.40 dex) = 25 templates -- old, near-solar-to-moderately-super-
solar populations, matching the physical expectation for an early-type/S0
target (NGC 5102) without pulling in young/metal-poor SSPs that aren't
relevant and would needlessly reintroduce a too-similar-templates
degeneracy problem. Extracted from the tar via `extract_curated_subgrid`
below (only the needed 25 of ~8900 files -- no need to unpack the whole
archive).

**LSF matching**: E-MILES SSP spectra are documented (Vazdekis et al.)
to have a constant 2.51 Angstrom FWHM from 3540-8950 Angstrom (the
MILES/MIUSCAT-built part of the grid); beyond 8950 Angstrom a coarser,
separately-sourced NIR extension takes over with a resolution this
script does not characterize. This package's fit windows (the CaII
triplet, ~8400-8800 Angstrom) sit safely inside the 2.51 Angstrom region,
so `TEMPLATE_FWHM_A = 2.51` is used as a constant across the whole
4750-9350 Angstrom output grid -- correct for everything actually used in
fitting, but do not trust the loaded templates redward of ~8950 Angstrom
without revisiting this. Matched to MUSE's own wavelength-dependent LSF
(`muse_lsf_fwhm`) via `kinextract.lsf.match_gaussian_lsf`. MUSE's
FWHM curve dips below 2.51 Angstrom over a broad stretch (~6760-8900
Angstrom, not just right at the CaT triplet) since its minimum sits near
the CaT wavelengths -- so `cannot_sharpen` is True for essentially the
WHOLE fit window (~46% of pixels across the full 4750-9350 Angstrom grid),
meaning the template-side correction is a no-op there: MUSE is already
at least as sharp as E-MILES throughout the region actually used for
fitting. The residual gap is real but small (~0.4 Angstrom FWHM at 8600
Angstrom, a ~6 km/s sigma-equivalent) -- negligible above sigma~100 km/s,
worth keeping in mind as a small confound specifically at the low-sigma
end (~20-30 km/s) of any E-MILES-vs-MUSE-template comparison. Not
corrected here (would require broadening the *data* slightly instead,
the reverse direction from what this function does); flagged rather than
silently ignored.

**Continuum normalization**: simple median-divide over the full output
grid -- an overall scale factor `fit_losvd`'s NNLS weight step absorbs
anyway; what matters is smooth and strictly positive, not the specific
normalization constant.

Usage: python prepare_emiles_templates.py [out_dir]
"""
import sys, os, tarfile, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
from astropy.io import fits

from kinextract.lsf import match_gaussian_lsf, muse_lsf_fwhm

EMILES_TAR = os.path.expanduser("~/Downloads/EMILES_BASTI_BASE_BI_FITS.tar.gz")
TEMPLATE_FWHM_A = 2.51  # constant, valid 3540-8950 A -- see module docstring

WAVEMIN, STEP, N_PIX = 4750.0, 1.25, 3681
WAVELENGTH = WAVEMIN + np.arange(N_PIX) * STEP

IMF_SLOPE = "1.30"
AGES_GYR = ["02.0000", "05.0000", "08.0000", "11.0000", "14.0000"]
METALS = ["m0.66", "m0.25", "p0.06", "p0.26", "p0.40"]


def _emiles_filename(imf_slope, metal_code, age_code):
    return f"Ebi{imf_slope}Z{metal_code}T{age_code}_iTp0.00_baseFe.fits"


def extract_curated_subgrid(tar_path=EMILES_TAR, ages=AGES_GYR, metals=METALS, imf_slope=IMF_SLOPE):
    """Read the selected (age, metallicity) FITS members directly out of
    the tar.gz without extracting the whole ~9000-file, ~1.4 GB archive.
    Returns a list of (name, wavelength, flux) tuples, one per member,
    where `wavelength` is built from the FITS header's own CRVAL1/CDELT1
    (linear WCS, as E-MILES ships it).
    """
    wanted = {_emiles_filename(imf_slope, z, t): (z, t) for z in metals for t in ages}
    found = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            base = os.path.basename(member.name)
            if base in wanted:
                data = tf.extractfile(member).read()
                with fits.open(io.BytesIO(data)) as hdul:
                    header = hdul[0].header
                    flux = hdul[0].data.astype(float)
                wave = header["CRVAL1"] + np.arange(header["NAXIS1"]) * header["CDELT1"]
                z, t = wanted[base]
                found[base] = (f"emiles_z{z}_t{t}", wave, flux)

    missing = set(wanted) - set(found)
    if missing:
        raise FileNotFoundError(f"{len(missing)} requested E-MILES members not found in {tar_path}: "
                                 f"{sorted(missing)[:5]}...")
    return [found[k] for k in wanted]


def build_emiles_template_set(out_dir, tar_path=EMILES_TAR):
    raw_templates = extract_curated_subgrid(tar_path)

    os.makedirs(out_dir, exist_ok=True)
    names = []
    n_cannot_sharpen_total = 0
    for name, wave, flux in raw_templates:
        flux_on_grid = np.interp(WAVELENGTH, wave, flux)
        matched, cannot_sharpen = match_gaussian_lsf(
            WAVELENGTH, flux_on_grid, template_fwhm=TEMPLATE_FWHM_A, target_fwhm=muse_lsf_fwhm,
        )
        n_cannot_sharpen_total += cannot_sharpen.sum()
        pos = matched > 0
        med = np.nanmedian(matched[pos]) if pos.any() else 1.0
        normalized = matched / med if med > 0 else matched

        fname = f"{name}.txt"
        np.savetxt(os.path.join(out_dir, fname), np.column_stack([WAVELENGTH, normalized]),
                   fmt="%10.4f  %14.8f")
        names.append(fname)

    tlist_path = os.path.join(out_dir, "Tlist")
    with open(tlist_path, "w") as f:
        f.write("\n".join(names) + "\n")

    frac_unmatched = n_cannot_sharpen_total / (len(names) * N_PIX)
    print(f"Wrote {len(names)} E-MILES templates to {out_dir}")
    print(f"cannot_sharpen fraction across all templates/pixels: {frac_unmatched:.4f} "
          f"(expected: ~0.46 -- MUSE's LSF dips below 2.51 A FWHM over ~6760-8900 A, covering the whole "
          f"CaT fit window, so the template-side correction is a no-op there; see module docstring)")
    return tlist_path


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "emiles_templates")
    build_emiles_template_set(out_dir)
