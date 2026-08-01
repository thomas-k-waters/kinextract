"""E-MILES SSP template loading, as a first-class template source --
`load_emiles_templates`'s output is a plain `(templates, wavelength)`
pair in exactly the format `forward_model.build_template_stack` already
expects, so E-MILES becomes "just another template source" through the
same core API used for the real 33-star MUSE library everywhere else in
this package, not a separate code path.

**Two-stage design, mirroring the OG (non-lite) `kinextract` package's own
`templates.pack_templates_to_npz`/`load_packed_templates`, with one
deliberate improvement.** Packing (`pack_emiles_templates`, run once,
slow) reads the user's own raw E-MILES FITS release and produces a single
portable `.npz` grid; loading (`load_emiles_templates`, fast) reads that
npz and optionally selects an (age, metallicity) subset. The OG package
defers LSF matching to fit time using a single *scalar* FWHM for the
whole template library and the whole data spectrum -- no wavelength
dependence at all. This module instead does LSF matching at *pack* time,
using this package's own wavelength-dependent MUSE LSF curve
(`lsf.match_gaussian_lsf`/`lsf.muse_lsf_fwhm`) -- more physically correct
(E-MILES's LSF is not constant across the full 3000-10500A range this
grid spans, even though it is close to constant over the narrow CaT
window this package's own validation has focused on), and it means a
packed grid is already fit-ready with no per-fit LSF bookkeeping needed.

**Source**: the user's own downloaded E-MILES release
(`EMILES_BASTI_BASE_BI_FITS.tar.gz`, BaSTI isochrones, baseFe/scaled-solar,
no alpha-enhancement variants), read directly via `astropy.io.fits` --
consistent with this package's own no-pPXF-dependency stance (see
`prepare_emiles_templates.py` in `dev_notes/emiles/`, the earlier,
narrower 25-template prototype this module supersedes for production use).
"""
from __future__ import annotations

import io
import re
import tarfile

import numpy as np
from astropy.io import fits

from .lsf import match_gaussian_lsf, muse_lsf_fwhm

_EMILES_FILENAME_RE = re.compile(
    r"^E(?P<imf_type>[a-z]{2})(?P<imf_slope>[0-9.]+)"
    r"Z(?P<metal_sign>[mp])(?P<metal>[0-9.]+)"
    r"T(?P<age>[0-9.]+)_iTp0\.00_baseFe\.fits$"
)


def parse_emiles_filename(name: str) -> dict | None:
    """Parse an E-MILES FITS member's base filename (e.g.
    `Ebi1.30Zp0.06T01.0000_iTp0.00_baseFe.fits`) into
    `{imf_type, imf_slope, metal, age}` (`metal` already signed). Returns
    `None` for a non-conforming name (e.g. an alpha-enhancement variant
    this package doesn't select) rather than raising -- callers filter on
    `None`, matching `pack_emiles_templates`'s own use.
    """
    m = _EMILES_FILENAME_RE.match(name)
    if m is None:
        return None
    sign = -1.0 if m.group("metal_sign") == "m" else 1.0
    return dict(
        imf_type=m.group("imf_type"),
        imf_slope=float(m.group("imf_slope")),
        metal=sign * float(m.group("metal")),
        age=float(m.group("age")),
    )


def pack_emiles_templates(tar_path: str, out_path: str,
                           imf_slope: str = "1.30", imf_type: str = "bi",
                           ages_gyr: list[float] | None = None,
                           metals: list[float] | None = None,
                           target_lsf_fwhm=muse_lsf_fwhm, template_fwhm: float = 2.51,
                           wave_range: tuple[float, float] | None = None,
                           verbose: bool = True) -> str:
    """Build a portable `.npz` E-MILES template grid from the user's own
    raw FITS release, LSF-matched to `target_lsf_fwhm` at pack time.

    Parameters
    ----------
    tar_path : path to the raw `EMILES_..._FITS.tar.gz` (or an already-
        extracted directory of `*.fits` files -- both accepted).
    out_path : where to write the packed `.npz`.
    imf_slope, imf_type : select one IMF (default bimodal, slope 1.30, the
        standard reference choice) -- E-MILES ships many IMF slopes/types
        per (age, metallicity) point; packing more than one requires a
        4th axis this module does not build (deliberately -- keep the
        packed grid the simplest thing that covers the actual use case,
        which is age/metallicity coverage, not IMF exploration).
    ages_gyr, metals : explicit lists to restrict which grid points get
        packed. Default `None` for both packs the *entire* native grid at
        this IMF (53 ages x 12 metallicities = 636 templates for the
        standard E-MILES/BaSTI/baseFe release) -- deliberately dense,
        for use cases like a template-mismatch/weight-sparsity diagnostic
        that need many candidate templates, not a small curated subset.
    target_lsf_fwhm : float or callable(wavelength) -> FWHM (Angstrom).
        Default MUSE's own wavelength-dependent curve
        (`lsf.muse_lsf_fwhm`); pass a different instrument's curve (or a
        constant, e.g. for HST STIS) to pack a grid matched to that
        instrument instead.
    template_fwhm : E-MILES's own native FWHM (Angstrom), constant per
        Vazdekis et al. across the MILES/MIUSCAT-built part of the grid
        (3540-8950A) -- see this package's own `dev_notes/emiles/
        prepare_emiles_templates.py` docstring for the caveat on trusting
        this redward of ~8950A, still applicable here.
    wave_range : optional `(wave_min, wave_max)` to trim the packed grid
        to (E-MILES's native range is far wider, ~1680-50000A, most of
        which is irrelevant to any optical/near-IR fit window). Default
        `None` keeps the full native range.
    verbose : print progress.

    Returns
    -------
    `out_path`, for chaining.
    """
    import os

    wanted_imf_slope = float(imf_slope)
    members: list[tuple[str, np.ndarray, np.ndarray]] = []  # (name, wave, flux)

    def _process(name: str, wave: np.ndarray, flux: np.ndarray, meta: dict):
        if wave_range is not None:
            mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
            wave, flux = wave[mask], flux[mask]
        members.append((name, wave, flux, meta))

    if os.path.isdir(tar_path):
        import glob
        for fpath in sorted(glob.glob(os.path.join(tar_path, "*.fits"))):
            base = os.path.basename(fpath)
            meta = parse_emiles_filename(base)
            if meta is None or meta["imf_type"] != imf_type or meta["imf_slope"] != wanted_imf_slope:
                continue
            if ages_gyr is not None and not any(abs(meta["age"] - a) < 1e-6 for a in ages_gyr):
                continue
            if metals is not None and not any(abs(meta["metal"] - z) < 1e-6 for z in metals):
                continue
            with fits.open(fpath) as hdul:
                header = hdul[0].header
                flux = hdul[0].data.astype(float)
            wave = header["CRVAL1"] + np.arange(header["NAXIS1"]) * header["CDELT1"]
            _process(base, wave, flux, meta)
    else:
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf:
                base = os.path.basename(member.name)
                meta = parse_emiles_filename(base)
                if meta is None or meta["imf_type"] != imf_type or meta["imf_slope"] != wanted_imf_slope:
                    continue
                if ages_gyr is not None and not any(abs(meta["age"] - a) < 1e-6 for a in ages_gyr):
                    continue
                if metals is not None and not any(abs(meta["metal"] - z) < 1e-6 for z in metals):
                    continue
                data = tf.extractfile(member).read()
                with fits.open(io.BytesIO(data)) as hdul:
                    header = hdul[0].header
                    flux = hdul[0].data.astype(float)
                wave = header["CRVAL1"] + np.arange(header["NAXIS1"]) * header["CDELT1"]
                _process(base, wave, flux, meta)

    if not members:
        raise ValueError(
            f"pack_emiles_templates: no matching FITS members found in {tar_path!r} for "
            f"imf_type={imf_type!r}, imf_slope={imf_slope!r} -- check the path and IMF selection."
        )

    # All native E-MILES members at a fixed IMF/isochrone/baseFe selection
    # share one identical wavelength grid -- verify rather than assume,
    # since a silent mismatch here would misassign flux to the wrong
    # wavelength for whichever templates don't match.
    wave0 = members[0][1]
    for name, wave, _, _ in members[1:]:
        if not np.allclose(wave, wave0):
            raise ValueError(
                f"pack_emiles_templates: {name} has a different wavelength grid than "
                f"{members[0][0]} -- expected every member to share one native grid."
            )

    names, ages, metals_out, fluxes = [], [], [], []
    n_cannot_sharpen_total = 0
    for i, (name, wave, flux, meta) in enumerate(members):
        matched, cannot_sharpen = match_gaussian_lsf(
            wave, flux, template_fwhm=template_fwhm, target_fwhm=target_lsf_fwhm,
        )
        n_cannot_sharpen_total += cannot_sharpen.sum()
        pos = matched > 0
        med = np.nanmedian(matched[pos]) if pos.any() else 1.0
        normalized = matched / med if med > 0 else matched
        names.append(name)
        ages.append(meta["age"])
        metals_out.append(meta["metal"])
        fluxes.append(normalized.astype(np.float32))
        if verbose and (i + 1) % 100 == 0:
            print(f"  LSF-matched {i + 1}/{len(members)}", flush=True)

    flux_arr = np.stack(fluxes, axis=1)  # (n_pix, n_templates), matching the OG package's own convention
    frac_unmatched = n_cannot_sharpen_total / (len(members) * len(wave0))
    source = (
        f"E-MILES BaSTI/baseFe, IMF {imf_type}{imf_slope}, {len(members)} templates "
        f"({len(set(ages))} ages x {len(set(metals_out))} metallicities), packed by "
        f"kinextract.emiles.pack_emiles_templates, LSF-matched to "
        f"{'MUSE (wavelength-dependent)' if target_lsf_fwhm is muse_lsf_fwhm else 'a custom curve'} "
        f"(cannot-sharpen fraction {frac_unmatched:.3f})."
    )
    np.savez(
        out_path, wavelength=wave0.astype(np.float64), flux=flux_arr,
        names=np.array(names), ages=np.array(ages), metals=np.array(metals_out),
        imf_slope=np.full(len(members), wanted_imf_slope), source=np.array(source),
    )
    if verbose:
        print(f"wrote {len(members)} templates to {out_path}")
        print(f"cannot-sharpen fraction: {frac_unmatched:.4f}")
    return out_path


def load_emiles_templates(npz_path: str, select: list[tuple[float, float]] | None = None,
                           atol: float = 1e-3) -> tuple[list[np.ndarray], np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Load a packed E-MILES grid (from `pack_emiles_templates`), ready to
    hand directly to `forward_model.build_template_stack` alongside any
    other template list.

    Parameters
    ----------
    npz_path : path from `pack_emiles_templates`.
    select : optional list of `(age_gyr, metallicity)` tuples to restrict
        to a subset (e.g. a physically-motivated curated selection for a
        specific target). Matched with `np.isclose(..., atol=atol)`,
        *not* nearest-neighbor -- raises `ValueError` listing every
        available `(age, metal)` pair if any requested pair isn't found,
        rather than silently substituting the closest one. `None`
        (default) returns every packed template.
    atol : tolerance for the `select` match (default 1e-3, well under the
        packed grid's own age/metallicity spacing).

    Returns
    -------
    `(templates, wavelength, names, ages, metals)` -- `templates` is a
    list of 1-D flux arrays (this package's own template-list convention,
    e.g. as used with the real 33-star MUSE library everywhere else),
    `wavelength` the shared rest-frame grid, `names`/`ages`/`metals`
    parallel metadata for whichever templates were selected.
    """
    d = np.load(npz_path, allow_pickle=False)
    wavelength = d["wavelength"]
    flux = d["flux"]  # (n_pix, n_templates)
    names_all = d["names"]
    ages_all = d["ages"]
    metals_all = d["metals"]

    if select is None:
        idx = np.arange(flux.shape[1])
    else:
        idx = []
        available = list(zip(ages_all.tolist(), metals_all.tolist()))
        for age, metal in select:
            matches = np.where(np.isclose(ages_all, age, atol=atol) & np.isclose(metals_all, metal, atol=atol))[0]
            if len(matches) == 0:
                raise ValueError(
                    f"load_emiles_templates: no packed template matches (age={age}, metal={metal}) "
                    f"within atol={atol} -- available (age, metal) pairs: {sorted(set(available))}"
                )
            idx.append(int(matches[0]))
        idx = np.array(idx)

    templates = [flux[:, i].astype(np.float64) for i in idx]
    names = [str(names_all[i]) for i in idx]
    return templates, wavelength, names, ages_all[idx], metals_all[idx]
