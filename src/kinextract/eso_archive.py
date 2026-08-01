"""ESO Science Archive query and download for MUSE Phase 3 datacubes --
generalized to any target, not hardcoded to any specific galaxy.

**Optional dependency, imported lazily.** `astroquery` is only needed for
this module's two functions, not for anything else in this package (the
synthetic mock and the shipped compact products need no network access
at all) -- declared as the `eso` extra in `pyproject.toml`
(`pip install kinextract[eso]`), and imported inside each function
body rather than at module level, so importing `kinextract` itself
never requires it.

**Off by default, on purpose.** A MUSE-DEEP Phase 3 cube is commonly
several GB. `download_dataset` refuses to pull anything unless called
with `confirm=True` -- this is the actual mechanism behind every
notebook's own `CONFIRM_DOWNLOAD` flag, not just a suggestion in
documentation.
"""
from __future__ import annotations


def query_muse_phase3(target: str | None = None, program_id: str | None = None,
                       ra: float | None = None, dec: float | None = None,
                       radius_arcmin: float = 5.0):
    """Query ESO's MUSE-DEEP Phase 3 collection.

    Parameters
    ----------
    target : a SIMBAD/NED-resolvable name (e.g. `"NGC5102"`) or the
        original PI-assigned target designation. Mutually exclusive with
        `ra`/`dec` in practice (both can be given, but `target` alone is
        the common case).
    program_id : restrict to a specific ESO program/run ID (e.g.
        `"60.A-9308(A)"`). `None` (default) doesn't filter on it.
    ra, dec : degrees (J2000), for a position-based cone search instead
        of a target name.
    radius_arcmin : cone-search radius, only used if `ra`/`dec` are given.

    Returns
    -------
    `astropy.table.Table` (possibly with zero rows) of matching Phase 3
    products -- see `ARCFILE` for the dataset id `download_dataset`
    needs, and `Exptime`/`Wavelength`/`Object`/`Run/Program ID` for
    identifying the right row among several.
    """
    from astroquery.eso import Eso  # lazy: only this function needs it

    eso = Eso()
    kwargs = {}
    if target is not None:
        kwargs["target"] = target
    if program_id is not None:
        kwargs["prog_id"] = program_id
    if ra is not None and dec is not None:
        kwargs["coord_sys"] = "eq"
        kwargs["coord1"] = ra
        kwargs["coord2"] = dec
        kwargs["box"] = radius_arcmin / 60.0  # arcmin -> degrees

    return eso.query_surveys(surveys="MUSE-DEEP", **kwargs)


def download_dataset(dp_id: str, dest_dir: str, confirm: bool = False,
                      size_estimate_gb: float | None = None) -> list[str]:
    """Download one dataset (by `ARCFILE`/`DP.ID`, from `query_muse_phase3`'s
    own result table) from the ESO archive.

    Parameters
    ----------
    dp_id : the dataset identifier (`ARCFILE` column from
        `query_muse_phase3`'s result, e.g.
        `"ADP.2017-03-22T15:25:58.757"`).
    dest_dir : directory to save the downloaded file(s) into.
    confirm : default `False` -- raises `ValueError` instead of
        downloading anything. Real MUSE-DEEP cubes are commonly several
        GB; this is the actual gate behind every notebook's own
        `CONFIRM_DOWNLOAD` flag, not just documentation.
    size_estimate_gb : if given, included in the refusal message when
        `confirm=False`, so the caller sees a concrete size before
        deciding.

    Returns
    -------
    List of local file paths downloaded.
    """
    if not confirm:
        size_note = f" (~{size_estimate_gb:.1f} GB)" if size_estimate_gb is not None else ""
        raise ValueError(
            f"download_dataset: refusing to download {dp_id!r}{size_note} without confirm=True -- "
            "MUSE Phase 3 cubes are typically several GB and this can take significant time/bandwidth. "
            "Pass confirm=True once you've decided to proceed."
        )
    from astroquery.eso import Eso  # lazy: only this function needs it

    eso = Eso()
    files = eso.retrieve_data(dp_id, destination=dest_dir)
    return [files] if isinstance(files, str) else list(files)
