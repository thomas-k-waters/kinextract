"""NGC 4751 NFM: is the bnn-side LOSVD "wing" (FINDINGS 30) made by the processing of the NFM cube on disk?

Compares the processed cube (`simple_extraction.CONFIG["NFM"]["cube"]`, empty primary header: rewritten, reportedly
ZAP + spaxel repair) with the raw ESO MUSE-DEEP product (data/N4751/MUSE/NFM/raw/), using one spaxel at a time:
  1. the difference cube (processed - raw) inside the fit window: where it is non-zero, and whether its spectrum
     has CaT-line structure (galaxy absorption removed or added);
  2. the winged bins and their mirrors re-extracted from the raw cube (same masks, bins, templates), refitted at
     lam = 4096 and compared with the processed-cube fits.

    python raw_vs_processed.py <outdir> [--jobs 4]
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/waterstk/Documents/Nagar_Astrophysics/EHT_Project/notebooks/N4751_notebooks/kinematics")
import simple_extraction as S  # noqa: E402

RAW = S.EHT / "data/N4751/MUSE/NFM/raw/ADP.2022-12-13T15:54:49.001.fits"
PAIRS = ("0403", "0402", "0501", "0502", "0601", "0802", "0904", "1102")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("~/.mplstyle")

    # ---- 1. difference cube in the fit window
    from astropy.io import fits

    def window(path):
        """The fit-window slice of DATA (memory-mapped: only these channels are read) and the rest wavelengths."""
        with fits.open(path, memmap=True) as f:
            h = f["DATA"].header
            wave = h["CRVAL3"] + (np.arange(h["NAXIS3"]) + 1 - h["CRPIX3"]) * h.get("CD3_3", h.get("CDELT3"))
            wr = wave / (1 + S.Z)
            sel = np.flatnonzero((wr >= S.WAVEFITMIN) & (wr <= S.WAVEFITMAX))
            return np.array(f["DATA"].data[sel[0]:sel[-1] + 1], dtype=float), wr[sel[0]:sel[-1] + 1], abs(h.get("CD2_2", 0)) * 3600
    fp, w, pix = window(str(S.CONFIG["NFM"]["cube"]))
    fr, w2, _ = window(str(RAW))
    assert np.allclose(w, w2) and fp.shape == fr.shape, "grids differ"
    d = fp - fr
    with np.errstate(invalid="ignore"):
        rel = np.nanmedian(np.abs(d), axis=0) / np.nanmedian(np.abs(fr), axis=0)
        changed = np.nanmax(np.abs(d), axis=0) > 0
    cy, cx = S.CONFIG["NFM"]["centre_yx"]
    yy, xx = np.mgrid[:rel.shape[0], :rel.shape[1]]
    pix = pix or 0.025
    r = np.hypot(yy - cy, xx - cx) * pix
    print(f"spaxels changed: {np.nansum(changed)} of {np.isfinite(rel).sum()}; "
          f"median |proc - raw| / |raw| in the window: {np.nanmedian(rel):.2e}")
    for lo, hi in ((0, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 6.0)):
        s = (r >= lo) & (r < hi) & np.isfinite(rel)
        print(f"  r {lo}-{hi}\": median rel. change {np.nanmedian(rel[s]):.2e}, 99th pct {np.nanpercentile(rel[s], 99):.2e}")
    # the mean difference spectrum (relative) inside 1", split by the two halves along the major axis
    t = np.radians(S.PA_NE)
    along = ((xx - cx) * np.cos(t + np.pi / 2) + (yy - cy) * np.sin(t + np.pi / 2)) * pix   # signed major-axis coordinate
    fig, ax = plt.subplots(2, 1, figsize=(13, 7))
    im = ax[0].imshow(np.log10(rel), origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax[0], label="log10 median |proc - raw| / |raw|")
    ax[0].plot(cx, cy, "w+"); ax[0].set_title("NFM: processed - raw, 8300-8850 A rest")
    for lab, sel, col in (("major-axis half +", (r < 1.0) & (along > 0), "tab:red"),
                          ("major-axis half -", (r < 1.0) & (along < 0), "tab:blue")):
        num = np.nansum(d[:, sel], axis=1); den = np.nansum(fr[:, sel], axis=1)
        ax[1].plot(w, num / np.nanmedian(den), color=col, lw=0.8, label=lab)
    ax[1].set(xlabel=r"rest wavelength [\AA]", ylabel="(proc - raw) / median raw, r < 1\"")
    ax[1].legend(); plt.tight_layout(); fig.savefig(out / "nfm_processed_minus_raw.png", dpi=100)
    del d, fp, fr
    np.save(out / "nfm_rel_change_map.npy", rel)

    # ---- 2. re-extract and refit the winged bins from the raw cube
    bins_p, meta = S.load("NFM")
    names = [p + n for n in PAIRS for p in ("bin", "bnn")]
    cfg = copy.deepcopy(S.CONFIG["NFM"])
    S.CONFIG["NFM_RAW"] = dict(cfg, cube=RAW)
    inp = S.load_inputs("NFM_RAW")
    keep = {b["bin_name"] for b in bins_p}
    inp["bin_table"] = [row for row in inp["bin_table"] if row["bin_name"] in names and row["bin_name"] in keep]
    bins_r = S.extract_spectra("NFM_RAW", inp)
    del inp
    T, tw, _ = S.base_templates("NFM")
    S.fit_bins("NFM_RAW", bins_r, T, tw, lam_override={b["bin_name"]: 4096.0 for b in bins_r}, n_jobs=a.jobs)
    P = {b["bin_name"]: b for b in pickle.load(open(S.OUT_ROOT / "regv2_check/NFM_fixed.pkl", "rb"))}   # lam 4096
    vsys = json.load(open(S.VSYS_FILE))["v_sys_kms"]

    def far(b):
        v = b["v_grid"] - vsys; bb = np.maximum(b["fit"]["b"], 0); bb = bb / np.trapezoid(bb, v)
        return 100 * np.trapezoid(bb[np.abs(v) > 1500], v[np.abs(v) > 1500])
    print("\nlam 4096 fits, processed | raw cube:  V - Vsys, sigma, h3, % light beyond 1500 km/s")
    for b in sorted(bins_r, key=lambda x: (x["bin_name"][3:], x["bin_name"])):
        q = P[b["bin_name"]]
        f = lambda x: (f"{x['fit']['moments']['vherm'] - vsys:+6.0f} {x['fit']['moments']['sherm']:5.0f} "  # noqa: E731
                       f"{x['fit']['moments']['h3']:+.3f} {far(x):5.1f}%")
        print(f"  {b['bin_name']}: {f(q)}  |  {f(b)}   (S/N proc {np.median(q['flux_fit'] / q['errors']):.0f}, "
              f"raw {np.median(b['flux_fit'] / b['errors']):.0f})")
    pickle.dump(bins_r, open(out / "nfm_raw_winged_bins.pkl", "wb"))


if __name__ == "__main__":
    main()
