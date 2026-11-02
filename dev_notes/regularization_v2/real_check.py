"""Real-data check of the step-1/3 choice (fixed lam 4096 + noise=skysplit + template error + model-bin errors) on the
NGC 4751 bins of kinematics/simple_extraction.py (same spectra, masks, templates and velocity grids).

    python real_check.py NFM <outdir> [--jobs 7]

Writes <outdir>/<ds>_fixed.pkl (the fixed-lam bins, with plain and skysplit bands and template refits) and three
spmc_in sets (aic = the cached adaptive-lam extraction, fixed = plain Laplace bands, split = skysplit + template spread
+ model-bin errors), then prints the bin/bnn pair chi2 per floor range and the moments of the known-problem bins.
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/waterstk/Documents/Nagar_Astrophysics/EHT_Project/notebooks/N4751_notebooks/kinematics")

LAM = 4096.0
LOG = None
WATCH = ("bnn1203", "bin1203", "bnn1103", "bin1103", "bnn0904", "bin0904", "bnn1201", "bin1201", "bin0105", "bnn0105",
         "bin0402", "bnn0402", "bnn1105", "bin1105")


def _split_one(args):
    """noise=skysplit bands for one bin: noise-model errors, Laplace at the adopted lam with the inflated errors scaled
    by err_nm/err, draws moved onto the plain fit."""
    warnings.filterwarnings("ignore")
    import dataclasses
    import simple_extraction as S
    from bench_dev import noise_model_errors
    from kinextract import build_template_stack, laplace_losvd_uncertainty_parallel
    from kinextract.noise_model import whiten
    b, T, tw, seed = args
    f = b["fit"]
    cfg = S._cfg(b, lam=f["lam_used"], rho=f["rho_used"])
    A = build_template_stack(b["wave_fit"], tw, T, b["v_grid"])
    err_nm, kappa, fmis = noise_model_errors(A, b["flux_fit"], b["errors"], b["wave_fit"],
                                             dataclasses.replace(cfg, rho=0.0))
    A_w, y_w, _ = whiten(A, b["flux_fit"], f["errors_inflated"] * err_nm / b["errors"], rho=cfg.rho)
    lap = laplace_losvd_uncertainty_parallel([A_w], [y_w], [cfg], n_samples=300, seed=seed, n_jobs=1,
                                             wave_fit=[b["wave_fit"]], n_continuum_knots=S.FIT_KW["continuum_knots"],
                                             lam_c=S.FIT_KW["lam_c"], ridge_c=S.FIT_KW["ridge_c"],
                                             clipped_mask=[f["clipped_mask"]])[0]
    v = b["v_grid"]
    b0 = np.maximum(f["b"], 0) / np.trapezoid(np.maximum(f["b"], 0), v)
    ratio = np.clip(b0 / np.maximum(lap["b_median"], 1e-300), 1e-3, 1e3)
    bs = lap["b_samples"] * ratio[None, :]
    bs = bs / np.trapezoid(bs, v, axis=1)[:, None]
    return dict(b_median=np.median(bs, 0), b_p16=np.percentile(bs, 16, 0), b_p84=np.percentile(bs, 84, 0),
                b_samples=bs[:200].astype(np.float32), kappa=float(kappa), f=float(fmis),
                infl=float(np.median(err_nm / b["errors"])))


def write_set(ds, bins, vsys, outdir, mode):
    """mode: 'plain' (Laplace bands, as S.write_spmc without spreads), 'aic' (S.error_bands with lam + template
    spreads, the cached extraction's own), 'split' (skysplit bands + template spread, model-bin errors)."""
    import simple_extraction as S
    from kinextract.legacy_io import write_spmc_in_laplace
    os.makedirs(outdir, exist_ok=True)
    for p in Path(outdir).glob("*spmc_in"):
        p.unlink()
    for b in bins:
        if "fit" not in b or b["fit"]["edge"] > S.EDGE_MAX:
            continue
        v = b["v_grid"] - vsys
        kw = {}
        if mode == "plain":
            L = b["laplace"]; med, lo, hi = L["b_median"], L["b_p16"], L["b_p84"]
        elif mode == "aic":
            med, lo, hi = S.error_bands(b, use_lambda=True, use_template=True)
        else:
            L = b["split"]
            tsp = S.template_spread(b) if "notmpl" not in mode else 0.0
            med = L["b_median"]
            lo = med - np.sqrt((med - L["b_p16"]) ** 2 + tsp ** 2)
            hi = med + np.sqrt((L["b_p84"] - med) ** 2 + tsp ** 2)
            if "nomb" not in mode:
                kw = dict(b_samples=L["b_samples"].astype(float), model_bins=(S.MODEL_VBINS, 150.0))
        write_spmc_in_laplace(S.CONFIG[ds]["prefix"] + b["bin_name"], v, med, lo, hi, str(outdir),
                              vmax_out=S.SPMC_VMAX, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("outdir"); ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--sig6", action="store_true",
                    help="also refit bins with sigma > 350 at lam = 4096 (sigma/350)^6 (cap 131072), from the fixed cache")
    a = ap.parse_args()
    import simple_extraction as S
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    bins_aic, _ = S.load(a.ds)
    T, tw, _ = S.base_templates(a.ds)
    cache = out / f"{a.ds}_fixed.pkl"
    global LOG
    LOG = str(out / f"{a.ds}_progress.log")
    t0 = time.time()
    if cache.exists():
        bins = pickle.load(open(cache, "rb"))
    else:
        bins = [{k: copy.deepcopy(v) for k, v in b.items() if k not in ("fit", "laplace", "variants", "lam_mode")}
                for b in bins_aic]
        S.fit_bins(a.ds, bins, T, tw, lam_override={b["bin_name"]: LAM for b in bins}, n_jobs=a.jobs, log=LOG)
        for fac in (2.0, 4.0):                     # retry ladder for fits that reach the grid edge
            bad = [b["bin_name"] for b in bins if b["fit"]["edge"] > S.EDGE_MAX]
            print(f"edge > {S.EDGE_MAX}: {len(bad)} bins -> lam x{fac:g}", flush=True)
            if not bad:
                break
            S.fit_bins(a.ds, bins, T, tw, lam_override={n: LAM * fac for n in bad}, n_jobs=a.jobs, only=bad,
                       log=LOG)
        S.fit_variants(a.ds, bins, S.template_sets(a.ds), n_jobs=a.jobs, log=LOG)
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.jobs) as ex:
            for b, r in zip(bins, ex.map(_split_one, [(b, T, tw, i) for i, b in enumerate(bins)])):
                b["split"] = r
        print(f"split bands done ({time.time() - t0:.0f} s)", flush=True)
        pickle.dump(bins, open(cache, "wb"))

    if a.sig6:
        cache6 = out / f"{a.ds}_sig6.pkl"
        if cache6.exists():
            bins6 = pickle.load(open(cache6, "rb"))
        else:
            bins6 = copy.deepcopy(bins)
            lam6 = {b["bin_name"]: float(np.clip(LAM * (b["fit"]["moments"]["sherm"] / 350.0) ** 6, LAM, 131072.0))
                    for b in bins6 if b["fit"]["moments"]["sherm"] > 350.0}
            print(f"sig6: refitting {len(lam6)} bins with sigma > 350", flush=True)
            names = list(lam6)
            S.fit_bins(a.ds, bins6, T, tw, lam_override=lam6, n_jobs=a.jobs, only=names, log=LOG)
            S.fit_variants(a.ds, bins6, S.template_sets(a.ds), n_jobs=a.jobs, only=names, log=LOG)
            from concurrent.futures import ProcessPoolExecutor
            sel = [b for b in bins6 if b["bin_name"] in lam6]
            with ProcessPoolExecutor(a.jobs) as ex:
                for b, r in zip(sel, ex.map(_split_one, [(b, T, tw, i) for i, b in enumerate(sel)])):
                    b["split"] = r
            pickle.dump(bins6, open(cache6, "wb"))

    vs = S.vsys_from_pairs(bins, {})
    vsys = vs["v_sys_kms"]
    print(f"V_sys from the fixed-lam pairs: {vsys:.2f} km/s ({vs['n_pairs']} pairs, 16-84 {vs['pair_midpoint_16_84']})")
    sets = {"aic": (bins_aic, "aic"), "fixed": (bins, "plain"), "split": (bins, "split"),
            "split_notmpl": (bins, "split_notmpl"), "split_nomb": (bins, "split_nomb"),
            "split_notmpl_nomb": (bins, "split_notmpl_nomb")}
    if a.sig6:
        sets.update({"sig6": (bins6, "plain"), "sig6_split": (bins6, "split"),
                     "sig6_split_notmpl": (bins6, "split_notmpl")})
    rows = {}
    for name, (bb, mode) in sets.items():
        d = out / f"spmc_{name}"
        write_set(a.ds, bb, vsys, d, mode)
        rows[name] = S.pair_chi2(a.ds, bb, d)
    print("\npair chi2 per model bin by floor range (n pairs):")
    for name, rr in rows.items():
        print(f"  {name:6s}", "  ".join(f"[{lo:g},{hi:g}) {c:6.2f} ({n})" for (lo, hi), (n, c, _) in
                                     zip(S.CONFIG[a.ds]["floor_ranges"], S.floor_scales(a.ds, rr))),
              f"| all {sum(x['chi2'] for x in rr) / max(sum(x['n'] for x in rr), 1):.2f} ({len(rr)})")
    print("\npair shape difference (rms of S_bin - S_bnn over the 13 model bins; error-free), median by floor range:")
    for name in ("aic", "fixed") + (("sig6",) if a.sig6 else ()):
        d = out / f"spmc_{name}"; p = S.CONFIG[a.ds]["prefix"]; r = {b["bin_name"]: abs(b["r_mid"]) for b in bins}
        dd = []
        for f in sorted(d.glob(f"{p}bin*spmc_in")):
            g = d / f.name.replace(f"{p}bin", f"{p}bnn", 1)
            if g.exists():
                dd.append((r[f.name[len(p):-len("spmc_in")]], np.sqrt(np.mean((S.spmc_binned(f)[0] - S.spmc_binned(g)[0]) ** 2))))
        dd = np.array(dd)
        print(f"  {name:6s}", "  ".join(f"[{lo:g},{hi:g}) {np.median(dd[(dd[:, 0] >= lo) & (dd[:, 0] < hi), 1]):.4f}"
                                     for lo, hi in S.CONFIG[a.ds]["floor_ranges"]), f"| all {np.median(dd[:, 1]):.4f}")
    print("\nwatch list: V sigma h3 h4 (lam, edge) -- aic | fixed" + (" | sig6" if a.sig6 else ""))
    ba = {b["bin_name"]: b for b in bins_aic}; bf = {b["bin_name"]: b for b in bins}
    b6 = {b["bin_name"]: b for b in bins6} if a.sig6 else {}
    for nm in WATCH:
        if nm not in bf:
            continue
        line = [nm]
        for b in (ba.get(nm), bf[nm]) + ((b6[nm],) if a.sig6 else ()):
            m = b["fit"]["moments"]
            line.append(f"{m['vherm']:7.1f} {m['sherm']:6.1f} {m['h3']:+.3f} {m['h4']:+.3f} "
                        f"({b['fit']['lam_used']:.0f}, {b['fit']['edge']:.3f})")
        print("  " + " | ".join(line))
    inf = np.array([b["split"]["infl"] for b in bins]); fm = np.array([b["split"]["f"] for b in bins])
    print(f"\nnoise model: median error inflation {np.median(inf):.2f} (16-84 {np.percentile(inf, 16):.2f}-"
          f"{np.percentile(inf, 84):.2f}), mismatch f median {np.median(fm):.4f}")


if __name__ == "__main__":
    main()
